/* Packed weight files: every array of a loaded model, in visitor order, behind
 * a small header. A model loaded from one maps the file read only, so start-up
 * is a mmap and several processes share the same page-cache copy. The packing
 * layout is architecture specific, so the header names it and a mismatch is
 * refused. */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "model.h"
#include "util.h"

#define PK_PACKED_MAGIC "PKWEIGHT"
#define PK_PACKED_VERSION 1
#define ALIGN 64

typedef struct {
    char magic[8];
    uint32_t version;
    char layout[32];
    uint32_t fast, has_vad, cache_n, n_arrays;
    uint64_t data_offset;
} packed_header;

typedef struct {
    uint64_t offset, bytes;
} packed_entry;

void pk_model_arrays(pk_model *m, pk_array_fn fn, void *user) {
    pk_features_arrays(&m->feats, fn, user);
    pk_subsampler_arrays(&m->sub, fn, user);
    pk_encoder_arrays(&m->enc, fn, user);
    pk_decoder_arrays(&m->dec, fn, user);
    if (m->has_vad) pk_vad_arrays(&m->vad, fn, user);
}

/* ---- saving ------------------------------------------------------------------ */

typedef struct {
    packed_entry *entries;
    void **arrays;
    uint32_t n, cap;
    uint64_t offset;
} collect_ctx;

static void collect(void *user, void **array, size_t bytes) {
    collect_ctx *c = user;
    if (c->n == c->cap) {
        c->cap = c->cap ? c->cap * 2 : 256;
        c->entries = xrealloc(c->entries, c->cap * sizeof *c->entries);
        c->arrays = xrealloc(c->arrays, c->cap * sizeof *c->arrays);
    }
    c->entries[c->n].offset = c->offset;
    c->entries[c->n].bytes = bytes;
    c->arrays[c->n] = *array;
    c->n++;
    c->offset += (bytes + ALIGN - 1) / ALIGN * ALIGN;
}

int pk_model_save(const pk_model *m, const char *path) {
    collect_ctx c = {0};
    pk_model_arrays((pk_model *)m, collect, &c);
    packed_header h = {{0}, PK_PACKED_VERSION, "", (uint32_t)m->fast, (uint32_t)m->has_vad, (uint32_t)m->enc.cache_n, c.n, 0};
    memcpy(h.magic, PK_PACKED_MAGIC, 8);
    strncpy(h.layout, pk_weight_layout(), sizeof h.layout - 1);
    uint64_t table = sizeof h + (uint64_t)c.n * sizeof(packed_entry);
    h.data_offset = (table + ALIGN - 1) / ALIGN * ALIGN;
    FILE *f = fopen(path, "wb");
    if (!f) {
        pk_set_error("cannot create %s", path);
        free(c.entries);
        free(c.arrays);
        return -1;
    }
    int ok = fwrite(&h, sizeof h, 1, f) == 1 && fwrite(c.entries, sizeof(packed_entry), c.n, f) == c.n;
    static const char zeros[ALIGN];
    uint64_t pos = table;
    while (ok && pos < h.data_offset) {
        size_t pad = (size_t)(h.data_offset - pos) < ALIGN ? (size_t)(h.data_offset - pos) : ALIGN;
        ok = fwrite(zeros, 1, pad, f) == pad;
        pos += pad;
    }
    for (uint32_t i = 0; ok && i < c.n; i++) {
        uint64_t want = h.data_offset + c.entries[i].offset;
        while (ok && pos < want) {
            size_t pad = (size_t)(want - pos) < ALIGN ? (size_t)(want - pos) : ALIGN;
            ok = fwrite(zeros, 1, pad, f) == pad;
            pos += pad;
        }
        if (c.entries[i].bytes) ok = fwrite(c.arrays[i], 1, c.entries[i].bytes, f) == c.entries[i].bytes;
        pos += c.entries[i].bytes;
    }
    if (fclose(f) != 0) ok = 0;
    free(c.entries);
    free(c.arrays);
    if (!ok) {
        pk_set_error("cannot write %s", path);
        unlink(path);
        return -1;
    }
    return 0;
}

/* ---- loading ------------------------------------------------------------------ */

static int read_header(const char *path, packed_header *h) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int ok = fread(h, sizeof *h, 1, f) == 1;
    fclose(f);
    if (!ok || memcmp(h->magic, PK_PACKED_MAGIC, 8) != 0 || h->version != PK_PACKED_VERSION) return 0;
    h->layout[sizeof h->layout - 1] = '\0';
    return 1;
}

int pk_packed_header(const char *path, int *fast, int *has_vad, int *cache_n) {
    struct stat st;
    if (stat(path, &st) != 0) return 0; /* nothing there yet */
    packed_header h;
    if (!read_header(path, &h)) {
        pk_set_error("%s exists but is not a packed weight file", path);
        return -1;
    }
    if (strcmp(h.layout, pk_weight_layout()) != 0) {
        pk_set_error("%s was packed for layout %s; this build uses %s (delete it to repack)", path, h.layout, pk_weight_layout());
        return -1;
    }
    *fast = (int)h.fast;
    *has_vad = (int)h.has_vad;
    *cache_n = (int)h.cache_n;
    return 1;
}

typedef struct {
    const uint8_t *data;
    const packed_entry *entries;
    uint32_t n, i;
    uint64_t data_len;
} assign_ctx;

static void assign(void *user, void **array, size_t bytes) {
    assign_ctx *a = user;
    if (a->i >= a->n) die("packed file has too few arrays");
    const packed_entry *e = &a->entries[a->i++];
    if (e->bytes != bytes) die("packed file array %u has %llu bytes, expected %zu", a->i - 1, (unsigned long long)e->bytes, bytes);
    if (e->offset + e->bytes > a->data_len) die("packed file is truncated");
    *array = bytes ? (void *)(a->data + e->offset) : NULL;
}

void pk_model_map_packed(pk_model *m, const char *path, int *fast, int *has_vad, int *cache_n) {
    packed_header h;
    if (!read_header(path, &h)) die("%s is not a packed weight file", path);
    if (strcmp(h.layout, pk_weight_layout()) != 0) die("%s was packed for layout %s, this build uses %s", path, h.layout, pk_weight_layout());
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("cannot open %s", path);
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        die("cannot read %s", path);
    }
    size_t len = (size_t)st.st_size;
    void *map = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) die("cannot map %s into memory", path);
    m->map = map;
    m->map_len = len;
    *fast = (int)h.fast;
    *has_vad = (int)h.has_vad;
    *cache_n = (int)h.cache_n;
    if (sizeof h + (uint64_t)h.n_arrays * sizeof(packed_entry) > len || h.data_offset > len) die("%s is truncated", path);
    assign_ctx a = {(const uint8_t *)map + h.data_offset, (const packed_entry *)((const uint8_t *)map + sizeof h), h.n_arrays, 0, len - h.data_offset};
    pk_model_arrays(m, assign, &a);
    if (a.i != a.n) die("packed file has %u arrays, this model needs %u", a.n, a.i);
}
