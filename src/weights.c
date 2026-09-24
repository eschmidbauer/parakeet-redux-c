#include "weights.h"

#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "util.h"

#define MAGIC "PRDX-BIN"
#define VERSION 1
#define HEADER_SIZE 256
#define ENTRY_SIZE 144

static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static uint64_t rd64(const uint8_t *p) { return (uint64_t)rd32(p) | (uint64_t)rd32(p + 4) << 32; }

static double rdf64(const uint8_t *p) {
    uint64_t bits = rd64(p);
    double v;
    memcpy(&v, &bits, 8);
    return v;
}

void pk_weights_open(pk_weights *w, const char *path) {
    memset(w, 0, sizeof *w);
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("cannot open %s", path);
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < HEADER_SIZE) {
        close(fd);
        die("%s is not a parakeet-redux weights file", path);
    }
    w->map_len = (size_t)st.st_size;
    w->map = mmap(NULL, w->map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (w->map == MAP_FAILED) {
        w->map = NULL;
        die("cannot map %s into memory", path);
    }
    const uint8_t *h = w->map;
    if (memcmp(h, MAGIC, 8) != 0) {
        pk_weights_close(w);
        die("%s is not a parakeet-redux weights file (bad magic)", path);
    }
    uint32_t version = rd32(h + 8);
    if (version != VERSION) {
        pk_weights_close(w);
        die("%s is weights format version %u, this build reads version %d", path, version, VERSION);
    }
    uint32_t n = rd32(h + 12);
    uint64_t table = rd64(h + 16), data = rd64(h + 24);
    w->vocab_size = (int)rd32(h + 32);
    w->blank_id = (int)rd32(h + 36);
    w->max_symbols = (int)rd32(h + 40);
    w->n_durations = (int)rd32(h + 44);
    if (w->n_durations > 8) w->n_durations = 8;
    for (int i = 0; i < 8; i++) w->durations[i] = (int32_t)rd32(h + 48 + 4 * i);
    w->frame_seconds = rdf64(h + 80);
    w->sample_rate = (int)rd32(h + 88);
    if (table + (uint64_t)n * ENTRY_SIZE > w->map_len || data > w->map_len) {
        pk_weights_close(w);
        die("%s is truncated", path);
    }
    w->tensors = xcalloc(n ? n : 1, sizeof *w->tensors);
    w->n_tensors = (int)n;
    for (uint32_t i = 0; i < n; i++) {
        const uint8_t *e = h + table + (uint64_t)i * ENTRY_SIZE;
        pk_tensor *t = &w->tensors[i];
        memcpy(t->name, e, sizeof t->name - 1);
        t->dtype = rd32(e + 80);
        t->ndim = rd32(e + 84);
        for (int d = 0; d < 4; d++) t->dims[d] = rd64(e + 88 + 8 * d);
        uint64_t offset = rd64(e + 120), bytes = rd64(e + 128);
        if (offset + bytes > w->map_len - data) {
            pk_weights_close(w);
            die("%s: tensor %s lies outside the file", path, t->name);
        }
        t->data = h + data + offset;
        t->bytes = (size_t)bytes;
    }
}

void pk_weights_close(pk_weights *w) {
    if (w->map) munmap(w->map, w->map_len);
    free(w->tensors);
    memset(w, 0, sizeof *w);
}

const pk_tensor *pk_weights_find(const pk_weights *w, const char *name) {
    for (int i = 0; i < w->n_tensors; i++)
        if (strcmp(w->tensors[i].name, name) == 0) return &w->tensors[i];
    return NULL;
}

static size_t elem_size(int dtype) { return dtype == PK_DT_F32 ? 4 : dtype == PK_DT_F16 ? 2 : 1; }

const pk_tensor *pk_weights_get(const pk_weights *w, const char *name, int dtype, size_t count) {
    const pk_tensor *t = pk_weights_find(w, name);
    if (!t) die("weights file has no tensor %s", name);
    if ((int)t->dtype != dtype) die("tensor %s has dtype %u, expected %d", name, t->dtype, dtype);
    size_t want = dtype == PK_DT_CODES ? count / 4 : count * elem_size(dtype);
    if (t->bytes != want) die("tensor %s has %zu bytes, expected %zu", name, t->bytes, want);
    return t;
}

float *pk_weights_f32(const pk_weights *w, const char *name, size_t count) {
    const pk_tensor *t = pk_weights_get(w, name, PK_DT_F32, count);
    float *out = xmalloc(count * sizeof(float));
    memcpy(out, t->data, count * sizeof(float));
    return out;
}

float *pk_weights_f32f(const pk_weights *w, size_t count, const char *fmt, ...) {
    char name[128];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof name, fmt, ap);
    va_end(ap);
    return pk_weights_f32(w, name, count);
}
