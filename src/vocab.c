#include "vocab.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "unicode_punct.h"
#include "util.h"

/* Decode one UTF-8 code point; returns its byte length (1 for malformed bytes). */
static int utf8_next(const unsigned char *s, uint32_t *cp) {
    if (s[0] < 0x80) { *cp = s[0]; return 1; }
    if ((s[0] & 0xE0) == 0xC0 && (s[1] & 0xC0) == 0x80) { *cp = ((uint32_t)(s[0] & 0x1F) << 6) | (s[1] & 0x3F); return 2; }
    if ((s[0] & 0xF0) == 0xE0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(s[0] & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        return 3;
    }
    if ((s[0] & 0xF8) == 0xF0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(s[0] & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) | ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        return 4;
    }
    *cp = s[0];
    return 1;
}

static int is_space_cp(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f' || c == 0x85 || c == 0xA0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
           c == 0x3000;
}

static int is_punct_cp(uint32_t c) {
    int lo = 0, hi = PK_PUNCT_RANGE_COUNT - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (c < pk_punct_ranges[mid][0]) hi = mid - 1;
        else if (c > pk_punct_ranges[mid][1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}

char *pk_strip(char *s) {
    unsigned char *p = (unsigned char *)s;
    size_t len = strlen(s);
    /* leading */
    size_t start = 0;
    while (start < len) {
        uint32_t cp;
        int n = utf8_next(p + start, &cp);
        if (!is_space_cp(cp)) break;
        start += (size_t)n;
    }
    /* trailing: scan forward remembering the last non-space end */
    size_t end = start, i = start;
    while (i < len) {
        uint32_t cp;
        int n = utf8_next(p + i, &cp);
        i += (size_t)n;
        if (!is_space_cp(cp)) end = i;
    }
    memmove(s, s + start, end - start);
    s[end - start] = '\0';
    return s;
}

static int all_punct(const char *text) {
    char *copy = xstrdup(text);
    pk_strip(copy);
    const unsigned char *p = (const unsigned char *)copy;
    if (!*p) {
        free(copy);
        return 0;
    }
    int ok = 1;
    while (*p) {
        uint32_t cp;
        p += utf8_next(p, &cp);
        if (!is_punct_cp(cp)) {
            ok = 0;
            break;
        }
    }
    free(copy);
    return ok;
}

/* replace U+2581 (E2 96 81) with a space */
static char *replace_underscore(const char *raw) {
    size_t len = strlen(raw);
    char *out = xmalloc(len + 1);
    size_t o = 0;
    for (size_t i = 0; i < len;) {
        if (i + 2 < len && (unsigned char)raw[i] == 0xE2 && (unsigned char)raw[i + 1] == 0x96 && (unsigned char)raw[i + 2] == 0x81) {
            out[o++] = ' ';
            i += 3;
        } else {
            out[o++] = raw[i++];
        }
    }
    out[o] = '\0';
    return out;
}

static void add_piece(pk_vocab *v, int *cap, char *line) {
    char *sp = strrchr(line, ' ');
    if (!sp) return;
    int id = atoi(sp + 1);
    *sp = '\0';
    if (id < 0) return;
    if (id >= *cap) {
        int ncap = *cap ? *cap : 1024;
        while (ncap <= id) ncap *= 2;
        v->pieces = xrealloc(v->pieces, (size_t)ncap * sizeof *v->pieces);
        memset(v->pieces + *cap, 0, (size_t)(ncap - *cap) * sizeof *v->pieces);
        *cap = ncap;
    }
    pk_piece *p = &v->pieces[id];
    free(p->raw);
    free(p->text);
    p->raw = xstrdup(line);
    p->text = replace_underscore(line);
    p->skip = strcmp(line, "<unk>") == 0 || strcmp(line, "<pad>") == 0 || strcmp(line, "<blk>") == 0;
    p->word_start = strncmp(line, "\xE2\x96\x81", 3) == 0;
    p->is_punct = all_punct(p->text);
    if (strcmp(line, "<blk>") == 0) v->blank_id = id;
    if (id + 1 > v->n) v->n = id + 1;
}

static void finish(pk_vocab *v) {
    for (int i = 0; i < v->n; i++) {
        if (!v->pieces[i].raw) {
            char name[32];
            snprintf(name, sizeof name, "<unused%d>", i);
            v->pieces[i].raw = xstrdup(name);
            v->pieces[i].text = xstrdup(name);
        }
    }
}

void pk_vocab_parse(pk_vocab *v, const char *text, size_t len) {
    memset(v, 0, sizeof *v);
    v->blank_id = -1;
    int cap = 0;
    char line[4096];
    size_t i = 0;
    while (i < len) {
        size_t start = i;
        while (i < len && text[i] != '\n') i++;
        size_t n = i - start;
        if (n >= sizeof line) n = sizeof line - 1;
        memcpy(line, text + start, n);
        line[n] = '\0';
        while (n && (line[n - 1] == '\r')) line[--n] = '\0';
        if (n) add_piece(v, &cap, line);
        i++;
    }
    finish(v);
}

int pk_vocab_load(pk_vocab *v, const char *path) {
    memset(v, 0, sizeof *v);
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    int cap = 0;
    char line[4096];
    v->blank_id = -1;
    while (fgets(line, sizeof line, f)) {
        size_t len = strlen(line);
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (len) add_piece(v, &cap, line);
    }
    fclose(f);
    finish(v);
    return 0;
}

void pk_vocab_free(pk_vocab *v) {
    for (int i = 0; i < v->n; i++) {
        free(v->pieces[i].raw);
        free(v->pieces[i].text);
    }
    free(v->pieces);
    memset(v, 0, sizeof *v);
}

char *pk_vocab_decode(const pk_vocab *v, const int *ids, int n) {
    size_t total = 1;
    for (int i = 0; i < n; i++)
        if (ids[i] >= 0 && ids[i] < v->n && !v->pieces[ids[i]].skip) total += strlen(v->pieces[ids[i]].text);
    char *out = xmalloc(total);
    size_t o = 0;
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= v->n || v->pieces[ids[i]].skip) continue;
        const char *t = v->pieces[ids[i]].text;
        size_t len = strlen(t);
        memcpy(out + o, t, len);
        o += len;
    }
    out[o] = '\0';
    if (out[0] == ' ') memmove(out, out + 1, o);
    return out;
}

typedef struct {
    int *ids;
    int n, cap;
} id_list;

static void push(id_list *l, int id) {
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 16;
        l->ids = xrealloc(l->ids, (size_t)l->cap * sizeof(int));
    }
    l->ids[l->n++] = id;
}

typedef struct {
    pk_word *words;
    int n, cap;
} word_list;

static void flush(const pk_vocab *v, id_list *current, double start, double end, word_list *words) {
    if (!current->n) return;
    char *text = pk_vocab_decode(v, current->ids, current->n);
    pk_strip(text);
    if (*text) {
        if (words->n == words->cap) {
            words->cap = words->cap ? words->cap * 2 : 64;
            words->words = xrealloc(words->words, (size_t)words->cap * sizeof *words->words);
        }
        words->words[words->n].word = text;
        words->words[words->n].start = start;
        words->words[words->n].end = end;
        words->n++;
    } else {
        free(text);
    }
    current->n = 0;
}

int pk_vocab_words(const pk_vocab *v, const int *ids, const int *durations, int n, double frame_seconds, pk_word **out) {
    long frame = 0;
    id_list current = {0};
    word_list words = {0};
    double current_start = 0.0, current_end = 0.0;
    for (int i = 0; i < n; i++) {
        double start = (double)frame * frame_seconds;
        frame += durations[i];
        int id = ids[i];
        if (id < 0 || id >= v->n || v->pieces[id].skip) continue;
        const pk_piece *p = &v->pieces[id];
        if (p->word_start) flush(v, &current, current_start, current_end, &words);
        if (p->is_punct) {
            if (current.n) {
                push(&current, id);
                current_end = (double)frame * frame_seconds;
            } else if (words.n) {
                pk_word *last = &words.words[words.n - 1];
                char *piece = xstrdup(p->text);
                pk_strip(piece);
                size_t a = strlen(last->word), b = strlen(piece);
                last->word = xrealloc(last->word, a + b + 1);
                memcpy(last->word + a, piece, b + 1);
                free(piece);
                last->end = (double)frame * frame_seconds;
            }
            continue;
        }
        if (!current.n) current_start = start;
        push(&current, id);
        current_end = (double)frame * frame_seconds;
    }
    flush(v, &current, current_start, current_end, &words);
    free(current.ids);
    *out = words.words;
    return words.n;
}

int pk_ends_sentence(const char *word) {
    char *copy = xstrdup(word);
    /* rstrip */
    size_t len = strlen(copy);
    while (len) {
        /* find start of last code point */
        size_t s = len - 1;
        while (s > 0 && ((unsigned char)copy[s] & 0xC0) == 0x80) s--;
        uint32_t cp;
        utf8_next((unsigned char *)copy + s, &cp);
        if (!is_space_cp(cp)) break;
        len = s;
    }
    int result = len > 0 && (copy[len - 1] == '.' || copy[len - 1] == '?' || copy[len - 1] == '!');
    free(copy);
    return result;
}
