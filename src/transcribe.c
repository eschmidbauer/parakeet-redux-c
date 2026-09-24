/* The pipeline: features -> subsampler -> encoder -> greedy decode, with the
 * VAD-based cutting of long recordings. Mirrors transcribe.py step for step. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "ctx.h"
#include "model.h"
#include "segment.h"
#include "subsample.h"
#include "util.h"
#include "vad.h"
#include "vocab.h"
#include "wav.h"

/* ---- contexts ------------------------------------------------------------- */

/* Arm an error trap for the calling thread: a die() anywhere below lands in
 * the `if (setjmp(...))` branch with pk_error_message() set. */
#define PK_TRAP(trap, previous) jmp_buf trap; jmp_buf *previous = pk_error_trap(&trap)

pk_context *pk_context_new(int threads) {
    pk_context *volatile c = NULL;
    PK_TRAP(trap, previous);
    if (setjmp(trap)) {
        pk_error_trap(previous);
        pk_context_free(c);
        return NULL;
    }
    c = xcalloc(1, sizeof *c);
    c->pool = pk_pool_new(threads);
    pk_error_trap(previous);
    return c;
}

void pk_context_free(pk_context *c) {
    if (!c) return;
    pk_pool_free(c->pool);
    pk_context_free_scratch(c);
    pk_encoder_free_scratch(c);
    free(c);
}

int pk_context_threads(const pk_context *c) { return pk_pool_size(c->pool); }
void pk_context_set_verbose(pk_context *c, int on) { c->verbose = on; }
void pk_context_set_dump_dir(pk_context *c, const char *dir) { c->dump_dir = dir; }

/* ---- .npy dumps for tests/stages.py ------------------------------------- */

static void npy_write(const pk_context *c, const char *name, const void *data, size_t elem, const char *descr, int ndim, const long *shape) {
    if (!c->dump_dir) return;
    char path[1024], header[256];
    snprintf(path, sizeof path, "%s/%s.npy", c->dump_dir, name);
    int len = snprintf(header, sizeof header, "{'descr': '%s', 'fortran_order': False, 'shape': (", descr);
    size_t count = 1;
    for (int i = 0; i < ndim; i++) {
        len += snprintf(header + len, sizeof header - (size_t)len, "%ld%s", shape[i], ndim == 1 ? "," : (i + 1 < ndim ? ", " : ""));
        count *= (size_t)shape[i];
    }
    len += snprintf(header + len, sizeof header - (size_t)len, "), }");
    int total = 10 + len + 1;
    int padding = (64 - total % 64) % 64;
    FILE *f = fopen(path, "wb");
    if (!f) die("cannot write %s", path);
    uint16_t hlen = (uint16_t)(len + padding + 1);
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fputc(hlen & 0xff, f);
    fputc(hlen >> 8, f);
    fwrite(header, 1, (size_t)len, f);
    for (int i = 0; i < padding; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, elem, count, f);
    fclose(f);
}

/* ---- settings from the weights file header ---------------------------------- */

static void read_settings(pk_model *m, const pk_weights *w) {
    m->vocab_size = w->vocab_size > 0 ? w->vocab_size : 8193;
    m->blank_id = w->blank_id;
    m->max_symbols = w->max_symbols > 0 ? w->max_symbols : 10;
    m->n_durations = w->n_durations > 0 ? w->n_durations : 5;
    for (int i = 0; i < m->n_durations; i++) m->durations[i] = w->durations[i];
    if (w->n_durations <= 0)
        for (int i = 0; i < 5; i++) m->durations[i] = i;
    m->frame_seconds = w->frame_seconds > 0 ? w->frame_seconds : 0.08;
}

/* ---- loading ---------------------------------------------------------------- */

/* `path` is the weights file, or a directory holding parakeet-redux.bin */
static char *resolve_weights(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) die("no such file or directory: %s", path);
    if (!S_ISDIR(st.st_mode)) return xstrdup(path);
    size_t n = strlen(path) + sizeof "/parakeet-redux.bin";
    char *p = xmalloc(n);
    snprintf(p, n, "%s/parakeet-redux.bin", path);
    if (stat(p, &st) != 0) die("%s has no parakeet-redux.bin (get it from https://huggingface.co/eschmidbauer/parakeet-redux-c)", path);
    return p;
}

static int max_frames_for(long samples) {
    /* the encoder frame count of a piece of that many samples */
    return pk_sub_valid((int)(samples / PK_HOP), 3);
}

/* everything the trap must release if loading fails half way */
typedef struct {
    pk_model *m;
    pk_weights *w;
    pk_context *loader;
    char *path;
} load_state;

static pk_model *model_load(const char *path, const pk_options *opt_in, load_state *st);

/* the trap lives in a function of its own so no argument is live across setjmp */
static pk_model *model_load_trapped(const char *path, const pk_options *opt) {
    load_state st = {0};
    load_state *volatile state = &st;
    PK_TRAP(trap, previous);
    if (setjmp(trap)) {
        pk_error_trap(previous);
        if (state->w) pk_weights_close(state->w);
        free(state->w);
        free(state->path);
        pk_context_free(state->loader);
        pk_model_free(state->m);
        return NULL;
    }
    pk_model *m = model_load(path, opt, state);
    pk_error_trap(previous);
    return m;
}

pk_model *pk_model_load(const char *path, const pk_options *options) {
    pk_options opt = {0};
    if (options) opt = *options;
    if (opt.max_window_seconds <= 0) opt.max_window_seconds = PK_SEGMENT_SECONDS;
    return model_load_trapped(path, &opt);
}

static pk_model *model_load(const char *path_in, const pk_options *opt_in, load_state *st) {
    pk_options opt = *opt_in;
    pk_model *m = st->m = xcalloc(1, sizeof *m);
    m->fast = opt.fast;
    m->max_window = (long)nearbyint(opt.max_window_seconds * PK_SAMPLE_RATE);

    /* the weights file gives the settings and the vocabulary in every case */
    char *path = st->path = resolve_weights(path_in);
    pk_weights *w = st->w = xcalloc(1, sizeof *w);
    pk_weights_open(w, path);
    read_settings(m, w);
    const pk_tensor *vocab = pk_weights_find(w, "vocab");
    if (!vocab || vocab->dtype != PK_DT_TEXT) die("%s has no vocabulary", path);
    pk_vocab_parse(&m->vocab, (const char *)vocab->data, vocab->bytes);
    if (m->blank_id < 0) m->blank_id = m->vocab.blank_id;
    if (m->blank_id < 0) die("no blank token in the vocabulary");

    int packed_fast, packed_vad, packed_cache;
    int packed_state = opt.packed_path ? pk_packed_header(opt.packed_path, &packed_fast, &packed_vad, &packed_cache) : 0;
    if (packed_state < 0) die("%s", pk_error_message());
    if (packed_state > 0) {
        /* shapes first, then every array points into the mapped cache file */
        m->fast = packed_fast;
        m->has_vad = packed_vad;
        pk_features_init_shapes(&m->feats);
        pk_subsampler_init_shapes(&m->sub);
        pk_encoder_init_shapes(&m->enc, m->fast, packed_cache);
        pk_decoder_init_shapes(&m->dec, m->fast);
        if (m->has_vad) pk_vad_init_shapes(&m->vad);
        pk_model_map_packed(m, opt.packed_path, &packed_fast, &packed_vad, &packed_cache);
        if (m->fast != opt.fast) pk_log(PK_LOG_WARNING, "%s was packed in %s mode; using that", opt.packed_path, m->fast ? "fast" : "exact");
        if (max_frames_for(m->max_window) > m->enc.cache_n)
            pk_log(PK_LOG_WARNING, "%s was packed for windows of up to %d frames; longer segments are projected per call", opt.packed_path, m->enc.cache_n);
        if (opt.lock_in_memory) {
            int rc = pk_lock_pages(m->map, m->map_len);
            if (rc == 0) m->locked = 1;
            else pk_log(PK_LOG_WARNING, "could not lock the packed model in memory (raise RLIMIT_MEMLOCK)");
        }
        pk_weights_close(w);
        free(w);
        st->w = NULL;
        free(path);
        st->path = NULL;
        return m;
    }

    st->loader = pk_context_new(opt.load_threads);
    if (!st->loader) die("%s", pk_error_message());
    pk_pool *pool = st->loader->pool;
    pk_features_load(&m->feats, w);
    pk_subsampler_load(&m->sub, w);
    pk_encoder_load(&m->enc, w, m->fast, pool);
    pk_decoder_load(&m->dec, w, m->fast);
    if (pk_weights_find(w, "vad.proj.weight")) {
        pk_vad_load(&m->vad, w);
        m->has_vad = 1;
    }
    pk_weights_close(w);
    free(w);
    st->w = NULL;
    free(path);
    st->path = NULL;

    pk_encoder_prepare(st->loader, &m->enc, max_frames_for(m->max_window));
    pk_context_free(st->loader);
    st->loader = NULL;

    if (opt.packed_path) {
        if (pk_model_save(m, opt.packed_path) == 0) pk_log(PK_LOG_INFO, "wrote packed weights to %s", opt.packed_path);
        else pk_log(PK_LOG_WARNING, "%s", pk_error_message());
    }
    if (opt.lock_in_memory) {
        int rc = pk_encoder_lock(&m->enc) | pk_decoder_lock(&m->dec) | pk_subsampler_lock(&m->sub) | pk_features_lock(&m->feats);
        if (m->has_vad) rc |= pk_vad_lock(&m->vad);
        if (rc == 0) m->locked = 1;
        else pk_log(PK_LOG_WARNING, "could not lock all model pages in memory (raise RLIMIT_MEMLOCK)");
    }
    return m;
}

int pk_model_is_mapped(const pk_model *m) { return m->map != NULL; }

const char *pk_model_backend(const pk_model *m) { return pk_gemm_backend(m->fast); }
int pk_model_is_fast(const pk_model *m) { return m->fast; }
int pk_model_is_locked(const pk_model *m) { return m->locked; }

size_t pk_model_weight_bytes(const pk_model *m) {
    size_t total = pk_encoder_bytes(&m->enc);
    total += (size_t)PK_DEC_VOCAB * PK_DEC_HIDDEN * sizeof(float); /* embedding */
    total += (size_t)PK_DEC_OUT * PK_DEC_HIDDEN * (m->fast ? 2 : 4);
    total += (size_t)2 * 4 * PK_DEC_HIDDEN * 2 * PK_DEC_HIDDEN * (m->fast ? 2 : 4);
    total += pk_weight_bytes(&m->dec.enc_proj) + pk_weight_bytes(&m->sub.lin) + pk_weight_bytes(&m->sub.pw3) + pk_weight_bytes(&m->sub.pw6);
    if (m->has_vad) total += pk_weight_bytes(&m->vad.proj) + pk_weight_bytes(&m->vad.ctx);
    total += (size_t)PK_ENC_LAYERS * (2 * (size_t)m->enc.cache_n - 1) * PK_ENC_D * sizeof(float);
    return total;
}

static void forget_array(void *user, void **array, size_t bytes) { *array = NULL; }

void pk_model_free(pk_model *m) {
    if (!m) return;
    if (m->map) { /* mapped arrays are not ours to free */
        pk_model_arrays(m, forget_array, NULL);
        munmap(m->map, m->map_len);
    }
    pk_features_free(&m->feats);
    pk_subsampler_free(&m->sub);
    pk_encoder_free(&m->enc);
    pk_decoder_free(&m->dec);
    if (m->has_vad) pk_vad_free(&m->vad);
    pk_vocab_free(&m->vocab);
    free(m);
}

/* ---- the pipeline -------------------------------------------------------------- */

static float *encode(pk_context *c, const pk_model *m, const float *pcm, int n, int seg, int *frames) {
    int T, valid, T3, len3;
    char name[64];
    double t0 = now_seconds();
    float *feat = pk_features_compute(c, &m->feats, pcm, n, &T, &valid);
    c->timing.features += now_seconds() - t0;
    snprintf(name, sizeof name, "seg%d_features", seg);
    npy_write(c, name, feat, sizeof(float), "<f4", 2, (long[]){T, PK_N_MEL});
    t0 = now_seconds();
    float *sub = pk_subsample(c, &m->sub, feat, T, valid, &T3, &len3);
    c->timing.subsample += now_seconds() - t0;
    free(feat);
    snprintf(name, sizeof name, "seg%d_subsample", seg);
    npy_write(c, name, sub, sizeof(float), "<f4", 2, (long[]){len3, PK_D_MODEL});
    float *enc = xmalloc((size_t)len3 * PK_ENC_D * sizeof(float));
    t0 = now_seconds();
    pk_encoder_run(c, &m->enc, sub, len3, enc);
    c->timing.encoder += now_seconds() - t0;
    free(sub);
    snprintf(name, sizeof name, "seg%d_encoder", seg);
    npy_write(c, name, enc, sizeof(float), "<f4", 2, (long[]){len3, PK_ENC_D});
    *frames = len3;
    return enc;
}

typedef struct {
    pk_span *items;
    int n, cap;
} span_list;

static void span_push(span_list *l, double start, double end) {
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 64;
        l->items = xrealloc(l->items, (size_t)l->cap * sizeof *l->items);
    }
    l->items[l->n].start = start;
    l->items[l->n].end = end;
    l->n++;
}

static span_list speech_regions(pk_context *c, const pk_model *m, const float *pcm, int n) {
    span_list regions = {0};
    double duration = (double)n / PK_SAMPLE_RATE;
    if (!m->has_vad) {
        span_push(&regions, 0.0, duration);
        return regions;
    }
    long block_samples = (long)nearbyint(PK_BLOCK_SECONDS * PK_SAMPLE_RATE);
    int block_index = 0;
    for (long offset = 0; offset < n; offset += block_samples, block_index++) {
        int len = (int)((n - offset) < block_samples ? (n - offset) : block_samples);
        double base = (double)offset / PK_SAMPLE_RATE;
        double block_duration = (double)len / PK_SAMPLE_RATE;
        if (len < PK_MIN_FEATURE_SAMPLES) {
            span_push(&regions, base, base + block_duration);
            continue;
        }
        int T, valid, T3, len3;
        double t0 = now_seconds();
        float *feat = pk_features_compute(c, &m->feats, pcm + offset, len, &T, &valid);
        float *sub = pk_subsample(c, &m->sub, feat, T, valid, &T3, &len3);
        free(feat);
        float *probs = pk_vad_run(c, &m->vad, sub, T3);
        free(sub);
        c->timing.vad += now_seconds() - t0;
        char name[64];
        snprintf(name, sizeof name, "vad%d_probs", block_index);
        npy_write(c, name, probs, sizeof(float), "<f4", 1, (long[]){len3});
        pk_span *spans = xmalloc((size_t)(len3 / 2 + 2) * sizeof *spans);
        int count = pk_speech_regions_from_probs(probs, len3, m->frame_seconds, spans);
        for (int i = 0; i < count; i++)
            span_push(&regions, base + spans[i].start, base + (spans[i].end < block_duration ? spans[i].end : block_duration));
        free(spans);
        free(probs);
    }
    return regions;
}

typedef struct {
    long start, end;
} piece;

static int relative_regions(const span_list *regions, double origin, pk_span *out) {
    int n = 0;
    for (int i = 0; i < regions->n; i++) {
        if (regions->items[i].end <= origin) continue;
        double s = regions->items[i].start - origin;
        out[n].start = s > 0.0 ? s : 0.0;
        out[n].end = regions->items[i].end - origin;
        n++;
    }
    return n;
}

static int cut_segments(pk_context *c, const pk_model *m, const float *pcm, int n, piece **out) {
    long cap = (long)nearbyint(PK_SEGMENT_SECONDS * PK_SAMPLE_RATE);
    piece *pieces = xmalloc(sizeof *pieces);
    int count = 0, capacity = 1;
    if (n <= cap) {
        pieces[0].start = 0;
        pieces[0].end = n;
        *out = pieces;
        return 1;
    }
    span_list regions = speech_regions(c, m, pcm, n);
    pk_span *relative = xmalloc((size_t)(regions.n + 1) * sizeof *relative);
    pk_span *pauses = xmalloc((size_t)(regions.n + 2) * sizeof *pauses);
    long start = 0;
    int cuts = 0;
    while (n - start > cap) {
        double origin = (double)start / PK_SAMPLE_RATE;
        int nrel = relative_regions(&regions, origin, relative);
        int npauses = pk_pauses_from_speech(relative, nrel, (double)(n - start) / PK_SAMPLE_RATE, pauses);
        double cut = pk_next_cut(pauses, npauses, PK_SEGMENT_SECONDS);
        long index = (long)nearbyint(cut * PK_SAMPLE_RATE);
        if (pk_has_speech(relative, nrel, (double)index / PK_SAMPLE_RATE)) {
            if (count == capacity) pieces = xrealloc(pieces, (size_t)(capacity *= 2) * sizeof *pieces);
            pieces[count].start = start;
            pieces[count].end = start + index;
            count++;
        }
        start += index;
        cuts++;
    }
    double origin = (double)start / PK_SAMPLE_RATE;
    int nrel = relative_regions(&regions, origin, relative);
    long tail = n - start;
    if (!(cuts && (tail < PK_MIN_FEATURE_SAMPLES || !pk_has_speech(relative, nrel, (double)tail / PK_SAMPLE_RATE)))) {
        if (count == capacity) pieces = xrealloc(pieces, (size_t)(capacity *= 2) * sizeof *pieces);
        pieces[count].start = start;
        pieces[count].end = n;
        count++;
    }
    free(relative);
    free(pauses);
    free(regions.items);
    *out = pieces;
    return count;
}

typedef struct {
    pk_segment *items;
    int n, cap;
} segment_list;

static pk_segment *segment_new(segment_list *l) {
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 16;
        l->items = xrealloc(l->items, (size_t)l->cap * sizeof *l->items);
    }
    pk_segment *s = &l->items[l->n++];
    memset(s, 0, sizeof *s);
    return s;
}

static void emit_sentence(segment_list *segments, pk_word *words, int n, int include_words) {
    pk_segment *s = segment_new(segments);
    size_t total = 1;
    for (int i = 0; i < n; i++) total += strlen(words[i].word) + 1;
    s->text = xmalloc(total);
    s->text[0] = '\0';
    for (int i = 0; i < n; i++) {
        if (i) strcat(s->text, " ");
        strcat(s->text, words[i].word);
    }
    s->start = words[0].start;
    s->end = words[n - 1].end;
    if (include_words) {
        s->words = xmalloc((size_t)n * sizeof *s->words);
        memcpy(s->words, words, (size_t)n * sizeof *s->words);
        s->n_words = n;
    } else {
        for (int i = 0; i < n; i++) free(words[i].word);
    }
}

static void sentence_segments(segment_list *segments, pk_word *words, int n, int include_words) {
    int first = 0;
    for (int i = 0; i < n; i++) {
        if (pk_ends_sentence(words[i].word) || words[i].end - words[first].start >= PK_SEGMENT_SECONDS) {
            emit_sentence(segments, words + first, i - first + 1, include_words);
            first = i + 1;
        }
    }
    if (first < n) emit_sentence(segments, words + first, n - first, include_words);
}

/* Transcribe one piece of audio (at most a window long); times are offset by base_seconds. */
static pk_result *transcribe_piece(pk_context *c, const pk_model *m, const float *pcm, int n, int timestamps, double base_seconds, int seg_index) {
    int frames;
    double t0;
    float *enc = encode(c, m, pcm, n, seg_index, &frames);
    int *tokens, *durations;
    t0 = now_seconds();
    int steps = pk_greedy_decode(c, &m->dec, enc, frames, m->durations, m->n_durations, m->max_symbols, m->blank_id, &tokens, &durations);
    c->timing.decode += now_seconds() - t0;
    free(enc);
    if (c->dump_dir) {
        int *pairs = xmalloc((size_t)steps * 2 * sizeof(int));
        for (int st = 0; st < steps; st++) {
            pairs[2 * st] = tokens[st];
            pairs[2 * st + 1] = durations[st];
        }
        char name[64];
        snprintf(name, sizeof name, "seg%d_steps", seg_index);
        npy_write(c, name, pairs, sizeof(int), "<i4", 2, (long[]){steps, 2});
        free(pairs);
    }
    pk_result *r = xcalloc(1, sizeof *r);
    r->duration = (double)n / PK_SAMPLE_RATE;
    r->start = base_seconds;
    char *piece_text = pk_vocab_decode(&m->vocab, tokens, steps);
    r->text = pk_strip(xstrdup(piece_text));
    segment_list segments = {0};
    if (timestamps != PK_TS_NONE && *piece_text) {
        double clip_start = base_seconds, clip_end = base_seconds + (double)n / PK_SAMPLE_RATE;
        pk_word *words;
        int n_words = pk_vocab_words(&m->vocab, tokens, durations, steps, m->frame_seconds, &words);
        for (int w = 0; w < n_words; w++) {
            double ws = words[w].start + clip_start, we = words[w].end + clip_start;
            words[w].start = ws < clip_end ? ws : clip_end;
            words[w].end = we < clip_end ? we : clip_end;
        }
        if (n_words) {
            sentence_segments(&segments, words, n_words, timestamps == PK_TS_WORD);
        } else {
            pk_segment *seg = segment_new(&segments);
            seg->text = xstrdup(piece_text);
            seg->start = clip_start;
            seg->end = clip_end;
        }
        free(words);
    }
    free(piece_text);
    free(tokens);
    free(durations);
    r->segments = segments.items;
    r->n_segments = segments.n;
    return r;
}

/* Append the text and segments of `piece` to `r` and free `piece`. */
static void result_append(pk_result *r, pk_result *piece) {
    if (*piece->text) {
        size_t a = strlen(r->text), b = strlen(piece->text);
        r->text = xrealloc(r->text, a + b + 2);
        if (a) strcat(r->text, " ");
        strcat(r->text, piece->text);
    }
    if (piece->n_segments) {
        r->segments = xrealloc(r->segments, (size_t)(r->n_segments + piece->n_segments) * sizeof *r->segments);
        memcpy(r->segments + r->n_segments, piece->segments, (size_t)piece->n_segments * sizeof *r->segments);
        r->n_segments += piece->n_segments;
    }
    free(piece->segments);
    free(piece->text);
    free(piece);
}

pk_result *pk_transcribe(pk_context *c, const pk_model *m, const float *pcm, int n, int timestamps) {
    if (n < PK_MIN_FEATURE_SAMPLES) {
        pk_set_error("audio is too short to transcribe (%d samples, need %d)", n, PK_MIN_FEATURE_SAMPLES);
        return NULL;
    }
    pk_result *volatile r = NULL;
    piece *volatile pieces = NULL;
    PK_TRAP(trap, previous);
    if (setjmp(trap)) { /* out of memory somewhere below; what was allocated in between is lost */
        pk_error_trap(previous);
        free(pieces);
        pk_result_free(r);
        return NULL;
    }
    memset(&c->timing, 0, sizeof c->timing);
    int n_pieces = cut_segments(c, m, pcm, n, (piece **)&pieces);
    r = xcalloc(1, sizeof *r);
    r->text = xstrdup("");
    r->duration = (double)n / PK_SAMPLE_RATE;
    for (int i = 0; i < n_pieces; i++) {
        long start = pieces[i].start, end = pieces[i].end;
        result_append(r, transcribe_piece(c, m, pcm + start, (int)(end - start), timestamps, (double)start / PK_SAMPLE_RATE, i));
    }
    if (c->verbose)
        pk_log(PK_LOG_INFO, "  %d segment(s): vad+cut %.2fs, features %.2fs, subsample %.2fs, encoder %.2fs, decode %.2fs", n_pieces,
               c->timing.vad, c->timing.features, c->timing.subsample, c->timing.encoder, c->timing.decode);
    pk_error_trap(previous);
    free(pieces);
    return r;
}

/* ---- streaming ------------------------------------------------------------------- */

struct pk_stream {
    pk_context *c;
    const pk_model *m;
    int timestamps;
    long window;      /* samples per result at most */
    float *buf;
    long n, cap;
    long base;        /* absolute sample index of buf[0] */
    int cuts;
    int finished;
    int failed;
    pk_result **queue;
    int q_head, q_tail, q_cap;
    /* VAD regions (absolute seconds) of completed 120 s blocks, and how far they reach */
    span_list vad_cache;
    long vad_scanned;
    /* end-of-utterance cuts and manual processing */
    long pause;         /* samples of silence after speech that end an utterance; 0 = off */
    int manual;
    long examined;      /* absolute sample index up to which the pause rule has looked */
    int results_ready;  /* results queued since the last take, for pk_stream_process's return value */
};

pk_stream *pk_stream_new(pk_context *c, const pk_model *m, int timestamps, double window_seconds) {
    pk_stream *s = xcalloc(1, sizeof *s);
    s->c = c;
    s->m = m;
    s->timestamps = timestamps;
    if (window_seconds <= 0) window_seconds = PK_SEGMENT_SECONDS;
    s->window = (long)nearbyint(window_seconds * PK_SAMPLE_RATE);
    return s;
}

static void stream_push(pk_stream *s, pk_result *r) {
    s->results_ready++;
    if (s->q_tail == s->q_cap) {
        if (s->q_head) {
            memmove(s->queue, s->queue + s->q_head, (size_t)(s->q_tail - s->q_head) * sizeof *s->queue);
            s->q_tail -= s->q_head;
            s->q_head = 0;
        }
        if (s->q_tail == s->q_cap) s->queue = xrealloc(s->queue, (size_t)(s->q_cap = s->q_cap ? s->q_cap * 2 : 8) * sizeof *s->queue);
    }
    s->queue[s->q_tail++] = r;
}

/* Speech regions for the buffered audio, relative to the buffer start: the
 * VAD head runs once over each completed block and again over the unfinished
 * tail, instead of over the whole buffer at every decision. */
static span_list stream_regions(pk_stream *s) {
    long block = (long)nearbyint(PK_BLOCK_SECONDS * PK_SAMPLE_RATE);
    long tail_start = s->vad_scanned > s->base ? s->vad_scanned : s->base;
    while (tail_start + block <= s->base + s->n) {
        span_list part = speech_regions(s->c, s->m, s->buf + (tail_start - s->base), (int)block);
        double t0 = (double)tail_start / PK_SAMPLE_RATE;
        for (int i = 0; i < part.n; i++) span_push(&s->vad_cache, t0 + part.items[i].start, t0 + part.items[i].end);
        free(part.items);
        tail_start += block;
        s->vad_scanned = tail_start;
    }
    /* drop cached regions that ended before the buffer */
    double origin = (double)s->base / PK_SAMPLE_RATE;
    int kept = 0;
    for (int i = 0; i < s->vad_cache.n; i++)
        if (s->vad_cache.items[i].end > origin) s->vad_cache.items[kept++] = s->vad_cache.items[i];
    s->vad_cache.n = kept;
    span_list out = {0};
    for (int i = 0; i < kept; i++) {
        double a = s->vad_cache.items[i].start - origin;
        span_push(&out, a > 0.0 ? a : 0.0, s->vad_cache.items[i].end - origin);
    }
    if (s->base + s->n - tail_start >= PK_MIN_FEATURE_SAMPLES) {
        span_list part = speech_regions(s->c, s->m, s->buf + (tail_start - s->base), (int)(s->base + s->n - tail_start));
        double t0 = (double)(tail_start - s->base) / PK_SAMPLE_RATE;
        for (int i = 0; i < part.n; i++) span_push(&out, t0 + part.items[i].start, t0 + part.items[i].end);
        free(part.items);
    } else if (s->base + s->n > tail_start) {
        span_push(&out, (double)(tail_start - s->base) / PK_SAMPLE_RATE, (double)s->n / PK_SAMPLE_RATE);
    }
    return out;
}

/* Cut the buffer at the last pause within the window and transcribe the piece. */
static void stream_cut(pk_stream *s, long index, int with_speech) {
    if (with_speech) {
        pk_result *r = transcribe_piece(s->c, s->m, s->buf, (int)index, s->timestamps, (double)s->base / PK_SAMPLE_RATE, s->cuts);
        stream_push(s, r);
    }
    memmove(s->buf, s->buf + index, (size_t)(s->n - index) * sizeof(float));
    s->n -= index;
    s->base += index;
    s->cuts++;
}

/* Run every cut that is due: the window rule, and the pause rule when enabled.
 * Called under an error trap. */
static void stream_decide(pk_stream *s) {
    for (;;) {
        if (s->n > s->window) { /* the window is full: cut at the last pause inside it */
            span_list regions = stream_regions(s);
            pk_span *pauses = xmalloc((size_t)(regions.n + 2) * sizeof *pauses);
            int npauses = pk_pauses_from_speech(regions.items, regions.n, (double)s->n / PK_SAMPLE_RATE, pauses);
            double cut = pk_next_cut(pauses, npauses, (double)s->window / PK_SAMPLE_RATE);
            long index = (long)nearbyint(cut * PK_SAMPLE_RATE);
            if (index < PK_MIN_FEATURE_SAMPLES) index = PK_MIN_FEATURE_SAMPLES;
            if (index > s->n) index = s->n;
            int with_speech = pk_has_speech(regions.items, regions.n, (double)index / PK_SAMPLE_RATE);
            free(pauses);
            free(regions.items);
            stream_cut(s, index, with_speech);
            continue;
        }
        if (s->pause <= 0) return;
        /* pause rule: look again once half a second of new audio has arrived */
        long quantum = PK_SAMPLE_RATE / 2;
        if (s->n < s->pause + quantum || s->base + s->n - s->examined < quantum) return;
        s->examined = s->base + s->n;
        span_list regions = stream_regions(s);
        double end = (double)s->n / PK_SAMPLE_RATE, last_speech = -1.0;
        for (int i = 0; i < regions.n; i++)
            if (regions.items[i].end > last_speech) last_speech = regions.items[i].end;
        free(regions.items);
        double pause = (double)s->pause / PK_SAMPLE_RATE;
        if (last_speech < 0) {
            /* nothing but silence: keep the last `pause` seconds as context, drop the rest */
            if (s->n > 2 * s->pause) stream_cut(s, s->n - s->pause, 0);
            return;
        }
        if (end - last_speech < pause) return; /* still speaking, or the pause is not long enough yet */
        long index = (long)nearbyint((last_speech + pause / 2) * PK_SAMPLE_RATE);
        if (index < PK_MIN_FEATURE_SAMPLES) index = PK_MIN_FEATURE_SAMPLES;
        if (index > s->n) index = s->n;
        stream_cut(s, index, 1);
    }
}

int pk_stream_feed(pk_stream *s, const float *pcm, int n) {
    if (s->failed) return -1;
    if (s->finished || n <= 0) return 0;
    if (s->n + n > s->cap) {
        s->cap = (s->n + n) * 2;
        s->buf = xrealloc(s->buf, (size_t)s->cap * sizeof(float));
    }
    memcpy(s->buf + s->n, pcm, (size_t)n * sizeof(float));
    s->n += n;
    if (s->manual) return 0;
    PK_TRAP(trap, previous);
    if (setjmp(trap)) {
        pk_error_trap(previous);
        s->failed = 1;
        return -1;
    }
    stream_decide(s);
    pk_error_trap(previous);
    return 0;
}

void pk_stream_set_context(pk_stream *s, pk_context *c) { s->c = c; }
void pk_stream_set_pause(pk_stream *s, double seconds) { s->pause = seconds > 0 ? (long)nearbyint(seconds * PK_SAMPLE_RATE) : 0; }
void pk_stream_set_manual(pk_stream *s, int manual) { s->manual = manual; }
double pk_stream_buffered(const pk_stream *s) { return (double)s->n / PK_SAMPLE_RATE; }

int pk_stream_process(pk_stream *s, pk_context *c) {
    if (s->failed) return -1;
    if (!c) c = s->c;
    if (!c) {
        pk_set_error("pk_stream_process needs a context");
        return -1;
    }
    pk_context *saved = s->c;
    s->c = c;
    s->results_ready = 0;
    PK_TRAP(trap, previous);
    if (setjmp(trap)) {
        pk_error_trap(previous);
        s->c = saved;
        s->failed = 1;
        return -1;
    }
    if (!s->finished) stream_decide(s);
    pk_error_trap(previous);
    s->c = saved;
    return s->results_ready;
}

int pk_stream_finish(pk_stream *s) {
    if (s->failed) return -1;
    if (s->finished) return 0;
    if (!s->c) {
        pk_set_error("pk_stream_finish needs a context (pk_stream_set_context)");
        s->failed = 1;
        return -1;
    }
    s->finished = 1;
    if (s->n < PK_MIN_FEATURE_SAMPLES) return 0;
    PK_TRAP(trap, previous);
    if (setjmp(trap)) {
        pk_error_trap(previous);
        s->failed = 1;
        return -1;
    }
    int with_speech = 1;
    if (s->cuts) {
        span_list regions = stream_regions(s);
        with_speech = pk_has_speech(regions.items, regions.n, (double)s->n / PK_SAMPLE_RATE);
        free(regions.items);
    }
    stream_cut(s, s->n, with_speech);
    pk_error_trap(previous);
    return 0;
}

pk_result *pk_stream_take(pk_stream *s) {
    if (s->q_head == s->q_tail) return NULL;
    return s->queue[s->q_head++];
}

void pk_stream_free(pk_stream *s) {
    if (!s) return;
    while (s->q_head < s->q_tail) pk_result_free(s->queue[s->q_head++]);
    free(s->queue);
    free(s->buf);
    free(s->vad_cache.items);
    free(s);
}

float *pk_read_audio(const char *path, int *n_samples) {
    float *volatile pcm = NULL;
    PK_TRAP(trap, previous);
    if (setjmp(trap)) {
        pk_error_trap(previous);
        return NULL;
    }
    pcm = pk_audio_read(path, PK_SAMPLE_RATE, n_samples);
    pk_error_trap(previous);
    return pcm;
}

void pk_result_free(pk_result *r) {
    if (!r) return;
    for (int i = 0; i < r->n_segments; i++) {
        free(r->segments[i].text);
        for (int w = 0; w < r->segments[i].n_words; w++) free(r->segments[i].words[w].word);
        free(r->segments[i].words);
    }
    free(r->segments);
    free(r->text);
    free(r);
}
