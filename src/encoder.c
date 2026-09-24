#include "encoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gemm.h"
#include "mathx.h"
#include "pool.h"
#include "util.h"

#define D PK_ENC_D
#define FF PK_ENC_FF
#define H PK_ENC_HEADS
#define DH PK_ENC_DHEAD
#define LN_EPS 1e-5f
#define BN_EPS 1e-5f

static float *layer_f32(const pk_weights *w, int layer, const char *suffix, size_t count) {
    return pk_weights_f32f(w, count, "encoder.layers.%d.%s", layer, suffix);
}

void pk_encoder_load(pk_encoder *e, const pk_weights *w, int fast, pk_pool *pool) {
    memset(e, 0, sizeof *e);
    e->inv_freq = pk_weights_f32(w, "encoder.inverse_frequency", D / 2);
    /* the positional projection is one [24 * 1024][1024] ternary tensor; each layer takes its rows */
    const pk_tensor *pc = pk_weights_get(w, "encoder.pos.codes", PK_DT_CODES, (size_t)PK_ENC_LAYERS * D * D);
    const pk_tensor *ps = pk_weights_get(w, "encoder.pos.scales", PK_DT_F16, (size_t)PK_ENC_LAYERS * D * (D / PK_TERNARY_BLOCK));
    static const char *const roles[] = {"ff1_in", "ff1_out", "qkv", "att_out", "conv_pw1", "conv_pw2", "ff2_in", "ff2_out"};
    static const int shape[][2] = {{FF, D}, {D, FF}, {3 * D, D}, {D, D}, {2 * D, D}, {D, D}, {FF, D}, {D, FF}};
    for (int l = 0; l < PK_ENC_LAYERS; l++) {
        pk_layer *L = &e->layers[l];
        pk_weight_from_codes(&L->pos, pc->data + (size_t)l * D * (D / 4), (const uint16_t *)ps->data + (size_t)l * D * (D / PK_TERNARY_BLOCK),
                             D, D, fast, pool);
        pk_linear *targets[] = {&L->ff1_in, &L->ff1_out, &L->qkv, &L->att_out, &L->conv_pw1, &L->conv_pw2, &L->ff2_in, &L->ff2_out};
        for (int r = 0; r < 8; r++) {
            char name[96];
            snprintf(name, sizeof name, "encoder.layers.%d.%s", l, roles[r]);
            pk_linear_from_ternary(targets[r], w, name, shape[r][0], shape[r][1], fast, pool);
        }
        L->pos_bias_u = layer_f32(w, l, "pos_bias_u", D);
        L->pos_bias_v = layer_f32(w, l, "pos_bias_v", D);
        L->ln_ff1_w = layer_f32(w, l, "norm_ff1.weight", D);
        L->ln_ff1_b = layer_f32(w, l, "norm_ff1.bias", D);
        L->ln_att_w = layer_f32(w, l, "norm_att.weight", D);
        L->ln_att_b = layer_f32(w, l, "norm_att.bias", D);
        L->ln_conv_w = layer_f32(w, l, "norm_conv.weight", D);
        L->ln_conv_b = layer_f32(w, l, "norm_conv.bias", D);
        L->ln_ff2_w = layer_f32(w, l, "norm_ff2.weight", D);
        L->ln_ff2_b = layer_f32(w, l, "norm_ff2.bias", D);
        L->ln_out_w = layer_f32(w, l, "norm_out.weight", D);
        L->ln_out_b = layer_f32(w, l, "norm_out.bias", D);
        float *dw = layer_f32(w, l, "conv.depthwise.weight", (size_t)D * PK_ENC_KERNEL); /* [1024][1][9] */
        L->dw_t = xmalloc((size_t)PK_ENC_KERNEL * D * sizeof(float));
        for (int c = 0; c < D; c++)
            for (int k = 0; k < PK_ENC_KERNEL; k++) L->dw_t[(size_t)k * D + c] = dw[(size_t)c * PK_ENC_KERNEL + k];
        free(dw);
        float *g = layer_f32(w, l, "conv.bn.weight", D);
        float *b = layer_f32(w, l, "conv.bn.bias", D);
        float *mean = layer_f32(w, l, "conv.bn.running_mean", D);
        float *var = layer_f32(w, l, "conv.bn.running_var", D);
        L->bn_s = xmalloc(D * sizeof(float));
        L->bn_t = xmalloc(D * sizeof(float));
        for (int c = 0; c < D; c++) {
            L->bn_s[c] = g[c] / sqrtf(var[c] + BN_EPS);
            L->bn_t[c] = b[c] - mean[c] * L->bn_s[c];
        }
        free(g); free(b); free(mean); free(var);
    }
}

void pk_encoder_free(pk_encoder *e) {
    for (int l = 0; l < PK_ENC_LAYERS; l++) {
        pk_layer *L = &e->layers[l];
        pk_linear_free(&L->pos);
        pk_linear_free(&L->ff1_in); pk_linear_free(&L->ff1_out); pk_linear_free(&L->qkv); pk_linear_free(&L->att_out);
        pk_linear_free(&L->conv_pw1); pk_linear_free(&L->conv_pw2); pk_linear_free(&L->ff2_in); pk_linear_free(&L->ff2_out);
        free(L->ln_ff1_w); free(L->ln_ff1_b); free(L->ln_att_w); free(L->ln_att_b); free(L->ln_conv_w); free(L->ln_conv_b);
        free(L->ln_ff2_w); free(L->ln_ff2_b); free(L->ln_out_w); free(L->ln_out_b);
        free(L->pos_bias_u); free(L->pos_bias_v); free(L->dw_t); free(L->bn_s); free(L->bn_t);
    }
    free(e->inv_freq);
    for (int l = 0; l < PK_ENC_LAYERS; l++) free(e->pos_cache[l]);
    memset(e, 0, sizeof *e);
}

/* relative positional encodings for positions n-1 down to -(n-1), interleaved sin/cos */
static float *positional_encoding(const pk_encoder *e, int n) {
    int P = 2 * n - 1;
    float *pe = xmalloc((size_t)P * D * sizeof(float));
    for (int k = 0; k < P; k++) {
        float pos = (float)(n - 1 - k);
        for (int i = 0; i < D / 2; i++) {
            float a = pos * e->inv_freq[i];
            pe[(size_t)k * D + 2 * i] = sinf(a);
            pe[(size_t)k * D + 2 * i + 1] = cosf(a);
        }
    }
    return pe;
}

/* The positional term depends only on the relative position, so the projected
 * encodings for positions n-1 .. -(n-1) are a contiguous slice of those for any
 * larger n. They are computed once, at load, for the longest expected segment;
 * the model is read only afterwards. */
void pk_encoder_prepare(pk_context *c, pk_encoder *e, int max_frames) {
    if (max_frames <= e->cache_n) return;
    int P = 2 * max_frames - 1;
    float *pe = positional_encoding(e, max_frames);
    for (int l = 0; l < PK_ENC_LAYERS; l++) {
        free(e->pos_cache[l]);
        e->pos_cache[l] = xmalloc((size_t)P * D * sizeof(float));
        pk_linear_gemm(c, &e->layers[l].pos, pe, P, D, e->pos_cache[l], D, NULL);
    }
    free(pe);
    e->cache_n = max_frames;
}

size_t pk_encoder_bytes(const pk_encoder *e) {
    size_t total = 0;
    for (int l = 0; l < PK_ENC_LAYERS; l++) {
        const pk_layer *L = &e->layers[l];
        const pk_linear *w[] = {&L->pos, &L->ff1_in, &L->ff1_out, &L->qkv, &L->att_out, &L->conv_pw1, &L->conv_pw2, &L->ff2_in, &L->ff2_out};
        for (size_t i = 0; i < sizeof w / sizeof *w; i++) total += pk_weight_bytes(w[i]);
    }
    return total;
}

void pk_encoder_init_shapes(pk_encoder *e, int fast, int cache_n) {
    memset(e, 0, sizeof *e);
    e->cache_n = cache_n;
    for (int l = 0; l < PK_ENC_LAYERS; l++) {
        pk_layer *L = &e->layers[l];
        pk_weight_init(&L->pos, D, D, 1, fast);
        pk_weight_init(&L->ff1_in, FF, D, 1, fast);
        pk_weight_init(&L->ff1_out, D, FF, 1, fast);
        pk_weight_init(&L->qkv, 3 * D, D, 1, fast);
        pk_weight_init(&L->att_out, D, D, 1, fast);
        pk_weight_init(&L->conv_pw1, 2 * D, D, 1, fast);
        pk_weight_init(&L->conv_pw2, D, D, 1, fast);
        pk_weight_init(&L->ff2_in, FF, D, 1, fast);
        pk_weight_init(&L->ff2_out, D, FF, 1, fast);
    }
}

void pk_encoder_arrays(pk_encoder *e, pk_array_fn fn, void *user) {
    fn(user, (void **)&e->inv_freq, (size_t)(D / 2) * sizeof(float));
    for (int l = 0; l < PK_ENC_LAYERS; l++) {
        pk_layer *L = &e->layers[l];
        pk_linear *w[] = {&L->pos, &L->ff1_in, &L->ff1_out, &L->qkv, &L->att_out, &L->conv_pw1, &L->conv_pw2, &L->ff2_in, &L->ff2_out};
        for (size_t i = 0; i < sizeof w / sizeof *w; i++) pk_weight_arrays(w[i], fn, user);
        float **v[] = {&L->ln_ff1_w, &L->ln_ff1_b, &L->ln_att_w, &L->ln_att_b, &L->ln_conv_w, &L->ln_conv_b, &L->ln_ff2_w, &L->ln_ff2_b,
                       &L->ln_out_w, &L->ln_out_b, &L->pos_bias_u, &L->pos_bias_v, &L->bn_s, &L->bn_t};
        for (size_t i = 0; i < sizeof v / sizeof *v; i++) fn(user, (void **)v[i], D * sizeof(float));
        fn(user, (void **)&L->dw_t, (size_t)PK_ENC_KERNEL * D * sizeof(float));
        fn(user, (void **)&e->pos_cache[l], e->cache_n > 0 ? (size_t)(2 * e->cache_n - 1) * D * sizeof(float) : 0);
    }
}

static void lock_visit(void *user, void **array, size_t bytes) {
    int *rc = user;
    if (bytes && pk_lock_pages(*array, bytes) != 0) *rc = -1;
}

int pk_encoder_lock(const pk_encoder *e) {
    int rc = 0;
    pk_encoder_arrays((pk_encoder *)e, lock_visit, &rc);
    return rc;
}

/* ---- per-call state; the buffers live in the context ------------------------- */

#ifndef PK_ATTENTION_BUDGET
#define PK_ATTENTION_BUDGET ((size_t)256 << 20) /* bytes of attention score buffers per context */
#endif

typedef struct {
    pk_context *c;
    int n;
    float *h;    /* [n][1024] normalized input */
    float *ff;   /* [n][4096] */
    float *qkv;  /* [n][3072] */
    const float *p; /* [2n-1][1024] positional projection for one layer */
    float *qu, *qv; /* [slots][n][128] */
    float *ac;   /* [slots][n][n] */
    float *bd;   /* [slots][n][2n-1] */
    float *ctx;  /* [n][1024] */
    float *g;    /* [n][2048] */
    float *a;    /* [n][1024] */
    float *pe;   /* [2n-1][1024], only when n exceeds the model's cache */
    float *pbuf; /* [2n-1][1024] projection scratch for that case */
    int slots;   /* attention heads processed at once */
    int head_base;
    const pk_layer *L;
    float *x;
} scratch;

/* Heads whose score buffers fit the budget together, at least one. */
static int heads_at_once(int n) {
    size_t per_head = ((size_t)n * n + (size_t)n * (2 * n - 1) + 2 * (size_t)n * DH) * sizeof(float);
    size_t fit = PK_ATTENTION_BUDGET / per_head;
    return fit < 1 ? 1 : fit > H ? H : (int)fit;
}

void pk_encoder_free_scratch(pk_context *c) {
    pk_enc_scratch *e = &c->enc;
    free(e->h); free(e->ff); free(e->qkv); free(e->ctx); free(e->g); free(e->a);
    free(e->qu); free(e->qv); free(e->ac); free(e->bd);
    memset(e, 0, sizeof *e);
}

static void scratch_prepare(pk_context *c, int n) {
    pk_enc_scratch *e = &c->enc;
    if (n <= e->n_cap) return;
    pk_encoder_free_scratch(c);
    int cap = n + n / 8; /* headroom so a slightly longer segment does not reallocate */
    int P = 2 * cap - 1;
    e->slots = heads_at_once(cap);
    e->h = xmalloc((size_t)cap * D * sizeof(float));
    e->ff = xmalloc((size_t)cap * FF * sizeof(float));
    e->qkv = xmalloc((size_t)cap * 3 * D * sizeof(float));
    e->ctx = xmalloc((size_t)cap * D * sizeof(float));
    e->g = xmalloc((size_t)cap * 2 * D * sizeof(float));
    e->a = xmalloc((size_t)cap * D * sizeof(float));
    e->qu = xmalloc((size_t)e->slots * cap * DH * sizeof(float));
    e->qv = xmalloc((size_t)e->slots * cap * DH * sizeof(float));
    e->ac = xmalloc((size_t)e->slots * cap * cap * sizeof(float));
    e->bd = xmalloc((size_t)e->slots * cap * P * sizeof(float));
    e->n_cap = cap;
}

/* ---- row-parallel elementwise passes -------------------------------------------- */

typedef struct {
    const float *x, *w, *b;
    float *y;
} ln_ctx;

static void ln_range(void *arg, int begin, int end, int thread) {
    ln_ctx *c = arg;
    pk_layer_norm(c->x + (size_t)begin * D, c->w, c->b, c->y + (size_t)begin * D, end - begin, D, LN_EPS);
}

static void layer_norm(pk_context *c, const float *x, const float *w, const float *b, float *y, int n) {
    ln_ctx ctx = {x, w, b, y};
    pk_pool_range(c->pool, n, 8, ln_range, &ctx);
}

static void glu_range(void *arg, int begin, int end, int thread) {
    scratch *s = arg;
    for (int i = begin; i < end; i++) {
        const float *g = s->g + (size_t)i * 2 * D;
        float *a = s->a + (size_t)i * D;
        for (int c = 0; c < D; c++) a[c] = g[c] * pk_sigmoid(g[D + c]);
    }
}

/* depthwise conv over time (kernel 9, zero padded), folded BatchNorm, SiLU */
static void dwconv_range(void *arg, int begin, int end, int thread) {
    scratch *s = arg;
    const pk_layer *L = s->L;
    int n = s->n;
    for (int t = begin; t < end; t++) {
        float *y = s->h + (size_t)t * D;
        memcpy(y, L->bn_t, D * sizeof(float));
        for (int k = 0; k < PK_ENC_KERNEL; k++) {
            int src = t + k - PK_ENC_KERNEL / 2;
            if (src < 0 || src >= n) continue;
            const float *a = s->a + (size_t)src * D, *w = L->dw_t + (size_t)k * D;
            for (int c = 0; c < D; c++) y[c] += a[c] * w[c] * L->bn_s[c];
        }
        pk_silu_inplace(y, D);
    }
}

/* ---- blocks ----------------------------------------------------------------------- */

static void feed_forward(const float *ln_w, const float *ln_b, const pk_linear *in, const pk_linear *out, float *x, scratch *s) {
    int n = s->n;
    layer_norm(s->c, x, ln_w, ln_b, s->h, n);
    pk_epilogue silu = {.act = PK_ACT_SILU};
    pk_linear_gemm(s->c, in, s->h, n, D, s->ff, FF, &silu);
    pk_epilogue residual = {.alpha = 0.5f, .accumulate = 1};
    pk_linear_gemm(s->c, out, s->ff, n, FF, x, D, &residual);
}

/* one attention head: scores from content and position terms, softmax, context */
static void head_task(void *arg, int slot, int thread) {
    scratch *s = arg;
    const pk_layer *L = s->L;
    int n = s->n, P = 2 * n - 1;
    int h = s->head_base + slot;
    const float scale = 0.08838834764831845f; /* 1 / sqrt(128) */
    const float *q = s->qkv + h * DH, *k = s->qkv + D + h * DH, *v = s->qkv + 2 * D + h * DH;
    const float *u = L->pos_bias_u + h * DH, *bv = L->pos_bias_v + h * DH;
    float *qu = s->qu + (size_t)slot * n * DH, *qv = s->qv + (size_t)slot * n * DH;
    float *ac = s->ac + (size_t)slot * n * n, *bd = s->bd + (size_t)slot * n * P;
    for (int i = 0; i < n; i++) {
        const float *qi = q + (size_t)i * 3 * D;
        for (int d = 0; d < DH; d++) {
            qu[i * DH + d] = qi[d] + u[d];
            qv[i * DH + d] = qi[d] + bv[d];
        }
    }
    pk_gemm_nt_st(n, n, DH, qu, DH, k, 3 * D, ac, n);
    pk_gemm_nt_st(n, P, DH, qv, DH, s->p + h * DH, D, bd, P);
    for (int i = 0; i < n; i++) {
        float *row = ac + (size_t)i * n;
        const float *rel = bd + (size_t)i * P + (n - 1 - i); /* rel[j] <-> position i - j */
        for (int j = 0; j < n; j++) row[j] = row[j] * scale + rel[j] * scale;
        pk_softmax_inplace(row, n);
    }
    pk_gemm_nn_st(n, DH, n, ac, n, v, 3 * D, s->ctx + h * DH, D);
}

static void attention(const pk_layer *L, float *x, scratch *s) {
    int n = s->n;
    layer_norm(s->c, x, L->ln_att_w, L->ln_att_b, s->h, n);
    pk_linear_gemm(s->c, &L->qkv, s->h, n, D, s->qkv, 3 * D, NULL);
    if (s->pe) { /* segment longer than the precomputed cache: project for this layer now */
        pk_linear_gemm(s->c, &L->pos, s->pe, 2 * n - 1, D, s->pbuf, D, NULL);
        s->p = s->pbuf;
    }
    for (s->head_base = 0; s->head_base < H; s->head_base += s->slots) {
        int group = H - s->head_base < s->slots ? H - s->head_base : s->slots;
        pk_pool_run(s->c->pool, group, head_task, s);
    }
    pk_epilogue residual = {.accumulate = 1};
    pk_linear_gemm(s->c, &L->att_out, s->ctx, n, D, x, D, &residual);
}

static void conv_module(const pk_layer *L, float *x, scratch *s) {
    int n = s->n;
    layer_norm(s->c, x, L->ln_conv_w, L->ln_conv_b, s->h, n);
    pk_linear_gemm(s->c, &L->conv_pw1, s->h, n, D, s->g, 2 * D, NULL);
    pk_pool_range(s->c->pool, n, 8, glu_range, s);
    pk_pool_range(s->c->pool, n, 8, dwconv_range, s);
    pk_epilogue residual = {.accumulate = 1};
    pk_linear_gemm(s->c, &L->conv_pw2, s->h, n, D, x, D, &residual);
}

void pk_encoder_run(pk_context *c, const pk_encoder *e, const float *x_in, int n, float *out) {
    int P = 2 * n - 1;
    scratch_prepare(c, n);
    pk_enc_scratch *w = &c->enc;
    scratch s = {.c = c, .n = n, .h = w->h, .ff = w->ff, .qkv = w->qkv, .qu = w->qu, .qv = w->qv, .ac = w->ac, .bd = w->bd,
                 .ctx = w->ctx, .g = w->g, .a = w->a, .slots = w->slots};
    if (n > e->cache_n) {
        s.pe = positional_encoding(e, n);
        s.pbuf = xmalloc((size_t)P * D * sizeof(float));
    }

    float *x = out;
    s.x = x;
    memcpy(x, x_in, (size_t)n * D * sizeof(float));
    for (int l = 0; l < PK_ENC_LAYERS; l++) {
        const pk_layer *L = &e->layers[l];
        s.L = L;
        if (!s.pe) s.p = e->pos_cache[l] + (size_t)(e->cache_n - n) * D; /* positions n-1 .. -(n-1) */
        feed_forward(L->ln_ff1_w, L->ln_ff1_b, &L->ff1_in, &L->ff1_out, x, &s);
        attention(L, x, &s);
        conv_module(L, x, &s);
        feed_forward(L->ln_ff2_w, L->ln_ff2_b, &L->ff2_in, &L->ff2_out, x, &s);
        layer_norm(c, x, L->ln_out_w, L->ln_out_b, s.h, n);
        memcpy(x, s.h, (size_t)n * D * sizeof(float));
    }
    free(s.pe);
    free(s.pbuf);
}
