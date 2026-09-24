#include "subsample.h"

#include <stdlib.h>
#include <string.h>

#include "gemm.h"
#include "mathx.h"
#include "pool.h"
#include "util.h"

#define C PK_SUB_CH

int pk_sub_frames(int T, int stages) {
    for (int i = 0; i < stages; i++) T = 1 + (T - 1) / 2;
    return T;
}

int pk_sub_valid(int len, int stages) {
    for (int i = 0; i < stages; i++) len = (int)floordiv(len - 1, 2) + 1;
    return len;
}

/* [256][1][3][3] -> [9][256] */
static float *transpose_taps(const pk_weights *w, const char *name) {
    float *src = pk_weights_f32(w, name, (size_t)C * 9);
    float *t = xmalloc((size_t)9 * C * sizeof(float));
    for (int c = 0; c < C; c++)
        for (int k = 0; k < 9; k++) t[(size_t)k * C + c] = src[(size_t)c * 9 + k];
    free(src);
    return t;
}

void pk_subsampler_load(pk_subsampler *s, const pk_weights *w) {
    memset(s, 0, sizeof *s);
    s->w0t = transpose_taps(w, "subsampling.conv0.weight");
    s->b0 = pk_weights_f32(w, "subsampling.conv0.bias", C);
    s->w2t = transpose_taps(w, "subsampling.dw2.weight");
    s->b2 = pk_weights_f32(w, "subsampling.dw2.bias", C);
    pk_linear_from_nk(&s->pw3, w, "subsampling.pw3.weight", C, C);
    s->b3 = pk_weights_f32(w, "subsampling.pw3.bias", C);
    s->w5t = transpose_taps(w, "subsampling.dw5.weight");
    s->b5 = pk_weights_f32(w, "subsampling.dw5.bias", C);
    pk_linear_from_nk(&s->pw6, w, "subsampling.pw6.weight", C, C);
    s->b6 = pk_weights_f32(w, "subsampling.pw6.bias", C);
    /* file weight is [4096][1024] with row index c*16+f; we want [1024][f*256+c] */
    float *kn = pk_weights_f32(w, "subsampling.linear.weight", (size_t)4096 * PK_D_MODEL);
    float *nk = xmalloc((size_t)PK_D_MODEL * 4096 * sizeof(float));
    for (int c = 0; c < C; c++)
        for (int f = 0; f < 16; f++)
            for (int n = 0; n < PK_D_MODEL; n++)
                nk[(size_t)n * 4096 + f * C + c] = kn[((size_t)c * 16 + f) * PK_D_MODEL + n];
    free(kn);
    pk_weight_from_dense(&s->lin, nk, PK_D_MODEL, 4096);
    free(nk);
    s->lin_b = pk_weights_f32(w, "subsampling.linear.bias", PK_D_MODEL);
}

void pk_subsampler_init_shapes(pk_subsampler *s) {
    memset(s, 0, sizeof *s);
    pk_weight_init(&s->pw3, C, C, 0, 0);
    pk_weight_init(&s->pw6, C, C, 0, 0);
    pk_weight_init(&s->lin, PK_D_MODEL, 4096, 0, 0);
}

void pk_subsampler_arrays(pk_subsampler *s, pk_array_fn fn, void *user) {
    fn(user, (void **)&s->w0t, (size_t)9 * C * sizeof(float));
    fn(user, (void **)&s->b0, C * sizeof(float));
    fn(user, (void **)&s->w2t, (size_t)9 * C * sizeof(float));
    fn(user, (void **)&s->b2, C * sizeof(float));
    pk_weight_arrays(&s->pw3, fn, user);
    fn(user, (void **)&s->b3, C * sizeof(float));
    fn(user, (void **)&s->w5t, (size_t)9 * C * sizeof(float));
    fn(user, (void **)&s->b5, C * sizeof(float));
    pk_weight_arrays(&s->pw6, fn, user);
    fn(user, (void **)&s->b6, C * sizeof(float));
    pk_weight_arrays(&s->lin, fn, user);
    fn(user, (void **)&s->lin_b, PK_D_MODEL * sizeof(float));
}

static void lock_visit(void *user, void **array, size_t bytes) {
    int *rc = user;
    if (bytes && pk_lock_pages(*array, bytes) != 0) *rc = -1;
}

int pk_subsampler_lock(const pk_subsampler *s) {
    int rc = 0;
    pk_subsampler_arrays((pk_subsampler *)s, lock_visit, &rc);
    return rc;
}

void pk_subsampler_free(pk_subsampler *s) {
    free(s->w0t); free(s->b0); free(s->w2t); free(s->b2); free(s->b3); free(s->w5t); free(s->b5); free(s->b6); free(s->lin_b);
    pk_linear_free(&s->pw3);
    pk_linear_free(&s->pw6);
    memset(s, 0, sizeof *s);
}

/* depthwise 3x3 stride 2 pad 1 over [T][F][256] -> [T2][F2][256], rows t2 in [begin, end) */
typedef struct {
    const float *in;
    int T, F;
    const float *wt, *bias;
    float *out;
    int T2, F2;
} dw_ctx;

static void depthwise_range(void *arg, int begin, int end, int thread) {
    dw_ctx *d = arg;
    for (int t2 = begin; t2 < end; t2++) {
        for (int f2 = 0; f2 < d->F2; f2++) {
            float *o = d->out + ((size_t)t2 * d->F2 + f2) * C;
            memcpy(o, d->bias, C * sizeof(float));
            for (int dt = 0; dt < 3; dt++) {
                int t = 2 * t2 - 1 + dt;
                if (t < 0 || t >= d->T) continue;
                for (int df = 0; df < 3; df++) {
                    int f = 2 * f2 - 1 + df;
                    if (f < 0 || f >= d->F) continue;
                    const float *x = d->in + ((size_t)t * d->F + f) * C;
                    const float *w = d->wt + (size_t)(dt * 3 + df) * C;
                    for (int c = 0; c < C; c++) o[c] += x[c] * w[c];
                }
            }
        }
    }
}

static void depthwise(pk_context *c, const float *in, int T, int F, const float *wt, const float *bias, float *out, int T2, int F2) {
    dw_ctx d = {in, T, F, wt, bias, out, T2, F2};
    pk_pool_range(c->pool, T2, 4, depthwise_range, &d);
}

/* stage 1: conv 1 -> 256 channels, rows t1 in [begin, end) */
typedef struct {
    const pk_subsampler *s;
    const float *feat;
    int T, T1, F1;
    float *s1;
} conv1_ctx;

static void conv1_range(void *arg, int begin, int end, int thread) {
    conv1_ctx *c = arg;
    const int F0 = 128;
    for (int t1 = begin; t1 < end; t1++) {
        for (int f1 = 0; f1 < c->F1; f1++) {
            float *o = c->s1 + ((size_t)t1 * c->F1 + f1) * C;
            memcpy(o, c->s->b0, C * sizeof(float));
            for (int dt = 0; dt < 3; dt++) {
                int t = 2 * t1 - 1 + dt;
                if (t < 0 || t >= c->T) continue;
                for (int df = 0; df < 3; df++) {
                    int f = 2 * f1 - 1 + df;
                    if (f < 0 || f >= F0) continue;
                    float v = c->feat[(size_t)t * F0 + f];
                    const float *w = c->s->w0t + (size_t)(dt * 3 + df) * C;
                    for (int k = 0; k < C; k++) o[k] += v * w[k];
                }
            }
        }
    }
}

static void zero_rows_from(float *x, int from, int T, size_t row) {
    if (from < T) memset(x + (size_t)from * row, 0, (size_t)(T - from) * row * sizeof(float));
}

float *pk_subsample(pk_context *c, const pk_subsampler *s, const float *feat, int T, int len, int *T3_out, int *len3_out) {
    int F1 = 64, F2 = 32, F3 = 16;
    int T1 = pk_sub_frames(T, 1), T2 = pk_sub_frames(T, 2), T3 = pk_sub_frames(T, 3);
    int len1 = pk_sub_valid(len, 1), len2 = pk_sub_valid(len, 2), len3 = pk_sub_valid(len, 3);

    /* stage 1: conv 1 -> 256 channels */
    float *s1 = xmalloc((size_t)T1 * F1 * C * sizeof(float));
    conv1_ctx c1 = {s, feat, T, T1, F1, s1};
    pk_pool_range(c->pool, T1, 4, conv1_range, &c1);
    zero_rows_from(s1, len1, T1, (size_t)F1 * C);
    pk_relu_inplace(s1, (size_t)T1 * F1 * C);

    /* stage 2: depthwise, mask, pointwise, relu */
    float *s2 = xmalloc((size_t)T2 * F2 * C * sizeof(float));
    depthwise(c, s1, T1, F1, s->w2t, s->b2, s2, T2, F2);
    free(s1);
    zero_rows_from(s2, len2, T2, (size_t)F2 * C);
    float *s2p = xmalloc((size_t)T2 * F2 * C * sizeof(float));
    pk_epilogue bias_relu3 = {.bias = s->b3, .act = PK_ACT_RELU};
    pk_linear_gemm(c, &s->pw3, s2, T2 * F2, C, s2p, C, &bias_relu3);
    free(s2);

    /* stage 3 */
    float *s3 = xmalloc((size_t)T3 * F3 * C * sizeof(float));
    depthwise(c, s2p, T2, F2, s->w5t, s->b5, s3, T3, F3);
    free(s2p);
    zero_rows_from(s3, len3, T3, (size_t)F3 * C);
    float *s3p = xmalloc((size_t)T3 * F3 * C * sizeof(float));
    pk_epilogue bias_relu6 = {.bias = s->b6, .act = PK_ACT_RELU};
    pk_linear_gemm(c, &s->pw6, s3, T3 * F3, C, s3p, C, &bias_relu6);
    free(s3);

    float *out = xmalloc((size_t)T3 * PK_D_MODEL * sizeof(float));
    pk_linear_apply(c, &s->lin, s3p, T3, s->lin_b, out);
    free(s3p);
    *T3_out = T3;
    *len3_out = len3;
    return out;
}
