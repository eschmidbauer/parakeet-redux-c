#include "vad.h"

#include <stdlib.h>
#include <string.h>

#include "gemm.h"
#include "mathx.h"
#include "util.h"

#define CH PK_VAD_CH
#define KCTX 5

void pk_vad_load(pk_vad *v, const pk_weights *w) {
    memset(v, 0, sizeof *v);
    pk_linear_from_nk(&v->proj, w, "vad.proj.weight", CH, 1024); /* [128][1024][1] */
    v->proj_b = pk_weights_f32(w, "vad.proj.bias", CH);
    float *src = pk_weights_f32(w, "vad.ctx.weight", (size_t)CH * CH * KCTX); /* [out][in][k] */
    float *ctx_w = xmalloc((size_t)CH * KCTX * CH * sizeof(float));
    for (int o = 0; o < CH; o++)
        for (int i = 0; i < CH; i++)
            for (int k = 0; k < KCTX; k++) ctx_w[(size_t)o * KCTX * CH + k * CH + i] = src[((size_t)o * CH + i) * KCTX + k];
    free(src);
    pk_weight_from_dense(&v->ctx, ctx_w, CH, KCTX * CH);
    free(ctx_w);
    v->ctx_b = pk_weights_f32(w, "vad.ctx.bias", CH);
    v->out_w = pk_weights_f32(w, "vad.out.weight", CH); /* [1][128][1] */
    float *ob = pk_weights_f32(w, "vad.out.bias", 1);
    v->out_b = ob[0];
    free(ob);
}

void pk_vad_init_shapes(pk_vad *v) {
    memset(v, 0, sizeof *v);
    pk_weight_init(&v->proj, CH, 1024, 0, 0);
    pk_weight_init(&v->ctx, CH, KCTX * CH, 0, 0);
}

void pk_vad_arrays(pk_vad *v, pk_array_fn fn, void *user) {
    pk_weight_arrays(&v->proj, fn, user);
    fn(user, (void **)&v->proj_b, CH * sizeof(float));
    pk_weight_arrays(&v->ctx, fn, user);
    fn(user, (void **)&v->ctx_b, CH * sizeof(float));
    fn(user, (void **)&v->out_w, CH * sizeof(float));
}

static void lock_visit(void *user, void **array, size_t bytes) {
    int *rc = user;
    if (bytes && pk_lock_pages(*array, bytes) != 0) *rc = -1;
}

int pk_vad_lock(const pk_vad *v) {
    int rc = 0;
    pk_vad_arrays((pk_vad *)v, lock_visit, &rc);
    return rc;
}

void pk_vad_free(pk_vad *v) {
    pk_linear_free(&v->proj);
    pk_linear_free(&v->ctx);
    free(v->proj_b); free(v->ctx_b); free(v->out_w);
    memset(v, 0, sizeof *v);
}

float *pk_vad_run(pk_context *c, const pk_vad *v, const float *sub, int T) {
    float *a = xmalloc((size_t)T * CH * sizeof(float));
    pk_epilogue proj_ep = {.bias = v->proj_b, .act = PK_ACT_SILU};
    pk_linear_gemm(c, &v->proj, sub, T, 1024, a, CH, &proj_ep);
    float *cols = xcalloc((size_t)T * KCTX * CH, sizeof(float));
    for (int t = 0; t < T; t++)
        for (int k = 0; k < KCTX; k++) {
            int src = t + k - KCTX / 2;
            if (src < 0 || src >= T) continue;
            memcpy(cols + ((size_t)t * KCTX + k) * CH, a + (size_t)src * CH, CH * sizeof(float));
        }
    float *y = xmalloc((size_t)T * CH * sizeof(float));
    pk_epilogue ctx_ep = {.bias = v->ctx_b, .act = PK_ACT_SILU};
    pk_linear_gemm(c, &v->ctx, cols, T, KCTX * CH, y, CH, &ctx_ep);
    free(cols);
    free(a);
    float *probs = xmalloc((size_t)T * sizeof(float));
    for (int t = 0; t < T; t++) {
        float *row = y + (size_t)t * CH;
        float acc = v->out_b;
        for (int c = 0; c < CH; c++) acc += row[c] * v->out_w[c];
        probs[t] = pk_sigmoid(acc);
    }
    free(y);
    return probs;
}
