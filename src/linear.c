#include "linear.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

void pk_linear_from_ternary(pk_linear *lin, const pk_weights *w, const char *name, int n, int k, int fast, pk_pool *pool) {
    char tensor[128];
    snprintf(tensor, sizeof tensor, "%s.codes", name);
    const pk_tensor *codes = pk_weights_get(w, tensor, PK_DT_CODES, (size_t)n * k);
    snprintf(tensor, sizeof tensor, "%s.scales", name);
    const pk_tensor *scales = pk_weights_get(w, tensor, PK_DT_F16, (size_t)n * (k / PK_TERNARY_BLOCK));
    pk_weight_from_codes(lin, codes->data, (const uint16_t *)scales->data, n, k, fast, pool);
}

void pk_linear_from_kn(pk_linear *lin, const pk_weights *w, const char *name, int n, int k) {
    float *kn = pk_weights_f32(w, name, (size_t)n * k);
    float *nk = xmalloc((size_t)n * k * sizeof(float));
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++) nk[(size_t)j * k + i] = kn[(size_t)i * n + j];
    free(kn);
    pk_weight_from_dense(lin, nk, n, k);
    free(nk);
}

void pk_linear_from_nk(pk_linear *lin, const pk_weights *w, const char *name, int n, int k) {
    float *nk = pk_weights_f32(w, name, (size_t)n * k);
    pk_weight_from_dense(lin, nk, n, k);
    free(nk);
}

void pk_linear_free(pk_linear *lin) { pk_weight_free(lin); }

void pk_linear_apply(pk_context *c, const pk_linear *lin, const float *x, int rows, const float *bias, float *y) {
    pk_epilogue ep = {.bias = bias};
    pk_gemm_w(c, lin, x, rows, lin->k, y, lin->n, &ep);
}
