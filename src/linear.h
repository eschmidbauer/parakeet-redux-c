/* Linear layers: a pk_weight loaded from the weights file plus y = x W^T helpers. */
#ifndef PK_LINEAR_H
#define PK_LINEAR_H

#include "gemm.h"
#include "pool.h"
#include "weights.h"

typedef pk_weight pk_linear;

/* Ternary weight from "<name>.codes" [n][k] and "<name>.scales" [n][k/128] (float16). */
void pk_linear_from_ternary(pk_linear *lin, const pk_weights *w, const char *name, int n, int k, int fast, pk_pool *pool);
/* Dense weight stored [k][n] in the file (X @ W). */
void pk_linear_from_kn(pk_linear *lin, const pk_weights *w, const char *name, int n, int k);
/* Dense weight stored [n][k] (W^T applied, e.g. Conv 1x1). */
void pk_linear_from_nk(pk_linear *lin, const pk_weights *w, const char *name, int n, int k);
void pk_linear_free(pk_linear *lin);

/* y[rows][n] = x[rows][k] W^T (+ bias) */
void pk_linear_apply(pk_context *c, const pk_linear *lin, const float *x, int rows, const float *bias, float *y);
/* the general form with leading dimensions and a fused epilogue */
static inline void pk_linear_gemm(pk_context *c, const pk_linear *lin, const float *x, int rows, int lda, float *y, int ldc,
                                  const pk_epilogue *ep) {
    pk_gemm_w(c, lin, x, rows, lda, y, ldc, ep);
}

#endif
