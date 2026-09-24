/* Dense matrix products. The default backend is our own: packed panels, an
 * architecture-specific micro-kernel (kernels.h) and a thread pool. With
 * PK_BLAS_ACCELERATE / PK_BLAS_OPENBLAS the same API forwards to CBLAS, which
 * serves as the reference implementation. All matrices are row major.
 *
 * Threaded products take a pk_context (its pool and scratch); the _st variants
 * run on the calling thread and may be used inside pool tasks. */
#ifndef PK_GEMM_H
#define PK_GEMM_H

#include <stddef.h>
#include <stdint.h>

#include "ctx.h"

#define PK_TERNARY_BLOCK 128

enum { PK_ACT_NONE = 0, PK_ACT_RELU = 1, PK_ACT_SILU = 2 };

typedef struct {
    const float *bias; /* [N], added before the activation; may be NULL */
    int act;
    float alpha;       /* scale on the product (0 means 1) */
    int accumulate;    /* C += alpha * A W^T; bias and act are ignored */
} pk_epilogue;

/* A weight matrix W[n][k], stored the way the backend wants it. */
typedef struct {
    int n, k, n_pad;
    int fast;        /* ternary weight to be used with int8 activations */
    float *dense;    /* CBLAS backend: [n][k] */
    float *packed;   /* own backend, float32: [n_pad/NR][k][NR] */
    uint8_t *codes;  /* own backend, ternary: [n_pad/NR][k][NR/4], 2 bits per weight (value + 1) */
    float *scales;   /* own backend, ternary: [n_pad/NR][k/128][NR] */
} pk_weight;

void pk_weight_from_dense(pk_weight *w, const float *W, int n, int k);
/* codes[n][k] in {-1, 0, 1}, scales[n][k/128] */
void pk_weight_from_ternary(pk_weight *w, const int8_t *codes, const float *scales, int n, int k, int fast);
/* the weights file's form: row-major 2-bit codes (value + 1, four per byte) and float16 scales;
 * packed on the pool when one is given */
struct pk_pool;
void pk_weight_from_codes(pk_weight *w, const uint8_t *codes, const uint16_t *scales, int n, int k, int fast, struct pk_pool *pool);
void pk_weight_free(pk_weight *w);
size_t pk_weight_bytes(const pk_weight *w);

/* Every heap array of a structure, in a fixed order: (address of the pointer,
 * size in bytes). Used to lock, save and map weights. */
typedef void (*pk_array_fn)(void *user, void **array, size_t bytes);
/* Sets the shape without allocating data, so the arrays can be mapped in. */
void pk_weight_init(pk_weight *w, int n, int k, int ternary, int fast);
void pk_weight_arrays(pk_weight *w, pk_array_fn fn, void *user);
int pk_weight_lock(const pk_weight *w);
/* The name of the packing layout, which a saved weight file must match. */
const char *pk_weight_layout(void);

/* C[M][ldc] = epilogue(A[M][lda] W^T). ep may be NULL. */
void pk_gemm_w(pk_context *c, const pk_weight *w, const float *A, int M, int lda, float *C, int ldc, const pk_epilogue *ep);

/* C[M][N] = A[M][K] B^T with B[N][K] (nt), or A B with B[K][N] (nn). */
void pk_gemm_nt(pk_context *c, int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc);
void pk_gemm_nn(pk_context *c, int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc);
void pk_gemm_nt_st(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc);
void pk_gemm_nn_st(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc);

/* y[N] = W[N][K] x[K] */
void pk_gemv(pk_context *c, const float *W, int N, int K, const float *x, float *y);
void pk_gemv_st(const float *W, int N, int K, const float *x, float *y);
/* the same with float16 weights */
void pk_gemv_f16(pk_context *c, const uint16_t *W, int N, int K, const float *x, float *y);
uint16_t pk_float_to_half(float f);

void pk_context_free_scratch(pk_context *c);
const char *pk_gemm_backend(int fast);

#endif
