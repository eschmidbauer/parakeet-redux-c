#include "gemm.h"

#include <stdlib.h>
#include <string.h>

#include "mathx.h"
#include "pool.h"
#include "util.h"

static void epilogue_rows(float *C, int ldc, int M, int col0, int cols, const pk_epilogue *ep) {
    if (!ep || ep->accumulate) return;
    float alpha = ep->alpha == 0.0f ? 1.0f : ep->alpha;
    if (alpha == 1.0f && !ep->bias && ep->act == PK_ACT_NONE) return;
    for (int i = 0; i < M; i++) {
        float *c = C + (size_t)i * ldc + col0;
        if (alpha != 1.0f)
            for (int j = 0; j < cols; j++) c[j] *= alpha;
        if (ep->bias)
            for (int j = 0; j < cols; j++) c[j] += ep->bias[col0 + j];
        if (ep->act == PK_ACT_RELU) pk_relu_inplace(c, (size_t)cols);
        else if (ep->act == PK_ACT_SILU) pk_silu_inplace(c, (size_t)cols);
    }
}

void pk_context_free_scratch(pk_context *c) {
    free(c->apack);
    free(c->aq);
    free(c->as);
    free(c->asum);
    c->apack = NULL;
    c->aq = NULL;
    c->as = NULL;
    c->asum = NULL;
    c->apack_cap = c->aq_cap = c->as_cap = 0;
}

#include "kernels.h"

uint16_t pk_float_to_half(float f) { return pk_f32_to_f16(f); }

#if defined(PK_BLAS_ACCELERATE) || defined(PK_BLAS_OPENBLAS)
/* ========================================================================== */
/* CBLAS backend                                                              */
/* ========================================================================== */
#if defined(PK_BLAS_ACCELERATE)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

void pk_weight_from_dense(pk_weight *w, const float *W, int n, int k) {
    memset(w, 0, sizeof *w);
    w->n = w->n_pad = n;
    w->k = k;
    w->dense = xmalloc((size_t)n * k * sizeof(float));
    memcpy(w->dense, W, (size_t)n * k * sizeof(float));
}

void pk_weight_from_ternary(pk_weight *w, const int8_t *codes, const float *scales, int n, int k, int fast) {
    memset(w, 0, sizeof *w);
    w->n = w->n_pad = n;
    w->k = k;
    w->fast = fast;
    w->dense = xmalloc((size_t)n * k * sizeof(float));
    int blocks = k / PK_TERNARY_BLOCK;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            w->dense[(size_t)i * k + j] = (float)codes[(size_t)i * k + j] * scales[(size_t)i * blocks + j / PK_TERNARY_BLOCK];
}

void pk_weight_from_codes(pk_weight *w, const uint8_t *codes, const uint16_t *scales, int n, int k, int fast, struct pk_pool *pool) {
    memset(w, 0, sizeof *w);
    w->n = w->n_pad = n;
    w->k = k;
    w->fast = fast;
    w->dense = xmalloc((size_t)n * k * sizeof(float));
    int blocks = k / PK_TERNARY_BLOCK;
    for (int i = 0; i < n; i++) {
        const uint8_t *row = codes + (size_t)i * (k / 4);
        for (int j = 0; j < k; j++) {
            int code = (row[j / 4] >> (2 * (j & 3))) & 3;
            w->dense[(size_t)i * k + j] = (float)(code - 1) * pk_f16_to_f32(scales[(size_t)i * blocks + j / PK_TERNARY_BLOCK]);
        }
    }
}

void pk_weight_free(pk_weight *w) {
    free(w->dense);
    memset(w, 0, sizeof *w);
}

size_t pk_weight_bytes(const pk_weight *w) { return (size_t)w->n * w->k * sizeof(float); }

void pk_weight_init(pk_weight *w, int n, int k, int ternary, int fast) {
    memset(w, 0, sizeof *w);
    w->n = w->n_pad = n;
    w->k = k;
    w->fast = ternary ? fast : 0;
}

void pk_weight_arrays(pk_weight *w, pk_array_fn fn, void *user) { fn(user, (void **)&w->dense, (size_t)w->n * w->k * sizeof(float)); }
int pk_weight_lock(const pk_weight *w) { return pk_lock_pages(w->dense, pk_weight_bytes(w)); }
const char *pk_weight_layout(void) { return "dense-f32"; }

void pk_gemm_w(pk_context *c, const pk_weight *w, const float *A, int M, int lda, float *C, int ldc, const pk_epilogue *ep) {
    if (M == 0) return;
    float alpha = ep && ep->accumulate ? (ep->alpha == 0.0f ? 1.0f : ep->alpha) : 1.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, w->n, w->k, alpha, A, lda, w->dense, w->k,
                ep && ep->accumulate ? 1.0f : 0.0f, C, ldc);
    epilogue_rows(C, ldc, M, 0, w->n, ep);
}

void pk_gemm_nt(pk_context *c, int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc) {
    if (M == 0 || N == 0) return;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
}

void pk_gemm_nn(pk_context *c, int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc) {
    if (M == 0 || N == 0) return;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
}

void pk_gemm_nt_st(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc) {
    pk_gemm_nt(NULL, M, N, K, A, lda, B, ldb, C, ldc);
}

void pk_gemm_nn_st(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc) {
    pk_gemm_nn(NULL, M, N, K, A, lda, B, ldb, C, ldc);
}

void pk_gemv_st(const float *W, int N, int K, const float *x, float *y) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K, 1.0f, W, K, x, 1, 0.0f, y, 1);
}

void pk_gemv(pk_context *c, const float *W, int N, int K, const float *x, float *y) { pk_gemv_st(W, N, K, x, y); }

void pk_gemv_f16(pk_context *c, const uint16_t *W, int N, int K, const float *x, float *y) {
    for (int i = 0; i < N; i++) y[i] = pk_dot_f16(W + (size_t)i * K, x, K);
}

const char *pk_gemm_backend(int fast) {
#if defined(PK_BLAS_ACCELERATE)
    return "cblas (Accelerate)";
#else
    return "cblas (OpenBLAS)";
#endif
}

#else
/* ========================================================================== */
/* Own backend                                                                */
/* ========================================================================== */

#define KC 512 /* K block: A panel MR x KC and B panel KC x NR stay in L1 */

/* per-thread scratch for B panels and, in the _st products, packed A; freed
 * when the thread exits through a pthread key destructor */
#include <pthread.h>

typedef struct {
    float *buf;
    size_t cap;
} scratch;

static _Thread_local scratch *tls;
static pthread_key_t tls_key;
static pthread_once_t tls_once = PTHREAD_ONCE_INIT;

static void tls_destroy(void *p) {
    scratch *s = p;
    free(s->buf);
    free(s);
}

static void tls_init(void) { pthread_key_create(&tls_key, tls_destroy); }

static float *tls_alloc(size_t floats) {
    if (!tls) {
        pthread_once(&tls_once, tls_init);
        tls = xcalloc(1, sizeof *tls);
        pthread_setspecific(tls_key, tls);
    }
    if (tls->cap < floats) {
        free(tls->buf);
        tls->buf = NULL;
        tls->cap = 0;
        tls->buf = xaligned((floats + floats / 4) * sizeof(float));
        tls->cap = floats + floats / 4;
    }
    return tls->buf;
}

static float *apack_alloc(pk_context *c, size_t floats) {
    if (c->apack_cap < floats) {
        free(c->apack);
        c->apack = NULL;
        c->apack_cap = 0;
        c->apack = xaligned((floats + floats / 4) * sizeof(float));
        c->apack_cap = floats + floats / 4;
    }
    return c->apack;
}

static void aq_alloc(pk_context *c, size_t bytes, size_t scales) {
    if (c->aq_cap < bytes) {
        free(c->aq);
        c->aq = NULL;
        c->aq_cap = 0;
        c->aq = xaligned(bytes + bytes / 4);
        c->aq_cap = bytes + bytes / 4;
    }
    if (c->as_cap < scales) {
        free(c->as);
        free(c->asum);
        c->as = NULL;
        c->asum = NULL;
        c->as_cap = 0;
        c->as = xaligned((scales + scales / 4) * sizeof(float));
        c->asum = xaligned((scales + scales / 4) * sizeof(int32_t));
        c->as_cap = scales + scales / 4;
    }
}

/* ---- packing ------------------------------------------------------------- */

/* Ap[(mp * K + k) * MR + i] = A[mp * MR + i][k], zero beyond M */
static void pack_a_panel(const float *A, int M, int lda, int K, int mp, float *Ap) {
    float *dst = Ap + (size_t)mp * K * PK_MR;
    int row0 = mp * PK_MR;
    int rows = M - row0 < PK_MR ? M - row0 : PK_MR;
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        for (int i = 0; i < PK_MR; i++) {
            if (i < rows) {
                const float *src = A + (size_t)(row0 + i) * lda + k;
                dst[(size_t)k * PK_MR + i] = src[0];
                dst[(size_t)(k + 1) * PK_MR + i] = src[1];
                dst[(size_t)(k + 2) * PK_MR + i] = src[2];
                dst[(size_t)(k + 3) * PK_MR + i] = src[3];
            } else {
                dst[(size_t)k * PK_MR + i] = dst[(size_t)(k + 1) * PK_MR + i] = 0.0f;
                dst[(size_t)(k + 2) * PK_MR + i] = dst[(size_t)(k + 3) * PK_MR + i] = 0.0f;
            }
        }
    }
    for (; k < K; k++)
        for (int i = 0; i < PK_MR; i++) dst[(size_t)k * PK_MR + i] = i < rows ? A[(size_t)(row0 + i) * lda + k] : 0.0f;
}

typedef struct {
    const float *A;
    int M, lda, K;
    float *Ap;
} pack_a_ctx;

static void pack_a_task(void *arg, int task, int thread) {
    pack_a_ctx *c = arg;
    pack_a_panel(c->A, c->M, c->lda, c->K, task, c->Ap);
}

static void pack_a_all(pk_pool *pool, const float *A, int M, int lda, int K, float *Ap) {
    int Mp = (M + PK_MR - 1) / PK_MR;
    pack_a_ctx c = {A, M, lda, K, Ap};
    pk_pool_run(pool, Mp, pack_a_task, &c);
}

/* int8 quantization of A rows with one scale per 128-block; zero rows beyond M */
typedef struct {
    const float *A;
    int M, lda, K;
    int8_t *aq;
    float *as;
    int32_t *asum;
} quant_ctx;

static void quant_range(void *arg, int begin, int end, int thread) {
    quant_ctx *c = arg;
    int blocks = c->K / PK_TERNARY_BLOCK;
    for (int i = begin; i < end; i++) {
        int8_t *q = c->aq + (size_t)i * c->K;
        float *sc = c->as + (size_t)i * blocks;
        int32_t *sums = c->asum + (size_t)i * blocks;
        if (i >= c->M) {
            memset(q, 0, (size_t)c->K);
            memset(sc, 0, (size_t)blocks * sizeof(float));
            memset(sums, 0, (size_t)blocks * sizeof(int32_t));
            continue;
        }
        const float *row = c->A + (size_t)i * c->lda;
        for (int b = 0; b < blocks; b++) {
            const float *x = row + b * PK_TERNARY_BLOCK;
            float amax = 0.0f;
            for (int k = 0; k < PK_TERNARY_BLOCK; k++) {
                float v = x[k] < 0 ? -x[k] : x[k];
                amax = v > amax ? v : amax;
            }
            float scale = amax / 127.0f, inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            sc[b] = scale;
            int8_t *dst = q + b * PK_TERNARY_BLOCK;
            int32_t sum = 0;
            for (int k = 0; k < PK_TERNARY_BLOCK; k++) {
                float v = x[k] * inv;
                v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);
                dst[k] = (int8_t)(v >= 0.0f ? (int)(v + 0.5f) : -(int)(-v + 0.5f));
                sum += dst[k];
            }
            sums[b] = sum;
        }
    }
}

static void quantize_a(pk_context *c, const float *A, int M, int lda, int K) {
    int Mp = (M + PK_MR - 1) / PK_MR;
    aq_alloc(c, (size_t)Mp * PK_MR * K, (size_t)Mp * PK_MR * (K / PK_TERNARY_BLOCK));
    quant_ctx q = {A, M, lda, K, c->aq, c->as, c->asum};
    pk_pool_range(c->pool, Mp * PK_MR, PK_MR, quant_range, &q);
}

/* ternary panel -> int8 Bp[k/4][NR][4] for the dot-product kernel; the kernel
 * says whether it wants the values as -1/0/1 or shifted up by one (unsigned) */
static void dequant_b_i8(const pk_weight *w, int panel, int k0, int kc, int8_t *Bp) {
    const uint8_t *codes = w->codes + ((size_t)panel * w->k + k0) * (PK_NR / 4);
    for (int k = 0; k < kc; k++) {
        uint32_t word = 0;
        for (int b = 0; b < PK_NR / 4; b++) word |= (uint32_t)codes[(size_t)k * (PK_NR / 4) + b] << (8 * b);
        int8_t *dst = Bp + (size_t)(k / 4) * PK_NR * 4 + (k & 3);
        for (int j = 0; j < PK_NR; j++) dst[j * 4] = (int8_t)((int)((word >> (2 * j)) & 3) - 1 + PK_I8_WEIGHT_OFFSET);
    }
}

/* B[N][K] -> Bp[k][NR] for columns n0.., zero beyond N */
static void pack_b_nt(const float *B, int N, int ldb, int n0, int k0, int kc, float *Bp) {
    int cols = N - n0 < PK_NR ? N - n0 : PK_NR;
    for (int j = 0; j < PK_NR; j++) {
        if (j < cols) {
            const float *src = B + (size_t)(n0 + j) * ldb + k0;
            for (int k = 0; k < kc; k++) Bp[(size_t)k * PK_NR + j] = src[k];
        } else {
            for (int k = 0; k < kc; k++) Bp[(size_t)k * PK_NR + j] = 0.0f;
        }
    }
}

/* B[K][N] -> Bp[k][NR] */
static void pack_b_nn(const float *B, int N, int ldb, int n0, int k0, int kc, float *Bp) {
    int cols = N - n0 < PK_NR ? N - n0 : PK_NR;
    for (int k = 0; k < kc; k++) {
        const float *src = B + (size_t)(k0 + k) * ldb + n0;
        float *dst = Bp + (size_t)k * PK_NR;
        if (cols == PK_NR) {
            memcpy(dst, src, PK_NR * sizeof(float));
        } else {
            for (int j = 0; j < cols; j++) dst[j] = src[j];
            for (int j = cols; j < PK_NR; j++) dst[j] = 0.0f;
        }
    }
}

/* ternary panel -> float32 Bp[k][NR] */
static void dequant_b(const pk_weight *w, int panel, int k0, int kc, float *Bp) {
    const uint8_t *codes = w->codes + ((size_t)panel * w->k + k0) * (PK_NR / 4);
    int blocks = w->k / PK_TERNARY_BLOCK;
    for (int k = 0; k < kc; k++) {
        const float *scale = w->scales + ((size_t)panel * blocks + (k0 + k) / PK_TERNARY_BLOCK) * PK_NR;
        uint32_t word = 0;
        for (int b = 0; b < PK_NR / 4; b++) word |= (uint32_t)codes[(size_t)k * (PK_NR / 4) + b] << (8 * b);
        float *dst = Bp + (size_t)k * PK_NR;
        for (int j = 0; j < PK_NR; j++) dst[j] = (float)((int)((word >> (2 * j)) & 3) - 1) * scale[j];
    }
}

/* ---- weights --------------------------------------------------------------- */

void pk_weight_from_dense(pk_weight *w, const float *W, int n, int k) {
    memset(w, 0, sizeof *w);
    w->n = n;
    w->k = k;
    w->n_pad = (n + PK_NR - 1) / PK_NR * PK_NR;
    w->packed = xaligned((size_t)w->n_pad * k * sizeof(float));
    for (int panel = 0; panel < w->n_pad / PK_NR; panel++) pack_b_nt(W, n, k, panel * PK_NR, 0, k, w->packed + (size_t)panel * k * PK_NR);
}

void pk_weight_from_ternary(pk_weight *w, const int8_t *codes, const float *scales, int n, int k, int fast) {
    memset(w, 0, sizeof *w);
    if (k % PK_TERNARY_BLOCK) die("ternary weight with k=%d, not a multiple of %d", k, PK_TERNARY_BLOCK);
    w->n = n;
    w->k = k;
    w->fast = fast;
    w->n_pad = (n + PK_NR - 1) / PK_NR * PK_NR;
    int panels = w->n_pad / PK_NR, blocks = k / PK_TERNARY_BLOCK;
    w->codes = xaligned((size_t)panels * k * (PK_NR / 4));
    w->scales = xaligned((size_t)panels * blocks * PK_NR * sizeof(float));
    for (int panel = 0; panel < panels; panel++) {
        for (int kk = 0; kk < k; kk++) {
            uint32_t word = 0;
            for (int j = 0; j < PK_NR; j++) {
                int row = panel * PK_NR + j;
                int code = row < n ? codes[(size_t)row * k + kk] + 1 : 1;
                word |= (uint32_t)code << (2 * j);
            }
            uint8_t *dst = w->codes + ((size_t)panel * k + kk) * (PK_NR / 4);
            for (int b = 0; b < PK_NR / 4; b++) dst[b] = (uint8_t)(word >> (8 * b));
        }
        for (int b = 0; b < blocks; b++)
            for (int j = 0; j < PK_NR; j++) {
                int row = panel * PK_NR + j;
                w->scales[((size_t)panel * blocks + b) * PK_NR + j] = row < n ? scales[(size_t)row * blocks + b] : 0.0f;
            }
    }
}

typedef struct {
    pk_weight *w;
    const uint8_t *codes;
    const uint16_t *scales;
} pack_codes_ctx;

/* one panel of NR rows: for each byte column (4 weights) gather one byte per row
 * and spread it into the four k-words of the panel */
static void pack_codes_task(void *arg, int panel, int thread) {
    pack_codes_ctx *c = arg;
    pk_weight *w = c->w;
    int k = w->k, n = w->n, blocks = k / PK_TERNARY_BLOCK;
    size_t row_bytes = (size_t)k / 4;
    uint8_t *dst = w->codes + (size_t)panel * k * (PK_NR / 4);
    for (int kb = 0; kb < k / 4; kb++) {
        uint8_t b[PK_NR];
        for (int j = 0; j < PK_NR; j++) {
            int row = panel * PK_NR + j;
            b[j] = row < n ? c->codes[(size_t)row * row_bytes + kb] : 0x55; /* 0x55: four "zero" codes */
        }
        for (int sub = 0; sub < 4; sub++) {
            uint32_t word = 0;
            for (int j = 0; j < PK_NR; j++) word |= (uint32_t)((b[j] >> (2 * sub)) & 3) << (2 * j);
            uint8_t *out = dst + ((size_t)kb * 4 + sub) * (PK_NR / 4);
            for (int q = 0; q < PK_NR / 4; q++) out[q] = (uint8_t)(word >> (8 * q));
        }
    }
    for (int blk = 0; blk < blocks; blk++)
        for (int j = 0; j < PK_NR; j++) {
            int row = panel * PK_NR + j;
            w->scales[((size_t)panel * blocks + blk) * PK_NR + j] = row < n ? pk_f16_to_f32(c->scales[(size_t)row * blocks + blk]) : 0.0f;
        }
}

void pk_weight_from_codes(pk_weight *w, const uint8_t *codes, const uint16_t *scales, int n, int k, int fast, struct pk_pool *pool) {
    memset(w, 0, sizeof *w);
    if (k % PK_TERNARY_BLOCK) die("ternary weight with k=%d, not a multiple of %d", k, PK_TERNARY_BLOCK);
    w->n = n;
    w->k = k;
    w->fast = fast;
    w->n_pad = (n + PK_NR - 1) / PK_NR * PK_NR;
    int panels = w->n_pad / PK_NR, blocks = k / PK_TERNARY_BLOCK;
    w->codes = xaligned((size_t)panels * k * (PK_NR / 4));
    w->scales = xaligned((size_t)panels * blocks * PK_NR * sizeof(float));
    pack_codes_ctx c = {w, codes, scales};
    pk_pool_run(pool, panels, pack_codes_task, &c);
}

void pk_weight_free(pk_weight *w) {
    free(w->packed);
    free(w->codes);
    free(w->scales);
    memset(w, 0, sizeof *w);
}

size_t pk_weight_bytes(const pk_weight *w) {
    if (w->packed) return (size_t)w->n_pad * w->k * sizeof(float);
    return (size_t)w->n_pad * w->k / 4 + (size_t)w->n_pad * (w->k / PK_TERNARY_BLOCK) * sizeof(float);
}

void pk_weight_init(pk_weight *w, int n, int k, int ternary, int fast) {
    memset(w, 0, sizeof *w);
    w->n = n;
    w->k = k;
    w->n_pad = (n + PK_NR - 1) / PK_NR * PK_NR;
    w->fast = ternary ? fast : 0;
    /* a non-NULL marker tells the visitor which arrays this weight has; the
     * visitor may replace the pointers, a mapped model never frees them */
    if (ternary) w->codes = (uint8_t *)1, w->scales = (float *)1;
    else w->packed = (float *)1;
}

void pk_weight_arrays(pk_weight *w, pk_array_fn fn, void *user) {
    if (w->packed) fn(user, (void **)&w->packed, (size_t)w->n_pad * w->k * sizeof(float));
    if (w->codes) fn(user, (void **)&w->codes, (size_t)w->n_pad * w->k / 4);
    if (w->scales) fn(user, (void **)&w->scales, (size_t)w->n_pad * (w->k / PK_TERNARY_BLOCK) * sizeof(float));
}

static void lock_array(void *user, void **array, size_t bytes) {
    int *rc = user;
    if (pk_lock_pages(*array, bytes) != 0) *rc = -1;
}

int pk_weight_lock(const pk_weight *w) {
    int rc = 0;
    pk_weight_arrays((pk_weight *)w, lock_array, &rc);
    return rc;
}

const char *pk_weight_layout(void) { return "panels-" PK_KERNEL_NAME; }

/* ---- the products ------------------------------------------------------------ */

typedef struct {
    const pk_weight *w;       /* packed weight, or NULL for a general B */
    const float *B;           /* general B */
    int ldb, b_is_nt;
    int M, N, K;
    const float *Ap;          /* packed A (float32 path) */
    const int8_t *aq;         /* quantized A (int8 path) */
    const float *as;
    const int32_t *asum;
    float *C;
    int ldc;
    const pk_epilogue *ep;
    int panels_per_task;
} gemm_ctx;

static void gemm_panels_i8(const gemm_ctx *g, int p0, int p1) {
    int Mp = (g->M + PK_MR - 1) / PK_MR;
    int accumulate = g->ep && g->ep->accumulate;
    float alpha = g->ep && g->ep->alpha != 0.0f ? g->ep->alpha : 1.0f;
    int blocks = g->K / PK_TERNARY_BLOCK;
    int8_t *Bbuf = (int8_t *)tls_alloc((size_t)KC * PK_NR / sizeof(float) + 16);
    for (int k0 = 0; k0 < g->K; k0 += KC) {
        int kc = g->K - k0 < KC ? g->K - k0 : KC;
        int mode = accumulate ? PK_ADD_SCALED : (k0 == 0 ? PK_STORE : PK_ADD);
        for (int panel = p0; panel < p1; panel++) {
            dequant_b_i8(g->w, panel, k0, kc, Bbuf);
            const float *wsc = g->w->scales + ((size_t)panel * blocks + k0 / PK_TERNARY_BLOCK) * PK_NR;
            int cols = g->N - panel * PK_NR < PK_NR ? g->N - panel * PK_NR : PK_NR;
            for (int mp = 0; mp < Mp; mp++) {
                int rows = g->M - mp * PK_MR < PK_MR ? g->M - mp * PK_MR : PK_MR;
                pk_ukernel_i8(kc, g->aq + (size_t)mp * PK_MR * g->K + k0, g->K, g->as + (size_t)mp * PK_MR * blocks + k0 / PK_TERNARY_BLOCK,
                              g->asum + (size_t)mp * PK_MR * blocks + k0 / PK_TERNARY_BLOCK, blocks, Bbuf, wsc,
                              g->C + (size_t)mp * PK_MR * g->ldc + panel * PK_NR, g->ldc, mode, alpha, rows, cols);
            }
        }
    }
    int col0 = p0 * PK_NR, col1 = p1 * PK_NR < g->N ? p1 * PK_NR : g->N;
    epilogue_rows(g->C, g->ldc, g->M, col0, col1 - col0, g->ep);
}

static void gemm_panels(const gemm_ctx *g, int p0, int p1) {
    if (g->aq) {
        gemm_panels_i8(g, p0, p1);
        return;
    }
    int Mp = (g->M + PK_MR - 1) / PK_MR;
    int accumulate = g->ep && g->ep->accumulate;
    float alpha = g->ep && g->ep->alpha != 0.0f ? g->ep->alpha : 1.0f;
    float *Bbuf = NULL;
    if (!(g->w && g->w->packed)) Bbuf = tls_alloc((size_t)KC * PK_NR);
    for (int k0 = 0; k0 < g->K; k0 += KC) {
        int kc = g->K - k0 < KC ? g->K - k0 : KC;
        int mode = accumulate ? PK_ADD_SCALED : (k0 == 0 ? PK_STORE : PK_ADD);
        for (int panel = p0; panel < p1; panel++) {
            const float *Bp;
            if (g->w && g->w->packed) {
                Bp = g->w->packed + ((size_t)panel * g->K + k0) * PK_NR;
            } else if (g->w) {
                dequant_b(g->w, panel, k0, kc, Bbuf);
                Bp = Bbuf;
            } else {
                if (g->b_is_nt) pack_b_nt(g->B, g->N, g->ldb, panel * PK_NR, k0, kc, Bbuf);
                else pack_b_nn(g->B, g->N, g->ldb, panel * PK_NR, k0, kc, Bbuf);
                Bp = Bbuf;
            }
            int cols = g->N - panel * PK_NR < PK_NR ? g->N - panel * PK_NR : PK_NR;
            for (int mp = 0; mp < Mp; mp++) {
                int rows = g->M - mp * PK_MR < PK_MR ? g->M - mp * PK_MR : PK_MR;
                pk_ukernel(kc, g->Ap + ((size_t)mp * g->K + k0) * PK_MR, Bp, g->C + (size_t)mp * PK_MR * g->ldc + panel * PK_NR,
                           g->ldc, mode, alpha, rows, cols);
            }
        }
    }
    int col0 = p0 * PK_NR, col1 = p1 * PK_NR < g->N ? p1 * PK_NR : g->N;
    epilogue_rows(g->C, g->ldc, g->M, col0, col1 - col0, g->ep);
}

static void gemm_task(void *arg, int task, int thread) {
    const gemm_ctx *g = arg;
    int panels = (g->N + PK_NR - 1) / PK_NR;
    int p0 = task * g->panels_per_task, p1 = p0 + g->panels_per_task;
    if (p1 > panels) p1 = panels;
    if (p0 < p1) gemm_panels(g, p0, p1);
}

static void gemm_run(pk_pool *pool, gemm_ctx *g) {
    if (g->M == 0 || g->N == 0) return;
    int panels = (g->N + PK_NR - 1) / PK_NR;
    int want = pk_pool_size(pool) * 3;
    g->panels_per_task = (panels + want - 1) / want;
    if (g->panels_per_task < 1) g->panels_per_task = 1;
    int tasks = (panels + g->panels_per_task - 1) / g->panels_per_task;
    pk_pool_run(pool, tasks, gemm_task, g);
}

void pk_gemm_w(pk_context *c, const pk_weight *w, const float *A, int M, int lda, float *C, int ldc, const pk_epilogue *ep) {
    if (M == 0) return;
    int Mp = (M + PK_MR - 1) / PK_MR;
    gemm_ctx g = {.w = w, .M = M, .N = w->n, .K = w->k, .C = C, .ldc = ldc, .ep = ep};
    if (w->codes && w->fast) {
        quantize_a(c, A, M, lda, w->k);
        g.aq = c->aq;
        g.as = c->as;
        g.asum = c->asum;
    } else {
        g.Ap = apack_alloc(c, (size_t)Mp * w->k * PK_MR);
        pack_a_all(c->pool, A, M, lda, w->k, c->apack);
    }
    gemm_run(c->pool, &g);
}

static void gemm_general(pk_context *c, int M, int N, int K, const float *A, int lda, const float *B, int ldb, int nt, float *C, int ldc) {
    if (M == 0 || N == 0) return;
    int Mp = (M + PK_MR - 1) / PK_MR;
    gemm_ctx g = {.B = B, .ldb = ldb, .b_is_nt = nt, .M = M, .N = N, .K = K, .C = C, .ldc = ldc};
    if (c) {
        g.Ap = apack_alloc(c, (size_t)Mp * K * PK_MR);
        pack_a_all(c->pool, A, M, lda, K, c->apack);
        gemm_run(c->pool, &g);
        return;
    }
    /* single thread: packed A followed by the B panel in the thread-local scratch */
    float *Ap = tls_alloc((size_t)Mp * K * PK_MR + (size_t)KC * PK_NR);
    pack_a_all(NULL, A, M, lda, K, Ap);
    float *Bbuf = Ap + (size_t)Mp * K * PK_MR;
    int panels = (N + PK_NR - 1) / PK_NR;
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kc = K - k0 < KC ? K - k0 : KC;
        int mode = k0 == 0 ? PK_STORE : PK_ADD;
        for (int panel = 0; panel < panels; panel++) {
            if (nt) pack_b_nt(B, N, ldb, panel * PK_NR, k0, kc, Bbuf);
            else pack_b_nn(B, N, ldb, panel * PK_NR, k0, kc, Bbuf);
            int cols = N - panel * PK_NR < PK_NR ? N - panel * PK_NR : PK_NR;
            for (int mp = 0; mp < Mp; mp++) {
                int rows = M - mp * PK_MR < PK_MR ? M - mp * PK_MR : PK_MR;
                pk_ukernel(kc, Ap + ((size_t)mp * K + k0) * PK_MR, Bbuf, C + (size_t)mp * PK_MR * ldc + panel * PK_NR, ldc, mode, 1.0f,
                           rows, cols);
            }
        }
    }
}

void pk_gemm_nt(pk_context *c, int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc) {
    gemm_general(c, M, N, K, A, lda, B, ldb, 1, C, ldc);
}

void pk_gemm_nn(pk_context *c, int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc) {
    gemm_general(c, M, N, K, A, lda, B, ldb, 0, C, ldc);
}

void pk_gemm_nt_st(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc) {
    gemm_general(NULL, M, N, K, A, lda, B, ldb, 1, C, ldc);
}

void pk_gemm_nn_st(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc) {
    gemm_general(NULL, M, N, K, A, lda, B, ldb, 0, C, ldc);
}

/* ---- matrix-vector -------------------------------------------------------------- */

typedef struct {
    const float *W;
    const uint16_t *Wh;
    const float *x;
    float *y;
    int N, K, rows_per_task;
} gemv_ctx;

static void gemv_task(void *arg, int task, int thread) {
    const gemv_ctx *g = arg;
    int r0 = task * g->rows_per_task, r1 = r0 + g->rows_per_task;
    if (r1 > g->N) r1 = g->N;
    if (g->Wh)
        for (int i = r0; i < r1; i++) g->y[i] = pk_dot_f16(g->Wh + (size_t)i * g->K, g->x, g->K);
    else
        for (int i = r0; i < r1; i++) g->y[i] = pk_dot(g->W + (size_t)i * g->K, g->x, g->K);
}

void pk_gemv_st(const float *W, int N, int K, const float *x, float *y) {
    for (int i = 0; i < N; i++) y[i] = pk_dot(W + (size_t)i * K, x, K);
}

static void gemv_run(pk_context *c, const float *W, const uint16_t *Wh, int N, int K, const float *x, float *y) {
    int threads = pk_pool_size(c ? c->pool : NULL);
    gemv_ctx g = {W, Wh, x, y, N, K, 0};
    if (threads == 1 || (size_t)N * K < 65536) {
        g.rows_per_task = N;
        gemv_task(&g, 0, 0);
        return;
    }
    g.rows_per_task = (N + threads * 2 - 1) / (threads * 2);
    pk_pool_run(c->pool, (N + g.rows_per_task - 1) / g.rows_per_task, gemv_task, &g);
}

void pk_gemv(pk_context *c, const float *W, int N, int K, const float *x, float *y) { gemv_run(c, W, NULL, N, K, x, y); }

void pk_gemv_f16(pk_context *c, const uint16_t *W, int N, int K, const float *x, float *y) { gemv_run(c, NULL, W, N, K, x, y); }

const char *pk_gemm_backend(int fast) {
    return fast ? "own kernels (" PK_KERNEL_NAME ", int8 " PK_I8_KERNEL_NAME ")" : "own kernels (" PK_KERNEL_NAME ")";
}

#endif
