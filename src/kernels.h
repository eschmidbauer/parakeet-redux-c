/* Architecture-specific inner loops: the GEMM micro-kernel (a PK_MR x PK_NR
 * tile of C accumulated over K from packed panels), a dot product, and the
 * 4x4 transposes used for packing. NEON, AVX2 and plain C variants; the C
 * variant is what every other target gets. Included by gemm.c only. */
#ifndef PK_KERNELS_H
#define PK_KERNELS_H

#include <stdint.h>
#include <string.h>

#if defined(__ARM_NEON) && !defined(PK_GENERIC_KERNELS)
#include <arm_neon.h>
#define PK_KERNEL_NAME "neon"
#define PK_MR 12
#define PK_NR 8
#elif defined(__AVX2__) && defined(__FMA__) && !defined(PK_GENERIC_KERNELS)
#include <immintrin.h>
#define PK_KERNEL_NAME "avx2"
#define PK_MR 6
#define PK_NR 16
#else
#define PK_KERNEL_NAME "generic"
#define PK_MR 4
#define PK_NR 8
#endif

enum { PK_STORE = 0, PK_ADD = 1, PK_ADD_SCALED = 2 };

/* tile[MR][NR] = sum_k Ap[k][MR] * Bp[k][NR]; then C (op)= tile for the first
 * `rows` rows and `cols` columns, op = store / add / add alpha*tile. */
static inline void pk_ukernel(int K, const float *restrict Ap, const float *restrict Bp, float *restrict C, int ldc,
                              int mode, float alpha, int rows, int cols);

static inline float pk_dot(const float *a, const float *b, int n);

/* -------------------------------------------------------------------------- */
#if defined(PK_KERNEL_NAME) && PK_MR == 12 && PK_NR == 8 && defined(__ARM_NEON)

static inline void pk_ukernel(int K, const float *restrict Ap, const float *restrict Bp, float *restrict C, int ldc,
                              int mode, float alpha, int rows, int cols) {
    float32x4_t c00 = vdupq_n_f32(0), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00, c30 = c00, c31 = c00;
    float32x4_t c40 = c00, c41 = c00, c50 = c00, c51 = c00, c60 = c00, c61 = c00, c70 = c00, c71 = c00;
    float32x4_t c80 = c00, c81 = c00, c90 = c00, c91 = c00, ca0 = c00, ca1 = c00, cb0 = c00, cb1 = c00;
    for (int k = 0; k < K; k++) {
        float32x4_t a0 = vld1q_f32(Ap), a1 = vld1q_f32(Ap + 4), a2 = vld1q_f32(Ap + 8);
        float32x4_t b0 = vld1q_f32(Bp), b1 = vld1q_f32(Bp + 4);
        Ap += 12;
        Bp += 8;
        c00 = vfmaq_laneq_f32(c00, b0, a0, 0); c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
        c10 = vfmaq_laneq_f32(c10, b0, a0, 1); c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
        c20 = vfmaq_laneq_f32(c20, b0, a0, 2); c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
        c30 = vfmaq_laneq_f32(c30, b0, a0, 3); c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
        c40 = vfmaq_laneq_f32(c40, b0, a1, 0); c41 = vfmaq_laneq_f32(c41, b1, a1, 0);
        c50 = vfmaq_laneq_f32(c50, b0, a1, 1); c51 = vfmaq_laneq_f32(c51, b1, a1, 1);
        c60 = vfmaq_laneq_f32(c60, b0, a1, 2); c61 = vfmaq_laneq_f32(c61, b1, a1, 2);
        c70 = vfmaq_laneq_f32(c70, b0, a1, 3); c71 = vfmaq_laneq_f32(c71, b1, a1, 3);
        c80 = vfmaq_laneq_f32(c80, b0, a2, 0); c81 = vfmaq_laneq_f32(c81, b1, a2, 0);
        c90 = vfmaq_laneq_f32(c90, b0, a2, 1); c91 = vfmaq_laneq_f32(c91, b1, a2, 1);
        ca0 = vfmaq_laneq_f32(ca0, b0, a2, 2); ca1 = vfmaq_laneq_f32(ca1, b1, a2, 2);
        cb0 = vfmaq_laneq_f32(cb0, b0, a2, 3); cb1 = vfmaq_laneq_f32(cb1, b1, a2, 3);
    }
    float32x4_t t[12][2] = {{c00, c01}, {c10, c11}, {c20, c21}, {c30, c31}, {c40, c41}, {c50, c51},
                            {c60, c61}, {c70, c71}, {c80, c81}, {c90, c91}, {ca0, ca1}, {cb0, cb1}};
    if (cols == 8) {
        for (int i = 0; i < rows; i++) {
            float *c = C + (size_t)i * ldc;
            if (mode == PK_STORE) {
                vst1q_f32(c, t[i][0]);
                vst1q_f32(c + 4, t[i][1]);
            } else if (mode == PK_ADD) {
                vst1q_f32(c, vaddq_f32(vld1q_f32(c), t[i][0]));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), t[i][1]));
            } else {
                vst1q_f32(c, vfmaq_n_f32(vld1q_f32(c), t[i][0], alpha));
                vst1q_f32(c + 4, vfmaq_n_f32(vld1q_f32(c + 4), t[i][1], alpha));
            }
        }
    } else {
        float tmp[8];
        for (int i = 0; i < rows; i++) {
            vst1q_f32(tmp, t[i][0]);
            vst1q_f32(tmp + 4, t[i][1]);
            float *c = C + (size_t)i * ldc;
            for (int j = 0; j < cols; j++) c[j] = mode == PK_STORE ? tmp[j] : mode == PK_ADD ? c[j] + tmp[j] : c[j] + alpha * tmp[j];
        }
    }
}

static inline float pk_dot(const float *a, const float *b, int n) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= n; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

/* -------------------------------------------------------------------------- */
#elif defined(PK_KERNEL_NAME) && PK_MR == 6 && PK_NR == 16 && defined(__AVX2__)

static inline void pk_ukernel(int K, const float *restrict Ap, const float *restrict Bp, float *restrict C, int ldc,
                              int mode, float alpha, int rows, int cols) {
    __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00;
    __m256 c30 = c00, c31 = c00, c40 = c00, c41 = c00, c50 = c00, c51 = c00;
    for (int k = 0; k < K; k++) {
        __m256 b0 = _mm256_loadu_ps(Bp), b1 = _mm256_loadu_ps(Bp + 8);
        __m256 a;
        a = _mm256_broadcast_ss(Ap + 0); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(Ap + 1); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(Ap + 2); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(Ap + 3); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(Ap + 4); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(Ap + 5); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
        Ap += 6;
        Bp += 16;
    }
    __m256 t[6][2] = {{c00, c01}, {c10, c11}, {c20, c21}, {c30, c31}, {c40, c41}, {c50, c51}};
    if (cols == 16) {
        __m256 va = _mm256_set1_ps(alpha);
        for (int i = 0; i < rows; i++) {
            float *c = C + (size_t)i * ldc;
            if (mode == PK_STORE) {
                _mm256_storeu_ps(c, t[i][0]);
                _mm256_storeu_ps(c + 8, t[i][1]);
            } else if (mode == PK_ADD) {
                _mm256_storeu_ps(c, _mm256_add_ps(_mm256_loadu_ps(c), t[i][0]));
                _mm256_storeu_ps(c + 8, _mm256_add_ps(_mm256_loadu_ps(c + 8), t[i][1]));
            } else {
                _mm256_storeu_ps(c, _mm256_fmadd_ps(t[i][0], va, _mm256_loadu_ps(c)));
                _mm256_storeu_ps(c + 8, _mm256_fmadd_ps(t[i][1], va, _mm256_loadu_ps(c + 8)));
            }
        }
    } else {
        float tmp[16];
        for (int i = 0; i < rows; i++) {
            _mm256_storeu_ps(tmp, t[i][0]);
            _mm256_storeu_ps(tmp + 8, t[i][1]);
            float *c = C + (size_t)i * ldc;
            for (int j = 0; j < cols; j++) c[j] = mode == PK_STORE ? tmp[j] : mode == PK_ADD ? c[j] + tmp[j] : c[j] + alpha * tmp[j];
        }
    }
}

static inline float pk_dot(const float *a, const float *b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = s0;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s0 = _mm256_add_ps(s0, s1);
    float tmp[8];
    _mm256_storeu_ps(tmp, s0);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

/* -------------------------------------------------------------------------- */
#else /* generic C: the compiler vectorizes the NR loop */

static inline void pk_ukernel(int K, const float *restrict Ap, const float *restrict Bp, float *restrict C, int ldc,
                              int mode, float alpha, int rows, int cols) {
    float t[PK_MR][PK_NR];
    memset(t, 0, sizeof t);
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < PK_MR; i++) {
            float a = Ap[i];
            for (int j = 0; j < PK_NR; j++) t[i][j] += a * Bp[j];
        }
        Ap += PK_MR;
        Bp += PK_NR;
    }
    for (int i = 0; i < rows; i++) {
        float *c = C + (size_t)i * ldc;
        if (mode == PK_STORE)
            for (int j = 0; j < cols; j++) c[j] = t[i][j];
        else if (mode == PK_ADD)
            for (int j = 0; j < cols; j++) c[j] += t[i][j];
        else
            for (int j = 0; j < cols; j++) c[j] += alpha * t[i][j];
    }
}

static inline float pk_dot(const float *a, const float *b, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    float sum = (s0 + s1) + (s2 + s3);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

#endif
#endif

/* ========================================================================== */
/* int8 path (--fast): activations quantized per row and 128-block, ternary   */
/* weights as int8, integer dot products, float32 accumulation per block.     */
/* ========================================================================== */

/* C tile (op)= sum over blocks of a_scale[i][blk] * w_scale[blk][j] * sum_k Aq[i][k] Bq[k][j].
 * Aq: MR rows with stride lda, this chunk's kc columns; a_scales and a_sums
 * (the sum of each block's quantized values): [MR][*] with stride sstride.
 * Bp: [kc/4][NR][4], values -1/0/1 plus PK_I8_WEIGHT_OFFSET; w_scales: [kc/128][NR].
 * kc is a multiple of 128. */
static inline void pk_ukernel_i8(int kc, const int8_t *restrict Aq, int lda, const float *restrict a_scales,
                                 const int32_t *restrict a_sums, int sstride, const int8_t *restrict Bp,
                                 const float *restrict w_scales, float *restrict C, int ldc, int mode, float alpha, int rows, int cols);

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && !defined(PK_GENERIC_KERNELS) && PK_MR == 12 && PK_NR == 8
#define PK_I8_KERNEL_NAME "neon-dotprod"
#define PK_I8_WEIGHT_OFFSET 0

static inline int32x4_t dup4(const int8_t *p) {
    int32_t v;
    memcpy(&v, p, 4);
    return vdupq_n_s32(v);
}

static inline void pk_ukernel_i8(int kc, const int8_t *restrict Aq, int lda, const float *restrict a_scales,
                                 const int32_t *restrict a_sums, int sstride, const int8_t *restrict Bp,
                                 const float *restrict w_scales, float *restrict C, int ldc, int mode, float alpha, int rows, int cols) {
    (void)a_sums;
    float32x4_t t[12][2];
    for (int i = 0; i < 12; i++) t[i][0] = t[i][1] = vdupq_n_f32(0);
    for (int blk = 0; blk < kc / 128; blk++) {
        int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00, c30 = c00, c31 = c00;
        int32x4_t c40 = c00, c41 = c00, c50 = c00, c51 = c00, c60 = c00, c61 = c00, c70 = c00, c71 = c00;
        int32x4_t c80 = c00, c81 = c00, c90 = c00, c91 = c00, ca0 = c00, ca1 = c00, cb0 = c00, cb1 = c00;
        const int8_t *bp = Bp + (size_t)blk * 128 * 8;
        const int8_t *ap = Aq + blk * 128;
        for (int k4 = 0; k4 < 32; k4++) {
            int8x16_t b0 = vld1q_s8(bp), b1 = vld1q_s8(bp + 16);
            bp += 32;
            int8x16_t a;
#define ROW(i, r0, r1) a = vreinterpretq_s8_s32(dup4(ap + (size_t)(i) * lda + k4 * 4)); r0 = vdotq_s32(r0, a, b0); r1 = vdotq_s32(r1, a, b1);
            ROW(0, c00, c01) ROW(1, c10, c11) ROW(2, c20, c21) ROW(3, c30, c31)
            ROW(4, c40, c41) ROW(5, c50, c51) ROW(6, c60, c61) ROW(7, c70, c71)
            ROW(8, c80, c81) ROW(9, c90, c91) ROW(10, ca0, ca1) ROW(11, cb0, cb1)
#undef ROW
        }
        int32x4_t acc[12][2] = {{c00, c01}, {c10, c11}, {c20, c21}, {c30, c31}, {c40, c41}, {c50, c51},
                                {c60, c61}, {c70, c71}, {c80, c81}, {c90, c91}, {ca0, ca1}, {cb0, cb1}};
        float32x4_t ws0 = vld1q_f32(w_scales + blk * 8), ws1 = vld1q_f32(w_scales + blk * 8 + 4);
        for (int i = 0; i < 12; i++) {
            float s = a_scales[(size_t)i * sstride + blk];
            t[i][0] = vfmaq_f32(t[i][0], vcvtq_f32_s32(acc[i][0]), vmulq_n_f32(ws0, s));
            t[i][1] = vfmaq_f32(t[i][1], vcvtq_f32_s32(acc[i][1]), vmulq_n_f32(ws1, s));
        }
    }
    if (cols == 8) {
        for (int i = 0; i < rows; i++) {
            float *c = C + (size_t)i * ldc;
            if (mode == PK_STORE) {
                vst1q_f32(c, t[i][0]);
                vst1q_f32(c + 4, t[i][1]);
            } else if (mode == PK_ADD) {
                vst1q_f32(c, vaddq_f32(vld1q_f32(c), t[i][0]));
                vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), t[i][1]));
            } else {
                vst1q_f32(c, vfmaq_n_f32(vld1q_f32(c), t[i][0], alpha));
                vst1q_f32(c + 4, vfmaq_n_f32(vld1q_f32(c + 4), t[i][1], alpha));
            }
        }
    } else {
        float tmp[8];
        for (int i = 0; i < rows; i++) {
            vst1q_f32(tmp, t[i][0]);
            vst1q_f32(tmp + 4, t[i][1]);
            float *c = C + (size_t)i * ldc;
            for (int j = 0; j < cols; j++) c[j] = mode == PK_STORE ? tmp[j] : mode == PK_ADD ? c[j] + tmp[j] : c[j] + alpha * tmp[j];
        }
    }
}

#elif defined(__AVX2__) && !defined(PK_GENERIC_KERNELS) && PK_MR == 6 && PK_NR == 16
#define PK_I8_KERNEL_NAME "avx2-maddubs"
#define PK_I8_WEIGHT_OFFSET 1 /* weights as 0/1/2 so they can be the unsigned operand of maddubs */

/* vpmaddubsw multiplies unsigned bytes (the weights, 0..2) by signed bytes (the
 * activations) and adds pairs; vpmaddwd with ones then adds the two pairs into
 * one int32 per group of four k. Using w + 1 adds sum(a) per block, which the
 * quantizer recorded in a_sums and is subtracted again here. */
static inline void pk_ukernel_i8(int kc, const int8_t *restrict Aq, int lda, const float *restrict a_scales,
                                 const int32_t *restrict a_sums, int sstride, const int8_t *restrict Bp,
                                 const float *restrict w_scales, float *restrict C, int ldc, int mode, float alpha, int rows, int cols) {
    float t[6][16] __attribute__((aligned(32)));
    memset(t, 0, sizeof t);
    const __m256i ones = _mm256_set1_epi16(1);
    for (int blk = 0; blk < kc / 128; blk++) {
        __m256i c00 = _mm256_setzero_si256(), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00;
        __m256i c30 = c00, c31 = c00, c40 = c00, c41 = c00, c50 = c00, c51 = c00;
        const int8_t *bp = Bp + (size_t)blk * 128 * 16;
        const int8_t *ap = Aq + blk * 128;
        for (int k4 = 0; k4 < 32; k4++) {
            __m256i b0 = _mm256_loadu_si256((const __m256i *)bp), b1 = _mm256_loadu_si256((const __m256i *)(bp + 32));
            bp += 64;
            __m256i a;
            int32_t v;
#define ROW(i, r0, r1)                                                                              \
    memcpy(&v, ap + (size_t)(i) * lda + k4 * 4, 4);                                                 \
    a = _mm256_set1_epi32(v);                                                                       \
    r0 = _mm256_add_epi32(r0, _mm256_madd_epi16(_mm256_maddubs_epi16(b0, a), ones));               \
    r1 = _mm256_add_epi32(r1, _mm256_madd_epi16(_mm256_maddubs_epi16(b1, a), ones));
            ROW(0, c00, c01) ROW(1, c10, c11) ROW(2, c20, c21) ROW(3, c30, c31) ROW(4, c40, c41) ROW(5, c50, c51)
#undef ROW
        }
        __m256i acc[6][2] = {{c00, c01}, {c10, c11}, {c20, c21}, {c30, c31}, {c40, c41}, {c50, c51}};
        __m256 ws0 = _mm256_loadu_ps(w_scales + blk * 16), ws1 = _mm256_loadu_ps(w_scales + blk * 16 + 8);
        for (int i = 0; i < 6; i++) {
            __m256i sum = _mm256_set1_epi32(a_sums[(size_t)i * sstride + blk]);
            __m256 s = _mm256_set1_ps(a_scales[(size_t)i * sstride + blk]);
            __m256 t0 = _mm256_load_ps(t[i]), t1 = _mm256_load_ps(t[i] + 8);
            t0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(acc[i][0], sum)), _mm256_mul_ps(ws0, s), t0);
            t1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(acc[i][1], sum)), _mm256_mul_ps(ws1, s), t1);
            _mm256_store_ps(t[i], t0);
            _mm256_store_ps(t[i] + 8, t1);
        }
    }
    for (int i = 0; i < rows; i++) {
        float *c = C + (size_t)i * ldc;
        if (mode == PK_STORE)
            for (int j = 0; j < cols; j++) c[j] = t[i][j];
        else if (mode == PK_ADD)
            for (int j = 0; j < cols; j++) c[j] += t[i][j];
        else
            for (int j = 0; j < cols; j++) c[j] += alpha * t[i][j];
    }
}

#else
#define PK_I8_KERNEL_NAME "generic"
#define PK_I8_WEIGHT_OFFSET 0

static inline void pk_ukernel_i8(int kc, const int8_t *restrict Aq, int lda, const float *restrict a_scales,
                                 const int32_t *restrict a_sums, int sstride, const int8_t *restrict Bp,
                                 const float *restrict w_scales, float *restrict C, int ldc, int mode, float alpha, int rows, int cols) {
    (void)a_sums;
    float t[PK_MR][PK_NR];
    memset(t, 0, sizeof t);
    for (int blk = 0; blk < kc / 128; blk++) {
        int32_t acc[PK_MR][PK_NR];
        memset(acc, 0, sizeof acc);
        const int8_t *bp = Bp + (size_t)blk * 128 * PK_NR;
        for (int k4 = 0; k4 < 32; k4++) {
            for (int i = 0; i < PK_MR; i++) {
                const int8_t *a = Aq + (size_t)i * lda + blk * 128 + k4 * 4;
                for (int j = 0; j < PK_NR; j++) {
                    const int8_t *b = bp + (k4 * PK_NR + j) * 4;
                    acc[i][j] += a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
                }
            }
        }
        for (int i = 0; i < PK_MR; i++) {
            float s = a_scales[(size_t)i * sstride + blk];
            for (int j = 0; j < PK_NR; j++) t[i][j] += (float)acc[i][j] * s * w_scales[blk * PK_NR + j];
        }
    }
    for (int i = 0; i < rows; i++) {
        float *c = C + (size_t)i * ldc;
        if (mode == PK_STORE)
            for (int j = 0; j < cols; j++) c[j] = t[i][j];
        else if (mode == PK_ADD)
            for (int j = 0; j < cols; j++) c[j] += t[i][j];
        else
            for (int j = 0; j < cols; j++) c[j] += alpha * t[i][j];
    }
}
#endif

/* ---- float16 -------------------------------------------------------------- */

static inline uint16_t pk_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0)); /* inf / nan */
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00);                                     /* overflow */
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1), mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1))) half++;
        return (uint16_t)(sign | half);
    }
    uint32_t half = ((uint32_t)exp << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) half++;
    return (uint16_t)(sign | half);
}

static inline float pk_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, mant = h & 0x3ff, x;
    if (exp == 0) {
        if (!mant) x = sign;
        else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            x = sign | (exp << 23) | ((mant & 0x3ff) << 13);
        }
    } else if (exp == 0x1f) {
        x = sign | 0x7f800000 | (mant << 13);
    } else {
        x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}

/* dot product of a float16 row with a float32 vector */
static inline float pk_dot_f16(const uint16_t *a, const float *b, int n) {
#if defined(__ARM_NEON) && !defined(PK_GENERIC_KERNELS)
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(a + i));
        s0 = vfmaq_f32(s0, vcvt_f32_f16(vget_low_f16(h)), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vcvt_f32_f16(vget_high_f16(h)), vld1q_f32(b + i + 4));
    }
    float sum = vaddvq_f32(vaddq_f32(s0, s1));
    for (; i < n; i++) sum += pk_f16_to_f32(a[i]) * b[i];
    return sum;
#elif defined(__AVX2__) && defined(__F16C__) && !defined(PK_GENERIC_KERNELS)
    __m256 s = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8)
        s = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a + i))), _mm256_loadu_ps(b + i), s);
    float tmp[8];
    _mm256_storeu_ps(tmp, s);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < n; i++) sum += pk_f16_to_f32(a[i]) * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += pk_f16_to_f32(a[i]) * b[i];
    return sum;
#endif
}
