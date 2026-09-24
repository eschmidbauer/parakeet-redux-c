/* Elementwise math that the compiler can vectorize: a Cephes-style expf on
 * plain arithmetic (no libm call in the loop), and the activations built on it. */
#ifndef PK_MATHX_H
#define PK_MATHX_H

#include <math.h>
#include <stdint.h>
#include <string.h>

static inline float pk_expf(float x) {
    x = x > 88.0f ? 88.0f : x;
    x = x < -87.3365447504019f ? -87.3365447504019f : x;
    float k = (x * 1.44269504088896341f + 12582912.0f) - 12582912.0f; /* round(x / ln 2) */
    float r = x - k * 0.693359375f + k * 2.12194440e-4f;              /* Cody-Waite reduction */
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * r * r + r + 1.0f;
    int32_t ki = (int32_t)k;
    uint32_t bits = (uint32_t)(ki + 127) << 23;
    float scale;
    memcpy(&scale, &bits, sizeof scale);
    return p * scale;
}

static inline float pk_sigmoid(float x) { return 1.0f / (1.0f + pk_expf(-x)); }
static inline float pk_silu(float x) { return x * pk_sigmoid(x); }

static inline void pk_silu_inplace(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = pk_silu(x[i]);
}

static inline void pk_relu_inplace(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

static inline void pk_softmax_inplace(float *x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) m = x[i] > m ? x[i] : m;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = pk_expf(x[i] - m);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* y = LayerNorm(x) * w + b over the last dimension d, for n rows. */
static inline void pk_layer_norm(const float *x, const float *w, const float *b, float *y, int n, int d,
                                 float eps) {
    for (int i = 0; i < n; i++) {
        const float *row = x + (size_t)i * d;
        float *out = y + (size_t)i * d;
        double sum = 0.0;
        for (int j = 0; j < d; j++) sum += row[j];
        float mean = (float)(sum / d);
        double var = 0.0;
        for (int j = 0; j < d; j++) {
            float t = row[j] - mean;
            var += (double)t * t;
        }
        float inv = 1.0f / sqrtf((float)(var / d) + eps);
        for (int j = 0; j < d; j++) out[j] = (row[j] - mean) * inv * w[j] + b[j];
    }
}

#endif
