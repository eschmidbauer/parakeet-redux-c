#include "logmel.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "fft.h"
#include "gemm.h"
#include "pool.h"
#include "util.h"

#define BINS (PK_N_FFT / 2 + 1)
#define PAD (PK_N_FFT / 2)
#define WIN 400

static void init_tables(pk_features *f) {
    /* symmetric Hann window of 400 samples, centred in the 512-point frame */
    f->window = xcalloc(PK_N_FFT, sizeof(float));
    for (int i = 0; i < WIN; i++) f->window[(PK_N_FFT - WIN) / 2 + i] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * i / (WIN - 1)));
    f->fft = xmalloc(sizeof(pk_fft));
    pk_fft_init(f->fft, PK_N_FFT);
}

void pk_features_load(pk_features *f, const pk_weights *w) {
    memset(f, 0, sizeof *f);
    float *mel = pk_weights_f32(w, "mel.filterbank", (size_t)PK_N_MEL * BINS);
    pk_weight_from_dense(&f->mel, mel, PK_N_MEL, BINS);
    free(mel);
    init_tables(f);
}

void pk_features_init_shapes(pk_features *f) {
    memset(f, 0, sizeof *f);
    pk_weight_init(&f->mel, PK_N_MEL, BINS, 0, 0);
    init_tables(f);
}

void pk_features_arrays(pk_features *f, pk_array_fn fn, void *user) { pk_weight_arrays(&f->mel, fn, user); }

int pk_features_lock(const pk_features *f) {
    return pk_weight_lock(&f->mel) | pk_lock_pages(f->window, PK_N_FFT * sizeof(float));
}

void pk_features_free(pk_features *f) {
    pk_linear_free(&f->mel);
    free(f->window);
    if (f->fft) pk_fft_free(f->fft);
    free(f->fft);
    f->fft = NULL;
}

typedef struct {
    const pk_features *f;
    const float *emph;
    int n;
    float *power;
} stft_ctx;

/* frame t covers emph[160 t - 256 .. 160 t + 256), zero outside the signal */
static void stft_range(void *arg, int begin, int end, int thread) {
    stft_ctx *c = arg;
    float frame[PK_N_FFT], spec[2 * BINS], work[PK_N_FFT];
    for (int t = begin; t < end; t++) {
        int start = t * PK_HOP - PAD;
        for (int j = 0; j < PK_N_FFT; j++) {
            int idx = start + j;
            frame[j] = (idx >= 0 && idx < c->n ? c->emph[idx] : 0.0f) * c->f->window[j];
        }
        pk_fft_real(c->f->fft, frame, spec, work);
        float *p = c->power + (size_t)t * BINS;
        for (int k = 0; k < BINS; k++) p[k] = spec[2 * k] * spec[2 * k] + spec[2 * k + 1] * spec[2 * k + 1];
    }
}

float *pk_features_compute(pk_context *c, const pk_features *f, const float *pcm, int n, int *T_out, int *valid_out) {
    int T = n / PK_HOP + 1;
    int valid = n / PK_HOP;

    float *emph = xmalloc((size_t)n * sizeof(float));
    emph[0] = pcm[0];
    for (int i = 1; i < n; i++) emph[i] = pcm[i] - 0.97f * pcm[i - 1];

    float *power = xmalloc((size_t)T * BINS * sizeof(float));
    stft_ctx sc = {f, emph, n, power};
    pk_pool_range(c->pool, T, 64, stft_range, &sc);
    free(emph);

    float *feat = xmalloc((size_t)T * PK_N_MEL * sizeof(float));
    pk_linear_apply(c, &f->mel, power, T, NULL, feat);
    free(power);
    for (size_t i = 0; i < (size_t)T * PK_N_MEL; i++) feat[i] = logf(feat[i] + 5.9604645e-8f);

    /* per-band normalization over the valid frames (unbiased variance) */
    for (int b = 0; b < PK_N_MEL; b++) {
        double sum = 0.0;
        for (int t = 0; t < valid; t++) sum += feat[(size_t)t * PK_N_MEL + b];
        float mean = (float)(sum / valid);
        double var = 0.0;
        for (int t = 0; t < valid; t++) {
            float d = feat[(size_t)t * PK_N_MEL + b] - mean;
            var += (double)d * d;
        }
        float denom = sqrtf((float)(var / (valid > 1 ? valid - 1 : 1))) + 1e-5f;
        for (int t = 0; t < valid; t++) feat[(size_t)t * PK_N_MEL + b] = (feat[(size_t)t * PK_N_MEL + b] - mean) / denom;
        for (int t = valid; t < T; t++) feat[(size_t)t * PK_N_MEL + b] = 0.0f;
    }
    *T_out = T;
    *valid_out = valid;
    return feat;
}
