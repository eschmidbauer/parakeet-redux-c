#ifndef PK_LOGMEL_H
#define PK_LOGMEL_H

#define PK_N_MEL 128
#define PK_HOP 160
#define PK_N_FFT 512

#include "linear.h"
#include "weights.h"

struct pk_fft;

typedef struct {
    pk_linear mel;      /* [128][257] Slaney mel filterbank from preprocessor.onnx */
    float *window;      /* [512] Hann(400) zero-padded */
    struct pk_fft *fft; /* 512-point real FFT */
} pk_features;

void pk_features_load(pk_features *f, const pk_weights *w);
/* Window and FFT tables without the mel filterbank (to be mapped in). */
void pk_features_init_shapes(pk_features *f);
void pk_features_arrays(pk_features *f, pk_array_fn fn, void *user);
void pk_features_free(pk_features *f);
int pk_features_lock(const pk_features *f);

/* Log-mel features feat[T][128] for n samples: T = n/160 + 1 frames, of which
 * the first *valid = n/160 are normalized and the rest are zero. */
float *pk_features_compute(pk_context *c, const pk_features *f, const float *pcm, int n, int *T, int *valid);

#endif
