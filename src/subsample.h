#ifndef PK_SUBSAMPLE_H
#define PK_SUBSAMPLE_H

#include "linear.h"
#include "weights.h"

#define PK_SUB_CH 256
#define PK_D_MODEL 1024

/* dw-striding subsampler: conv 3x3/2 (1->256), relu, dw 3x3/2, pw 1x1, relu,
 * dw 3x3/2, pw 1x1, relu, linear 4096->1024. Activations are channels-last. */
typedef struct {
    float *w0t, *b0; /* [9][256] */
    float *w2t, *b2; /* depthwise [9][256] */
    pk_linear pw3;
    float *b3;
    float *w5t, *b5;
    pk_linear pw6;
    float *b6;
    pk_linear lin; /* [1024][4096], columns permuted to the channels-last order */
    float *lin_b;
} pk_subsampler;

void pk_subsampler_load(pk_subsampler *s, const pk_weights *w);
void pk_subsampler_free(pk_subsampler *s);
int pk_subsampler_lock(const pk_subsampler *s);
void pk_subsampler_init_shapes(pk_subsampler *s);
void pk_subsampler_arrays(pk_subsampler *s, pk_array_fn fn, void *user);

/* feat[T][128] with `len` valid frames -> out[T3][1024]; *len3 rows are valid. */
float *pk_subsample(pk_context *c, const pk_subsampler *s, const float *feat, int T, int len, int *T3, int *len3);

/* Output frame count and valid length after k stride-2 stages (ONNX semantics). */
int pk_sub_frames(int T, int stages);
int pk_sub_valid(int len, int stages);

#endif
