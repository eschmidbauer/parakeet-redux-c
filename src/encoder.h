#ifndef PK_ENCODER_H
#define PK_ENCODER_H

#include <stddef.h>

#include "linear.h"
#include "weights.h"

#define PK_ENC_LAYERS 24
#define PK_ENC_D 1024
#define PK_ENC_FF 4096
#define PK_ENC_HEADS 8
#define PK_ENC_DHEAD 128
#define PK_ENC_KERNEL 9

typedef struct {
    pk_linear pos; /* this layer's slice of the positional projection, [1024][1024] */
    float *ln_ff1_w, *ln_ff1_b;
    pk_linear ff1_in, ff1_out;
    float *ln_att_w, *ln_att_b;
    pk_linear qkv;
    float *pos_bias_u, *pos_bias_v; /* [1024] = heads x dhead */
    pk_linear att_out;
    float *ln_conv_w, *ln_conv_b;
    pk_linear conv_pw1;
    float *dw_t; /* depthwise weight [9][1024] */
    float *bn_s, *bn_t; /* folded BatchNorm: y = x * s + t */
    pk_linear conv_pw2;
    float *ln_ff2_w, *ln_ff2_b;
    pk_linear ff2_in, ff2_out;
    float *ln_out_w, *ln_out_b;
} pk_layer;

typedef struct {
    pk_layer layers[PK_ENC_LAYERS];
    float *inv_freq; /* [512] */
    /* positional projections precomputed for segments up to cache_n frames: [layer][2*cache_n-1][1024] */
    int cache_n;
    float *pos_cache[PK_ENC_LAYERS];
} pk_encoder;

void pk_encoder_load(pk_encoder *e, const pk_weights *w, int fast, pk_pool *pool);
/* Precompute the positional projections for segments of up to max_frames. */
void pk_encoder_prepare(pk_context *c, pk_encoder *e, int max_frames);
void pk_encoder_free(pk_encoder *e);
size_t pk_encoder_bytes(const pk_encoder *e);
int pk_encoder_lock(const pk_encoder *e);
/* Shapes only (for mapping a saved file in), and every array in a fixed order. */
void pk_encoder_init_shapes(pk_encoder *e, int fast, int cache_n);
void pk_encoder_arrays(pk_encoder *e, pk_array_fn fn, void *user);

/* x[n][1024] (subsampler output, valid rows only) -> out[n][1024]. Thread safe
 * for a shared encoder as long as every caller brings its own context. */
void pk_encoder_run(pk_context *c, const pk_encoder *e, const float *x, int n, float *out);
void pk_encoder_free_scratch(pk_context *c);

#endif
