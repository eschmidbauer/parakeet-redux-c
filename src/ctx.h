/* The per-thread execution context behind the public pk_context: a thread
 * pool plus the scratch buffers of the matrix products, timing and debug
 * settings. Contexts are never shared between threads; models are. */
#ifndef PK_CTX_H
#define PK_CTX_H

#include <stddef.h>
#include <stdint.h>

#include "pool.h"

/* encoder working memory, sized for segments of up to n_cap frames */
typedef struct {
    int n_cap;
    int slots;      /* attention heads processed at once */
    float *h, *ff, *qkv, *ctx, *g, *a;
    float *qu, *qv, *ac, *bd; /* [slots][...] */
} pk_enc_scratch;

struct pk_context {
    pk_pool *pool;
    /* packed activations for the float32 kernels, [Mp][K][MR] */
    float *apack;
    size_t apack_cap;
    /* quantized activations for the int8 kernels: rows [Mp*MR][K], scales and
     * per-block row sums [Mp*MR][K/128] */
    int8_t *aq;
    float *as;
    int32_t *asum;
    size_t aq_cap, as_cap;
    pk_enc_scratch enc;
    int verbose;
    const char *dump_dir;
    struct {
        double features, subsample, encoder, decode, vad;
    } timing;
};

typedef struct pk_context pk_context;

#endif
