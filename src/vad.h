#ifndef PK_VAD_H
#define PK_VAD_H

#include "linear.h"
#include "weights.h"

#define PK_VAD_CH 128

typedef struct {
    pk_linear proj; /* [128][1024] */
    float *proj_b;
    pk_linear ctx;  /* [128][5 * 128], im2col order (tap-major) */
    float *ctx_b;
    float *out_w;   /* [128] */
    float out_b;
} pk_vad;

void pk_vad_load(pk_vad *v, const pk_weights *w);
void pk_vad_free(pk_vad *v);
int pk_vad_lock(const pk_vad *v);
void pk_vad_init_shapes(pk_vad *v);
void pk_vad_arrays(pk_vad *v, pk_array_fn fn, void *user);

/* Speech probabilities for every subsampler output frame sub[T][1024]. */
float *pk_vad_run(pk_context *c, const pk_vad *v, const float *sub, int T);

#endif
