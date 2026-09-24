#ifndef PK_DECODER_H
#define PK_DECODER_H

#include "linear.h"
#include "weights.h"

#define PK_DEC_HIDDEN 640
#define PK_DEC_LAYERS 2
#define PK_DEC_VOCAB 8193
#define PK_DEC_OUT 8198
#define PK_DEC_ENC 1024

/* Prediction network (embedding + 2-layer LSTM) and the joint network. */
typedef struct {
    pk_linear enc_proj;     /* [640][1024] */
    float *enc_b;           /* [640] */
    float *emb;             /* [8193][640] */
    float *cell_w[2];       /* [2560][1280]: gates i,f,g,o over concat(input, h) */
    float *cell_b[2];       /* [2560] */
    float *dec_w, *dec_b;   /* [640][640] (W x), [640] */
    float *head_w, *head_b; /* [8198][640] (W x), [8198] */
    /* fast mode: float16 copies of the LSTM and head weights (the float32 ones are dropped) */
    uint16_t *cell_h[2], *head_h;
} pk_decoder;

void pk_decoder_load(pk_decoder *d, const pk_weights *w, int fast);
void pk_decoder_free(pk_decoder *d);
int pk_decoder_lock(const pk_decoder *d);
void pk_decoder_init_shapes(pk_decoder *d, int fast);
void pk_decoder_arrays(pk_decoder *d, pk_array_fn fn, void *user);

/* Greedy token-and-duration decoding over enc[n][1024]. Returns the number of
 * steps; *tokens and *durations are malloc'd arrays of that length (blanks
 * included, as the word timing needs their durations). */
int pk_greedy_decode(pk_context *c, const pk_decoder *d, const float *enc, int n, const int *durations, int n_durations,
                     int max_symbols_per_step, int blank_id, int **tokens, int **step_durations);

#endif
