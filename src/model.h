/* The model structure behind the public pk_model, shared by transcribe.c and packed.c. */
#ifndef PK_MODEL_H
#define PK_MODEL_H

#include <stddef.h>

#include "decoder.h"
#include "encoder.h"
#include "logmel.h"
#include "pk.h"
#include "subsample.h"
#include "vad.h"
#include "vocab.h"
#include "weights.h"

struct pk_model {
    pk_features feats;
    pk_subsampler sub;
    pk_encoder enc;
    pk_decoder dec;
    pk_vad vad;
    int has_vad;
    pk_vocab vocab;
    int vocab_size, blank_id, max_symbols;
    int durations[16];
    int n_durations;
    double frame_seconds;
    int fast;
    int locked;
    long max_window; /* samples the encoder is prepared for */
    /* weights mapped from a packed file: the arrays point into `map` and are not freed */
    void *map;
    size_t map_len;
};

/* Every weight array of the model, in a fixed order. */
void pk_model_arrays(pk_model *m, pk_array_fn fn, void *user);
/* Map a packed file's arrays into a model whose shapes are initialized. Fails through die(). */
void pk_model_map_packed(pk_model *m, const char *path, int *fast, int *has_vad, int *cache_n);
/* Peek at a packed file's header: 1 usable, 0 no such file, -1 unusable (error message set). */
int pk_packed_header(const char *path, int *fast, int *has_vad, int *cache_n);

#endif
