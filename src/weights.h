/* The parakeet-redux.bin weights file: a header with the decoding settings,
 * a table of named tensors, and the data. The file is mapped read only and
 * tensors are referenced in place; loaders copy or repack what they keep.
 * The layout is documented in convert.py of the Hugging Face repository. */
#ifndef PK_WEIGHTS_H
#define PK_WEIGHTS_H

#include <stddef.h>
#include <stdint.h>

enum { PK_DT_F32 = 0, PK_DT_F16 = 1, PK_DT_CODES = 2, PK_DT_TEXT = 3 };

typedef struct {
    char name[80];
    uint32_t dtype, ndim;
    uint64_t dims[4];
    const uint8_t *data;
    size_t bytes;
} pk_tensor;

typedef struct {
    void *map;
    size_t map_len;
    pk_tensor *tensors;
    int n_tensors;
    /* decoding settings from the header */
    int vocab_size, blank_id, max_symbols, n_durations;
    int durations[8];
    double frame_seconds;
    int sample_rate;
} pk_weights;

/* Fails through die(); callers arm an error trap. */
void pk_weights_open(pk_weights *w, const char *path);
void pk_weights_close(pk_weights *w);

const pk_tensor *pk_weights_find(const pk_weights *w, const char *name);
/* The named tensor, or die() if it is missing or not of that dtype and element count. */
const pk_tensor *pk_weights_get(const pk_weights *w, const char *name, int dtype, size_t count);
/* A malloc'd copy of a float32 tensor with `count` elements. */
float *pk_weights_f32(const pk_weights *w, const char *name, size_t count);
/* The same for a name built with printf formatting. */
float *pk_weights_f32f(const pk_weights *w, size_t count, const char *fmt, ...) __attribute__((format(printf, 3, 4)));

#endif
