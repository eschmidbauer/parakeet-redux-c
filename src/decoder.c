#include "decoder.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "gemm.h"
#include "util.h"

#define HID PK_DEC_HIDDEN

/* [k][n] in the file (x @ W) -> [n][k] */
static float *transposed(const pk_weights *w, const char *name, int k, int n) {
    float *kn = pk_weights_f32(w, name, (size_t)k * n);
    float *nk = xmalloc((size_t)n * k * sizeof(float));
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++) nk[(size_t)j * k + i] = kn[(size_t)i * n + j];
    free(kn);
    return nk;
}

void pk_decoder_load(pk_decoder *d, const pk_weights *w, int fast) {
    memset(d, 0, sizeof *d);
    pk_linear_from_kn(&d->enc_proj, w, "decoder.encoder_proj.weight", HID, PK_DEC_ENC);
    d->enc_b = pk_weights_f32(w, "decoder.encoder_proj.bias", HID);
    d->emb = pk_weights_f32(w, "decoder.embedding", (size_t)PK_DEC_VOCAB * HID);
    d->cell_w[0] = pk_weights_f32(w, "decoder.lstm.0.weight", (size_t)4 * HID * 2 * HID);
    d->cell_b[0] = pk_weights_f32(w, "decoder.lstm.0.bias", (size_t)4 * HID);
    d->cell_w[1] = pk_weights_f32(w, "decoder.lstm.1.weight", (size_t)4 * HID * 2 * HID);
    d->cell_b[1] = pk_weights_f32(w, "decoder.lstm.1.bias", (size_t)4 * HID);
    d->dec_w = transposed(w, "decoder.proj.weight", HID, HID);
    d->dec_b = pk_weights_f32(w, "decoder.proj.bias", HID);
    d->head_w = transposed(w, "joint.head.weight", HID, PK_DEC_OUT);
    d->head_b = pk_weights_f32(w, "joint.head.bias", PK_DEC_OUT);
    if (fast) {
        float *src[3] = {d->cell_w[0], d->cell_w[1], d->head_w};
        uint16_t **dst[3] = {&d->cell_h[0], &d->cell_h[1], &d->head_h};
        size_t count[3] = {(size_t)4 * HID * 2 * HID, (size_t)4 * HID * 2 * HID, (size_t)PK_DEC_OUT * HID};
        for (int i = 0; i < 3; i++) {
            *dst[i] = xmalloc(count[i] * sizeof(uint16_t));
            for (size_t j = 0; j < count[i]; j++) (*dst[i])[j] = pk_float_to_half(src[i][j]);
            free(src[i]);
        }
        d->cell_w[0] = d->cell_w[1] = d->head_w = NULL;
    }
}

void pk_decoder_init_shapes(pk_decoder *d, int fast) {
    memset(d, 0, sizeof *d);
    pk_weight_init(&d->enc_proj, HID, PK_DEC_ENC, 0, 0);
    /* markers for the arrays this configuration has (see pk_weight_init) */
    d->enc_b = d->emb = d->cell_b[0] = d->cell_b[1] = d->dec_w = d->dec_b = d->head_b = (float *)1;
    if (fast) d->cell_h[0] = d->cell_h[1] = d->head_h = (uint16_t *)1;
    else d->cell_w[0] = d->cell_w[1] = d->head_w = (float *)1;
}

void pk_decoder_arrays(pk_decoder *d, pk_array_fn fn, void *user) {
    pk_weight_arrays(&d->enc_proj, fn, user);
    fn(user, (void **)&d->enc_b, HID * sizeof(float));
    fn(user, (void **)&d->emb, (size_t)PK_DEC_VOCAB * HID * sizeof(float));
    for (int i = 0; i < 2; i++) {
        if (d->cell_w[i]) fn(user, (void **)&d->cell_w[i], (size_t)4 * HID * 2 * HID * sizeof(float));
        if (d->cell_h[i]) fn(user, (void **)&d->cell_h[i], (size_t)4 * HID * 2 * HID * sizeof(uint16_t));
        fn(user, (void **)&d->cell_b[i], (size_t)4 * HID * sizeof(float));
    }
    fn(user, (void **)&d->dec_w, (size_t)HID * HID * sizeof(float));
    fn(user, (void **)&d->dec_b, HID * sizeof(float));
    if (d->head_w) fn(user, (void **)&d->head_w, (size_t)PK_DEC_OUT * HID * sizeof(float));
    if (d->head_h) fn(user, (void **)&d->head_h, (size_t)PK_DEC_OUT * HID * sizeof(uint16_t));
    fn(user, (void **)&d->head_b, PK_DEC_OUT * sizeof(float));
}

static void lock_visit(void *user, void **array, size_t bytes) {
    int *rc = user;
    if (bytes && pk_lock_pages(*array, bytes) != 0) *rc = -1;
}

int pk_decoder_lock(const pk_decoder *d) {
    int rc = 0;
    pk_decoder_arrays((pk_decoder *)d, lock_visit, &rc);
    return rc;
}

void pk_decoder_free(pk_decoder *d) {
    pk_linear_free(&d->enc_proj);
    free(d->enc_b); free(d->emb); free(d->cell_w[0]); free(d->cell_b[0]); free(d->cell_w[1]); free(d->cell_b[1]);
    free(d->dec_w); free(d->dec_b); free(d->head_w); free(d->head_b);
    free(d->cell_h[0]); free(d->cell_h[1]); free(d->head_h);
    memset(d, 0, sizeof *d);
}

typedef struct {
    float h[PK_DEC_LAYERS][HID];
    float c[PK_DEC_LAYERS][HID];
} lstm_state;

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static void lstm_cell(pk_context *c, const float *w, const uint16_t *wh, const float *b, const float *x, const float *h_in,
                      const float *c_in, float *h_out, float *c_out) {
    float xh[2 * HID], gates[4 * HID];
    memcpy(xh, x, HID * sizeof(float));
    memcpy(xh + HID, h_in, HID * sizeof(float));
    if (wh) pk_gemv_f16(c, wh, 4 * HID, 2 * HID, xh, gates);
    else pk_gemv(c, w, 4 * HID, 2 * HID, xh, gates);
    for (int j = 0; j < HID; j++) {
        float i = sigmoid(gates[j] + b[j]);
        float f = sigmoid(gates[HID + j] + b[HID + j]);
        float g = tanhf(gates[2 * HID + j] + b[2 * HID + j]);
        float o = sigmoid(gates[3 * HID + j] + b[3 * HID + j]);
        float c = f * c_in[j] + i * g;
        c_out[j] = c;
        h_out[j] = o * tanhf(c);
    }
}

/* One prediction-network step: consume `token` from `in`, giving `out` and the projected output. */
static void predict(pk_context *c, const pk_decoder *d, int token, const lstm_state *in, lstm_state *out, float *pred) {
    lstm_cell(c, d->cell_w[0], d->cell_h[0], d->cell_b[0], d->emb + (size_t)token * HID, in->h[0], in->c[0], out->h[0], out->c[0]);
    lstm_cell(c, d->cell_w[1], d->cell_h[1], d->cell_b[1], out->h[0], in->h[1], in->c[1], out->h[1], out->c[1]);
    pk_gemv(c, d->dec_w, HID, HID, out->h[1], pred);
    for (int j = 0; j < HID; j++) pred[j] += d->dec_b[j];
}

int pk_greedy_decode(pk_context *c, const pk_decoder *d, const float *enc, int n, const int *durations, int n_durations,
                     int max_symbols_per_step, int blank_id, int **tokens_out, int **durations_out) {
    float *enc_proj = xmalloc((size_t)n * HID * sizeof(float));
    pk_linear_apply(c, &d->enc_proj, enc, n, d->enc_b, enc_proj);

    long capacity = (long)max_symbols_per_step * n + 1;
    int *tokens = xmalloc((size_t)capacity * sizeof(int));
    int *durs = xmalloc((size_t)capacity * sizeof(int));
    int steps = 0;

    lstm_state state, next;
    memset(&state, 0, sizeof state);
    float pred[HID], z[HID], logits[PK_DEC_OUT];
    int last = blank_id;
    predict(c, d, last, &state, &next, pred);

    int frame = 0;
    long budget = (long)max_symbols_per_step * n;
    while (frame < n && budget > 0) {
        const float *ep = enc_proj + (size_t)frame * HID;
        for (int j = 0; j < HID; j++) {
            float v = ep[j] + pred[j];
            z[j] = v > 0.0f ? v : 0.0f;
        }
        if (d->head_h) pk_gemv_f16(c, d->head_h, PK_DEC_OUT, HID, z, logits);
        else pk_gemv(c, d->head_w, PK_DEC_OUT, HID, z, logits);
        int token = 0;
        float best = logits[0] + d->head_b[0];
        for (int j = 1; j < PK_DEC_VOCAB; j++) {
            float v = logits[j] + d->head_b[j];
            if (v > best) {
                best = v;
                token = j;
            }
        }
        int di = 0;
        best = logits[PK_DEC_VOCAB] + d->head_b[PK_DEC_VOCAB];
        for (int j = 1; j < n_durations; j++) {
            float v = logits[PK_DEC_VOCAB + j] + d->head_b[PK_DEC_VOCAB + j];
            if (v > best) {
                best = v;
                di = j;
            }
        }
        int duration = durations[di];
        if (token == blank_id && duration == 0) duration = 1;
        tokens[steps] = token;
        durs[steps] = duration;
        steps++;
        frame += duration;
        if (token != blank_id) {
            last = token;
            state = next;
            predict(c, d, last, &state, &next, pred);
        }
        budget--;
    }
    free(enc_proj);
    *tokens_out = tokens;
    *durations_out = durs;
    return steps;
}
