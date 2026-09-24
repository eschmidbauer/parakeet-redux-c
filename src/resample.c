#include "resample.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

#define TAPS_PER_SIDE 24 /* at the lower of the two Nyquist rates */
#define MAX_PHASES 4096
#define KAISER_BETA 9.0

struct pk_resampler {
    int in_rate, out_rate;
    long L, M;      /* output j sits at input position j * M / L */
    int half, taps, phases;
    float *bank;    /* [phases][taps] */
    float *hist;    /* the last `taps` input samples seen, plus pending input */
    long hist_len, hist_cap;
    long consumed;  /* absolute index of hist[0] in the input stream */
    long next_out;  /* absolute index of the next output sample */
    long total_in;  /* input samples received */
    int flushed;
};

static double bessel_i0(double x) {
    double sum = 1.0, term = 1.0, y = x * x / 4.0;
    for (int k = 1; k < 64; k++) {
        term *= y / ((double)k * k);
        sum += term;
        if (term < sum * 1e-12) break;
    }
    return sum;
}

static long gcd(long a, long b) {
    while (b) {
        long t = a % b;
        a = b;
        b = t;
    }
    return a;
}

pk_resampler *pk_resampler_new(int in_rate, int out_rate) {
    pk_resampler *r = xcalloc(1, sizeof *r);
    r->in_rate = in_rate;
    r->out_rate = out_rate;
    long g = gcd(in_rate, out_rate);
    r->L = out_rate / g;
    r->M = in_rate / g;
    double ratio = (double)out_rate / in_rate;
    double cutoff = ratio < 1.0 ? ratio : 1.0;
    r->half = (int)ceil(TAPS_PER_SIDE / cutoff);
    r->taps = 2 * r->half;
    r->phases = r->L <= MAX_PHASES ? (int)r->L : MAX_PHASES;
    r->bank = xmalloc((size_t)r->phases * r->taps * sizeof(float));
    double norm = bessel_i0(KAISER_BETA);
    for (int p = 0; p < r->phases; p++) {
        double frac = (double)p / r->phases, sum = 0.0;
        float *h = r->bank + (size_t)p * r->taps;
        for (int t = 0; t < r->taps; t++) {
            double u = (t - r->half + 1) - frac;
            double x = M_PI * u * cutoff;
            double sinc = fabs(x) < 1e-9 ? 1.0 : sin(x) / x;
            double w = u / r->half;
            double window = fabs(w) >= 1.0 ? 0.0 : bessel_i0(KAISER_BETA * sqrt(1.0 - w * w)) / norm;
            h[t] = (float)(sinc * window * cutoff);
            sum += sinc * window * cutoff;
        }
        for (int t = 0; t < r->taps; t++) h[t] = (float)(h[t] / sum);
    }
    /* the filter looks half-1 samples back: pretend that many zeros preceded the stream */
    r->hist_cap = (long)r->taps * 4;
    r->hist = xcalloc((size_t)r->hist_cap, sizeof(float));
    r->hist_len = r->half - 1;
    r->consumed = -(r->half - 1);
    return r;
}

void pk_resampler_free(pk_resampler *r) {
    if (!r) return;
    free(r->bank);
    free(r->hist);
    free(r);
}

int pk_resampler_max_output(const pk_resampler *r, int n) { return (int)(((long)n * r->L) / r->M + 2); }

/* Produce every output whose filter window lies within the history buffer. */
static int drain(pk_resampler *r, float *out) {
    int produced = 0;
    for (;;) {
        long num = r->next_out * r->M;
        long base = num / r->L;                     /* input index of the sample just at or before the output position */
        long first = base - (r->half - 1), last = base + r->half;
        if (last >= r->consumed + r->hist_len) break; /* needs input that has not arrived */
        double frac = (double)(num % r->L) / r->L;
        int p = (int)(frac * r->phases + 0.5);
        if (p >= r->phases) p = r->phases - 1;
        const float *h = r->bank + (size_t)p * r->taps;
        const float *x = r->hist + (first - r->consumed);
        double acc = 0.0;
        for (int t = 0; t < r->taps; t++) acc += (double)x[t] * h[t];
        out[produced++] = (float)acc;
        r->next_out++;
    }
    /* keep only what future outputs can still need */
    long keep_from = (r->next_out * r->M) / r->L - (r->half - 1);
    long drop = keep_from - r->consumed;
    if (drop > 0) {
        if (drop > r->hist_len) drop = r->hist_len;
        memmove(r->hist, r->hist + drop, (size_t)(r->hist_len - drop) * sizeof(float));
        r->hist_len -= drop;
        r->consumed += drop;
    }
    return produced;
}

int pk_resampler_process(pk_resampler *r, const float *in, int n, float *out) {
    if (n <= 0) return 0;
    if (r->hist_len + n > r->hist_cap) {
        r->hist_cap = (r->hist_len + n) * 2;
        r->hist = xrealloc(r->hist, (size_t)r->hist_cap * sizeof(float));
    }
    memcpy(r->hist + r->hist_len, in, (size_t)n * sizeof(float));
    r->hist_len += n;
    r->total_in += n;
    return drain(r, out);
}

int pk_resampler_flush(pk_resampler *r, float *out) {
    if (r->flushed) return 0;
    r->flushed = 1;
    /* zeros after the end, then every output positioned inside the real input */
    long want = (r->total_in * r->L + r->M - 1) / r->M; /* ceil(n * L / M) outputs in total */
    int produced = 0;
    long pad = r->half + 1;
    if (r->hist_len + pad > r->hist_cap) {
        r->hist_cap = r->hist_len + pad;
        r->hist = xrealloc(r->hist, (size_t)r->hist_cap * sizeof(float));
    }
    memset(r->hist + r->hist_len, 0, (size_t)pad * sizeof(float));
    r->hist_len += pad;
    produced = drain(r, out);
    if (r->next_out > want) produced -= (int)(r->next_out - want);
    return produced < 0 ? 0 : produced;
}

float *pk_resample(const float *in, int n, int in_rate, int out_rate, int *out_n) {
    if (in_rate == out_rate) {
        float *out = xmalloc((size_t)(n ? n : 1) * sizeof(float));
        memcpy(out, in, (size_t)n * sizeof(float));
        *out_n = n;
        return out;
    }
    pk_resampler *r = pk_resampler_new(in_rate, out_rate);
    long cap = ((long)n * r->L + r->M - 1) / r->M + r->taps + 2;
    float *out = xmalloc((size_t)cap * sizeof(float));
    int count = pk_resampler_process(r, in, n, out);
    count += pk_resampler_flush(r, out + count);
    pk_resampler_free(r);
    *out_n = count;
    return out;
}
