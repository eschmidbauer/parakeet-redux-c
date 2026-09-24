#include "fft.h"

#include <math.h>
#include <stdlib.h>

#include "util.h"

void pk_fft_init(pk_fft *f, int n) {
    int h = n / 2;
    f->n = n;
    f->twiddle = xmalloc((size_t)h * sizeof(float)); /* [h/2][2] cos, -sin */
    for (int k = 0; k < h / 2; k++) {
        double a = -2.0 * M_PI * k / h;
        f->twiddle[2 * k] = (float)cos(a);
        f->twiddle[2 * k + 1] = (float)sin(a);
    }
    f->rev = xmalloc((size_t)h * sizeof(int));
    int bits = 0;
    while ((1 << bits) < h) bits++;
    for (int i = 0; i < h; i++) {
        int r = 0;
        for (int b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
        f->rev[i] = r;
    }
    f->split = xmalloc((size_t)(h + 1) * 2 * sizeof(float));
    for (int k = 0; k <= h; k++) {
        double a = -2.0 * M_PI * k / n;
        f->split[2 * k] = (float)cos(a);
        f->split[2 * k + 1] = (float)sin(a);
    }
}

void pk_fft_free(pk_fft *f) {
    free(f->twiddle);
    free(f->rev);
    free(f->split);
}

/* in-place iterative radix-2 complex FFT of h points, z[2h] interleaved */
static void cfft(const pk_fft *f, float *z) {
    int h = f->n / 2;
    for (int i = 0; i < h; i++) {
        int j = f->rev[i];
        if (j > i) {
            float tr = z[2 * i], ti = z[2 * i + 1];
            z[2 * i] = z[2 * j];
            z[2 * i + 1] = z[2 * j + 1];
            z[2 * j] = tr;
            z[2 * j + 1] = ti;
        }
    }
    for (int len = 2; len <= h; len *= 2) {
        int half = len / 2, step = h / len;
        for (int start = 0; start < h; start += len) {
            for (int k = 0; k < half; k++) {
                float wr = f->twiddle[2 * k * step], wi = f->twiddle[2 * k * step + 1];
                float *a = z + 2 * (start + k), *b = z + 2 * (start + k + half);
                float br = b[0] * wr - b[1] * wi, bi = b[0] * wi + b[1] * wr;
                b[0] = a[0] - br;
                b[1] = a[1] - bi;
                a[0] += br;
                a[1] += bi;
            }
        }
    }
}

void pk_fft_real(const pk_fft *f, const float *in, float *out, float *z) {
    int h = f->n / 2;
    for (int m = 0; m < h; m++) {
        z[2 * m] = in[2 * m];
        z[2 * m + 1] = in[2 * m + 1];
    }
    cfft(f, z);
    for (int k = 0; k <= h; k++) {
        int a = k % h, b = (h - k) % h;
        float zr = z[2 * a], zi = z[2 * a + 1];
        float cr = z[2 * b], ci = -z[2 * b + 1]; /* conj(Z[h-k]) */
        float er = 0.5f * (zr + cr), ei = 0.5f * (zi + ci);         /* even part */
        float dr = 0.5f * (zr - cr), di = 0.5f * (zi - ci);         /* (Z - conj) / 2 */
        float or_ = di, oi = -dr;                                   /* odd part: divided by i */
        float wr = f->split[2 * k], wi = f->split[2 * k + 1];
        out[2 * k] = er + (or_ * wr - oi * wi);
        out[2 * k + 1] = ei + (or_ * wi + oi * wr);
    }
}
