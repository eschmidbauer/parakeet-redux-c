/* Real-input FFT of a fixed power-of-two size, via a half-size complex FFT. */
#ifndef PK_FFT_H
#define PK_FFT_H

typedef struct pk_fft {
    int n;          /* real length (power of two) */
    float *twiddle; /* cos/sin tables for the n/2-point complex FFT and the final split */
    int *rev;       /* bit reversal for n/2 */
    float *split;   /* [n/2][2]: e^{-2 pi i k / n} */
} pk_fft;

void pk_fft_init(pk_fft *f, int n);
void pk_fft_free(pk_fft *f);
/* in[n] real -> out[2 * (n/2 + 1)]: re[0], im[0], re[1], im[1], ... (forward transform) */
void pk_fft_real(const pk_fft *f, const float *in, float *out, float *work /* n floats */);

#endif
