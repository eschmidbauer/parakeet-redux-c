/* Sample-rate conversion with a Kaiser windowed-sinc polyphase filter, as a
 * stateful converter for streams or in one call for whole buffers. */
#ifndef PK_RESAMPLE_H
#define PK_RESAMPLE_H

typedef struct pk_resampler pk_resampler;

pk_resampler *pk_resampler_new(int in_rate, int out_rate);
void pk_resampler_free(pk_resampler *r);
/* Output samples produced so far for n more input samples. Returns the count;
 * the buffer must have room for pk_resampler_max_output(r, n). */
int pk_resampler_max_output(const pk_resampler *r, int n);
int pk_resampler_process(pk_resampler *r, const float *in, int n, float *out);
/* Drain the filter delay at the end of a stream. */
int pk_resampler_flush(pk_resampler *r, float *out);

/* Whole buffers: malloc'd output, *out_n samples. */
float *pk_resample(const float *in, int n, int in_rate, int out_rate, int *out_n);

#endif
