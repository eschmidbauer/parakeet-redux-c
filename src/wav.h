#ifndef PK_WAV_H
#define PK_WAV_H
/* Reads a RIFF/WAVE file (8/16/24/32-bit PCM or 32-bit float, any channel
 * count averaged to mono). Returns malloc'd samples in [-1, 1) at the file's
 * own rate, or NULL on error with a message on stderr. */
float *pk_wav_read(const char *path, int *n_samples, int *sample_rate);

/* Decodes any audio file through an ffmpeg pipe to mono float at `rate`.
 * Returns NULL if ffmpeg is not available or fails. */
float *pk_ffmpeg_read(const char *path, int rate, int *n_samples);

/* Whatever it takes: WAV directly (resampled to `rate` if needed), else ffmpeg. */
float *pk_audio_read(const char *path, int rate, int *n_samples);

#endif
