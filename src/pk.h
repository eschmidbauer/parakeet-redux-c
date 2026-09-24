/* Public API of the C port.
 *
 * A pk_model is loaded once and is read only afterwards, so any number of
 * threads may use it at the same time. Each thread brings its own pk_context
 * (a thread pool plus scratch memory); contexts must not be shared between
 * threads. A context with one thread runs entirely on the calling thread, which
 * is the right choice when concurrency comes from many requests. */
#ifndef PK_H
#define PK_H

#include <stddef.h>

#define PK_SAMPLE_RATE 16000

enum { PK_TS_NONE = 0, PK_TS_SEGMENT = 1, PK_TS_WORD = 2 };

typedef struct {
    char *word;
    double start, end;
} pk_word;

typedef struct {
    char *text;
    double start, end;
    pk_word *words; /* only with PK_TS_WORD */
    int n_words;
} pk_segment;

typedef struct {
    char *text;
    double duration;
    pk_segment *segments;
    int n_segments;
    double start; /* where this result begins, in seconds: 0 offline, absolute in a stream */
} pk_result;

typedef struct pk_model pk_model;
typedef struct pk_context pk_context;

typedef struct {
    int fast;                  /* int8 activations and float16 decoder weights: faster, may change a few words */
    double max_window_seconds; /* longest piece the encoder is prepared for; 0 means 30 (the offline cut) */
    int lock_in_memory;        /* mlock the weights so they are never paged out */
    int load_threads;          /* threads used while loading; 0 means all cores */
    const char *packed_path;   /* if set: map the weights from this file when it exists (instant start,
                                  shared between processes), otherwise load the ONNX files and write it */
} pk_options;

/* Load the weights: `path` is parakeet-redux.bin or a directory containing it
 * (https://huggingface.co/eschmidbauer/parakeet-redux-c). NULL on failure
 * with pk_error_message() set. */
pk_model *pk_model_load(const char *path, const pk_options *options);
/* Write the loaded weights as a packed file for pk_options.packed_path. 0 or -1. */
int pk_model_save(const pk_model *m, const char *path);
int pk_model_is_mapped(const pk_model *m);
void pk_model_free(pk_model *m);
const char *pk_model_backend(const pk_model *m);
size_t pk_model_weight_bytes(const pk_model *m);
int pk_model_is_fast(const pk_model *m);
int pk_model_is_locked(const pk_model *m);

/* Every function that can fail returns NULL or -1 and leaves a message here
 * (thread local). Nothing in the library exits or writes to stderr on its own;
 * warnings go to the log callback, which defaults to stderr. */
const char *pk_error_message(void);
enum { PK_LOG_INFO = 0, PK_LOG_WARNING = 1, PK_LOG_ERROR = 2 };
typedef void (*pk_log_fn)(int level, const char *message, void *user);
void pk_set_log(pk_log_fn fn, void *user);

/* threads <= 0 uses every online CPU; 1 runs on the calling thread only. NULL on failure. */
pk_context *pk_context_new(int threads);
void pk_context_free(pk_context *c);
int pk_context_threads(const pk_context *c);
/* Print per-stage timings to stderr after each transcription. */
void pk_context_set_verbose(pk_context *c, int on);
/* When set, intermediate tensors are written to this directory as .npy files. */
void pk_context_set_dump_dir(pk_context *c, const char *dir);

/* pcm: mono float samples at 16 kHz. NULL on error (too short, out of memory). */
pk_result *pk_transcribe(pk_context *c, const pk_model *m, const float *pcm, int n, int timestamps);
void pk_result_free(pk_result *r);

/* Any audio file, decoded to mono float32 at 16 kHz: WAV directly (resampled
 * if needed), anything else through ffmpeg if it is on the PATH. NULL on error. */
float *pk_read_audio(const char *path, int *n_samples);

/* Streaming: feed audio as it arrives; results come out one window at a time.
 * A window is cut at the last pause found by the VAD head, at most
 * window_seconds long (30, like the offline mode), so the first result appears
 * once that much audio is in. Times in results are absolute. */
typedef struct pk_stream pk_stream;
pk_stream *pk_stream_new(pk_context *c, const pk_model *m, int timestamps, double window_seconds);
/* Both return 0, or -1 after which the stream is unusable and pk_error_message() says why. */
int pk_stream_feed(pk_stream *s, const float *pcm, int n);
int pk_stream_finish(pk_stream *s);

/* End-of-utterance cuts: once speech has been followed by this much silence,
 * the buffered audio is decoded without waiting for the window to fill.
 * 0 (the default) cuts only when the window is full. */
void pk_stream_set_pause(pk_stream *s, double seconds);
/* Manual mode for worker pools: pk_stream_feed only buffers, and
 * pk_stream_process runs the VAD, cuts and transcriptions on the calling
 * thread with the given context (or the stream's own if NULL). Returns the
 * number of results queued, or -1. The stream may be created with a NULL
 * context in this mode. */
void pk_stream_set_manual(pk_stream *s, int manual);
int pk_stream_process(pk_stream *s, pk_context *c);
/* The context pk_stream_finish (and pk_stream_process with NULL) will use;
 * a worker that picks up a stream sets its own before touching it. */
void pk_stream_set_context(pk_stream *s, pk_context *c);
/* Seconds of audio buffered and not yet decoded. */
double pk_stream_buffered(const pk_stream *s);
/* The next finished result, or NULL when none is ready. Free with pk_result_free. */
pk_result *pk_stream_take(pk_stream *s);
void pk_stream_free(pk_stream *s);

#endif
