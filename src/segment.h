/* Long-audio cutting rules, identical to transcribe.py. Times in seconds. */
#ifndef PK_SEGMENT_H
#define PK_SEGMENT_H

#define PK_SEGMENT_SECONDS 30.0
#define PK_MIN_PAUSE_SECONDS 0.2
#define PK_MIN_SEGMENT_SECONDS 1.0
#define PK_BLOCK_SECONDS 120.0
#define PK_MIN_FEATURE_SAMPLES 320
#define PK_PAUSE_EPSILON 1e-6
#define PK_SPEECH_THRESHOLD 0.5f
#define PK_MIN_SPEECH_SECONDS 0.1
#define PK_MIN_GAP_SECONDS 0.1

typedef struct {
    double start, end;
} pk_span;

/* Speech regions from per-frame probabilities; `out` needs room for n/2+1 spans. */
int pk_speech_regions_from_probs(const float *probs, int n, double frame_seconds, pk_span *out);
/* Pauses between sorted speech spans within [0, duration]; `out` needs room for n+1. */
int pk_pauses_from_speech(const pk_span *speech, int n, double duration, pk_span *out);
double pk_next_cut(const pk_span *pauses, int n, double max_seconds);
int pk_has_speech(const pk_span *regions, int n, double seconds);

#endif
