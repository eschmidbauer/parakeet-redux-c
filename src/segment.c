#include "segment.h"

#include <stdlib.h>

int pk_speech_regions_from_probs(const float *probs, int n, double frame_seconds, pk_span *out) {
    int count = 0;
    int i = 0;
    while (i < n) {
        if (probs[i] < PK_SPEECH_THRESHOLD) {
            i++;
            continue;
        }
        int j = i;
        while (j < n && probs[j] >= PK_SPEECH_THRESHOLD) j++;
        double start = (double)i * frame_seconds, end = (double)j * frame_seconds;
        if (count && start - out[count - 1].end < PK_MIN_GAP_SECONDS) {
            out[count - 1].end = end;
        } else {
            out[count].start = start;
            out[count].end = end;
            count++;
        }
        i = j;
    }
    int kept = 0;
    for (int k = 0; k < count; k++)
        if (out[k].end - out[k].start >= PK_MIN_SPEECH_SECONDS) out[kept++] = out[k];
    return kept;
}

static int span_cmp(const void *a, const void *b) {
    const pk_span *x = a, *y = b;
    if (x->start != y->start) return x->start < y->start ? -1 : 1;
    if (x->end != y->end) return x->end < y->end ? -1 : 1;
    return 0;
}

int pk_pauses_from_speech(const pk_span *speech_in, int n, double duration, pk_span *out) {
    pk_span *speech = malloc((size_t)(n ? n : 1) * sizeof *speech);
    for (int i = 0; i < n; i++) speech[i] = speech_in[i];
    qsort(speech, (size_t)n, sizeof *speech, span_cmp);
    double least = PK_MIN_PAUSE_SECONDS - PK_PAUSE_EPSILON;
    double previous = 0.0;
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (speech[i].start - previous >= least) {
            out[count].start = previous;
            out[count].end = speech[i].start;
            count++;
        }
        if (speech[i].end > previous) previous = speech[i].end;
    }
    if (duration - previous >= least) {
        out[count].start = previous;
        out[count].end = duration;
        count++;
    }
    free(speech);
    return count;
}

double pk_next_cut(const pk_span *pauses, int n, double max_seconds) {
    int last = -1;
    for (int i = 0; i < n; i++)
        if (pauses[i].start >= PK_MIN_SEGMENT_SECONDS && pauses[i].end <= max_seconds) last = i;
    if (last < 0) {
        for (int i = 0; i < n; i++) {
            double mid = (pauses[i].start + pauses[i].end) / 2;
            if (mid >= PK_MIN_SEGMENT_SECONDS && mid <= max_seconds) last = i;
        }
    }
    if (last < 0) return max_seconds;
    return (pauses[last].start + pauses[last].end) / 2;
}

int pk_has_speech(const pk_span *regions, int n, double seconds) {
    for (int i = 0; i < n; i++)
        if (regions[i].end > 0 && regions[i].start < seconds) return 1;
    return 0;
}
