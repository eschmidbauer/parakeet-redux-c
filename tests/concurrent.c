/* Concurrency test for the C API: one model shared by several threads, each
 * with its own context, transcribing at the same time. Every thread gets a
 * clip of a different length so the results differ; each is compared with a
 * sequential run of the same clip.
 *
 *   concurrent_test MODEL AUDIO.wav [threads] [threads_per_context] [--fast] [--lock] [--stream] [--packed FILE]
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pk.h"
#include "util.h"

typedef struct {
    const pk_model *model;
    const float *pcm;
    int n;
    int threads_per_context;
    int use_stream;
    char *text;
    double elapsed;
} job;

static char *transcribe_text(pk_context *ctx, const pk_model *model, const float *pcm, int n, int use_stream) {
    if (!use_stream) {
        pk_result *r = pk_transcribe(ctx, model, pcm, n, PK_TS_WORD);
        if (!r) return NULL;
        char *text = xstrdup(r->text);
        pk_result_free(r);
        return text;
    }
    /* feed one second at a time and join the window texts */
    pk_stream *s = pk_stream_new(ctx, model, PK_TS_SEGMENT, 0);
    char *text = xstrdup("");
    for (int off = 0; off < n; off += PK_SAMPLE_RATE) {
        int got = n - off < PK_SAMPLE_RATE ? n - off : PK_SAMPLE_RATE;
        pk_stream_feed(s, pcm + off, got);
        pk_result *r;
        while ((r = pk_stream_take(s))) {
            if (*r->text) {
                text = xrealloc(text, strlen(text) + strlen(r->text) + 2);
                if (*text) strcat(text, " ");
                strcat(text, r->text);
            }
            pk_result_free(r);
        }
    }
    pk_stream_finish(s);
    pk_result *r;
    while ((r = pk_stream_take(s))) {
        if (*r->text) {
            text = xrealloc(text, strlen(text) + strlen(r->text) + 2);
            if (*text) strcat(text, " ");
            strcat(text, r->text);
        }
        pk_result_free(r);
    }
    pk_stream_free(s);
    return text;
}

static void *worker(void *arg) {
    job *j = arg;
    pk_context *ctx = pk_context_new(j->threads_per_context);
    double started = now_seconds();
    j->text = transcribe_text(ctx, j->model, j->pcm, j->n, j->use_stream);
    j->elapsed = now_seconds() - started;
    pk_context_free(ctx);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: concurrent_test MODEL AUDIO.wav [threads] [threads_per_context] [--fast] [--lock] [--stream] [--packed FILE]\n");
        return 2;
    }
    int threads = 4, per_context = 1;
    pk_options opt = {0};
    int use_stream = 0, positional = 0;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--fast")) opt.fast = 1;
        else if (!strcmp(argv[i], "--lock")) opt.lock_in_memory = 1;
        else if (!strcmp(argv[i], "--stream")) use_stream = 1;
        else if (!strcmp(argv[i], "--packed") && i + 1 < argc) opt.packed_path = argv[++i];
        else if (positional == 0) threads = atoi(argv[i]), positional++;
        else per_context = atoi(argv[i]);
    }
    int n;
    float *pcm = pk_read_audio(argv[2], &n);
    if (!pcm) return 1;

    double t0 = now_seconds();
    pk_model *model = pk_model_load(argv[1], &opt);
    if (!model) {
        fprintf(stderr, "load failed: %s\n", pk_error_message());
        return 1;
    }
    printf("model loaded in %.2fs (%s, %zu MB, %s%s)\n", now_seconds() - t0, pk_model_backend(model), pk_model_weight_bytes(model) >> 20,
           pk_model_is_mapped(model) ? "mapped, " : "", pk_model_is_locked(model) ? "locked in memory" : "not locked");

    /* sequential references, one clip length per thread */
    job *jobs = xcalloc((size_t)threads, sizeof *jobs);
    char **reference = xcalloc((size_t)threads, sizeof *reference);
    pk_context *ctx = pk_context_new(0);
    t0 = now_seconds();
    double audio = 0.0;
    for (int t = 0; t < threads; t++) {
        jobs[t].model = model;
        jobs[t].pcm = pcm;
        jobs[t].n = n - t * PK_SAMPLE_RATE / 2; /* half a second shorter each time */
        if (jobs[t].n < PK_SAMPLE_RATE) jobs[t].n = PK_SAMPLE_RATE;
        jobs[t].threads_per_context = per_context;
        jobs[t].use_stream = use_stream;
        reference[t] = transcribe_text(ctx, model, pcm, jobs[t].n, use_stream);
        audio += (double)jobs[t].n / PK_SAMPLE_RATE;
    }
    double sequential = now_seconds() - t0;
    pk_context_free(ctx);
    printf("sequential: %d clips, %.1fs of audio in %.2fs (%.0fx real time, all cores)\n", threads, audio, sequential, audio / sequential);

    pthread_t *tids = xmalloc((size_t)threads * sizeof *tids);
    t0 = now_seconds();
    for (int t = 0; t < threads; t++) pthread_create(&tids[t], NULL, worker, &jobs[t]);
    for (int t = 0; t < threads; t++) pthread_join(tids[t], NULL);
    double concurrent = now_seconds() - t0;
    printf("concurrent: %d threads x %d pool threads, %.1fs of audio in %.2fs (%.0fx real time)\n", threads, per_context, audio, concurrent,
           audio / concurrent);

    int failures = 0;
    for (int t = 0; t < threads; t++) {
        int ok = jobs[t].text && reference[t] && strcmp(jobs[t].text, reference[t]) == 0;
        if (!ok) {
            failures++;
            printf("  thread %d: MISMATCH\n    sequential: %s\n    concurrent: %s\n", t, reference[t] ? reference[t] : "(null)",
                   jobs[t].text ? jobs[t].text : "(null)");
        }
        free(jobs[t].text);
        free(reference[t]);
    }
    printf("%s: %d of %d concurrent results identical to their sequential run\n", failures ? "FAIL" : "PASS", threads - failures, threads);
    free(jobs);
    free(reference);
    free(tids);
    free(pcm);
    pk_model_free(model);
    return failures ? 1 : 0;
}
