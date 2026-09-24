/* parakeet: speech to text with the C port of parakeet-redux.
 *
 *   parakeet speech.wav
 *   parakeet speech.wav --timestamps segment     one line per sentence
 *   parakeet speech.wav --timestamps word
 *   parakeet talk.mp3 --json                     anything else than WAV goes through ffmpeg
 *   parakeet --jobs 4 *.wav                      four files at a time, one model in memory
 *   arecord -f S16_LE -r 16000 -c 1 | parakeet --stream -
 */
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pk.h"
#include "resample.h"
#include "pool.h"
#include "util.h"

static void usage(FILE *f) {
    fputs("usage: parakeet [options] audio.wav [audio2.wav ...]\n"
          "\n"
          "  --model PATH         parakeet-redux.bin, or a directory holding it (default: the first of\n"
          "                       $PARAKEET_MODEL, ./parakeet-redux.bin, models/parakeet-redux-c)\n"
          "  --timestamps MODE    none (default), segment, or word; --json defaults to segment\n"
          "  --json               emit the full result as JSON\n"
          "  --threads N          worker threads per job (default: all cores, divided among jobs)\n"
          "  --jobs N             transcribe N files at a time, sharing one model in memory\n"
          "  --fast               int8 activations and float16 decoder weights: about 1.6x faster,\n"
          "                       may change a few words (like ONNX Runtime's accuracy_level 4)\n"
          "  --lock               lock the model's memory so it is never paged out\n"
          "  --packed FILE        map the weights from FILE (written on first use): instant start,\n"
          "                       and processes using the same file share the memory\n"
          "  --stream             print results as the audio comes in; '-' reads 16-bit PCM from\n"
          "                       stdin, raw (see --rate, --channels) or with a WAV header\n"
          "  --rate HZ            sample rate of raw stdin audio (default 16000; resampled if needed)\n"
          "  --channels N         channels of raw stdin audio (default 1; averaged to mono)\n"
          "  --window SEC         longest piece decoded at once in --stream mode (default 30)\n"
          "  --pause SEC          in --stream mode, also decode as soon as speech is followed by\n"
          "                       this much silence (end of utterance); default: window only\n"
          "  --dump DIR           write intermediate tensors as .npy files into DIR\n"
          "  --verbose            print per-stage timings\n"
          "  -h, --help           this text\n"
          "\n"
          "WAV files (8/16/24/32-bit PCM or float, any rate) are read directly and resampled to\n"
          "16 kHz if needed; other formats are decoded through ffmpeg when it is on the PATH.\n",
          f);
}

static void print_json_string(const char *s) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '"': fputs("\\\"", stdout); break;
        case '\\': fputs("\\\\", stdout); break;
        case '\n': fputs("\\n", stdout); break;
        case '\r': fputs("\\r", stdout); break;
        case '\t': fputs("\\t", stdout); break;
        default:
            if (*p < 0x20) printf("\\u%04x", *p);
            else putchar(*p);
        }
    }
    putchar('"');
}

/* Shortest representation that round-trips, with a ".0" for whole numbers, like Python's repr. */
static void print_json_double(double v) {
    char buf[40];
    for (int prec = 1; prec <= 17; prec++) {
        snprintf(buf, sizeof buf, "%.*g", prec, v);
        if (strtod(buf, NULL) == v) break;
    }
    if (!strpbrk(buf, ".eEn")) strcat(buf, ".0");
    fputs(buf, stdout);
}

static void print_json(const pk_result *r, int include_words) {
    printf("{\n  \"text\": ");
    print_json_string(r->text);
    printf(",\n  \"language\": null,\n  \"task\": \"transcribe\",\n  \"duration_seconds\": ");
    print_json_double(r->duration);
    printf(",\n  \"source_duration_seconds\": ");
    print_json_double(r->duration);
    printf(",\n  \"clip_start_seconds\": 0.0,\n  \"clip_end_seconds\": ");
    print_json_double(r->duration);
    printf(",\n  \"segments\": [");
    for (int i = 0; i < r->n_segments; i++) {
        const pk_segment *s = &r->segments[i];
        printf("%s\n    {\n      \"text\": ", i ? "," : "");
        print_json_string(s->text);
        printf(",\n      \"start\": ");
        print_json_double(s->start);
        printf(",\n      \"end\": ");
        print_json_double(s->end);
        if (include_words) {
            printf(",\n      \"words\": [");
            for (int w = 0; w < s->n_words; w++) {
                printf("%s\n        {\n          \"word\": ", w ? "," : "");
                print_json_string(s->words[w].word);
                printf(",\n          \"start\": ");
                print_json_double(s->words[w].start);
                printf(",\n          \"end\": ");
                print_json_double(s->words[w].end);
                printf("\n        }");
            }
            printf("%s]", s->n_words ? "\n      " : "");
        }
        printf("\n    }");
    }
    printf("%s]\n}\n", r->n_segments ? "\n  " : "");
}

static void print_text(const pk_result *r, int timestamps, const char *header) {
    if (header) printf("== %s\n", header);
    if (timestamps == PK_TS_NONE) {
        if (*r->text || !header) printf("%s\n", r->text);
    } else {
        for (int i = 0; i < r->n_segments; i++) {
            const pk_segment *s = &r->segments[i];
            printf("[%7.2f - %7.2f]  %s\n", s->start, s->end, s->text);
            for (int w = 0; w < s->n_words; w++) printf("    %7.2f - %7.2f  %s\n", s->words[w].start, s->words[w].end, s->words[w].word);
        }
    }
    if (header) printf("\n");
}

/* ---- streaming output ------------------------------------------------------- */

static void print_stream_json(const pk_result *r, double start, int include_words) {
    printf("{\"start\": ");
    print_json_double(start);
    printf(", \"end\": ");
    print_json_double(start + r->duration);
    printf(", \"text\": ");
    print_json_string(r->text);
    printf(", \"segments\": [");
    for (int i = 0; i < r->n_segments; i++) {
        const pk_segment *seg = &r->segments[i];
        printf("%s{\"text\": ", i ? ", " : "");
        print_json_string(seg->text);
        printf(", \"start\": ");
        print_json_double(seg->start);
        printf(", \"end\": ");
        print_json_double(seg->end);
        if (include_words) {
            printf(", \"words\": [");
            for (int w = 0; w < seg->n_words; w++) {
                printf("%s{\"word\": ", w ? ", " : "");
                print_json_string(seg->words[w].word);
                printf(", \"start\": ");
                print_json_double(seg->words[w].start);
                printf(", \"end\": ");
                print_json_double(seg->words[w].end);
                printf("}");
            }
            printf("]");
        }
        printf("}");
    }
    printf("]}\n");
    fflush(stdout);
}

static void drain_stream(pk_stream *stream, int json, int timestamps, double *emitted) {
    pk_result *r;
    while ((r = pk_stream_take(stream))) {
        if (json) print_stream_json(r, r->start, timestamps == PK_TS_WORD);
        else if (timestamps == PK_TS_NONE) {
            if (*r->text) {
                printf("%s\n", r->text);
                fflush(stdout);
            }
        } else {
            print_text(r, timestamps, NULL);
            fflush(stdout);
        }
        *emitted += r->duration;
        pk_result_free(r);
    }
}

/* 16-bit PCM from stdin, raw or with a WAV header, any rate and channel count;
 * delivers mono 16 kHz samples, resampling on the fly. Returns the sample count. */
typedef struct {
    int rate, channels, header_done, eof;
    pk_resampler *resampler;
    unsigned char pending[4];
    int pending_n;
} stdin_reader;

static int read_stdin_chunk(stdin_reader *rd, float *pcm, int max) {
    if (!rd->header_done) {
        rd->header_done = 1;
        rd->pending_n = (int)fread(rd->pending, 1, 4, stdin);
        if (rd->pending_n == 4 && memcmp(rd->pending, "RIFF", 4) == 0) {
            unsigned char rest[8];
            if (fread(rest, 1, 8, stdin) != 8 || memcmp(rest + 4, "WAVE", 4) != 0) die("stdin: not a WAV stream");
            rd->pending_n = 0;
            for (;;) { /* walk the chunks until the data chunk */
                unsigned char head[8];
                if (fread(head, 1, 8, stdin) != 8) return 0;
                unsigned len = head[4] | head[5] << 8 | head[6] << 16 | (unsigned)head[7] << 24;
                if (memcmp(head, "data", 4) == 0) break;
                if (memcmp(head, "fmt ", 4) == 0) {
                    unsigned char fmt[64];
                    unsigned take = len > sizeof fmt ? sizeof fmt : len;
                    if (fread(fmt, 1, take, stdin) != take) return 0;
                    for (unsigned i = take; i < len + (len & 1); i++) getchar();
                    int format = fmt[0] | fmt[1] << 8, bits = fmt[14] | fmt[15] << 8;
                    rd->channels = fmt[2] | fmt[3] << 8;
                    rd->rate = fmt[4] | fmt[5] << 8 | fmt[6] << 16 | fmt[7] << 24;
                    if (format == 0xFFFE && take >= 26) format = fmt[24] | fmt[25] << 8;
                    if (format != 1 || bits != 16) die("--stream expects 16-bit PCM WAV on stdin (got format %d, %d bits)", format, bits);
                } else {
                    for (unsigned i = 0; i < len + (len & 1); i++) getchar();
                }
            }
        }
        if (rd->channels < 1) die("--channels must be at least 1");
        if (rd->rate != PK_SAMPLE_RATE) rd->resampler = pk_resampler_new(rd->rate, PK_SAMPLE_RATE);
    }
    if (rd->eof) return 0;
    /* read about a second of input frames */
    int frames = rd->rate;
    int frame_bytes = 2 * rd->channels;
    if (rd->resampler) {
        int fit = max / pk_resampler_max_output(rd->resampler, 1) - 4;
        if (fit < 64) fit = 64;
        if (frames > fit) frames = fit;
    } else if (frames > max) {
        frames = max;
    }
    size_t want = (size_t)frames * frame_bytes;
    unsigned char *raw = xmalloc(want + 4);
    memcpy(raw, rd->pending, (size_t)rd->pending_n);
    size_t have = (size_t)rd->pending_n + fread(raw + rd->pending_n, 1, want - (size_t)rd->pending_n, stdin);
    rd->pending_n = 0;
    int got = (int)(have / frame_bytes);
    size_t leftover = have - (size_t)got * frame_bytes;
    memcpy(rd->pending, raw + (size_t)got * frame_bytes, leftover);
    rd->pending_n = (int)leftover;
    if (got < frames) rd->eof = 1;
    float *mono = xmalloc((size_t)(got ? got : 1) * sizeof(float));
    for (int i = 0; i < got; i++) {
        double acc = 0.0;
        for (int c = 0; c < rd->channels; c++) {
            const unsigned char *p = raw + ((size_t)i * rd->channels + c) * 2;
            acc += (short)(p[0] | p[1] << 8) / 32768.0;
        }
        mono[i] = (float)(acc / rd->channels);
    }
    free(raw);
    int samples;
    if (rd->resampler) {
        samples = pk_resampler_process(rd->resampler, mono, got, pcm);
        if (rd->eof) samples += pk_resampler_flush(rd->resampler, pcm + samples);
    } else {
        memcpy(pcm, mono, (size_t)got * sizeof(float));
        samples = got;
    }
    free(mono);
    return samples;
}

static int run_stream(pk_context *ctx, const pk_model *model, const char *file, int json, int timestamps, double window, double pause, int rate,
                      int channels) {
    pk_stream *stream = pk_stream_new(ctx, model, timestamps, window);
    if (!stream) die("%s", pk_error_message());
    pk_stream_set_pause(stream, pause);
    double emitted = 0.0, total = 0.0;
    double started = now_seconds();
    int failed = 0;
    if (!strcmp(file, "-")) {
        stdin_reader rd = {.rate = rate, .channels = channels};
        float *chunk = xmalloc((size_t)(2 * PK_SAMPLE_RATE + 64) * sizeof(float));
        int got;
        while ((got = read_stdin_chunk(&rd, chunk, 2 * PK_SAMPLE_RATE)) > 0) {
            if (pk_stream_feed(stream, chunk, got) != 0) {
                failed = 1;
                break;
            }
            total += got;
            drain_stream(stream, json, timestamps, &emitted);
        }
        free(chunk);
        pk_resampler_free(rd.resampler);
    } else {
        int n;
        float *pcm = pk_read_audio(file, &n);
        if (!pcm) {
            pk_log(PK_LOG_ERROR, "%s", pk_error_message());
            pk_stream_free(stream);
            return 1;
        }
        for (int off = 0; off < n && !failed; off += PK_SAMPLE_RATE) {
            int got = n - off < PK_SAMPLE_RATE ? n - off : PK_SAMPLE_RATE;
            failed = pk_stream_feed(stream, pcm + off, got) != 0;
            drain_stream(stream, json, timestamps, &emitted);
        }
        total = n;
        free(pcm);
    }
    if (!failed) failed = pk_stream_finish(stream) != 0;
    drain_stream(stream, json, timestamps, &emitted);
    pk_stream_free(stream);
    if (failed) {
        pk_log(PK_LOG_ERROR, "%s: %s", file, pk_error_message());
        return 1;
    }
    fprintf(stderr, "%s: %.1fs of audio in %.2fs\n", file, total / PK_SAMPLE_RATE, now_seconds() - started);
    return 0;
}

/* ---- concurrent jobs: one shared model, one context per worker ---------------- */

typedef struct {
    const pk_model *model;
    const char **files;
    int n_files, timestamps, threads_per_job, verbose;
    const char *dump;
    atomic_int next;
    pk_result **results;
    double *elapsed;
    int *failed;
    atomic_int *done;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} jobs;

static void *job_worker(void *arg) {
    jobs *j = arg;
    pk_context *ctx = pk_context_new(j->threads_per_job);
    if (!ctx) die("%s", pk_error_message());
    pk_context_set_verbose(ctx, j->verbose);
    pk_context_set_dump_dir(ctx, j->dump);
    for (;;) {
        int i = atomic_fetch_add(&j->next, 1);
        if (i >= j->n_files) break;
        int n;
        float *pcm = pk_read_audio(j->files[i], &n);
        double started = now_seconds();
        pk_result *r = pcm ? pk_transcribe(ctx, j->model, pcm, n, j->timestamps) : NULL;
        if (!r) pk_log(PK_LOG_ERROR, "%s: %s", j->files[i], pk_error_message());
        free(pcm);
        pthread_mutex_lock(&j->mutex);
        j->results[i] = r;
        j->elapsed[i] = now_seconds() - started;
        j->failed[i] = r == NULL;
        atomic_store(&j->done[i], 1);
        pthread_cond_broadcast(&j->cond);
        pthread_mutex_unlock(&j->mutex);
    }
    pk_context_free(ctx);
    return NULL;
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *dump = NULL, *packed = NULL;
    int timestamps = -1, json = 0, threads = 0, verbose = 0, fast = 0, stream_mode = 0, n_jobs = 1, lock = 0;
    int rate = PK_SAMPLE_RATE, channels = 1;
    double window = 0.0, pause = 0.0;
    const char **files = xmalloc((size_t)argc * sizeof *files);
    int n_files = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout);
            return 0;
        } else if ((!strcmp(a, "--model") || !strcmp(a, "--model-dir")) && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (!strcmp(a, "--timestamps") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "none")) timestamps = PK_TS_NONE;
            else if (!strcmp(v, "segment")) timestamps = PK_TS_SEGMENT;
            else if (!strcmp(v, "word")) timestamps = PK_TS_WORD;
            else die("--timestamps must be none, segment or word");
        } else if (!strcmp(a, "--json")) {
            json = 1;
        } else if (!strcmp(a, "--threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (!strcmp(a, "--jobs") && i + 1 < argc) {
            n_jobs = atoi(argv[++i]);
            if (n_jobs < 1) n_jobs = 1;
        } else if (!strcmp(a, "--dump") && i + 1 < argc) {
            dump = argv[++i];
        } else if (!strcmp(a, "--verbose")) {
            verbose = 1;
        } else if (!strcmp(a, "--fast")) {
            fast = 1;
        } else if (!strcmp(a, "--lock")) {
            lock = 1;
        } else if (!strcmp(a, "--packed") && i + 1 < argc) {
            packed = argv[++i];
        } else if (!strcmp(a, "--stream")) {
            stream_mode = 1;
        } else if (!strcmp(a, "--window") && i + 1 < argc) {
            window = atof(argv[++i]);
        } else if (!strcmp(a, "--pause") && i + 1 < argc) {
            pause = atof(argv[++i]);
        } else if (!strcmp(a, "--rate") && i + 1 < argc) {
            rate = atoi(argv[++i]);
        } else if (!strcmp(a, "--channels") && i + 1 < argc) {
            channels = atoi(argv[++i]);
        } else if (a[0] == '-' && a[1]) {
            usage(stderr);
            die("unknown option %s", a);
        } else {
            files[n_files++] = a;
        }
    }
    if (!n_files) {
        usage(stderr);
        return 2;
    }
    if (timestamps < 0) timestamps = json ? PK_TS_SEGMENT : PK_TS_NONE;
    for (int i = 0; i < n_files; i++) {
        if (!strcmp(files[i], "-")) {
            if (!stream_mode) die("reading stdin needs --stream");
            continue;
        }
        FILE *f = fopen(files[i], "rb");
        if (!f) die("no such file: %s", files[i]);
        fclose(f);
    }
    if (!model_dir) model_dir = getenv("PARAKEET_MODEL");
    if (!model_dir) {
        const char *candidates[] = {"parakeet-redux.bin", "models/parakeet-redux-c/parakeet-redux.bin"};
        for (size_t i = 0; i < sizeof candidates / sizeof *candidates && !model_dir; i++) {
            FILE *f = fopen(candidates[i], "rb");
            if (f) {
                fclose(f);
                model_dir = candidates[i];
            }
        }
        if (!model_dir)
            die("no weights found: pass --model, or clone https://huggingface.co/eschmidbauer/parakeet-redux-c into models/parakeet-redux-c");
    }
    if (threads <= 0) threads = pk_cpu_count();
    if (n_jobs > n_files) n_jobs = n_files;
    int threads_per_job = threads / n_jobs > 0 ? threads / n_jobs : 1;

    double started = now_seconds();
    pk_options options = {.fast = fast, .lock_in_memory = lock, .max_window_seconds = window > 30.0 ? window : 0.0, .packed_path = packed};
    pk_model *model = pk_model_load(model_dir, &options);
    if (!model) die("%s", pk_error_message());
    fprintf(stderr, "model loaded in %.2fs from %s (%s, %zu MB of weights%s%s, %d job%s x %d threads)\n", now_seconds() - started,
            pk_model_is_mapped(model) ? packed : model_dir, pk_model_backend(model), pk_model_weight_bytes(model) >> 20,
            pk_model_is_mapped(model) ? ", mapped" : "", pk_model_is_locked(model) ? ", locked" : "", n_jobs, n_jobs > 1 ? "s" : "", threads_per_job);

    int status = 0;
    if (stream_mode || n_jobs == 1) {
        pk_context *ctx = pk_context_new(threads_per_job);
        if (!ctx) die("%s", pk_error_message());
        pk_context_set_verbose(ctx, verbose);
        pk_context_set_dump_dir(ctx, dump);
        if (stream_mode) {
            for (int i = 0; i < n_files; i++) status |= run_stream(ctx, model, files[i], json, timestamps, window, pause, rate, channels);
        } else {
            if (json && n_files > 1) printf("{\n");
            for (int i = 0; i < n_files; i++) {
                int n;
                float *pcm = pk_read_audio(files[i], &n);
                if (!pcm) {
                    pk_log(PK_LOG_ERROR, "%s", pk_error_message());
                    status = 1;
                    continue;
                }
                started = now_seconds();
                pk_result *r = pk_transcribe(ctx, model, pcm, n, timestamps);
                double elapsed = now_seconds() - started;
                free(pcm);
                if (!r) {
                    pk_log(PK_LOG_ERROR, "%s: %s", files[i], pk_error_message());
                    status = 1;
                    continue;
                }
                fprintf(stderr, "%s: %.1fs of audio in %.2fs, %.0fx real time\n", files[i], r->duration, elapsed, r->duration / elapsed);
                if (json) {
                    if (n_files > 1) {
                        printf("%s", i ? ",\n" : "");
                        print_json_string(files[i]);
                        printf(": ");
                    }
                    print_json(r, timestamps == PK_TS_WORD);
                } else {
                    const char *base = strrchr(files[i], '/');
                    print_text(r, timestamps, n_files > 1 ? (base ? base + 1 : files[i]) : NULL);
                }
                pk_result_free(r);
            }
            if (json && n_files > 1) printf("}\n");
        }
        pk_context_free(ctx);
    } else {
        jobs j = {.model = model, .files = files, .n_files = n_files, .timestamps = timestamps, .threads_per_job = threads_per_job,
                  .verbose = verbose, .dump = dump};
        j.results = xcalloc((size_t)n_files, sizeof *j.results);
        j.elapsed = xcalloc((size_t)n_files, sizeof *j.elapsed);
        j.failed = xcalloc((size_t)n_files, sizeof *j.failed);
        j.done = xcalloc((size_t)n_files, sizeof *j.done);
        pthread_mutex_init(&j.mutex, NULL);
        pthread_cond_init(&j.cond, NULL);
        pthread_t *workers = xmalloc((size_t)n_jobs * sizeof *workers);
        double batch_started = now_seconds();
        for (int w = 0; w < n_jobs; w++)
            if (pthread_create(&workers[w], NULL, job_worker, &j) != 0) die("cannot create job thread");
        double audio_total = 0.0;
        if (json) printf("{\n");
        for (int i = 0; i < n_files; i++) { /* print in argument order as results arrive */
            pthread_mutex_lock(&j.mutex);
            while (!atomic_load(&j.done[i])) pthread_cond_wait(&j.cond, &j.mutex);
            pthread_mutex_unlock(&j.mutex);
            pk_result *r = j.results[i];
            if (!r) {
                status = 1;
                continue;
            }
            audio_total += r->duration;
            fprintf(stderr, "%s: %.1fs of audio in %.2fs\n", files[i], r->duration, j.elapsed[i]);
            if (json) {
                printf("%s", i ? ",\n" : "");
                print_json_string(files[i]);
                printf(": ");
                print_json(r, timestamps == PK_TS_WORD);
            } else {
                const char *base = strrchr(files[i], '/');
                print_text(r, timestamps, base ? base + 1 : files[i]);
            }
            fflush(stdout);
            pk_result_free(r);
        }
        if (json) printf("}\n");
        for (int w = 0; w < n_jobs; w++) pthread_join(workers[w], NULL);
        double elapsed = now_seconds() - batch_started;
        fprintf(stderr, "%d files, %.1fs of audio in %.2fs with %d jobs, %.0fx real time\n", n_files, audio_total, elapsed, n_jobs, audio_total / elapsed);
        free(workers);
        free(j.results); free(j.elapsed); free(j.failed); free(j.done);
    }
    pk_model_free(model);
    free(files);
    return status;
}
