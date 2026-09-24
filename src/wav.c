#include "wav.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "resample.h"
#include "util.h"

static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }

static uint8_t *read_file(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        pk_set_error("cannot open %s", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (*size < 0) *size = 0;
    uint8_t *buf = xmalloc((size_t)*size + 1);
    if (fread(buf, 1, (size_t)*size, f) != (size_t)*size) {
        fclose(f);
        free(buf);
        pk_set_error("cannot read %s", path);
        return NULL;
    }
    fclose(f);
    return buf;
}

static int is_wav(const uint8_t *buf, long size) {
    return size >= 12 && memcmp(buf, "RIFF", 4) == 0 && memcmp(buf + 8, "WAVE", 4) == 0;
}

static float *decode_wav(const uint8_t *buf, long size, const char *path, int *n_samples, int *sample_rate, int quiet) {
    int format = 0, channels = 0, rate = 0, bits = 0;
    const uint8_t *data = NULL;
    size_t data_len = 0;
    size_t off = 12;
    while (off + 8 <= (size_t)size) {
        const uint8_t *id = buf + off;
        size_t len = rd32(buf + off + 4);
        const uint8_t *body = buf + off + 8;
        if (len > (size_t)size - off - 8) len = (size_t)size - off - 8;
        if (memcmp(id, "fmt ", 4) == 0 && len >= 16) {
            format = rd16(body);
            channels = rd16(body + 2);
            rate = (int)rd32(body + 4);
            bits = rd16(body + 14);
            if (format == 0xFFFE && len >= 26) format = rd16(body + 24);
        } else if (memcmp(id, "data", 4) == 0) {
            data = body;
            data_len = len;
        }
        off += 8 + len + (len & 1);
    }
    if (!data || channels <= 0 || rate <= 0) {
        if (!quiet) pk_set_error("%s has no usable fmt/data chunks", path);
        return NULL;
    }
    int bytes = bits / 8;
    int ok = (format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) || (format == 3 && bits == 32);
    if (!ok || bytes == 0) {
        if (!quiet) pk_set_error("%s: unsupported WAV encoding (format %d, %d bits)", path, format, bits);
        return NULL;
    }
    size_t frames = data_len / ((size_t)bytes * channels);
    float *pcm = xmalloc((frames ? frames : 1) * sizeof(float));
    for (size_t i = 0; i < frames; i++) {
        double acc = 0.0;
        for (int c = 0; c < channels; c++) {
            const uint8_t *p = data + (i * channels + c) * bytes;
            if (format == 3) {
                uint32_t u = rd32(p);
                float v;
                memcpy(&v, &u, 4);
                acc += v;
            } else if (bits == 8) {
                acc += ((int)p[0] - 128) / 128.0;
            } else if (bits == 16) {
                acc += (int16_t)rd16(p) / 32768.0;
            } else if (bits == 24) {
                int32_t v = (int32_t)((uint32_t)p[0] << 8 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 24) >> 8;
                acc += v / 8388608.0;
            } else {
                acc += (int32_t)rd32(p) / 2147483648.0;
            }
        }
        pcm[i] = (float)(acc / channels);
    }
    *n_samples = (int)frames;
    *sample_rate = rate;
    return pcm;
}

float *pk_wav_read(const char *path, int *n_samples, int *sample_rate) {
    long size;
    uint8_t *buf = read_file(path, &size);
    if (!buf) return NULL;
    if (!is_wav(buf, size)) {
        free(buf);
        pk_set_error("%s is not a WAV file", path);
        return NULL;
    }
    float *pcm = decode_wav(buf, size, path, n_samples, sample_rate, 0);
    free(buf);
    return pcm;
}

float *pk_ffmpeg_read(const char *path, int rate, int *n_samples) {
    /* single-quote the path for the shell */
    size_t len = strlen(path), extra = 0;
    for (size_t i = 0; i < len; i++) extra += path[i] == '\'' ? 3 : 0;
    char *quoted = xmalloc(len + extra + 3);
    char *q = quoted;
    *q++ = '\'';
    for (size_t i = 0; i < len; i++) {
        if (path[i] == '\'') {
            memcpy(q, "'\\''", 4);
            q += 4;
        } else {
            *q++ = path[i];
        }
    }
    *q++ = '\'';
    *q = '\0';
    char cmd[4096];
    snprintf(cmd, sizeof cmd, "ffmpeg -v error -nostdin -i %s -f f32le -acodec pcm_f32le -ac 1 -ar %d - 2>/dev/null", quoted, rate);
    free(quoted);
    FILE *pipe = popen(cmd, "r");
    if (!pipe) return NULL;
    size_t cap = 1 << 20, n = 0;
    float *pcm = xmalloc(cap * sizeof(float));
    for (;;) {
        if (n == cap) pcm = xrealloc(pcm, (cap *= 2) * sizeof(float));
        size_t got = fread(pcm + n, sizeof(float), cap - n, pipe);
        if (!got) break;
        n += got;
    }
    int status = pclose(pipe);
    if (status != 0 || n == 0) {
        free(pcm);
        return NULL;
    }
    *n_samples = (int)n;
    return pcm;
}

float *pk_audio_read(const char *path, int rate, int *n_samples) {
    long size;
    uint8_t *buf = read_file(path, &size);
    if (!buf) return NULL;
    float *pcm = NULL;
    int file_rate = 0;
    if (is_wav(buf, size)) pcm = decode_wav(buf, size, path, n_samples, &file_rate, 1);
    free(buf);
    if (pcm && file_rate != rate) {
        int m;
        float *converted = pk_resample(pcm, *n_samples, file_rate, rate, &m);
        free(pcm);
        pcm = converted;
        *n_samples = m;
    }
    if (pcm) return pcm;
    pcm = pk_ffmpeg_read(path, rate, n_samples);
    if (!pcm) pk_set_error("%s: not a readable WAV file and ffmpeg could not decode it", path);
    return pcm;
}
