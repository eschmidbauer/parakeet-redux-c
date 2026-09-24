#include "util.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static _Thread_local jmp_buf *error_trap;
static _Thread_local char error_message[1024];

jmp_buf *pk_error_trap(jmp_buf *trap) {
    jmp_buf *previous = error_trap;
    error_trap = trap;
    return previous;
}

const char *pk_error_message(void) { return error_message; }

static pk_log_fn log_fn;
static void *log_user;

static void default_log(int level, const char *message, void *user) {
    (void)user;
    fprintf(stderr, "%s%s\n", level == PK_LOG_ERROR ? "error: " : level == PK_LOG_WARNING ? "warning: " : "", message);
}

void pk_set_log(pk_log_fn fn, void *user) {
    log_fn = fn;
    log_user = user;
}

void pk_log(int level, const char *fmt, ...) {
    char message[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(message, sizeof message, fmt, ap);
    va_end(ap);
    (log_fn ? log_fn : default_log)(level, message, log_user);
}

/* format into a scratch buffer first: the arguments may point at error_message itself */
void pk_set_error(const char *fmt, ...) {
    char scratch[sizeof error_message];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(scratch, sizeof scratch, fmt, ap);
    va_end(ap);
    memcpy(error_message, scratch, sizeof scratch);
}

void die(const char *fmt, ...) {
    char scratch[sizeof error_message];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(scratch, sizeof scratch, fmt, ap);
    va_end(ap);
    memcpy(error_message, scratch, sizeof scratch);
    if (error_trap) longjmp(*error_trap, 1);
    pk_log(PK_LOG_ERROR, "%s", error_message);
    exit(1);
}

void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory allocating %zu bytes", n);
    return p;
}

void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n ? n : 1, size ? size : 1);
    if (!p) die("out of memory allocating %zu bytes", n * size);
    return p;
}

void *xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) die("out of memory allocating %zu bytes", n);
    return p;
}

void *xaligned(size_t n) {
    void *p = NULL;
    size_t size = (n + 63) / 64 * 64;
    if (posix_memalign(&p, 64, size ? size : 64) != 0) die("out of memory allocating %zu bytes", n);
    return p;
}

char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = xmalloc(n);
    memcpy(p, s, n);
    return p;
}

double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int pk_lock_pages(const void *p, size_t n) {
    if (!p || !n) return 0;
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) page = 4096;
    uintptr_t start = (uintptr_t)p & ~((uintptr_t)page - 1);
    uintptr_t end = ((uintptr_t)p + n + (uintptr_t)page - 1) & ~((uintptr_t)page - 1);
    return mlock((void *)start, end - start) == 0 ? 0 : -1;
}
