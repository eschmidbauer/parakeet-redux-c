#ifndef PK_UTIL_H
#define PK_UTIL_H

#include <setjmp.h>
#include <stddef.h>

#include "pk.h"

/* die() records the message and returns control to the calling thread's error
 * trap; with no trap armed it logs the message and exits. */
void die(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
jmp_buf *pk_error_trap(jmp_buf *trap); /* returns the previous trap */
const char *pk_error_message(void);    /* thread local, valid after a trapped die() or pk_set_error() */
void pk_set_error(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

/* Library messages: default to stderr, or the callback set with pk_set_log() (pk.h). */
void pk_log(int level, const char *fmt, ...) __attribute__((format(printf, 2, 3)));

void *xmalloc(size_t n);
void *xcalloc(size_t n, size_t size);
void *xrealloc(void *p, size_t n);
/* 64-byte aligned allocation, freed with free() */
void *xaligned(size_t n);
char *xstrdup(const char *s);
double now_seconds(void);

/* Pin memory so it is never paged out; returns 0 or -1. */
int pk_lock_pages(const void *p, size_t n);

/* Python-style floor division for non-negative divisors. */
static inline long floordiv(long a, long b) {
    long q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q--;
    return q;
}

#endif
