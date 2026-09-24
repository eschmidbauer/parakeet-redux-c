#include "pool.h"

#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "util.h"

struct pk_pool {
    int count;          /* threads including the caller */
    pthread_t *workers; /* count - 1 */
    pthread_mutex_t mutex;
    pthread_cond_t wake;
    atomic_int generation;
    atomic_int shutdown;
    /* current job; every worker runs it to exhaustion and then reports in,
     * so no worker is ever still inside a previous region */
    pk_task_fn fn;
    void *ctx;
    int n_tasks;
    atomic_int next;
    atomic_int finished;
    atomic_int failed;
    char error[1024];
};

int pk_cpu_count(void) {
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    return online > 0 ? (int)online : 1;
}

/* Runs tasks until none are left. A die() inside a task (out of memory, in
 * practice) lands here: the message is kept, the remaining tasks are skipped
 * and the caller re-raises it once every worker has reached the barrier. */
static void run_tasks(pk_pool *p, int thread) {
    jmp_buf trap;
    jmp_buf *previous = pk_error_trap(&trap);
    if (setjmp(trap)) {
        if (!atomic_exchange(&p->failed, 1)) strncpy(p->error, pk_error_message(), sizeof p->error - 1);
        atomic_store(&p->next, INT_MAX / 2);
        pk_error_trap(previous);
        return;
    }
    for (;;) {
        int t = atomic_fetch_add(&p->next, 1);
        if (t >= p->n_tasks) break;
        p->fn(p->ctx, t, thread);
    }
    pk_error_trap(previous);
}

typedef struct {
    pk_pool *pool;
    int id;
} worker_arg;

static void *worker(void *arg) {
    worker_arg *w = arg;
    pk_pool *p = w->pool;
    int id = w->id;
    free(w);
    int seen = 0;
    for (;;) {
        /* spin briefly for the next job, then sleep on the condition variable */
        int gen = atomic_load(&p->generation);
        for (int spin = 0; spin < 20000 && gen == seen && !atomic_load(&p->shutdown); spin++) gen = atomic_load(&p->generation);
        if (gen == seen && !atomic_load(&p->shutdown)) {
            pthread_mutex_lock(&p->mutex);
            while (atomic_load(&p->generation) == seen && !atomic_load(&p->shutdown)) pthread_cond_wait(&p->wake, &p->mutex);
            gen = atomic_load(&p->generation);
            pthread_mutex_unlock(&p->mutex);
        }
        if (atomic_load(&p->shutdown)) return NULL;
        seen = gen;
        run_tasks(p, id);
        atomic_fetch_add(&p->finished, 1);
    }
}

pk_pool *pk_pool_new(int n) {
    if (n <= 0) n = pk_cpu_count();
    if (n > 256) n = 256;
    pk_pool *p = xcalloc(1, sizeof *p);
    p->count = n;
    pthread_mutex_init(&p->mutex, NULL);
    pthread_cond_init(&p->wake, NULL);
    p->workers = xmalloc((size_t)(n > 1 ? n - 1 : 1) * sizeof *p->workers);
    for (int i = 1; i < n; i++) {
        worker_arg *w = xmalloc(sizeof *w);
        w->pool = p;
        w->id = i;
        if (pthread_create(&p->workers[i - 1], NULL, worker, w) != 0) die("cannot create worker thread");
    }
    return p;
}

int pk_pool_size(const pk_pool *p) { return p ? p->count : 1; }

void pk_pool_run(pk_pool *p, int n_tasks, pk_task_fn fn, void *ctx) {
    if (n_tasks <= 0) return;
    if (!p || p->count == 1 || n_tasks == 1) {
        for (int t = 0; t < n_tasks; t++) fn(ctx, t, 0); /* on the calling thread: its own trap applies */
        return;
    }
    p->fn = fn;
    p->ctx = ctx;
    p->n_tasks = n_tasks;
    atomic_store(&p->failed, 0);
    atomic_store(&p->finished, 0);
    atomic_store(&p->next, 0);
    pthread_mutex_lock(&p->mutex);
    atomic_fetch_add(&p->generation, 1);
    pthread_cond_broadcast(&p->wake);
    pthread_mutex_unlock(&p->mutex);
    run_tasks(p, 0);
    /* barrier: wait for every worker to leave this generation */
    for (long spin = 0; atomic_load(&p->finished) < p->count - 1; spin++)
        if (spin > 2000) sched_yield();
    if (atomic_load(&p->failed)) die("%s", p->error);
}

typedef struct {
    pk_range_fn fn;
    void *ctx;
    int n, chunk;
} range_ctx;

static void range_task(void *arg, int task, int thread) {
    range_ctx *r = arg;
    int begin = task * r->chunk, end = begin + r->chunk;
    if (end > r->n) end = r->n;
    if (begin < end) r->fn(r->ctx, begin, end, thread);
}

void pk_pool_range(pk_pool *p, int n, int min_chunk, pk_range_fn fn, void *ctx) {
    if (n <= 0) return;
    int threads = pk_pool_size(p);
    int chunk = (n + threads * 2 - 1) / (threads * 2);
    if (chunk < min_chunk) chunk = min_chunk;
    range_ctx r = {fn, ctx, n, chunk};
    pk_pool_run(p, (n + chunk - 1) / chunk, range_task, &r);
}

void pk_pool_free(pk_pool *p) {
    if (!p) return;
    pthread_mutex_lock(&p->mutex);
    atomic_store(&p->shutdown, 1);
    pthread_cond_broadcast(&p->wake);
    pthread_mutex_unlock(&p->mutex);
    for (int i = 1; i < p->count; i++) pthread_join(p->workers[i - 1], NULL);
    pthread_mutex_destroy(&p->mutex);
    pthread_cond_destroy(&p->wake);
    free(p->workers);
    free(p);
}
