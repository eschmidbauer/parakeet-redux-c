/* A small persistent thread pool. A pool belongs to one context and is driven
 * from one thread at a time; the caller takes part in running the tasks. */
#ifndef PK_POOL_H
#define PK_POOL_H

typedef struct pk_pool pk_pool;
typedef void (*pk_task_fn)(void *ctx, int task, int thread);
typedef void (*pk_range_fn)(void *ctx, int begin, int end, int thread);

/* n threads including the caller; n <= 0 uses every online CPU. */
pk_pool *pk_pool_new(int n);
void pk_pool_free(pk_pool *p);
int pk_pool_size(const pk_pool *p);
int pk_cpu_count(void);

/* Run n_tasks tasks and return when all are done. p may be NULL (run inline). */
void pk_pool_run(pk_pool *p, int n_tasks, pk_task_fn fn, void *ctx);
/* Split [0, n) into chunks of at least min_chunk and run fn on each. */
void pk_pool_range(pk_pool *p, int n, int min_chunk, pk_range_fn fn, void *ctx);

#endif
