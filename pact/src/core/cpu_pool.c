#include "cpu_pool.h"
#include "thread.h"

#include <stdlib.h>

typedef struct pact_cpu_pool pact_cpu_pool;

typedef struct pact_cpu_worker {
    pact_cpu_pool *pool;
    pact_thread thread;
} pact_cpu_worker;

struct pact_cpu_pool {
    pact_mutex mutex;
    pact_condition ready;
    pact_condition completed;
    pact_cpu_worker *workers;
    size_t worker_count;
    uint64_t generation;
    size_t next_item;
    size_t item_count;
    size_t remaining;
    unsigned char *items;
    size_t item_size;
    pact_cpu_work_fn fn;
    int stop;
};

static int pact_cpu_worker_main(void *opaque)
{
    pact_cpu_pool *pool = ((pact_cpu_worker *)opaque)->pool;
    uint64_t seen = 0;
    pact_mutex_lock(&pool->mutex);
    for (;;) {
        while (!pool->stop && seen == pool->generation)
            pact_condition_wait(&pool->ready, &pool->mutex);
        if (pool->stop) break;
        seen = pool->generation;
        while (pool->next_item < pool->item_count) {
            size_t index = pool->next_item++;
            void *item = pool->items + index * pool->item_size;
            pact_cpu_work_fn fn = pool->fn;
            pact_mutex_unlock(&pool->mutex);
            fn(item);
            pact_mutex_lock(&pool->mutex);
            if (--pool->remaining == 0)
                pact_condition_signal(&pool->completed);
        }
    }
    pact_mutex_unlock(&pool->mutex);
    return 0;
}

static pact_cpu_pool *pact_cpu_pool_create(size_t workers)
{
    pact_cpu_pool *pool = calloc(1, sizeof(*pool));
    if (pool == NULL) return NULL;
    if (!pact_mutex_init(&pool->mutex)) { free(pool); return NULL; }
    if (!pact_condition_init(&pool->ready)) {
        pact_mutex_destroy(&pool->mutex); free(pool); return NULL;
    }
    if (!pact_condition_init(&pool->completed)) {
        pact_condition_destroy(&pool->ready);
        pact_mutex_destroy(&pool->mutex); free(pool); return NULL;
    }
    pool->workers = calloc(workers, sizeof(*pool->workers));
    if (pool->workers == NULL) { pact_cpu_pool_destroy(pool); return NULL; }
    for (size_t i = 0; i < workers; ++i) {
        pool->workers[i].pool = pool;
        if (!pact_thread_start(&pool->workers[i].thread,
                               pact_cpu_worker_main, &pool->workers[i])) {
            pact_cpu_pool_destroy(pool);
            return NULL;
        }
        ++pool->worker_count;
    }
    return pool;
}

void pact_cpu_pool_destroy(void *opaque)
{
    pact_cpu_pool *pool = opaque;
    if (pool == NULL) return;
    pact_mutex_lock(&pool->mutex);
    pool->stop = 1;
    pact_condition_broadcast(&pool->ready);
    pact_mutex_unlock(&pool->mutex);
    for (size_t i = 0; i < pool->worker_count; ++i)
        pact_thread_join(pool->workers[i].thread);
    free(pool->workers);
    pact_condition_destroy(&pool->completed);
    pact_condition_destroy(&pool->ready);
    pact_mutex_destroy(&pool->mutex);
    free(pool);
}

void pact_cpu_pool_prepare(pact_runtime *runtime)
{
    if (runtime->cpu_pool != NULL) return;
    uint32_t logical = runtime->devices[0].info.logical_processor_count;
    size_t workers = logical > 1 ? logical - 1 : 0;
    if (workers > 7) workers = 7;
    if (workers != 0) runtime->cpu_pool = pact_cpu_pool_create(workers);
}

void pact_cpu_pool_run(
    pact_runtime *runtime, pact_cpu_work_fn fn, void *items,
    size_t item_size, size_t item_count)
{
    if (item_count == 0) return;
    if (item_count == 1) { fn(items); return; }
    pact_cpu_pool_prepare(runtime);
    pact_cpu_pool *pool = runtime->cpu_pool;
    if (pool == NULL) {
        for (size_t i = 0; i < item_count; ++i)
            fn((unsigned char *)items + i * item_size);
        return;
    }
    pact_mutex_lock(&pool->mutex);
    pool->items = items;
    pool->item_size = item_size;
    pool->item_count = item_count;
    pool->next_item = 0;
    pool->remaining = item_count;
    pool->fn = fn;
    ++pool->generation;
    pact_condition_broadcast(&pool->ready);
    while (pool->next_item < pool->item_count) {
        size_t index = pool->next_item++;
        void *item = pool->items + index * pool->item_size;
        pact_mutex_unlock(&pool->mutex);
        fn(item);
        pact_mutex_lock(&pool->mutex);
        if (--pool->remaining == 0)
            pact_condition_signal(&pool->completed);
    }
    while (pool->remaining != 0)
        pact_condition_wait(&pool->completed, &pool->mutex);
    pool->items = NULL;
    pact_mutex_unlock(&pool->mutex);
}
