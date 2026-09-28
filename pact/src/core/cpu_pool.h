#ifndef PACT_CPU_POOL_H
#define PACT_CPU_POOL_H

#include "runtime_internal.h"

typedef void (*pact_cpu_work_fn)(void *);

void pact_cpu_pool_prepare(pact_runtime *runtime);
void pact_cpu_pool_run(
    pact_runtime *runtime, pact_cpu_work_fn fn, void *items,
    size_t item_size, size_t item_count);
void pact_cpu_pool_destroy(void *opaque);

#endif
