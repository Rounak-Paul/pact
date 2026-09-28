#ifndef PACT_TENSOR_INTERNAL_H
#define PACT_TENSOR_INTERNAL_H

#include "runtime_internal.h"

#include <stddef.h>
#include <stdint.h>

typedef enum pact_op_kind {
    PACT_OP_INPUT = 1,
    PACT_OP_ADD = 2
} pact_op_kind;

struct pact_tensor {
    pact_runtime *runtime;
    size_t ref_count;
    uint64_t value_id;
    uint64_t version;
    pact_dtype dtype;
    uint32_t rank;
    uint64_t *shape;
    size_t element_count;
    size_t byte_count;
    pact_op_kind op;
    pact_tensor *input_a;
    pact_tensor *input_b;
    unsigned char *host_data;
    pact_result failure;
    int materialized;
    uint64_t visit_epoch;
    unsigned char visit_state;
    size_t plan_index;
    pact_tensor *release_next;
};

typedef struct pact_graph {
    pact_tensor **nodes;
    size_t count;
} pact_graph;

typedef enum pact_task_kind {
    PACT_TASK_SOURCE = 1,
    PACT_TASK_ADD = 2,
    PACT_TASK_DOWNLOAD = 3
} pact_task_kind;

typedef struct pact_task {
    pact_task_kind kind;
    size_t node_index;
    uint64_t value_id;
    uint64_t version;
    size_t dependencies[2];
    size_t start;
    size_t count;
    uint32_t device_index;
} pact_task;

pact_result pact_graph_collect(pact_tensor *root, pact_graph *out_graph);
void pact_graph_release(pact_graph *graph);
pact_result pact_plan_execute(
    pact_tensor *root, const pact_graph *graph, unsigned char *output);
typedef struct pact_gpu_timing {
    double setup_us;
    double upload_us;
    double record_us;
    double queue_us;
    double download_us;
    double active_us;
} pact_gpu_timing;

pact_result pact_vulkan_execute_tasks(
    pact_runtime *runtime, uint32_t device_index, const pact_graph *graph,
    const pact_task *tasks, size_t task_count, unsigned char *output,
    pact_gpu_timing *out_timing);

#endif
