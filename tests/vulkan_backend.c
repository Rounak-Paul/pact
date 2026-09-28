#undef NDEBUG
#include <assert.h>
#include <pact/pact.h>
#include "tensor_internal.h"

#include <stdio.h>
#include <stdlib.h>

static void run_on_device(pact_runtime *runtime, uint32_t device, size_t n)
{
    float *a = malloc(n * sizeof(float));
    float *b = malloc(n * sizeof(float));
    float *output = malloc(n * sizeof(float));
    assert(a && b && output);
    for (size_t i = 0; i < n; ++i) {
        a[i] = (float)(i % 100);
        b[i] = (float)(i % 11);
    }
    uint64_t shape = n;
    pact_tensor *x = NULL, *y = NULL, *z = NULL, *w = NULL;
    assert(pact_tensor_from_data(runtime, PACT_DTYPE_F32, &shape, 1,
                                 a, n * sizeof(float), &x) == PACT_SUCCESS);
    assert(pact_tensor_from_data(runtime, PACT_DTYPE_F32, &shape, 1,
                                 b, n * sizeof(float), &y) == PACT_SUCCESS);
    assert(pact_tensor_add(x, y, &z) == PACT_SUCCESS);
    assert(pact_tensor_add(z, z, &w) == PACT_SUCCESS);
    pact_graph graph = {0};
    assert(pact_graph_collect(w, &graph) == PACT_SUCCESS);
    assert(graph.count == 4);
    pact_task tasks[5] = {0};
    for (size_t i = 0; i < graph.count; ++i) {
        tasks[i].node_index = i;
        tasks[i].value_id = graph.nodes[i]->value_id;
        tasks[i].version = graph.nodes[i]->version;
        tasks[i].count = n;
        tasks[i].device_index = device;
        tasks[i].kind = graph.nodes[i]->materialized
                            ? PACT_TASK_SOURCE : PACT_TASK_ADD;
        if (tasks[i].kind == PACT_TASK_ADD) {
            tasks[i].dependencies[0] = graph.nodes[i]->input_a->plan_index;
            tasks[i].dependencies[1] = graph.nodes[i]->input_b->plan_index;
        }
    }
    tasks[4].kind = PACT_TASK_DOWNLOAD;
    tasks[4].dependencies[0] = 3;
    tasks[4].count = n;
    tasks[4].device_index = device;
    for (unsigned pass = 0; pass < 2; ++pass) {
        pact_gpu_timing timing = {0};
        pact_result result = pact_vulkan_execute_tasks(
            runtime, device, &graph, tasks, 5, (unsigned char *)output, &timing);
        if (result != PACT_SUCCESS) {
            fprintf(stderr, "GPU submit failed: %s\n", pact_result_string(result));
            abort();
        }
        assert(timing.active_us > 0.0);
        for (size_t i = 0; i < n; ++i)
            assert(output[i] == 2 * (a[i] + b[i]));
    }
    pact_graph_release(&graph);
    pact_tensor_release(w); pact_tensor_release(z);
    pact_tensor_release(x); pact_tensor_release(y);
    free(a); free(b); free(output);
}

int main(void)
{
    pact_runtime *runtime = NULL;
    assert(pact_runtime_create(&runtime) == PACT_SUCCESS);
    uint32_t device_count = 0;
    assert(pact_runtime_get_device_count(runtime, &device_count) == PACT_SUCCESS);
    uint32_t checked = 0;
    for (uint32_t device = 0; device < device_count; ++device) {
        pact_device_info info = {0};
        info.struct_size = sizeof(info);
        assert(pact_runtime_get_device_info(runtime, device, &info) == PACT_SUCCESS);
        if (info.backend != PACT_BACKEND_VULKAN) continue;
        run_on_device(runtime, device, 1);
        run_on_device(runtime, device, 4096);
        run_on_device(runtime, device, 1200000);
        ++checked;
    }
    printf("verified persistent contexts and streaming on %u Vulkan devices\n", checked);
    pact_runtime_destroy(runtime);
    return 0;
}
