#undef NDEBUG
#include <assert.h>
#include <pact/pact.h>
#include "runtime_internal.h"

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    pact_runtime *runtime = NULL;
    assert(pact_runtime_create(&runtime) == PACT_SUCCESS);
    uint32_t gpu_count = 0;
    for (uint32_t i = 0; i < runtime->device_count; ++i)
        if (runtime->devices[i].info.backend == PACT_BACKEND_VULKAN) {
            runtime->devices[i].performance = (pact_performance_sample){
                16, 0.0001, 262144, 1};
            ++gpu_count;
        }
    if (gpu_count == 0) {
        pact_runtime_destroy(runtime);
        puts("no Vulkan GPUs; mixed plan skipped");
        return 0;
    }
    runtime->cpu_performance = (pact_performance_sample){16, 1.0, 1048576, 1, 2};
    const size_t n = 1048576;
    float *a = malloc(n * sizeof(float));
    float *b = malloc(n * sizeof(float));
    float *result = malloc(n * sizeof(float));
    assert(a && b && result);
    for (size_t i = 0; i < n; ++i) {
        a[i] = (float)(i % 97);
        b[i] = (float)(i % 13);
    }
    uint64_t shape = n;
    pact_tensor *x = NULL, *y = NULL, *current = NULL;
    assert(pact_tensor_from_data(runtime, PACT_DTYPE_F32, &shape, 1,
                                 a, n * sizeof(float), &x) == PACT_SUCCESS);
    assert(pact_tensor_from_data(runtime, PACT_DTYPE_F32, &shape, 1,
                                 b, n * sizeof(float), &y) == PACT_SUCCESS);
    current = x;
    pact_tensor_retain(current);
    for (unsigned i = 0; i < 16; ++i) {
        pact_tensor *next = NULL;
        assert(pact_tensor_add(current, y, &next) == PACT_SUCCESS);
        pact_tensor_release(current);
        current = next;
    }
    assert(pact_tensor_read(current, result, n * sizeof(float)) == PACT_SUCCESS);
    for (size_t i = 0; i < n; ++i)
        assert(result[i] == a[i] + 16 * b[i]);
    pact_execution_info info = {0};
    info.struct_size = sizeof(info);
    assert(pact_runtime_get_last_execution(runtime, &info) == PACT_SUCCESS);
    assert(info.vulkan_shards == gpu_count);
    assert(info.vulkan_elements > 0);
    assert(info.cpu_elements > 0);
    assert(info.cpu_elements + info.vulkan_elements == n);
    assert(info.vulkan_fallback == PACT_SUCCESS);
    pact_tensor_release(current);
    pact_tensor_release(x);
    pact_tensor_release(y);
    pact_runtime_destroy(runtime);
    free(a); free(b); free(result);
    return 0;
}
