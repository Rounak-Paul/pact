#include <pact/pact.h>

#undef NDEBUG
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void small_cases(pact_runtime *runtime)
{
    uint64_t shape[] = {4};
    float a[] = {1, 2, 3, 4};
    float b[] = {5, 6, 7, 8};
    pact_tensor *x = NULL, *y = NULL, *z = NULL, *w = NULL;
    assert(pact_tensor_from_data(runtime, PACT_DTYPE_F32, shape, 1,
                                 a, sizeof(a), &x) == PACT_SUCCESS);
    assert(pact_tensor_from_data(runtime, PACT_DTYPE_F32, shape, 1,
                                 b, sizeof(b), &y) == PACT_SUCCESS);
    assert(pact_tensor_add(x, y, &z) == PACT_SUCCESS);
    assert(pact_tensor_add(z, x, &w) == PACT_SUCCESS);
    pact_tensor_release(z);
    pact_tensor_release(x);
    pact_tensor_release(y);
    float result[4] = {0};
    assert(pact_tensor_read(w, result, sizeof(result)) == PACT_SUCCESS);
    for (int i = 0; i < 4; ++i) assert(result[i] == 2 * a[i] + b[i]);
    assert(pact_tensor_read(w, result, sizeof(result) - 1) == PACT_ERROR_BUFFER_SIZE);
    uint32_t rank = 0;
    assert(pact_tensor_get_rank(w, &rank) == PACT_SUCCESS && rank == 1);
    uint64_t actual_shape = 0;
    assert(pact_tensor_get_shape(w, &actual_shape, 1) == PACT_SUCCESS);
    assert(actual_shape == 4);
    pact_tensor_release(w);

    pact_tensor *first = NULL, *second = NULL, *wrong = NULL;
    uint64_t other_shape[] = {2, 2};
    assert(pact_tensor_from_data(runtime, PACT_DTYPE_F32, shape, 1,
                                 a, sizeof(a), &first) == PACT_SUCCESS);
    assert(pact_tensor_from_data(runtime, PACT_DTYPE_F32, other_shape, 2,
                                 b, sizeof(b), &second) == PACT_SUCCESS);
    assert(pact_tensor_add(first, second, &wrong) == PACT_ERROR_SHAPE_MISMATCH);
    assert(wrong == NULL);
    assert(pact_tensor_from_data(runtime, (pact_dtype)999, shape, 1,
                                 a, sizeof(a), &wrong) == PACT_ERROR_UNSUPPORTED_DTYPE);
    assert(pact_tensor_from_data(runtime, PACT_DTYPE_F32, shape, 1,
                                 a, sizeof(a) - 1, &wrong) == PACT_ERROR_BUFFER_SIZE);
    assert(pact_tensor_add(first, first, &wrong) == PACT_SUCCESS);
    a[0] = -1000.0f;
    assert(pact_tensor_read(wrong, result, sizeof(result)) == PACT_SUCCESS);
    assert(result[0] == 2.0f);
    pact_tensor_release(wrong);
    pact_tensor_release(first);
    pact_tensor_release(second);

    pact_tensor *scalar = NULL;
    float value = 3.5f;
    assert(pact_tensor_from_data(runtime, PACT_DTYPE_F32, NULL, 0,
                                 &value, sizeof(value), &scalar) == PACT_SUCCESS);
    assert(pact_tensor_read(scalar, &value, sizeof(value)) == PACT_SUCCESS);
    assert(value == 3.5f);
    pact_tensor_release(scalar);
    uint64_t empty_shape[] = {0};
    pact_tensor *empty = NULL;
    assert(pact_tensor_from_data(runtime, PACT_DTYPE_F32, empty_shape, 1,
                                 NULL, 0, &empty) == PACT_SUCCESS);
    assert(pact_tensor_read(empty, NULL, 0) == PACT_SUCCESS);
    pact_tensor_release(empty);
}

static void deep_graph(pact_runtime *runtime)
{
    float one = 1.0f;
    pact_tensor *base = NULL;
    assert(pact_tensor_from_data(runtime, PACT_DTYPE_F32, NULL, 0,
                                 &one, sizeof(one), &base) == PACT_SUCCESS);
    pact_tensor *current = base;
    pact_tensor_retain(current);
    for (int i = 0; i < 4096; ++i) {
        pact_tensor *next = NULL;
        assert(pact_tensor_add(current, base, &next) == PACT_SUCCESS);
        pact_tensor_release(current);
        current = next;
    }
    float result = 0.0f;
    assert(pact_tensor_read(current, &result, sizeof(result)) == PACT_SUCCESS);
    assert(result == 4097.0f);
    pact_tensor_release(current);
    pact_tensor_release(base);
}

static void parallel_branch(pact_runtime *runtime)
{
    const size_t n = 262144;
    float *a = malloc(n * sizeof(float));
    float *b = malloc(n * sizeof(float));
    float *result = malloc(n * sizeof(float));
    assert(a && b && result);
    for (size_t i = 0; i < n; ++i) {
        a[i] = (float)(i % 37);
        b[i] = (float)(i % 19);
    }
    uint64_t shape = n;
    pact_tensor *x = NULL, *y = NULL, *z = NULL, *w = NULL;
    assert(pact_tensor_from_data(runtime, PACT_DTYPE_F32, &shape, 1,
                                 a, n * sizeof(float), &x) == PACT_SUCCESS);
    assert(pact_tensor_from_data(runtime, PACT_DTYPE_F32, &shape, 1,
                                 b, n * sizeof(float), &y) == PACT_SUCCESS);
    assert(pact_tensor_add(x, y, &z) == PACT_SUCCESS);
    assert(pact_tensor_add(z, z, &w) == PACT_SUCCESS);
    assert(pact_tensor_read(w, result, n * sizeof(float)) == PACT_SUCCESS);
    for (size_t i = 0; i < n; ++i) assert(result[i] == 2 * (a[i] + b[i]));
    pact_execution_info execution = {0};
    execution.struct_size = sizeof(execution);
    assert(pact_runtime_get_last_execution(runtime, &execution) == PACT_SUCCESS);
    pact_device_info cpu = {0};
    cpu.struct_size = sizeof(cpu);
    assert(pact_runtime_get_device_info(runtime, 0, &cpu) == PACT_SUCCESS);
    if (cpu.logical_processor_count > 1) assert(execution.cpu_tasks > 1);
    pact_tensor_release(w); pact_tensor_release(z);
    pact_tensor_release(x); pact_tensor_release(y);
    free(a); free(b); free(result);
}

static void large_case(pact_runtime *runtime)
{
    const size_t n = 200000;
    float *a = malloc(n * sizeof(float));
    float *b = malloc(n * sizeof(float));
    float *result = malloc(n * sizeof(float));
    assert(a && b && result);
    for (size_t i = 0; i < n; ++i) {
        a[i] = (float)(i % 1000);
        b[i] = (float)(i % 17);
    }
    uint64_t shape = n;
    pact_tensor *x = NULL, *y = NULL, *z = NULL, *w = NULL;
    assert(pact_tensor_from_data(runtime, PACT_DTYPE_F32, &shape, 1,
                                 a, n * sizeof(float), &x) == PACT_SUCCESS);
    assert(pact_tensor_from_data(runtime, PACT_DTYPE_F32, &shape, 1,
                                 b, n * sizeof(float), &y) == PACT_SUCCESS);
    assert(pact_tensor_add(x, y, &z) == PACT_SUCCESS);
    assert(pact_tensor_add(z, x, &w) == PACT_SUCCESS);
    assert(pact_tensor_read(w, result, n * sizeof(float)) == PACT_SUCCESS);
    for (size_t i = 0; i < n; ++i) assert(result[i] == 2 * a[i] + b[i]);
    pact_execution_info execution = {0};
    execution.struct_size = sizeof(execution);
    assert(pact_runtime_get_last_execution(runtime, &execution) == PACT_SUCCESS);
    assert(execution.cpu_elements + execution.vulkan_elements == n);
    uint32_t devices = 0;
    assert(pact_runtime_get_device_count(runtime, &devices) == PACT_SUCCESS);
    uint32_t vulkan_devices = 0;
    for (uint32_t i = 0; i < devices; ++i) {
        pact_device_info info = {0};
        info.struct_size = sizeof(info);
        assert(pact_runtime_get_device_info(runtime, i, &info) == PACT_SUCCESS);
        if (info.backend == PACT_BACKEND_VULKAN) ++vulkan_devices;
    }
    assert(execution.vulkan_shards <= vulkan_devices);
    assert(execution.vulkan_fallback == PACT_SUCCESS);
    pact_tensor_release(w); pact_tensor_release(z);
    pact_tensor_release(x); pact_tensor_release(y);
    free(a); free(b); free(result);
}

int main(void)
{
    pact_runtime *runtime = NULL;
    assert(pact_runtime_create(&runtime) == PACT_SUCCESS);
    small_cases(runtime);
    deep_graph(runtime);
    parallel_branch(runtime);
    large_case(runtime);
    pact_runtime_destroy(runtime);
    runtime = NULL;
    assert(pact_runtime_create(&runtime) == PACT_SUCCESS);
    float value = 4.0f, out = 0.0f;
    pact_tensor *held = NULL;
    assert(pact_tensor_from_data(runtime, PACT_DTYPE_F32, NULL, 0,
                                 &value, sizeof(value), &held) == PACT_SUCCESS);
    pact_runtime_destroy(runtime);
    assert(pact_tensor_read(held, &out, sizeof(out)) == PACT_SUCCESS);
    assert(out == value);
    pact_tensor_release(held);
    puts("tensor tests passed");
    return 0;
}
