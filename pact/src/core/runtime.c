#include "runtime_internal.h"
#include "cpu_pool.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static int pact_allocation_size(uint32_t count, size_t item_size, size_t *out_size)
{
    size_t item_count = (size_t)count;
    if (item_size != 0 && item_count > SIZE_MAX / item_size) {
        return 0;
    }
    *out_size = item_count * item_size;
    return 1;
}

pact_result pact_runtime_add_device(
    pact_runtime *runtime, const pact_device_info *info, void *native_device)
{
    if (runtime == NULL || info == NULL) {
        return PACT_ERROR_INVALID_ARGUMENT;
    }
    if (runtime->device_count == UINT32_MAX) {
        return PACT_ERROR_OUT_OF_MEMORY;
    }

    if (runtime->device_count == runtime->device_capacity) {
        uint32_t new_capacity = runtime->device_capacity == 0 ? 4 : runtime->device_capacity * 2;
        size_t allocation_size = 0;
        if (new_capacity < runtime->device_capacity ||
            !pact_allocation_size(
                new_capacity, sizeof(*runtime->devices), &allocation_size)) {
            return PACT_ERROR_OUT_OF_MEMORY;
        }
        pact_device_record *new_devices = (pact_device_record *)realloc(
            runtime->devices, allocation_size);
        if (new_devices == NULL) {
            return PACT_ERROR_OUT_OF_MEMORY;
        }
        runtime->devices = new_devices;
        runtime->device_capacity = new_capacity;
    }

    pact_device_record *record = &runtime->devices[runtime->device_count++];
    record->info = *info;
    record->native_device = native_device;
    record->execution_context = NULL;
    record->execution_status = PACT_SUCCESS;
    return PACT_SUCCESS;
}

pact_result pact_runtime_create(pact_runtime **out_runtime)
{
    if (out_runtime == NULL) {
        return PACT_ERROR_INVALID_ARGUMENT;
    }
    *out_runtime = NULL;

    pact_runtime *runtime = (pact_runtime *)calloc(1, sizeof(*runtime));
    if (runtime == NULL) {
        return PACT_ERROR_OUT_OF_MEMORY;
    }
    runtime->ref_count = 1;
    runtime->next_value_id = 1;
    runtime->vulkan_status = PACT_ERROR_BACKEND_UNAVAILABLE;
    runtime->last_execution.struct_size = sizeof(pact_execution_info);

    pact_result result = pact_cpu_discover(runtime);
    if (result != PACT_SUCCESS) {
        pact_runtime_destroy(runtime);
        return result;
    }

    uint32_t cpu_device_count = runtime->device_count;
    runtime->vulkan_status = pact_vulkan_discover(runtime);
    if (runtime->vulkan_status == PACT_ERROR_OUT_OF_MEMORY) {
        pact_runtime_destroy(runtime);
        return PACT_ERROR_OUT_OF_MEMORY;
    }
    if (runtime->vulkan_status != PACT_SUCCESS) {
        runtime->device_count = cpu_device_count;
        pact_vulkan_release(runtime);
    }

    *out_runtime = runtime;
    return PACT_SUCCESS;
}

void pact_runtime_retain_internal(pact_runtime *runtime)
{
    ++runtime->ref_count;
}

void pact_runtime_release_internal(pact_runtime *runtime)
{
    if (runtime == NULL) {
        return;
    }
    if (--runtime->ref_count != 0) {
        return;
    }
    pact_cpu_pool_destroy(runtime->cpu_pool);
    for (uint32_t i = 0; i < runtime->device_count; ++i)
        pact_vulkan_context_destroy(runtime->devices[i].execution_context);
    pact_vulkan_release(runtime);
    free(runtime->devices);
    free(runtime);
}

void pact_runtime_destroy(pact_runtime *runtime)
{
    pact_runtime_release_internal(runtime);
}

pact_result pact_runtime_get_device_count(
    const pact_runtime *runtime, uint32_t *out_count)
{
    if (runtime == NULL || out_count == NULL) {
        return PACT_ERROR_INVALID_ARGUMENT;
    }
    *out_count = runtime->device_count;
    return PACT_SUCCESS;
}

pact_result pact_runtime_get_device_info(
    const pact_runtime *runtime, uint32_t index, pact_device_info *out_info)
{
    if (runtime == NULL || out_info == NULL || index >= runtime->device_count) {
        return PACT_ERROR_INVALID_ARGUMENT;
    }

    uint32_t supplied_size = out_info->struct_size;
    if (supplied_size < sizeof(pact_device_info)) {
        return PACT_ERROR_INVALID_ARGUMENT;
    }

    size_t copy_size = supplied_size < sizeof(*out_info) ? supplied_size : sizeof(*out_info);
    memcpy(out_info, &runtime->devices[index].info, copy_size);
    return PACT_SUCCESS;
}

pact_result pact_runtime_get_backend_status(
    const pact_runtime *runtime, pact_backend backend, pact_result *out_status)
{
    if (runtime == NULL || out_status == NULL) {
        return PACT_ERROR_INVALID_ARGUMENT;
    }
    switch (backend) {
    case PACT_BACKEND_CPU:
        *out_status = PACT_SUCCESS;
        return PACT_SUCCESS;
    case PACT_BACKEND_VULKAN:
        *out_status = runtime->vulkan_status;
        return PACT_SUCCESS;
    default:
        return PACT_ERROR_INVALID_ARGUMENT;
    }
}

pact_result pact_runtime_get_last_execution(
    const pact_runtime *runtime, pact_execution_info *out_info)
{
    if (runtime == NULL || out_info == NULL ||
        out_info->struct_size < sizeof(pact_execution_info))
        return PACT_ERROR_INVALID_ARGUMENT;
    *out_info = runtime->last_execution;
    return PACT_SUCCESS;
}

const char *pact_result_string(pact_result result)
{
    switch (result) {
    case PACT_SUCCESS:
        return "success";
    case PACT_ERROR_INVALID_ARGUMENT:
        return "invalid argument";
    case PACT_ERROR_OUT_OF_MEMORY:
        return "out of memory";
    case PACT_ERROR_BACKEND_UNAVAILABLE:
        return "backend unavailable";
    case PACT_ERROR_INTERNAL:
        return "internal error";
    case PACT_ERROR_SHAPE_MISMATCH:
        return "shape mismatch";
    case PACT_ERROR_UNSUPPORTED_DTYPE:
        return "unsupported dtype";
    case PACT_ERROR_BUFFER_SIZE:
        return "buffer size mismatch";
    case PACT_ERROR_EXECUTION:
        return "execution failed";
    default:
        return "unknown result";
    }
}
