#ifndef PACT_PACT_H
#define PACT_PACT_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(PACT_SHARED)
#  if defined(PACT_BUILDING_LIBRARY)
#    define PACT_API __declspec(dllexport)
#  else
#    define PACT_API __declspec(dllimport)
#  endif
#elif defined(PACT_SHARED) && defined(__GNUC__)
#  define PACT_API __attribute__((visibility("default")))
#else
#  define PACT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PACT_VERSION_MAJOR 0
#define PACT_VERSION_MINOR 1
#define PACT_VERSION_PATCH 0
#define PACT_DEVICE_NAME_CAPACITY 256

typedef struct pact_runtime pact_runtime;
typedef struct pact_tensor pact_tensor;

typedef enum pact_result {
    PACT_SUCCESS = 0,
    PACT_ERROR_INVALID_ARGUMENT = 1,
    PACT_ERROR_OUT_OF_MEMORY = 2,
    PACT_ERROR_BACKEND_UNAVAILABLE = 3,
    PACT_ERROR_INTERNAL = 4,
    PACT_ERROR_SHAPE_MISMATCH = 5,
    PACT_ERROR_UNSUPPORTED_DTYPE = 6,
    PACT_ERROR_BUFFER_SIZE = 7,
    PACT_ERROR_EXECUTION = 8
} pact_result;

typedef enum pact_dtype {
    PACT_DTYPE_F32 = 1
} pact_dtype;

typedef enum pact_backend {
    PACT_BACKEND_CPU = 1,
    PACT_BACKEND_VULKAN = 2
} pact_backend;

typedef enum pact_device_type {
    PACT_DEVICE_TYPE_CPU = 1,
    PACT_DEVICE_TYPE_GPU = 2,
    PACT_DEVICE_TYPE_OTHER = 3
} pact_device_type;

typedef struct pact_execution_info {
    uint32_t struct_size;
    uint32_t cpu_shards;
    uint32_t vulkan_shards;
    uint64_t cpu_elements;
    uint64_t vulkan_elements;
    pact_result vulkan_fallback;
    uint32_t cpu_tasks;
    double cpu_time_us;
    double vulkan_active_us;
    double vulkan_setup_us;
    double vulkan_upload_us;
    double vulkan_record_us;
    double vulkan_queue_us;
    double vulkan_download_us;
} pact_execution_info;

typedef struct pact_device_info {
    /* Set to sizeof(pact_device_info) before calling pact_runtime_get_device_info. */
    uint32_t struct_size;
    pact_backend backend;
    pact_device_type type;
    char name[PACT_DEVICE_NAME_CAPACITY];
    uint32_t logical_processor_count;
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t backend_api_version;
} pact_device_info;

/* Creates a runtime and discovers supported compute devices. The CPU is always available. */
PACT_API pact_result pact_runtime_create(pact_runtime **out_runtime);

/* Accepts NULL. Previously returned device-info copies remain valid. */
PACT_API void pact_runtime_destroy(pact_runtime *runtime);

PACT_API pact_result pact_runtime_get_device_count(
    const pact_runtime *runtime, uint32_t *out_count);

/* Device indices are runtime-local and are not stable across processes or runs. */
PACT_API pact_result pact_runtime_get_device_info(
    const pact_runtime *runtime, uint32_t index, pact_device_info *out_info);

/* Reports the backend discovery outcome; Vulkan may be unavailable or fail independently. */
PACT_API pact_result pact_runtime_get_backend_status(
    const pact_runtime *runtime, pact_backend backend, pact_result *out_status);

/* Reports the last materialization on this runtime; calls are not thread-safe. */
PACT_API pact_result pact_runtime_get_last_execution(
    const pact_runtime *runtime, pact_execution_info *out_info);

PACT_API const char *pact_result_string(pact_result result);

/* Input bytes are copied before this function returns. Rank zero denotes one scalar. */
PACT_API pact_result pact_tensor_from_data(
    pact_runtime *runtime, pact_dtype dtype, const uint64_t *shape, uint32_t rank,
    const void *data, size_t data_bytes, pact_tensor **out_tensor);

/* This operation validates its inputs and adds a node to the semantic DAG. */
PACT_API pact_result pact_tensor_add(
    pact_tensor *a, pact_tensor *b, pact_tensor **out_tensor);

/* Materializes the value and copies exactly tensor_size_bytes into out_data. */
PACT_API pact_result pact_tensor_read(
    pact_tensor *tensor, void *out_data, size_t out_bytes);

PACT_API pact_result pact_tensor_get_rank(
    const pact_tensor *tensor, uint32_t *out_rank);
PACT_API pact_result pact_tensor_get_shape(
    const pact_tensor *tensor, uint64_t *out_shape, uint32_t shape_capacity);
PACT_API pact_result pact_tensor_get_size_bytes(
    const pact_tensor *tensor, size_t *out_bytes);
PACT_API void pact_tensor_retain(pact_tensor *tensor);
PACT_API void pact_tensor_release(pact_tensor *tensor);

#ifdef __cplusplus
}
#endif

#endif
