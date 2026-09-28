#ifndef PACT_RUNTIME_INTERNAL_H
#define PACT_RUNTIME_INTERNAL_H

#include <pact/pact.h>
#include <stddef.h>

typedef struct pact_performance_sample {
    uint32_t additions;
    double microseconds_per_element;
    size_t sample_elements;
    int valid;
    uint32_t observations;
} pact_performance_sample;

typedef struct pact_device_record {
    pact_device_info info;
    void *native_device;
    void *execution_context;
    pact_result execution_status;
    pact_performance_sample performance;
} pact_device_record;

struct pact_runtime {
    uint32_t ref_count;
    pact_device_record *devices;
    uint32_t device_count;
    uint32_t device_capacity;
    pact_result vulkan_status;
    void *vulkan_state;
    uint64_t next_value_id;
    uint64_t visit_epoch;
    pact_execution_info last_execution;
    void *cpu_pool;
    pact_performance_sample cpu_performance;
    uint32_t mixed_additions;
    size_t mixed_elements;
    uint32_t mixed_losses;
};

pact_result pact_runtime_add_device(
    pact_runtime *runtime, const pact_device_info *info, void *native_device);

pact_result pact_cpu_discover(pact_runtime *runtime);

void pact_runtime_retain_internal(pact_runtime *runtime);
void pact_runtime_release_internal(pact_runtime *runtime);

pact_result pact_vulkan_discover(pact_runtime *runtime);
void pact_vulkan_release(pact_runtime *runtime);
void pact_vulkan_context_destroy(void *context);

#endif
