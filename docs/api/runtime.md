# Runtime and hardware API

**Status:** Device discovery and basic last-execution counters are implemented in
[pact.h](../../include/pact/pact.h). Capability queries and full tracing remain planned.

## Create and destroy a runtime

```c
pact_result pact_runtime_create(pact_runtime **out_runtime);
void pact_runtime_destroy(pact_runtime *runtime);
```

`pact_runtime_create` allocates a runtime and discovers devices. It sets `*out_runtime` to `NULL`
before starting. Passing a null output pointer returns `PACT_ERROR_INVALID_ARGUMENT`. A failure
to allocate required runtime state returns `PACT_ERROR_OUT_OF_MEMORY`. The native CPU device is
always registered. Failure to initialize the optional Vulkan backend does not prevent CPU
discovery; inspect its status separately. Destroy accepts `NULL`.

The runtime owns its discovery records and, when Vulkan is available, its Vulkan loader and
instance. Tensor handles keep an internal runtime reference, so caller-owned runtime references
may be destroyed before those tensors are released. Concurrent runtime calls are not supported yet.

## Enumerate devices

```c
pact_result pact_runtime_get_device_count(
    const pact_runtime *runtime, uint32_t *out_count);
pact_result pact_runtime_get_device_info(
    const pact_runtime *runtime, uint32_t index, pact_device_info *out_info);
```

Set `out_info->struct_size = sizeof(*out_info)` before calling the information function. The
function copies the device record into caller-owned storage. A smaller size is invalid; a larger
size is accepted so future versions can append fields. Indices range from zero to `count - 1`
and are stable only during one runtime's lifetime.

`pact_device_info` currently includes backend kind, device type, a null-terminated name, logical
processor count, vendor and device IDs, and backend API version. CPU discovery reports one native
`PACT_BACKEND_CPU` device and its online logical processor count. Vulkan discovery reports each
compute-capable, non-CPU Vulkan physical device separately. A software Vulkan CPU is excluded
because it would duplicate the native CPU. Vulkan vendor and device IDs and API version are raw
Vulkan property values; the corresponding CPU fields are zero.

Backend kind and device type are different concepts. Two Vulkan GPUs have the same backend kind
and separate indices. Device names and indices are diagnostic values, not persistent identifiers.

## Backend status and errors

```c
pact_result pact_runtime_get_backend_status(
    const pact_runtime *runtime, pact_backend backend, pact_result *out_status);
const char *pact_result_string(pact_result result);
```

The call itself returns `PACT_SUCCESS` for a recognized backend and writes its discovery outcome
to `out_status`. CPU status is `PACT_SUCCESS`. Vulkan status is `PACT_SUCCESS` when at least one
usable Vulkan compute device was found, `PACT_ERROR_BACKEND_UNAVAILABLE` when the loader or a
usable device was absent, or another result when discovery failed. Unknown backend enum values
return `PACT_ERROR_INVALID_ARGUMENT`.

## Example

```c
#include <pact/pact.h>
#include <stdio.h>

int main(void) {
    pact_runtime *runtime = NULL;
    if (pact_runtime_create(&runtime) != PACT_SUCCESS) return 1;

    uint32_t count = 0;
    if (pact_runtime_get_device_count(runtime, &count) != PACT_SUCCESS) return 1;
    for (uint32_t i = 0; i < count; ++i) {
        pact_device_info info = {0};
        info.struct_size = sizeof(info);
        if (pact_runtime_get_device_info(runtime, i, &info) != PACT_SUCCESS) return 1;
        printf("%u: %s\n", i, info.name);
    }
    pact_runtime_destroy(runtime);
    return 0;
}
```

## Planned runtime capabilities

Device selection should not be required for normal tensor programs. Future revisions will report
operation support and execution capabilities.

## Device identity and capabilities

Future device information should include supported dtypes and operations, relevant memory
properties, and execution capabilities. Persistent identification, if offered, requires a separate
field from the runtime-local index.

The implemented backend kinds are CPU and Vulkan. Native backends can be added later without
reinterpreting existing enum values.

## Planned execution and diagnostics

Normal operations may complete asynchronously. The runtime should offer an explicit wait and a way
to inspect failures. Tracing should reveal worker choice, task timing, transfers, memory use, and
reasons for fallback. Transfer diagnostics should include route, bytes moved, staging pool use, and
time spent waiting for transfer space. These are observation tools; they should not force every
application to configure scheduling manually.

Backend availability is separate from physical hardware availability. A GPU that lacks a usable
driver or a required feature may be reported as unavailable rather than treated as a working
execution device.

## Unresolved API details

- How configuration and diagnostics are represented in an ABI-stable C interface.
- How asynchronous failures are reported and associated with tensor operations.

## Last execution diagnostics

```c
pact_result pact_runtime_get_last_execution(
    const pact_runtime *runtime, pact_execution_info *out_info);
```

Set `out_info->struct_size = sizeof(pact_execution_info)` before calling. The result reports CPU
and Vulkan shard counts and element counts for the last tensor materialization on that runtime.
`vulkan_fallback` is the most recent Vulkan failure that caused a CPU retry, or `PACT_SUCCESS`.
`cpu_tasks` reports CPU work items, and `cpu_time_us` is the elapsed CPU shard time. The Vulkan
timings report the maximum across participating devices: `vulkan_active_us` covers chunk execution
after workspace setup, while `vulkan_setup_us`, `vulkan_upload_us`, `vulkan_record_us`,
`vulkan_queue_us`, and `vulkan_download_us` break down the main phases. These are host wall-clock
measurements, not device hardware timestamps. Values are zero when the corresponding backend did
no work. The structure is part of the pre-1.0 API and has grown in this revision.
The runtime must still be owned by the caller when reading these diagnostics. Concurrent calls on
one runtime are not yet supported.
