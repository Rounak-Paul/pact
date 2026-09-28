# Tensors and host access API

**Status:** The float32 creation and read functions below are implemented.

Include `<pact/pact.h>`. Tensor handles are opaque. A tensor owns a reference to its runtime, so
its handle remains valid after `pact_runtime_destroy` until the tensor is released. These calls
are not thread-safe when used concurrently on one runtime.

```c
pact_result pact_tensor_from_data(
    pact_runtime *runtime, pact_dtype dtype, const uint64_t *shape, uint32_t rank,
    const void *data, size_t data_bytes, pact_tensor **out_tensor);
void pact_tensor_retain(pact_tensor *tensor);
void pact_tensor_release(pact_tensor *tensor);
pact_result pact_tensor_get_rank(const pact_tensor *tensor, uint32_t *out_rank);
pact_result pact_tensor_get_shape(
    const pact_tensor *tensor, uint64_t *out_shape, uint32_t shape_capacity);
pact_result pact_tensor_get_size_bytes(const pact_tensor *tensor, size_t *out_bytes);
pact_result pact_tensor_read(pact_tensor *tensor, void *out_data, size_t out_bytes);
```

`pact_tensor_from_data` currently accepts only `PACT_DTYPE_F32`. Its shape is dense and row-major.
Rank zero is one scalar. A shape containing zero has zero elements. `data_bytes` must equal the
product of dimensions times `sizeof(float)`. The input bytes are copied before the call returns;
the caller may then reuse its buffer. On success, the caller owns one tensor reference. On failure,
`*out_tensor` is null. `pact_tensor_release(NULL)` is harmless.

`pact_tensor_read` materializes any reachable operation graph, waits for CPU and GPU shards, and
copies exactly the tensor's byte count into the caller's buffer. A repeated read copies the cached
host result. The output buffer remains caller-owned. There is no permanent host pointer or mapping
API. A zero-byte tensor can be read with a null output pointer.

Creation validates dtype, size, arguments, and overflow. Read reports a materialization failure;
that failure stays on the tensor handle. A later addition using a failed input reports the failure.

Physical memory placement is automatic. At present, original inputs and materialized outputs live
in host memory. Vulkan shards evaluate intermediate additions per invocation and retain
bounded device and staging workspace for reuse during the runtime's lifetime. Tensor values are
not retained in GPU memory across reads. Long-lived GPU residency, eviction, and aggregate RAM/VRAM
capacity remain future work.
