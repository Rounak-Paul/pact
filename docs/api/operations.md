# Operations and execution API

**Status:** Float32 elementwise addition is implemented.

```c
pact_result pact_tensor_add(
    pact_tensor *a, pact_tensor *b, pact_tensor **out_tensor);
```

Both inputs must belong to the same runtime, have dtype `PACT_DTYPE_F32`, and have exactly the same
rank and shape. Broadcasting is not implemented. On success, the caller owns one output reference.
The output retains its inputs, so callers may release the input handles before reading the output.
Addition is lazy: it appends a typed value to the semantic DAG and does not compute immediately.

A read collects the reachable DAG in topological order and lowers it to CPU and, for sufficiently
large work, Vulkan task shards. Source, addition, and download tasks have explicit dependencies.
Independent CPU and GPU shards run concurrently. Linear addition chains are fused on CPU and GPU;
the Vulkan path uploads source chunks, evaluates additions per invocation, and downloads
result chunks. Large CPU shards use a worker pool. The scheduler combines size thresholds with
measured throughput, a GPU pilot, and a backoff after slow mixed runs. Each GPU retains a bounded
workspace across reads. Placement is heuristic, not a performance guarantee. A GPU submission or
completion failure can be retried on the CPU;
`pact_runtime_get_last_execution` records the fallback result.

The numerical contract is elementwise IEEE 754 float32 addition under the selected backend's
floating-point behavior. CPU and Vulkan results can differ at the last bits for exceptional values
or platform-specific floating-point modes. The current tests use exactly representable values.
There are no mutation, broadcasting, reduction, or custom-kernel APIs yet.
