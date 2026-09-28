# PACT

**Portable Accelerated Compute for Tensors**

PACT is a C library being built for tensor computation across the hardware available to a process:
CPUs, GPUs, and future platform-specific accelerators. Its goal is a small, approachable API in
which applications describe tensor operations while PACT manages execution, memory placement,
transfers, and synchronization.

```c
#include <pact/pact.h>

/* Error checks and handle releases omitted here for brevity. */
pact_runtime *runtime = NULL;
pact_runtime_create(&runtime);
uint64_t shape[] = {4};
float input[] = {1, 2, 3, 4};
pact_tensor *a = NULL;
pact_tensor *sum = NULL;
pact_tensor_from_data(runtime, PACT_DTYPE_F32, shape, 1,
                      input, sizeof(input), &a);
pact_tensor_add(a, a, &sum);
float output[4];
pact_tensor_read(sum, output, sizeof(output));
pact_tensor_release(sum);
pact_tensor_release(a);
pact_runtime_destroy(runtime);
```

The same program should work on a CPU-only machine, one GPU, or a system with several CPUs and GPUs.
PACT should discover supported devices, choose useful workers for each operation, and keep tensor
data near the work that uses it. Users should not have to assign tensors to RAM or VRAM.

## Design goals

- **One logical tensor space.** A tensor may be split or replicated across physical memory pools.
  The user works with a tensor handle rather than a permanent pointer to its storage.
- **Automatic heterogeneous execution.** CPU and GPU implementations of an operation can run
  concurrently when splitting the work improves completion time.
- **Locality-aware scheduling.** Decisions account for transfer time, queued work, compute speed,
  available memory, and dependencies. An idle device is not automatically the fastest choice.
- **Reusable transfer space.** PACT manages staging memory so independent transfers and compute
  can overlap when the hardware allows it.
- **Portable C interface.** Backend details stay behind opaque handles. A later Python binding can
  wrap the C API without changing the runtime model.
- **Incremental backends.** Vulkan is the first GPU backend. A CPU backend is essential to the
  design; native backends such as Metal may follow.
- **Observable behavior.** Diagnostics should make device use, transfers, memory pressure, and
  scheduling decisions understandable.

The logical memory pool can draw on system RAM and GPU memory, but their advertised capacities are
not a promise that one contiguous allocation of their sum will fit. The operating system, drivers,
and temporary operation and transfer storage also consume memory. On unified-memory machines, CPU
and GPU memory may be the same physical pool and must not be counted twice.

## Status

PACT now implements float32 tensor creation, lazy addition, and host read through a semantic DAG and
physical CPU/Vulkan task plan. The same build works without a Vulkan loader. Addition chains are
fused for CPU and Vulkan execution. Larger CPU shards use a persistent worker pool. Each GPU has a
persistent context and bounded, reusable compute and transfer workspace; large shards stream
through that workspace in chunks. The scheduler measures CPU and GPU throughput for each runtime
and can assign CPU and multiple Vulkan GPUs concurrently. Placement is heuristic and can vary by
read. Persistent tensor residency across reads, a logical aggregate RAM/VRAM allocator, and a
broad operation set remain future work.

## Build

With a C11 compiler, CMake 3.21 or newer, and Vulkan headers:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

This is one hardware-capable build. At runtime PACT tries to load the platform's Vulkan loader and
discover compute devices. If the loader or a usable GPU is absent, the same library continues with
the CPU. Applications do not need Vulkan headers or a Vulkan library to use PACT's public C API.
See the [runtime API](docs/api/runtime.md) for the implemented interface.

## Documentation

- [Design and implementation plan](docs/design.md)
- [Graph and execution architecture](docs/graph.md)
- [API documentation](docs/api/README.md)
- [Development guidance](AGENTS.md)
- [Addition benchmark before refactor](benchmarks/results-2026-09-28.md)
- [Addition benchmark after refactor](benchmarks/results-2026-09-29-refactor.md)

## License

PACT is licensed under the [Apache License 2.0](LICENSE). Third-party components keep their own
licenses.
