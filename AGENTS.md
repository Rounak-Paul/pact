# PACT development guidance

This file applies to the repository root and its children. It records project decisions for Codex,
Claude, and other development agents. Read it before changing PACT. User instructions in the current
session take precedence over this file.

## Source of truth

Read [README.md](README.md), [docs/design.md](docs/design.md), and [docs/graph.md](docs/graph.md)
before architectural work. Read the relevant pages under [docs/api](docs/api/README.md) before
changing the public API. The implemented public headers and tests become the authority for actual
symbols and behavior once they exist. Keep the documents consistent with implementation changes.

The repository has a runtime and CPU/Vulkan device discovery API. Float32 tensor creation,
lazy addition, graph planning, mixed CPU/Vulkan execution, and host read are implemented. CPU worker
threads and per-GPU Vulkan contexts and bounded transfer workspaces persist for a runtime. The
scheduler calibrates observed throughput and can back off mixed placement. Persistent tensor
residency, aggregate memory allocation, and additional operations remain designs. Keep implemented
and planned behavior distinct.

## Product contract

- PACT is a portable tensor runtime with a public C API. A later pybind11 layer should wrap the C
  API and preserve its execution and ownership semantics.
- The normal user experience is tensor creation, operations, and result access. Users should not
  have to choose a device or place tensors in RAM or VRAM.
- PACT discovers supported CPUs and accelerators, then chooses workers based on expected completion
  time, data locality, queue load, memory pressure, transfer cost, and synchronization.
- A backend kind is an enum; a discovered device has a separate identity. Multiple CPUs or GPUs
  may use the same backend kind. Vulkan is the first GPU backend; CPU execution is a core backend.
- Build one hardware-capable library. Load Vulkan at runtime so the same binary runs on CPU-only
  systems without requiring the Vulkan loader to be installed.
- System RAM and discrete GPU VRAM may back one logical tensor space. Their advertised sizes are
  not guaranteed allocatable capacity. Unified physical memory must not be counted twice.
- The scheduler may use CPU and GPU concurrently when that helps. Do not promise 100% utilization
  of every device or force every device to participate in every operation.
- PACT manages bounded, reusable transfer workspace. Transfers, staging, and temporary operation
  storage count against memory budgets. Use shared access or direct peer routes when supported.
- Tensor handles do not expose a permanent host pointer. Host read or map operations must define
  synchronization, ownership, pointer lifetime, and error behavior.

## Architecture requirements

- Every tensor operation enters a typed semantic operation DAG from the first implementation.
  Lower reachable work to a separate physical task DAG. Do not add a temporary direct execution
  path that bypasses graph validation, versioning, residency, or error propagation.
- Keep operation semantics and the public API independent of CPU, Vulkan, and later backends.
  Backend implementations report capabilities and provide kernels, memory, transfers, and queues.
- Track value versions, shard regions, replicas, read/write hazards, and task completion. Never
  read stale data or reuse staging memory while a backend still references it.
- Plan peak live memory, temporary workspace, and transfer space before submitting work. Account
  for failures and resource lifetimes across asynchronous operations.
- An optimization must preserve documented shape, dtype, aliasing, and numerical behavior. Keep
  transformations testable and traceable back to public operations.
- Ordinary C functions are not automatically portable GPU kernels. Portable custom kernels need
  an explicit language or intermediate representation designed for supported backends.
- Keep Vulkan objects and platform-specific types out of the public tensor API.

## Implementation and verification

- Build in complete, reviewable milestones using production interfaces. The first tensor operation
  must pass through the real graph, planner, task, residency, completion, and error paths.
- Verify CPU-only, Vulkan-only, and mixed execution where hardware is available. Provide CPU-only
  verification paths for development environments without a Vulkan device.
- Test semantics and failure cases: shapes, dtypes, numerical results, ownership, lifetimes,
  synchronization, transfers, and memory pressure. Add performance measurements when changing the
  scheduler; a faster kernel alone does not prove a faster workload.
- Treat backend availability and hardware availability separately. Report unsupported features
  clearly instead of silently changing operation semantics.
- Preserve the C ABI deliberately. Document public ownership, thread safety, error reporting, and
  versioning before expanding the surface.
- Do not edit the vendored VMA submodule as part of ordinary PACT changes. Its implementation is
  C++ even though it exposes a C-compatible interface; decide and document that build tradeoff
  before integrating it into the C library.

## Documentation conventions

- Wrap Markdown prose at 100 characters or fewer per line. Keep lists, links, tables, and code
  blocks valid.
- Update README and design pages when behavior or architecture changes. Update API pages with the
  exact exported signatures, ownership rules, errors, and compilable examples once headers exist.
- Distinguish planned behavior from implemented behavior. Record important design decisions and
  unresolved tradeoffs in the docs rather than relying on a chat transcript.
