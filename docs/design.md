# PACT design

**Status:** The first float32 addition path is implemented. This document also describes planned
memory management and scheduling capabilities that are not yet available.

## Goal and scope

PACT is a C tensor runtime that automatically uses supported CPUs, GPUs, and eventually other
accelerators. The user expresses tensor operations. PACT chooses where work executes and where the
data needed by that work resides. The public interface should remain useful across one-device,
multi-device, discrete-memory, and unified-memory systems.

The first GPU backend will be Vulkan. A CPU backend must be implemented alongside the first tensor
operation to prove that operation semantics and scheduling are genuinely portable. Other backends,
including native platform backends, can be added without changing the basic tensor API.

PACT does not promise that every device is busy for every operation. The scheduler's objective is to
finish the workload efficiently, including transfer and synchronization costs. Small tasks or serial
dependencies may run best on one worker.

## Layers

1. **Public C API:** Opaque runtime and tensor handles, shape and dtype descriptions, operations,
   host data access, result codes, and diagnostics. Backend objects do not appear in public tensor
   signatures.
2. **Operation DAG:** Defines typed tensor values, operations, versions, and legal dependencies.
   Every tensor operation uses this representation from the first implementation.
3. **Planner and task DAG:** Optimizes reachable work, partitions it, assigns workers and memory
   routes, and lowers it to physical tasks with explicit dependencies.
4. **Memory manager:** Tracks allocations, shard locations, valid replicas, pending writes,
   capacity, eviction, and host access.
5. **Backends:** Discover devices, report capabilities, allocate storage, transfer data, submit
   kernels, and report completion. A backend kind is an enum; each discovered device has a distinct
   runtime identity. Multiple devices can share a backend kind.

The operation definition, shape rules, and numerical contract belong above the backends. Each
backend supplies an implementation for the operations it supports. An unsupported operation or dtype
must produce a clear error rather than silently changing numerical behavior.

The [graph architecture](graph.md) defines construction, planning, task execution, memory hazards,
errors, and observability. It is a core contract, not a later optimization layer.

## Logical memory and tensor storage

The user sees tensors, not individual allocations. A tensor's physical representation may include
multiple shards, one or more valid copies of a shard, and temporary workspace. Internal metadata
tracks which copy contains the latest completed version and which operations may read or write it.

PACT may use RAM and VRAM together as backing capacity. This is an **aggregate logical pool**, not a
single uniformly fast address space. For example, 64 GB of RAM and 16 GB of discrete GPU VRAM offer
up to roughly 80 GB of physical capacity before reservations, duplication, fragmentation, and
temporary storage. An operation may require additional space and still fail even when its input
tensors fit. PACT must report out-of-memory honestly rather than imply guaranteed oversubscription.

On unified-memory systems, CPU and GPU may share the same physical memory. PACT must detect this and
avoid counting it twice. Shared physical memory can remove some copies, but access ordering, cache
visibility, and contention still matter. Memory strategy is a backend capability, not a global
assumption.

Tensor handles cannot promise a permanently stable, host-dereferenceable pointer. Host read or map
requests form explicit observation points: PACT completes required work, makes the requested data
CPU-accessible, and returns it under a defined lifetime. A write mapping must invalidate or update
affected replicas before later device work reads them. The precise mapping API remains to be
designed.

## Transfer workspace

Moving shards between separate memory pools requires transfer space. PACT should maintain bounded,
reusable staging pools in host-visible memory and, where useful, device memory. For paths that
benefit from page-locked host storage, the backend may request it within a measured budget. PACT
should establish a small working reserve when a transfer path is active, then grow or shrink pools
according to demand, available capacity, and observed throughput. It must not reserve a fixed
fraction of every device's memory regardless of use.

Transfers should operate on chunks so a large tensor does not require an equally large staging
allocation. Multiple reusable chunks can let one transfer proceed while another chunk is being
filled or computed, when the backend supports overlap. Staging allocations are temporary workspace,
not extra tensor capacity; the memory manager must account for them and reclaim idle space under
pressure. If staging cannot be obtained, PACT may use a smaller chunk or serialize transfers before
reporting an out-of-memory error.

For each source and destination, the planner should choose among supported paths: shared access
without a copy, direct peer copy, or a staged route through host memory. Copies are necessary for
many discrete-memory paths but not for every device pair. The scheduler must include the chosen
route, its available bandwidth, and any synchronization in its cost estimate.

## Execution and scheduling

Every operation enters the typed semantic DAG. The planner lowers reachable work to a task DAG
containing compute, transfer, allocation, and synchronization tasks. It may split independent
ranges into shards and submit them to different workers. A host read or explicit wait materializes
the required result and reports any dependent asynchronous error.

The scheduler compares candidate plans using:

```text
completion cost = queue delay + required transfer + compute + output movement/synchronization
```

The planner must account for live tensors, transfer workspace, and intermediate results before
submitting a plan. It should avoid moving a tensor repeatedly between devices for consecutive
operations. Profiling and transfer tracing are necessary to tune the cost model and explain its
decisions, including transfer workspace use and stall time.

Automatic scheduling is only possible for operations with an implementation on the selected worker.
Built-in tensor operations will get CPU and Vulkan implementations. Portable user-defined kernels
are a later design problem; arbitrary C functions cannot directly execute as Vulkan shaders.
Backend-specific kernels may be supported before a portable kernel language exists.

## Backends and hardware

Device discovery distinguishes backend kind from device instance. Two Vulkan GPUs are two devices;
one machine may also have one or more CPU execution domains. CPU partitioning should be based on
actual topology and measured usefulness rather than assuming one logical CPU thread equals a device.

The Vulkan backend will discover compute-capable physical devices, create execution queues, manage
buffers, and submit compute shaders. Cross-device copies may use a supported direct path or staging
through host memory. Those routes have different costs and must be represented in the scheduler.
Vulkan device groups are an optional capability, not the foundation of PACT's multi-device model.

PACT builds one hardware-capable library. Vulkan is loaded and discovered at runtime; its absence
leaves CPU execution available in the same binary. Backend availability is a runtime capability,
not a separate application build.

On Apple platforms, Vulkan can initially run through a portability implementation such as MoltenVK.
A native Metal backend may later provide better access to platform capabilities. Neither path
changes the public tensor model.

## Correctness rules

- A task cannot read a shard until the latest relevant write and any required transfer have
  completed.
- Only one writable version of a shard may be authoritative at a time. Read-only replicas may
  coexist.
- Host access must have defined synchronization and lifetime rules.
- A failed transfer or kernel must propagate to dependent operations and to the next observable API
  boundary.
- Resource destruction must not release allocations still used by queued work.
- A staging chunk cannot be reused until the transfer that references it has completed.
- Every submitted task must be reachable from a validated, typed operation and have declared
  access to versioned inputs and outputs.
- A plan must respect peak live memory and preserve the operation's numerical contract.
- Data movement and fallback behavior may be automatic, but should be visible through diagnostics.

## Implementation sequence and quality bar

1. Establish a portable CMake build, backend isolation, API and ABI conventions, and device
   capability discovery for CPU and Vulkan.
2. Implement the typed operation DAG, value versioning, validation, planner interface, task DAG,
   completion events, error propagation, tracing identities, and resource ownership rules.
3. Implement backend allocation, transfer routes, bounded staging pools, synchronization, and
   residency tracking with explicit memory budgets.
4. Implement the first operation on CPU and Vulkan through the complete graph path. Verify shape,
   dtype, numerical, lifetime, failure, and concurrent execution behavior.
5. Add heterogeneous partitioning and multiple devices. Measure CPU-only, GPU-only, and mixed plans
   to validate placement decisions and refine the cost model.
6. Extend operations, optimization passes, backends, diagnostics, and performance regression tests
   while preserving the graph and API contracts.

Each milestone must use production interfaces and invariants, even when few operations are
implemented. The first end-to-end proof creates tensors without specifying placement, runs one
operation across CPU and GPU where beneficial, and reads the correct result. The float32 addition
path now demonstrates this proof with CPU and multiple Vulkan GPU shards.
The current scheduler combines minimum work thresholds with observed CPU and Vulkan throughput.
It pilots GPU use, can run CPU and GPU shards concurrently, and backs off mixed placement when it
loses to a measured CPU baseline. CPU workers and bounded per-device Vulkan transfer workspaces
persist across reads. Persistent tensor residency, topology-aware transfer routes, memory-pressure
aware placement, and aggregate memory remain future milestones.

## Open design decisions

- Exact C function names, ownership conventions, and ABI versioning.
- Tensor views, strides, broadcasting, and mutation rules.
- Initial dtype and operation set, including numerical tolerance across backends.
- Memory budgets, eviction policy, and behavior under host or device memory pressure.
- Transfer pool sizing, page locking, and backpressure under concurrent transfers.
- Whether and how portable custom kernels are represented.
