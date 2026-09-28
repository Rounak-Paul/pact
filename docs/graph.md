# Graph and execution architecture

**Status:** The float32 addition path uses a semantic graph and CPU/Vulkan task plan. The more
advanced memory, optimization, and tracing rules below remain architectural requirements.

## Two graphs with different responsibilities

PACT uses a semantic operation DAG and a physical task DAG. Every tensor operation enters the
semantic DAG. The planner lowers a reachable portion of it into a task DAG for execution. Keeping
these representations separate lets PACT change placement and scheduling without changing the
meaning of the user's program.

| Representation | Nodes | Edges | Purpose |
| --- | --- | --- | --- |
| Operation DAG | Typed ops and values | Value versions | Defines results |
| Task DAG | Compute and transfer tasks | Readiness and hazards | Executes a plan |

The operation DAG is typed and backend-independent. The task DAG names concrete devices, memory
pools, queues, ranges, and completion events. A task plan may be rebuilt when capacity, device
availability, or measured performance changes, provided no completed result changes meaning.

## Tensor values and operation nodes

A tensor handle refers to a logical value with dtype, shape, strides or layout, and a producing
node. Each operation records its schema, input values, attributes, output values, and source
location or trace identity. Schema validation checks rank, shape, dtype, broadcasting, and overflow
before execution is submitted. Immutable value versions make data dependencies explicit.

Mutation, views, and aliases require explicit rules. A write creates a new version of the affected
logical value or a versioned side-effect edge; it must not silently alter a value still needed by
another node. A view records its relationship to backing storage so overlapping reads and writes
can be ordered correctly. Host writes enter the graph as operations with the same hazard rules.

The graph is acyclic. Construction rejects invalid dependencies. Resource lifetimes are tied to
reachable values and outstanding tasks, not merely to the lifetime of a C handle.

## Construction and materialization

Public tensor operations append validated nodes and return handles without waiting for completion.
PACT may submit ready work before a host read, but a read, explicit wait, or export must materialize
the required reachable outputs. Submission should work in bounded graph windows so long-running
programs do not retain an unbounded history.

The runtime retains unfinished nodes, reachable values, and failures until their public results are
observed or released. Completed subgraphs can be pruned after their needed outputs and diagnostics
are retained. Repeated use of a tensor handle must refer to the same logical value version.

## Planning and optimization

Planning proceeds through explicit stages:

1. Validate and normalize operation schemas, shapes, layouts, and capabilities.
2. Find the subgraph reachable from requested outputs and remove dead work.
3. Apply only semantics-preserving transformations, such as safe fusion or constant folding.
4. Partition operations into shards and determine required communication, including reductions.
5. Choose workers and memory routes using a cost model and current capacity information.
6. Plan allocation lifetimes, reusable workspace, transfers, and queue dependencies.
7. Lower the plan to backend commands and retain the mapping for tracing and error reporting.

An optimization pass must state its preconditions and preserve the operation's numerical contract.
Fusion cannot change observable writes or read a stale replica. A plan must account for temporary
workspace and peak live memory, not just input and output sizes. If a candidate plan does not fit,
the planner can adjust shard sizes, serialize tasks, evict eligible replicas, or choose another
worker before reporting failure.

The first cost model may use conservative estimates, but it belongs in the planner from the start.
It should compare queue delay, compute time, transfer time, synchronization, and memory pressure.
Measurements can refine those estimates without changing graph semantics.

## Task dependencies and residency

Each compute task declares the value version and region it reads or writes. A transfer task names a
source replica, destination pool, byte range, and route. Allocation and staging tasks reserve
workspace before dependent work starts. Completion events connect tasks across queues and devices.
The task DAG must not submit a consumer until its input version is ready in an accessible location.

The memory manager tracks residency and validity per shard. Read-only replicas may coexist. A write
creates a new authoritative version and invalidates or supersedes older replicas. A staging buffer
cannot be reused until every task that references it has completed. On shared-memory hardware, the
route may require synchronization without a physical copy.

CPU and GPU tasks may run concurrently when their dependencies allow it. The scheduler chooses
among ready workers based on expected completion time, including the time to make inputs available.
It should avoid moving data back and forth merely because another worker became idle.

## Errors, cancellation, and observability

Graph construction errors return immediately. Asynchronous execution errors attach to the failed
task and propagate to dependent nodes. The next wait or host read for an affected value reports the
failure with its operation and backend context. Independent graph branches may continue.

Cancellation may stop tasks that have not been submitted; already submitted backend work must be
retired safely before its resources are released. The API must define whether cancellation is
best-effort and what values remain readable afterward.

Tracing should connect a public operation to its optimized nodes and physical tasks. It must show
selected devices, partitioning, transfer routes and bytes, queue delays, kernel times, workspace,
and reasons for fallback. Performance claims should be tested against CPU-only, GPU-only, and mixed
plans on representative workloads.

## Initial implementation bar

The first tensor operation must use the real operation DAG, planner interface, task DAG, residency
tracker, and completion/error path. Its planners and transformations may be conservative, but they
must obey the same contracts that later operations and backends will use. CPU-only, Vulkan-only, and
mixed execution should share the graph path rather than having separate public fast paths.

## Current implementation boundary

Each addition produces an immutable float32 value with a unique runtime-local ID and retained input
references. A read collects reachable values and builds ordered source, add, and download tasks for
each shard. CPU and Vulkan shards use the same graph. GPU command buffers keep inputs and
intermediate values within each shader invocation until final readback. A fence completes each GPU
chunk before its workspace is reused. Backend failures are recorded and the affected shard is
recomputed on the CPU. Each runtime retains CPU workers and a bounded workspace and pipeline per
Vulkan device. Large shards stream through that workspace in chunks.

The scheduler calibrates CPU and GPU throughput and can back off mixed execution after measured
losses. It does not yet retain GPU tensor replicas across reads, track arbitrary aliases or
mutations, or account for full transfer topology and memory pressure. Those requirements above
remain active design constraints for future operations.
