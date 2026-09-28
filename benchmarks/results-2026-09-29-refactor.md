# Float32 addition after the performance refactor: 2026-09-29

Machine: AMD Ryzen 7 9700X (16 logical processors), Radeon RX 9060 XT, and Radeon integrated GPU
(RADV RAPHAEL_MENDOCINO). Linux 7.0.0-34-generic, GCC 15.2.0, CMake Release build. Both Vulkan
devices were discovered; direct GPU and mixed-plan correctness tests passed.

Each result is the median of 31 timed reads after three warm-ups. The benchmark creates fresh
inputs and a lazy addition chain for each read; input creation and DAG construction are outside the
timer. A timed read includes planning, CPU or GPU execution, synchronization, and the final host
copy. CPU-only used the same library with Vulkan loading blocked by the Linux test interposer. The
before column is the [2026-09-28 baseline](results-2026-09-28.md) on the same machine. Speedup
compares automatic placement before and after the refactor.

| Elements | Adds | Before auto (ms) | After auto (ms) | After CPU (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 1 | 1.507 | 0.043 | 0.043 | 35.1x |
| 65,536 | 16 | 2.425 | 0.179 | 0.155 | 13.5x |
| 262,144 | 2 | 2.851 | 0.076 | 0.087 | 37.4x |
| 262,144 | 16 | 3.653 | 0.192 | 0.168 | 19.0x |
| 1,048,576 | 2 | 4.749 | 0.290 | 0.290 | 16.4x |
| 1,048,576 | 16 | 12.492 | 0.758 | 0.540 | 16.5x |
| 8,388,608 | 16 | 91.493 | 5.590 | 5.665 | 16.4x |

Automatic placement assigned zero GPU elements in all 31 measured samples for every case in this
run. The scheduler can still pilot and choose GPU shards for other workloads or hardware. These
results do not demonstrate a heterogeneous speedup; they show that the new scheduler avoids the
large regressions caused by the old fixed GPU split. The 1,048,576-element, 16-add automatic case
was 0.758 ms versus 0.540 ms CPU-only despite assigning no measured GPU work. That difference
needs further profiling. Runtime discovery, thread scheduling, and process-to-process variation
have not been isolated as causes.

The refactor fuses addition chains, uses a persistent CPU worker pool, reuses Vulkan contexts and
bounded transfer workspace, streams large GPU shards in chunks, and measures per-runtime throughput
for placement. It does not yet keep tensor values resident on GPUs across reads. The
[raw measurements](refactor-raw-2026-09-29.json) include quartiles, warm-up timings, and observed
GPU shard ranges. Reproduce with the commands in [README.md](README.md).
