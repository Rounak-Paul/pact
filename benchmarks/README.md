# Addition benchmark

This benchmark measures `pact_tensor_read` for a prebuilt lazy addition chain. Each row uses two
float32 input tensors, `N` elements each, and applies `A` additions (`x + A*y`). Input allocation,
input copying, DAG construction, verification, and handle release occur outside the timer. The
timed call includes graph collection, planning, CPU work, Vulkan transfers, dispatch,
synchronization, and the final host copy. Initial samples may also include CPU worker and Vulkan
context, pipeline, and workspace creation. Later samples reuse these resources.

The program performs three warm-up iterations, then prints individual samples. The runner alternates
which mode goes first by case and reports the median of 31 samples. CPU-only mode blocks Vulkan
loading with the existing Linux test interposer; both modes use the same compiled PACT library.
`CPU/auto` above 1 means automatic placement is faster. The CPU backend uses a persistent worker
pool on larger workloads. GPU share is the fraction of output elements assigned to Vulkan GPUs;
the runner rejects results with a backend fallback.

```sh
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DPACT_BUILD_BENCHMARKS=ON
cmake --build build-bench -j
python3 benchmarks/run.py --build build-bench
```

The benchmark program also accepts `elements additions samples` and emits CSV. It requires no
Python packages; the runner is Linux-specific because of its CPU-only interposer.
