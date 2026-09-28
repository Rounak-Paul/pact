#!/usr/bin/env python3
"""Compare PACT's automatic scheduler with the same binary in CPU-only mode."""

import argparse
import csv
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

CASES = [
    (65_536, 1),
    (65_536, 16),
    (262_144, 2),
    (262_144, 16),
    (1_048_576, 2),
    (1_048_576, 16),
    (8_388_608, 16),
]


def measure(binary: Path, blocker: Path, elements: int, adds: int,
            samples: int, cpu_only: bool) -> dict:
    env = os.environ.copy()
    if cpu_only:
        env["LD_PRELOAD"] = str(blocker)
    command = [str(binary), str(elements), str(adds), str(samples)]
    completed = subprocess.run(command, env=env, check=True,
                               capture_output=True, text=True)
    records = list(csv.DictReader(io.StringIO(completed.stdout)))
    if len(records) != samples + 3:
        raise RuntimeError(f"expected {samples + 3} samples, got {len(records)}")
    warmups = records[:3]
    measured = records[3:]
    times = sorted(float(record["read_us"]) for record in measured)
    fallback = {int(record["fallback"]) for record in measured}
    shards = {int(record["vulkan_shards"]) for record in measured}
    gpu_elements = {int(record["vulkan_elements"]) for record in measured}
    if fallback != {0}:
        raise RuntimeError("backend fallback occurred during a timed sample")
    return {
        "median_us": statistics.median(times),
        "q1_us": times[samples // 4],
        "q3_us": times[(3 * samples) // 4],
        "vulkan_shards": sorted(shards),
        "vulkan_elements": statistics.median(
            int(record["vulkan_elements"]) for record in measured),
        "vulkan_elements_min": min(gpu_elements),
        "vulkan_elements_max": max(gpu_elements),
        "first_read_us": float(warmups[0]["read_us"]),
        "first_read_vulkan_elements": int(warmups[0]["vulkan_elements"]),
        "second_read_us": float(warmups[1]["read_us"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build-bench"))
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("CPU-only comparison uses the Linux Vulkan-loader interposer")
    if args.samples < 5 or args.samples % 2 == 0:
        parser.error("--samples must be an odd number of at least 5")
    binary = (args.build / "pact_add_benchmark").resolve()
    blocker = (args.build / "libpact_test_block_vulkan.so").resolve()
    if not binary.exists() or not blocker.exists():
        parser.error("build with -DPACT_BUILD_BENCHMARKS=ON and BUILD_TESTING=ON")
    results = []
    print("| Elements | Adds | CPU only (ms) | Auto (ms) | Auto GPU share | CPU/auto |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for index, (elements, adds) in enumerate(CASES):
        modes = ("auto", "cpu") if index % 2 == 0 else ("cpu", "auto")
        row = {"elements": elements, "adds": adds}
        for mode in modes:
            row[mode] = measure(binary, blocker, elements, adds, args.samples,
                                cpu_only=mode == "cpu")
        gpu_share = row["auto"]["vulkan_elements"] / elements
        ratio = row["cpu"]["median_us"] / row["auto"]["median_us"]
        print(f"| {elements:,} | {adds} | {row['cpu']['median_us'] / 1000:.3f} | "
              f"{row['auto']['median_us'] / 1000:.3f} | {gpu_share:.1%} | "
              f"{ratio:.2f}x |", flush=True)
        results.append(row)
    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
