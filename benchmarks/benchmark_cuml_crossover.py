#!/usr/bin/env python3
"""
Benchmark: cKDTree (CPU) vs cuML NearestNeighbors (GPU) for 2D crossmatch.

This script measures the crossover point where GPU crossmatch becomes faster
than CPU KDTree for 2D source matching (the core operation in catalog
crossmatching). It helps users determine optimal GPUPHOT_CUML_MIN_SOURCES
and GPUPHOT_CUML_MAX_SOURCES thresholds for their specific GPU.

Key finding: cuML uses brute-force O(N^2) for low-dimensional data, while
cKDTree is O(N log N). GPU only wins in a narrow window (~2K-50K sources)
on powerful GPUs, and never wins on small GPUs (<= 4 GB VRAM).

In the real GPUPhot pipeline (multiple crossmatch calls per image), cuML
is slower than cKDTree for ALL tested source counts (112-18888) on an A100.
The synthetic crossover observed here does NOT translate to real pipeline gains.

Usage
-----
::

    # Inside a py3.12 profiler container with cuML:
    docker exec <container> python3 /app/benchmarks/benchmark_cuml_crossover.py

    # Custom sizes and repeats:
    docker exec <container> python3 /app/benchmarks/benchmark_cuml_crossover.py \\
        --sizes 100 500 1000 5000 10000 50000 --repeats 10

    # Output is CSV to stdout, redirect to file:
    docker exec <container> python3 /app/benchmarks/benchmark_cuml_crossover.py > crossover.csv

Results for reference (2026-03-28):

    GPU              cuML wins from    Peak speedup    Notes
    ----------------------------------------------------------------
    H100 PCIe        5K sources        4.6x @ 20K     Sustained
    A100-SXM4-80GB   5K sources        3.5x @ 20K     Shrinks at 100K
    L40S             2K sources        5.8x @ 20K     Best overall
    RTX 3090         2K sources        3.2x @ 20K     Sustained
    RTX 3060         5K sources        1.9x @ 10K     Loses at 50K
    RTX 3050 Ti      NEVER             —              CPU always wins

    WARNING: These are synthetic (single call). Real pipeline overhead
    makes cuML slower for ALL source counts on A100. See:
    benchmarks/results_collected/cuml_ablation_a100_20260329.csv
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import cupy as cp
import numpy as np
from scipy.spatial import cKDTree

try:
    from cuml.neighbors import NearestNeighbors
    HAS_CUML = True
except ImportError:
    HAS_CUML = False


def get_gpu_name() -> str:
    """Get GPU name for reporting."""
    try:
        import subprocess
        return subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader",
            shell=True, stderr=subprocess.DEVNULL
        ).decode().strip().split('\n')[0]
    except Exception:
        return "unknown"


def benchmark_crossover(sizes: list[int], repeats: int, warmup: int) -> None:
    """Run the crossover benchmark and print CSV results."""
    if not HAS_CUML:
        print("ERROR: cuML not available. Install RAPIDS cuML.", file=sys.stderr)
        sys.exit(1)

    gpu_name = get_gpu_name()
    print(f"# GPU: {gpu_name}", file=sys.stderr)
    print(f"# Sizes: {sizes}, Repeats: {repeats}, Warmup: {warmup}", file=sys.stderr)

    print("N,cpu_ms,gpu_e2e_ms,speedup_e2e,winner")

    for N in sizes:
        rng = np.random.default_rng(42)
        src = (rng.random((N, 2)) * 4096).astype(np.float32)
        ref = (rng.random((N, 2)) * 4096).astype(np.float32)

        # Warmup
        for _ in range(warmup):
            cKDTree(ref).query(src, k=1)
            sg = cp.asarray(src)
            rg = cp.asarray(ref)
            nn = NearestNeighbors(n_neighbors=1, algorithm='brute')
            nn.fit(rg)
            nn.kneighbors(sg)
            cp.cuda.Stream.null.synchronize()

        cpu_times: list[float] = []
        gpu_times: list[float] = []

        for _ in range(repeats):
            # CPU: cKDTree
            t0 = time.perf_counter()
            cKDTree(ref).query(src, k=1)
            cpu_times.append((time.perf_counter() - t0) * 1000)

            # GPU: cuML NearestNeighbors (end-to-end including transfer)
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            sg = cp.asarray(src)
            rg = cp.asarray(ref)
            nn = NearestNeighbors(n_neighbors=1, algorithm='brute')
            nn.fit(rg)
            nn.kneighbors(sg)
            cp.cuda.Stream.null.synchronize()
            gpu_times.append((time.perf_counter() - t0) * 1000)

        cpu_med = statistics.median(cpu_times)
        gpu_med = statistics.median(gpu_times)
        speedup = cpu_med / gpu_med if gpu_med > 0 else float('inf')
        winner = 'GPU' if speedup > 1.0 else 'CPU'

        print(f"{N},{cpu_med:.2f},{gpu_med:.2f},{speedup:.3f},{winner}")
        print(f"  N={N:>6}: CPU={cpu_med:.1f}ms  GPU={gpu_med:.1f}ms  "
              f"speedup={speedup:.2f}x  {winner}", file=sys.stderr)

    # Summary
    print("", file=sys.stderr)
    print(f"# Recommendation for {gpu_name}:", file=sys.stderr)
    print("#   If GPU never wins: set GPUPHOT_USE_CUML_CROSSMATCH=0", file=sys.stderr)
    print("#   If GPU wins in a range: consider GPUPHOT_CUML_MIN_SOURCES=<min_win>", file=sys.stderr)
    print("#   But note: real pipeline overhead makes cuML slower than these", file=sys.stderr)
    print("#   synthetic results suggest. Recommend keeping cKDTree as default.", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark cKDTree vs cuML NearestNeighbors crossover point"
    )
    p.add_argument(
        "--sizes", type=int, nargs="+",
        default=[100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
        help="Source counts to test",
    )
    p.add_argument("--repeats", type=int, default=5, help="Reps per N (default 5)")
    p.add_argument("--warmup", type=int, default=2, help="Warmup runs per N (default 2)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark_crossover(sizes=args.sizes, repeats=args.repeats, warmup=args.warmup)
