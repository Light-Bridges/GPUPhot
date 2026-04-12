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

    # Log-spaced sweep (recommended — better coverage of crossover zone):
    docker exec <container> python3 /app/benchmarks/benchmark_cuml_crossover.py \\
        --logspace 30 100 200000

    # Custom explicit sizes:
    docker exec <container> python3 /app/benchmarks/benchmark_cuml_crossover.py \\
        --sizes 100 500 1000 5000 10000 50000 --repeats 10

    # Auto-refine around crossover zones (recommended for production):
    docker exec <container> python3 /app/benchmarks/benchmark_cuml_crossover.py \\
        --logspace 20 100 200000 --auto-refine --repeats 10

    # Output CSV to file:
    docker exec <container> python3 /app/benchmarks/benchmark_cuml_crossover.py \\
        --auto-refine > crossover_$(hostname)_$(date +%Y%m%d).csv

Results for reference (2026-03-28 / 2026-04-02):

    GPU              cuML wins from    Peak speedup    Upper limit
    ----------------------------------------------------------------
    H100 PCIe        ~5K sources       4.6x @ 20K      > 500K
    A100-SXM4-80GB   ~5K sources       3.5x @ 20K      > 100K
    L40S             ~2K sources       5.8x @ 20K      > 200K
    RTX 3090         ~2K sources       3.2x @ 20K      > 100K
    RTX 3060         ~5K sources       1.9x @ 10K      ~40K
    RTX 3050 Ti      ~2K sources       2.3x @ 10K      ~50K  (2026-04-02 clean system)

    WARNING: These are synthetic (single call). Real pipeline overhead
    makes cuML slower for ALL source counts on A100. See:
    benchmarks/results_collected/cuml_ablation_a100_20260329.csv

    Set GPUPHOT_CUML_MIN_SOURCES and GPUPHOT_CUML_MAX_SOURCES based on
    the output of this script for your specific GPU. If cuML never wins,
    set GPUPHOT_USE_CUML_CROSSMATCH=0 (default).
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


def measure_one(
    N: int, repeats: int, warmup: int
) -> tuple[float, float, float, float, float, float]:
    """Measure CPU and GPU times for source count N.

    Returns (cpu_med, gpu_med, cpu_q25, cpu_q75, gpu_q25, gpu_q75) in ms.
    Percentiles enable IQR bands in the crossover figure.
    """
    rng = np.random.default_rng(42)
    src = (rng.random((N, 2)) * 4096).astype(np.float32)
    ref = (rng.random((N, 2)) * 4096).astype(np.float32)

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
        t0 = time.perf_counter()
        cKDTree(ref).query(src, k=1)
        cpu_times.append((time.perf_counter() - t0) * 1000)

        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        sg = cp.asarray(src)
        rg = cp.asarray(ref)
        nn = NearestNeighbors(n_neighbors=1, algorithm='brute')
        nn.fit(rg)
        nn.kneighbors(sg)
        cp.cuda.Stream.null.synchronize()
        gpu_times.append((time.perf_counter() - t0) * 1000)

    cpu_arr = np.array(cpu_times)
    gpu_arr = np.array(gpu_times)
    return (
        float(np.median(cpu_arr)),
        float(np.median(gpu_arr)),
        float(np.percentile(cpu_arr, 25)),
        float(np.percentile(cpu_arr, 75)),
        float(np.percentile(gpu_arr, 25)),
        float(np.percentile(gpu_arr, 75)),
    )


def print_row(
    N: int,
    cpu_ms: float, gpu_ms: float,
    cpu_q25: float, cpu_q75: float,
    gpu_q25: float, gpu_q75: float,
) -> str:
    """Print CSV row and human-readable line; return winner."""
    speedup = cpu_ms / gpu_ms if gpu_ms > 0 else float('inf')
    winner = 'GPU' if speedup > 1.0 else 'CPU'
    print(
        f"{N},{cpu_ms:.2f},{gpu_ms:.2f},{speedup:.3f},{winner},"
        f"{cpu_q25:.2f},{cpu_q75:.2f},{gpu_q25:.2f},{gpu_q75:.2f}"
    )
    print(f"  N={N:>7}: CPU={cpu_ms:.1f}ms  GPU={gpu_ms:.1f}ms  "
          f"speedup={speedup:.2f}x  [{winner}]", file=sys.stderr)
    return winner


def refine_between(N_lo: int, N_hi: int, n_points: int = 5) -> list[int]:
    """Generate n_points log-spaced integers between N_lo and N_hi (exclusive endpoints)."""
    return [
        int(round(v))
        for v in np.logspace(np.log10(N_lo), np.log10(N_hi), n_points + 2)[1:-1]
        if int(round(v)) not in (N_lo, N_hi)
    ]


def benchmark_crossover(
    sizes: list[int],
    repeats: int,
    warmup: int,
    auto_refine: bool,
    refine_points: int,
) -> None:
    """Run the crossover benchmark and print CSV results."""
    if not HAS_CUML:
        print("ERROR: cuML not available. Run inside a py3.12 profiler container.", file=sys.stderr)
        sys.exit(1)

    gpu_name = get_gpu_name()
    print(f"# GPU: {gpu_name}", file=sys.stderr)
    print(f"# Sizes: {sizes}", file=sys.stderr)
    print(f"# Repeats: {repeats}, Warmup: {warmup}, Auto-refine: {auto_refine}", file=sys.stderr)
    print("", file=sys.stderr)

    print("N,cpu_ms,gpu_ms,speedup_e2e,winner,cpu_q25,cpu_q75,gpu_q25,gpu_q75")

    results: list[tuple[int, float, float, str]] = []

    # ── Phase 1: initial sweep ────────────────────────────────────────────────
    print("=== Phase 1: initial sweep ===", file=sys.stderr)
    for N in sizes:
        cpu_ms, gpu_ms, cpu_q25, cpu_q75, gpu_q25, gpu_q75 = measure_one(N, repeats, warmup)
        winner = print_row(N, cpu_ms, gpu_ms, cpu_q25, cpu_q75, gpu_q25, gpu_q75)
        results.append((N, cpu_ms, gpu_ms, winner))

    # ── Phase 2: auto-refine around crossover zones ───────────────────────────
    if auto_refine and len(results) >= 2:
        print("", file=sys.stderr)
        print("=== Phase 2: refining around crossover zones ===", file=sys.stderr)

        refined_sizes: list[int] = []
        for i in range(len(results) - 1):
            N_lo, _, _, w_lo = results[i]
            N_hi, _, _, w_hi = results[i + 1]
            if w_lo != w_hi:
                new_pts = refine_between(N_lo, N_hi, n_points=refine_points)
                refined_sizes.extend(new_pts)
                print(f"  Crossover between N={N_lo} and N={N_hi} → "
                      f"adding {new_pts}", file=sys.stderr)

        refined_sizes = sorted(set(refined_sizes))
        for N in refined_sizes:
            cpu_ms, gpu_ms, cpu_q25, cpu_q75, gpu_q25, gpu_q75 = measure_one(N, repeats, warmup)
            winner = print_row(N, cpu_ms, gpu_ms, cpu_q25, cpu_q75, gpu_q25, gpu_q75)
            results.append((N, cpu_ms, gpu_ms, winner))

    # ── Summary ───────────────────────────────────────────────────────────────
    results.sort(key=lambda r: r[0])

    gpu_wins = [(N, cpu, gpu) for N, cpu, gpu, w in results if w == 'GPU']
    print("", file=sys.stderr)
    print(f"# ── Recommendation for {gpu_name} ──", file=sys.stderr)

    if not gpu_wins:
        print("#   GPU never wins over cKDTree for any tested N.", file=sys.stderr)
        print("#   → Set GPUPHOT_USE_CUML_CROSSMATCH=0 (keep cKDTree, default).", file=sys.stderr)
        print("#   → Do NOT set GPUPHOT_CUML_MIN_SOURCES / MAX_SOURCES.", file=sys.stderr)
    else:
        N_min = min(N for N, _, _ in gpu_wins)
        N_max = max(N for N, _, _ in gpu_wins)
        peak_speedup = max(cpu / gpu for _, cpu, gpu in gpu_wins)
        peak_N = max(gpu_wins, key=lambda t: t[1] / t[2])[0]
        print(f"#   GPU wins for N ≈ {N_min}–{N_max} sources  "
              f"(peak {peak_speedup:.1f}x at N={peak_N})", file=sys.stderr)
        print(f"#   → GPUPHOT_CUML_MIN_SOURCES={N_min}", file=sys.stderr)
        print(f"#   → GPUPHOT_CUML_MAX_SOURCES={N_max}", file=sys.stderr)
        print("#", file=sys.stderr)
        print("#   NOTE: These are synthetic (single crossmatch call).", file=sys.stderr)
        print("#   Real pipeline overhead (init + multiple calls/image) reduces", file=sys.stderr)
        print("#   the advantage. Verify with a real-image ablation before enabling.", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark cKDTree vs cuML NearestNeighbors crossover point.\n"
                    "Recommended: --logspace 25 100 200000 --auto-refine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    size_group = p.add_mutually_exclusive_group()
    size_group.add_argument(
        "--sizes", type=int, nargs="+",
        default=None,
        metavar="N",
        help="Explicit source counts to test (default: log-spaced 100..200000)",
    )
    size_group.add_argument(
        "--logspace", type=int, nargs=3,
        metavar=("NPOINTS", "MIN", "MAX"),
        default=None,
        help="Generate NPOINTS log-spaced sizes from MIN to MAX "
             "(e.g. --logspace 25 100 200000)",
    )

    p.add_argument(
        "--repeats", type=int, default=5,
        help="Measured repetitions per N (default: 5)",
    )
    p.add_argument(
        "--warmup", type=int, default=2,
        help="Warmup runs per N (default: 2)",
    )
    p.add_argument(
        "--auto-refine", action="store_true",
        help="After initial sweep, add extra points between N where winner changes "
             "(recommended for finding exact crossover boundary)",
    )
    p.add_argument(
        "--refine-points", type=int, default=5,
        help="Number of extra points to insert per crossover zone (default: 5)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.logspace is not None:
        npts, lo, hi = args.logspace
        sizes = [int(round(v)) for v in np.logspace(np.log10(lo), np.log10(hi), npts)]
        sizes = sorted(set(sizes))
    elif args.sizes is not None:
        sizes = sorted(set(args.sizes))
    else:
        # Default: log-spaced, 20 points, 100 to 200000
        sizes = [int(round(v)) for v in np.logspace(np.log10(100), np.log10(200_000), 20)]
        sizes = sorted(set(sizes))

    benchmark_crossover(
        sizes=sizes,
        repeats=args.repeats,
        warmup=args.warmup,
        auto_refine=args.auto_refine,
        refine_points=args.refine_points,
    )
