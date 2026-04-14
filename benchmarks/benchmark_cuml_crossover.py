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
    N: int,
    target_cv: float = 0.08,
    min_samples: int = 5,
    max_samples: int = 20,
    window: int = 5,
) -> tuple[float, float, float, float, float, float, int, bool]:
    """Measure CPU and GPU times adaptively until timing stabilises.

    Runs CPU+GPU pairs in a single interleaved loop — no separate warmup
    phase.  Early iterations heat up the GPU/CUDA state naturally and are
    discarded once the tail converges.  Convergence is declared when the
    coefficient of variation (CV = stdev/mean) of the last `window` samples
    falls below `target_cv` for **both** CPU and GPU.

    Returns
    -------
    cpu_med, gpu_med, cpu_q25, cpu_q75, gpu_q25, gpu_q75 : float
        Median and IQR bounds computed from the final stable window, in ms.
    n_samples : int
        Total iterations run (including early/warming ones).
    converged : bool
        True if CV threshold was met; False if max_samples was reached first.
    """
    rng = np.random.default_rng(42)
    src = (rng.random((N, 2)) * 4096).astype(np.float32)
    ref = (rng.random((N, 2)) * 4096).astype(np.float32)

    cpu_times: list[float] = []
    gpu_times: list[float] = []
    converged = False

    while len(cpu_times) < max_samples:
        # CPU measurement
        t0 = time.perf_counter()
        cKDTree(ref).query(src, k=1)
        cpu_times.append((time.perf_counter() - t0) * 1000)

        # GPU measurement immediately after — keeps GPU warm between samples
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        sg = cp.asarray(src)
        rg = cp.asarray(ref)
        nn = NearestNeighbors(n_neighbors=1, algorithm='brute')
        nn.fit(rg)
        nn.kneighbors(sg)
        cp.cuda.Stream.null.synchronize()
        gpu_times.append((time.perf_counter() - t0) * 1000)

        # Check convergence on the trailing window once enough samples exist
        if len(cpu_times) >= min_samples:
            cpu_win = cpu_times[-window:]
            gpu_win = gpu_times[-window:]
            cpu_mean = statistics.mean(cpu_win)
            gpu_mean = statistics.mean(gpu_win)
            if cpu_mean > 0 and gpu_mean > 0:
                cpu_cv = statistics.stdev(cpu_win) / cpu_mean
                gpu_cv = statistics.stdev(gpu_win) / gpu_mean
                if cpu_cv < target_cv and gpu_cv < target_cv:
                    converged = True
                    break

    # Compute statistics from the stable tail only
    n_use = min(window, len(cpu_times))
    cpu_arr = np.array(cpu_times[-n_use:])
    gpu_arr = np.array(gpu_times[-n_use:])
    return (
        float(np.median(cpu_arr)),
        float(np.median(gpu_arr)),
        float(np.percentile(cpu_arr, 25)),
        float(np.percentile(cpu_arr, 75)),
        float(np.percentile(gpu_arr, 25)),
        float(np.percentile(gpu_arr, 75)),
        len(cpu_times),
        converged,
    )


def print_row(
    N: int,
    cpu_ms: float, gpu_ms: float,
    cpu_q25: float, cpu_q75: float,
    gpu_q25: float, gpu_q75: float,
    n_samples: int = 0,
    converged: bool = True,
) -> str:
    """Print CSV row and human-readable line; return winner."""
    speedup = cpu_ms / gpu_ms if gpu_ms > 0 else float('inf')
    winner = 'GPU' if speedup > 1.0 else 'CPU'
    print(
        f"{N},{cpu_ms:.2f},{gpu_ms:.2f},{speedup:.3f},{winner},"
        f"{cpu_q25:.2f},{cpu_q75:.2f},{gpu_q25:.2f},{gpu_q75:.2f},"
        f"{n_samples},{'Y' if converged else 'N'}"
    )
    conv_note = "" if converged else "  (!max_samples)"
    print(f"  N={N:>7}: CPU={cpu_ms:.1f}ms  GPU={gpu_ms:.1f}ms  "
          f"speedup={speedup:.2f}x  [{winner}]  n={n_samples}{conv_note}",
          file=sys.stderr)
    return winner


def refine_between(N_lo: int, N_hi: int, n_points: int = 5) -> list[int]:
    """Generate n_points log-spaced integers between N_lo and N_hi (exclusive endpoints)."""
    return [
        int(round(v))
        for v in np.logspace(np.log10(N_lo), np.log10(N_hi), n_points + 2)[1:-1]
        if int(round(v)) not in (N_lo, N_hi)
    ]


def global_gpu_warmup(N_warm: int, iterations: int = 8) -> None:
    """Heat the GPU to steady thermal state before Phase 2.

    Runs cuML NearestNeighbors at N_warm for `iterations` rounds.  The goal is
    to recreate the same GPU clock / memory-pool state that existed at the end
    of Phase 1, so Phase 2 measurements are comparable.
    """
    print(f"  [pre-warm] GPU warm-up at N={N_warm} × {iterations} iterations …",
          file=sys.stderr)
    rng = np.random.default_rng(0)
    src = (rng.random((N_warm, 2)) * 4096).astype(np.float32)
    ref = (rng.random((N_warm, 2)) * 4096).astype(np.float32)
    sg = cp.asarray(src)
    rg = cp.asarray(ref)
    for _ in range(iterations):
        nn = NearestNeighbors(n_neighbors=1, algorithm='brute')
        nn.fit(rg)
        nn.kneighbors(sg)
        cp.cuda.Stream.null.synchronize()
    print("  [pre-warm] Done.", file=sys.stderr)


def check_boundary_drift(
    results_p1: list[tuple[int, float, float, str]],
    target_cv: float,
    min_samples: int,
    max_samples: int,
    window: int,
    threshold: float = 0.15,
) -> bool:
    """Re-measure Phase 1 boundary points and warn if GPU timing has drifted.

    Returns True if consistent (all GPU drifts within threshold), False otherwise.
    Prints a per-point drift report to stderr so the user can judge validity.
    """
    boundary: list[int] = []
    for i in range(len(results_p1) - 1):
        N_lo, _, _, w_lo = results_p1[i]
        N_hi, _, _, w_hi = results_p1[i + 1]
        if w_lo != w_hi:
            boundary.extend([N_lo, N_hi])

    if not boundary:
        return True

    print("  [validate] Re-measuring Phase 1 boundary points …", file=sys.stderr)
    p1 = {N: (cpu, gpu) for N, cpu, gpu, _ in results_p1}
    consistent = True
    for N in sorted(set(boundary)):
        cpu_new, gpu_new, *_ = measure_one(
            N, target_cv=target_cv, min_samples=min_samples,
            max_samples=max_samples, window=window,
        )
        cpu_old, gpu_old = p1[N]
        gpu_drift = abs(gpu_new - gpu_old) / gpu_old
        cpu_drift = abs(cpu_new - cpu_old) / cpu_old
        flag = " *** WARN" if gpu_drift > threshold else ""
        print(f"    N={N:>7}: GPU {gpu_old:.1f}→{gpu_new:.1f} ms "
              f"(drift {gpu_drift:+.1%})  CPU drift {cpu_drift:+.1%}{flag}",
              file=sys.stderr)
        if gpu_drift > threshold:
            consistent = False

    if not consistent:
        print(f"  [validate] GPU drift > {threshold:.0%} detected — "
              "Phase 2 results may not be fully reliable.", file=sys.stderr)
    else:
        print("  [validate] All boundary points within drift threshold — "
              "Phase 2 measurements look consistent.", file=sys.stderr)
    return consistent


def benchmark_crossover(
    sizes: list[int],
    target_cv: float,
    min_samples: int,
    max_samples: int,
    window: int,
    auto_refine: bool,
    refine_points: int,
    warmup_n: int | None,
    validate: bool,
    min_speedup: float = 1.15,
) -> None:
    """Run the crossover benchmark and print CSV results."""
    if not HAS_CUML:
        print("ERROR: cuML not available. Run inside a py3.12 profiler container.", file=sys.stderr)
        sys.exit(1)

    gpu_name = get_gpu_name()
    print(f"# GPU: {gpu_name}", file=sys.stderr)
    print(f"# Sizes: {sizes}", file=sys.stderr)
    print(f"# Adaptive convergence: target_cv={target_cv:.0%}  "
          f"min={min_samples}  max={max_samples}  window={window}",
          file=sys.stderr)
    print(f"# Auto-refine: {auto_refine}", file=sys.stderr)
    print("", file=sys.stderr)

    print("N,cpu_ms,gpu_ms,speedup_e2e,winner,cpu_q25,cpu_q75,gpu_q25,gpu_q75,n_samples,converged")

    results: list[tuple[int, float, float, str]] = []

    def _measure_and_record(N: int) -> str:
        cpu_ms, gpu_ms, cpu_q25, cpu_q75, gpu_q25, gpu_q75, n, conv = measure_one(
            N, target_cv=target_cv, min_samples=min_samples,
            max_samples=max_samples, window=window,
        )
        winner = print_row(N, cpu_ms, gpu_ms, cpu_q25, cpu_q75, gpu_q25, gpu_q75, n, conv)
        results.append((N, cpu_ms, gpu_ms, winner))
        return winner

    # ── Phase 1: initial sweep ────────────────────────────────────────────────
    print("=== Phase 1: initial sweep ===", file=sys.stderr)
    for N in sizes:
        _measure_and_record(N)

    # ── Phase 2: auto-refine around crossover zones ───────────────────────────
    if auto_refine and len(results) >= 2:
        print("", file=sys.stderr)
        print("=== Phase 2: refining around crossover zones ===", file=sys.stderr)

        # Pre-warm GPU between phases. With adaptive convergence each Phase 2
        # measurement self-stabilises, but a pre-warm still helps the first few
        # iterations converge faster.
        N_heat = warmup_n if warmup_n is not None else max(sizes)
        global_gpu_warmup(N_heat)

        if validate:
            print("", file=sys.stderr)
            check_boundary_drift(results, target_cv, min_samples, max_samples, window)

        print("", file=sys.stderr)
        refined_sizes: list[int] = []
        for i in range(len(results) - 1):
            N_lo, _, _, w_lo = results[i]
            N_hi, _, _, w_hi = results[i + 1]
            if w_lo != w_hi:
                new_pts = refine_between(N_lo, N_hi, n_points=refine_points)
                refined_sizes.extend(new_pts)
                print(f"  Crossover between N={N_lo} and N={N_hi} → "
                      f"adding {new_pts}", file=sys.stderr)

        for N in sorted(set(refined_sizes)):
            _measure_and_record(N)

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
        peak_speedup = max(cpu / gpu for _, cpu, gpu in gpu_wins)
        peak_N = max(gpu_wins, key=lambda t: t[1] / t[2])[0]

        # Full win zone: every point where GPU was faster (may include noisy edges)
        N_min_all = min(N for N, _, _ in gpu_wins)
        N_max_all = max(N for N, _, _ in gpu_wins)
        print(f"#   GPU win zone (all):    N ≈ {N_min_all}–{N_max_all}  "
              f"(peak {peak_speedup:.2f}x at N={peak_N})", file=sys.stderr)

        # Robust zone: only points with speedup >= min_speedup threshold.
        # The crossover boundaries sit where speedup ≈ 1.0 and are sensitive
        # to measurement noise; this filter excludes those fragile edges.
        robust_wins = [(N, cpu, gpu) for N, cpu, gpu in gpu_wins
                       if cpu / gpu >= min_speedup]
        if robust_wins:
            N_min_rob = min(N for N, _, _ in robust_wins)
            N_max_rob = max(N for N, _, _ in robust_wins)
            print(f"#   GPU win zone (robust): N ≈ {N_min_rob}–{N_max_rob}  "
                  f"(speedup ≥ {min_speedup:.2f}x throughout)", file=sys.stderr)
            print("#", file=sys.stderr)
            print(f"#   → GPUPHOT_CUML_MIN_SOURCES={N_min_rob}  "
                  f"  # robust boundary (speedup ≥ {min_speedup:.2f}x)", file=sys.stderr)
            print(f"#   → GPUPHOT_CUML_MAX_SOURCES={N_max_rob}  "
                  f"  # robust boundary (speedup ≥ {min_speedup:.2f}x)", file=sys.stderr)
        else:
            print(f"#   No point exceeds {min_speedup:.2f}x speedup — "
                  f"GPU advantage is too marginal to rely on.", file=sys.stderr)
            print("#   → Set GPUPHOT_USE_CUML_CROSSMATCH=0 (keep cKDTree, default).", file=sys.stderr)
            print("#   → Do NOT set GPUPHOT_CUML_MIN_SOURCES / MAX_SOURCES.", file=sys.stderr)

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
        "--target-cv", type=float, default=0.08, metavar="CV",
        help="Convergence threshold: stop when CV of last --window samples "
             "is below this value for both CPU and GPU (default: 0.08 = 8%%)",
    )
    p.add_argument(
        "--min-samples", type=int, default=5,
        help="Minimum iterations before checking convergence (default: 5)",
    )
    p.add_argument(
        "--max-samples", type=int, default=20,
        help="Hard cap on iterations per N — measurement stops here even if "
             "CV has not converged (default: 20)",
    )
    p.add_argument(
        "--window", type=int, default=5,
        help="Number of trailing samples used for CV and final statistics "
             "(default: 5)",
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
    p.add_argument(
        "--warmup-n", type=int, default=None,
        metavar="N",
        help="N used for the global GPU pre-warm before Phase 2 "
             "(default: max of --sizes / --logspace)",
    )
    p.add_argument(
        "--validate", action="store_true",
        help="After the pre-warm, re-measure Phase 1 boundary points and report "
             "GPU drift to confirm Phase 2 consistency.",
    )
    p.add_argument(
        "--min-speedup", type=float, default=1.15, metavar="X",
        help="Minimum GPU speedup to count as a reliable win (default: 1.15). "
             "Points with speedup < X are excluded from the robust recommendation. "
             "Raise this on noisy systems; lower to 1.05 on very stable ones.",
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
        # Default: log-spaced, 25 points, 100 to 200000
        sizes = [int(round(v)) for v in np.logspace(np.log10(100), np.log10(200_000), 25)]
        sizes = sorted(set(sizes))

    benchmark_crossover(
        sizes=sizes,
        target_cv=args.target_cv,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        window=args.window,
        auto_refine=args.auto_refine,
        refine_points=args.refine_points,
        warmup_n=args.warmup_n,
        validate=args.validate,
        min_speedup=args.min_speedup,
    )
