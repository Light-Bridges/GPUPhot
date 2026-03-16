#!/usr/bin/env python3
"""
Automated Benchmark Suite for GPUPhot Profiling.

Supports two execution modes:

- **Local** (``--local``): Runs profiling via ``docker run`` on the local machine.
  No SSH required — ideal for development and validation.

- **Remote** (default): Connects via SSH to profiler containers on remote machines.
  Used for production benchmarking on GPU servers.

Architecture
------------
Local mode:
  docker run --gpus all gpuphotfinal-profiler  →  warmup + measured runs
  docker run --gpus all gpuphotfinal-profiler_38  →  warmup + measured runs

Remote mode:
  SSH → Remote profiler Docker container (port 2222/2223)
    ├─ nvidia-smi  →  GPU state check
    ├─ warmup iterations (discarded)
    └─ measured iterations → .nsys-rep + .sqlite

Benchmark Images
----------------
10 reference images covering all camera/size combinations::

    iKon936  2048×2048   →  QSO0957+561 (SDSSg), C2025A6 (Lum)
    QHY600-3 3191×2129   →  C2025R2 (Lum)
    QHY600-4 4787×3193   →  WASP-43-b (SDSSg), NGC2903 (Ha)
    QHY411-1 7100×5325   →  2012QD8 (Lum), GaiaDR3... (SDSSi)
    QHY411-1 14200×10650 →  2025PR1 (Lum)
    QHY411-3 14200×10650 →  24P (Lum), M81 (SDSSr)

Usage
-----
::

    # Local: run all images on both profilers (3.12 + 3.8)
    python profiling_scripts/run_benchmark_suite.py --local

    # Local: only small images (quick test)
    python profiling_scripts/run_benchmark_suite.py --local --max-megapixels 20

    # Local: specific profiler only
    python profiling_scripts/run_benchmark_suite.py --local --profilers 312

    # Local: custom warmup and repetitions
    python profiling_scripts/run_benchmark_suite.py --local --warmup 1 --repetitions 3

    # Remote: run on configured targets
    python profiling_scripts/run_benchmark_suite.py --run --targets lenovo

    # Check GPU status (both local and remote)
    python profiling_scripts/run_benchmark_suite.py --check-only
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TargetMachine:
    """A remote machine with a profiler Docker container."""
    name: str
    ssh_host: str
    ssh_port: int = 2222
    ssh_opts: str = ""
    results_path: str = "/app/profiling_results"
    images_path: str = "/data/images"
    gpu_id: int = 0
    enabled: bool = True


@dataclass
class ReferenceImage:
    """A canonical test image for benchmarking."""
    oblineid: str
    instrument: str
    camera: str
    filename: str
    size_label: str
    filter_name: str
    megapixels: float
    expected_sources: int
    description: str = ""


# -- Reference images (the 10-image benchmark suite) -----------------------
REFERENCE_IMAGES: List[ReferenceImage] = [
    ReferenceImage(
        oblineid="5108788", instrument="iKon936", camera="iKon936-1",
        filename="TTT3_iKon936-1_2026-01-15-06-05-00-020013_QSO0957+561_SDSSg.fits",
        size_label="2048x2048", filter_name="SDSSg", megapixels=4.2,
        expected_sources=402, description="iKon936 2048x2048 SDSSg (QSO0957+561)",
    ),
    ReferenceImage(
        oblineid="4124611", instrument="iKon936", camera="iKon936-1",
        filename="TTT3_iKon936-1_2025-09-15-05-31-52-256207_C2025A6_Lum.fits",
        size_label="2048x2048", filter_name="Lum", megapixels=4.2,
        expected_sources=257, description="iKon936 2048x2048 Lum (C2025A6)",
    ),
    ReferenceImage(
        oblineid="4778615", instrument="QHY600M", camera="QHY600-3",
        filename="TTT3_QHY600-3_2025-12-01-23-02-47-487643_C2025R2_Lum.fits",
        size_label="3191x2129", filter_name="Lum", megapixels=6.8,
        expected_sources=124, description="QHY600-3 3191x2129 Lum (C2025R2)",
    ),
    ReferenceImage(
        oblineid="5411873", instrument="QHY600M", camera="QHY600-4",
        filename="TTT2_QHY600-4_2026-02-14-23-46-19-497908_WASP-43-b_SDSSg.fits",
        size_label="4787x3193", filter_name="SDSSg", megapixels=15.3,
        expected_sources=215, description="QHY600-4 4787x3193 SDSSg (WASP-43-b)",
    ),
    ReferenceImage(
        oblineid="5405929", instrument="QHY600M", camera="QHY600-4",
        filename="TTT2_QHY600-4_2026-02-14-00-13-51-310869_NGC2903_Ha.fits",
        size_label="4787x3193", filter_name="Ha", megapixels=15.3,
        expected_sources=375, description="QHY600-4 4787x3193 Ha (NGC2903)",
    ),
    ReferenceImage(
        oblineid="5597395", instrument="QHY411MERIS", camera="QHY411-1",
        filename="TTT1_QHY411-1_2026-03-09-21-22-48-661122_2012QD8_Lum.fits",
        size_label="7100x5325", filter_name="Lum", megapixels=37.8,
        expected_sources=155, description="QHY411-1 7100x5325 Lum (2012QD8)",
    ),
    ReferenceImage(
        oblineid="5406447", instrument="QHY411MERIS", camera="QHY411-1",
        filename="TTT1_QHY411-1_2026-02-14-03-52-12-471080_GaiaDR33534005919872722560_SDSSi.fits",
        size_label="7100x5325", filter_name="SDSSi", megapixels=37.8,
        expected_sources=245, description="QHY411-1 7100x5325 SDSSi (GaiaDR3...)",
    ),
    ReferenceImage(
        oblineid="3903693", instrument="QHY411MERIS", camera="QHY411-1",
        filename="TTT1_QHY411-1_2025-08-14-23-52-42-828197_2025PR1_Lum.fits",
        size_label="14200x10650", filter_name="Lum", megapixels=151.2,
        expected_sources=512, description="QHY411-1 14200x10650 Lum (2025PR1)",
    ),
    ReferenceImage(
        oblineid="5404937", instrument="QHY411MERIS", camera="QHY411-3",
        filename="TST_QHY411-3_2026-02-14-06-35-10-547282_24P_Lum.fits",
        size_label="14200x10650", filter_name="Lum", megapixels=151.2,
        expected_sources=19077, description="QHY411-3 14200x10650 Lum (24P)",
    ),
    ReferenceImage(
        oblineid="5412552", instrument="QHY411MERIS", camera="QHY411-3",
        filename="TST_QHY411-3_2026-02-14-23-09-41-463160_M81_SDSSr.fits",
        size_label="14200x10650", filter_name="SDSSr", megapixels=151.2,
        expected_sources=14239, description="QHY411-3 14200x10650 SDSSr (M81)",
    ),
]

# -- Local Docker profiler images ------------------------------------------
LOCAL_PROFILERS = {
    "312": {
        "image": "gpuphotfinal-profiler",
        "label": "Python 3.12 + cuML",
    },
    "38": {
        "image": "gpuphotfinal-profiler_38",
        "label": "Python 3.8 (legacy)",
    },
}

# -- Remote targets (edit to match your infrastructure) ---------------------
TARGETS: Dict[str, TargetMachine] = {
    "lenovo": TargetMachine(
        name="lenovo",
        ssh_host="lenovo_slemes",
        ssh_port=2223,
        ssh_opts="-o IdentitiesOnly=yes -o PasswordAuthentication=yes",
    ),
}

# Profiling parameters
DEFAULT_WARMUP = 1        # warmup iterations (discarded)
DEFAULT_REPETITIONS = 3   # measured iterations
PROFILE_TIMEOUT_S = 900   # timeout per profiling run (15 min, for 151MP images)

# GPU contention thresholds
GPU_LOAD_THRESHOLD_PCT = 15
GPU_TEMP_THRESHOLD_C = 50
GPU_MEM_USED_THRESHOLD_PCT = 20

# Local paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "benchmarks" / "benchmark_images"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "benchmark_results"
DEFAULT_CONFIGS_DIR = Path("/mnt/vast/samueltest/gpuphot/cameras_config")
DEFAULT_ASTRO_CACHE = PROJECT_ROOT / "tests" / "astronomy_cache"


# ---------------------------------------------------------------------------
# GPU state checking (local)
# ---------------------------------------------------------------------------
@dataclass
class GPUState:
    name: str = "Unknown"
    gpu_util: float = -1
    mem_used_mb: float = -1
    mem_total_mb: float = -1
    temperature: float = -1
    ok: bool = False
    reason: str = ""


def check_local_gpu() -> GPUState:
    """Check local GPU state via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return GPUState(reason=f"nvidia-smi failed: {result.stderr.strip()}")

        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        state = GPUState(
            name=parts[0], gpu_util=float(parts[1]),
            mem_used_mb=float(parts[2]), mem_total_mb=float(parts[3]),
            temperature=float(parts[4]),
        )
        issues = []
        if state.gpu_util > GPU_LOAD_THRESHOLD_PCT:
            issues.append(f"load {state.gpu_util:.0f}%")
        if state.temperature > GPU_TEMP_THRESHOLD_C:
            issues.append(f"temp {state.temperature:.0f}C")
        mem_pct = (state.mem_used_mb / state.mem_total_mb * 100) if state.mem_total_mb > 0 else 0
        if mem_pct > GPU_MEM_USED_THRESHOLD_PCT:
            issues.append(f"mem {mem_pct:.0f}%")
        state.ok = not issues
        state.reason = "; ".join(issues) if issues else "idle"
        return state
    except Exception as e:
        return GPUState(reason=str(e))


# ---------------------------------------------------------------------------
# Local execution via docker run
# ---------------------------------------------------------------------------
def run_local_profiling(
    docker_image: str,
    image: ReferenceImage,
    images_dir: Path,
    results_dir: Path,
    configs_dir: Path,
    astro_cache: Path,
    warmup: int,
    repetitions: int,
    verbose: bool = False,
) -> Tuple[bool, float]:
    """
    Run profiling locally via docker run.

    Returns (success, elapsed_seconds).
    """
    total_runs = warmup + repetitions
    image_path = images_dir / image.filename

    if not image_path.exists():
        print(f"    SKIP: image not found at {image_path}")
        return False, 0

    # Build docker run command
    docker_cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--privileged",
        "--cap-add", "SYS_ADMIN", "--cap-add", "SYS_PTRACE",
        "-v", f"{images_dir}:/data/images:z",
        "-v", f"{configs_dir}:/data/instrument_configs:z",
        "-v", f"{astro_cache}:/data/astrometry_cache:z",
        "-v", f"{results_dir}:/app/profiling_results:z",
        "-e", "INSTRUMENT_CONFIG_BASE_PATH=/data/instrument_configs",
        "-e", "IMAGE_BASE_PATH=/data/images",
        "-e", "GPU_ID=0",
        docker_image,
    ]

    # -- Warmup --
    if warmup > 0:
        print(f"    Warmup: {warmup} run(s)...")
        warmup_cmd = docker_cmd + [
            "python3", "/app/profiling_scripts/profile_image_processing.py",
            image.filename, image.instrument,
        ]
        for w in range(warmup):
            t0 = time.monotonic()
            result = subprocess.run(
                warmup_cmd, capture_output=True, text=True,
                timeout=PROFILE_TIMEOUT_S,
            )
            elapsed_w = time.monotonic() - t0
            if result.returncode != 0:
                print(f"    Warmup {w+1} FAILED ({elapsed_w:.1f}s)")
                if verbose:
                    for line in result.stderr.strip().split("\n")[-5:]:
                        print(f"      {line}")
                return False, 0
            # Extract time and objects from output
            for line in result.stdout.split("\n"):
                if "Time elapsed" in line:
                    print(f"      warmup {w+1}: {line.strip().split(': ', 1)[-1]} ({elapsed_w:.1f}s wall)")
                    break

    # -- Measured runs with nsys --
    print(f"    Measured: {repetitions} run(s) with nsys profiling...")
    nsys_cmd = docker_cmd + [
        "bash", "/app/profiling_scripts/run_profiling.sh",
        image.filename, image.instrument, str(repetitions),
    ]

    t0 = time.monotonic()
    result = subprocess.run(
        nsys_cmd, capture_output=True, text=True,
        timeout=PROFILE_TIMEOUT_S * repetitions,
    )
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        print(f"    FAILED after {elapsed:.1f}s")
        if verbose or True:  # always show errors
            stderr_lines = result.stderr.strip().split("\n")
            for line in stderr_lines[-10:]:
                print(f"      {line}")
        return False, elapsed

    # Parse timing from output
    times = []
    for line in result.stdout.split("\n"):
        if "Time elapsed" in line:
            time_str = line.strip().split(": ", 1)[-1]
            times.append(time_str)
        if verbose and line.strip():
            print(f"      {line.strip()}")

    for i, t_str in enumerate(times):
        print(f"      run {i+1}: {t_str}")

    return True, elapsed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GPUPhot Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--local", action="store_true",
                   help="Run benchmarks locally via docker run (no SSH)")
    p.add_argument("--run", action="store_true", default=True,
                   help="Execute profiling (default)")
    p.add_argument("--check-only", action="store_true",
                   help="Only check GPU state, don't profile")
    p.add_argument("--profilers", nargs="*", default=None,
                   choices=list(LOCAL_PROFILERS.keys()),
                   help=f"Which profilers to use (default: all). Options: {', '.join(LOCAL_PROFILERS.keys())}")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                   help=f"Warmup iterations before measurement (default: {DEFAULT_WARMUP})")
    p.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS,
                   help=f"Measured iterations per image (default: {DEFAULT_REPETITIONS})")
    p.add_argument("--max-megapixels", type=float, default=0,
                   help="Skip images larger than this (0 = no limit). "
                        "Use 20 for quick tests (skips 151MP images)")
    p.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR,
                   help=f"Directory with benchmark FITS images (default: {DEFAULT_IMAGES_DIR})")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                   help=f"Directory for profiling results (default: {DEFAULT_RESULTS_DIR})")
    p.add_argument("--configs-dir", type=Path, default=DEFAULT_CONFIGS_DIR,
                   help=f"Instrument config directory (default: {DEFAULT_CONFIGS_DIR})")
    p.add_argument("--astro-cache", type=Path, default=DEFAULT_ASTRO_CACHE,
                   help=f"Astrometry cache directory (default: {DEFAULT_ASTRO_CACHE})")
    p.add_argument("--skip-gpu-check", action="store_true",
                   help="Skip GPU contention check before profiling")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show detailed output")
    # Remote-only options
    p.add_argument("--targets", nargs="*", default=None,
                   help=f"Remote targets (for non-local mode). Available: {', '.join(TARGETS.keys())}")
    p.add_argument("--force-rerun", action="store_true",
                   help="Force re-profiling even if results exist")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    now = datetime.now(tz=timezone.utc)

    print("=" * 70)
    print("GPUPhot Benchmark Suite")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Mode: {'LOCAL (docker run)' if args.local else 'REMOTE (SSH)'}")
    print("=" * 70)

    # Select profilers
    profiler_keys = args.profilers or list(LOCAL_PROFILERS.keys())

    # Select images (optionally filter by size)
    images = REFERENCE_IMAGES
    if args.max_megapixels > 0:
        images = [img for img in images if img.megapixels <= args.max_megapixels]
        print(f"\n  Filtered to images <= {args.max_megapixels} MP: {len(images)} of {len(REFERENCE_IMAGES)}")

    print(f"  Images: {len(images)}")
    print(f"  Profilers: {', '.join(profiler_keys)}")
    print(f"  Warmup: {args.warmup}, Measured: {args.repetitions}")

    # ---- Phase 1: GPU State Check ----
    print(f"\n{'─' * 70}")
    print("Phase 1: GPU State Check")
    print(f"{'─' * 70}")

    if args.local:
        gpu_state = check_local_gpu()
        status = "OK" if gpu_state.ok else "!!"
        print(f"  [{status}] local  {gpu_state.name:30s}  "
              f"util={gpu_state.gpu_util:3.0f}%  temp={gpu_state.temperature:2.0f}C  "
              f"mem={gpu_state.mem_used_mb:.0f}/{gpu_state.mem_total_mb:.0f}MB  "
              f"[{gpu_state.reason}]")

        if not gpu_state.ok and not args.skip_gpu_check:
            print(f"\n  GPU contention detected. Use --skip-gpu-check to override.")
            if args.check_only:
                return
    else:
        print("  (remote mode — GPU checks happen per target)")

    if args.check_only:
        print("\nDone (--check-only).")
        return

    # ---- Phase 2: Image Availability ----
    print(f"\n{'─' * 70}")
    print("Phase 2: Image Availability")
    print(f"{'─' * 70}")

    available_images = []
    for img in images:
        path = args.images_dir / img.filename
        exists = path.exists()
        icon = "OK" if exists else "!!"
        size_str = f"{path.stat().st_size / 1e6:.0f}MB" if exists else "MISSING"
        print(f"  [{icon}] {img.description:50s}  {size_str}")
        if exists:
            available_images.append(img)

    if not available_images:
        print("\n  No images available. Run benchmarks/fetch_benchmark_images.sh first.")
        sys.exit(1)

    # ---- Phase 3: Run Profiling ----
    if args.local:
        print(f"\n{'─' * 70}")
        print(f"Phase 3: Local Profiling (warmup={args.warmup}, measured={args.repetitions})")
        print(f"{'─' * 70}")

        args.results_dir.mkdir(parents=True, exist_ok=True)
        results_log = []

        for pkey in profiler_keys:
            profiler = LOCAL_PROFILERS[pkey]
            print(f"\n  ┌─ Profiler: {profiler['label']} ({profiler['image']})")
            print(f"  │")

            # Create separate results subdirectory per profiler
            profiler_results = args.results_dir / f"py{pkey}"
            profiler_results.mkdir(parents=True, exist_ok=True)

            for i, img in enumerate(available_images, 1):
                print(f"  ├─ [{i}/{len(available_images)}] {img.description}")
                print(f"  │   {img.size_label} | {img.megapixels:.1f} MP | ~{img.expected_sources} sources")

                success, elapsed = run_local_profiling(
                    docker_image=profiler["image"],
                    image=img,
                    images_dir=args.images_dir,
                    results_dir=profiler_results,
                    configs_dir=args.configs_dir,
                    astro_cache=args.astro_cache,
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                    verbose=args.verbose,
                )

                status = "OK" if success else "FAIL"
                print(f"  │   -> {status} ({elapsed:.1f}s total)")

                results_log.append({
                    "profiler": pkey,
                    "profiler_label": profiler["label"],
                    "image": img.filename,
                    "oblineid": img.oblineid,
                    "camera": img.camera,
                    "size": img.size_label,
                    "filter": img.filter_name,
                    "megapixels": img.megapixels,
                    "success": success,
                    "elapsed_s": round(elapsed, 1),
                    "timestamp": now.isoformat(),
                })

            print(f"  └─ Done ({profiler['label']})")

        # Save run log
        log_file = args.results_dir / "benchmark_run_log.json"
        with open(log_file, "w") as f:
            json.dump({
                "metadata": {
                    "mode": "local",
                    "gpu": gpu_state.name if args.local else "unknown",
                    "warmup": args.warmup,
                    "repetitions": args.repetitions,
                    "timestamp": now.isoformat(),
                },
                "runs": results_log,
            }, f, indent=2)
        print(f"\n  Run log saved to {log_file}")

        # Summary
        print(f"\n{'─' * 70}")
        print("Summary")
        print(f"{'─' * 70}")
        ok_count = sum(1 for r in results_log if r["success"])
        fail_count = sum(1 for r in results_log if not r["success"])
        print(f"  Total runs: {len(results_log)} ({ok_count} OK, {fail_count} FAILED)")
        print(f"  Results in: {args.results_dir}/")

        # List generated files
        for pkey in profiler_keys:
            pdir = args.results_dir / f"py{pkey}"
            nsys_files = sorted(pdir.glob("*.nsys-rep"))
            sqlite_files = sorted(pdir.glob("*.sqlite"))
            print(f"    py{pkey}/: {len(nsys_files)} .nsys-rep, {len(sqlite_files)} .sqlite")

    print(f"\n{'=' * 70}")
    print("Benchmark suite complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
