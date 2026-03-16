#!/usr/bin/env python3
"""
Automated Benchmark Suite for GPUPhot Profiling.

Connects to profiler Docker containers on multiple machines, checks GPU state,
runs warmup + measured profiling iterations, and collects aggregated NVTX
results for publication-quality analysis.

Architecture
------------
Local machine (this script)
  └─ SSH ─→ Remote profiler Docker container (port 2222/2223)
               ├─ nvidia-smi  →  GPU state check
               ├─ nsys profile →  warmup iterations (discarded)
               ├─ nsys profile →  measured iterations
               └─ NVTX extraction → aggregated CSV (stdout)

Modes
-----
- ``--run``           : Execute profiling on all targets (default).
- ``--analyze-only``  : Skip profiling, only re-analyze existing results.
- ``--force-rerun``   : Force re-profiling even if results exist.
- ``--check-only``    : Only check GPU state on all targets.

Reference Images
----------------
Each camera/instrument has a canonical test image defined in
``REFERENCE_IMAGES``.  These images must be accessible inside the
profiler container at ``/data/images/<path>``.

Usage
-----
::

    # Check GPU status on all machines
    python profiling_scripts/run_benchmark_suite.py --check-only

    # Run full benchmark suite on all configured targets
    python profiling_scripts/run_benchmark_suite.py --run

    # Analyze existing results without re-running
    python profiling_scripts/run_benchmark_suite.py --analyze-only

    # Force re-run on a specific target
    python profiling_scripts/run_benchmark_suite.py --run --force-rerun --targets lenovo

    # Custom warmup and repetitions
    python profiling_scripts/run_benchmark_suite.py --run --warmup 3 --repetitions 5
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

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
    instrument: str
    camera: str
    filename: str
    description: str = ""


# -- Target machines (edit to match your infrastructure) -------------------
# ssh_host can be a hostname, alias from ~/.ssh/config, or user@host.
# ssh_port is the profiler container's exposed SSH port.
TARGETS: Dict[str, TargetMachine] = {
    "lenovo": TargetMachine(
        name="lenovo",
        ssh_host="lenovo_slemes",
        ssh_port=2223,
        ssh_opts="-o IdentitiesOnly=yes -o PasswordAuthentication=yes",
        results_path="/app/profiling_results",
    ),
    # Add more targets as needed:
    # "icr": TargetMachine(
    #     name="icr", ssh_host="icr_slemes", ssh_port=2222,
    # ),
    # "hp1": TargetMachine(
    #     name="hp1", ssh_host="hp1_slemes", ssh_port=2222,
    # ),
}

# -- Reference images (must exist at <images_path>/<filename> inside container)
REFERENCE_IMAGES: List[ReferenceImage] = [
    ReferenceImage(
        instrument="iKon936",
        camera="iKon-L 936",
        filename="iKon936/2024-01-15_SDSSi_001.fits",
        description="iKon936 SDSSi - moderate star density",
    ),
    ReferenceImage(
        instrument="QHY411MERIS",
        camera="QHY411MERIS",
        filename="QHY411MERIS/2024-02-20_Lum_001.fits",
        description="QHY411M full-frame - high star density",
    ),
    ReferenceImage(
        instrument="QHY600M",
        camera="QHY600M",
        filename="QHY600M/2024-03-10_SDSSr_001.fits",
        description="QHY600M - medium field",
    ),
]

# Thresholds for GPU contention detection
GPU_LOAD_THRESHOLD_PCT = 15       # max acceptable GPU utilization %
GPU_TEMP_THRESHOLD_C = 45         # max acceptable idle GPU temperature
GPU_MEM_USED_THRESHOLD_PCT = 20   # max acceptable GPU memory usage %

# Profiling parameters
DEFAULT_WARMUP = 2        # warmup iterations (discarded)
DEFAULT_REPETITIONS = 3   # measured iterations
PROFILE_TIMEOUT_S = 600   # timeout per profiling run (10 min)


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------
def _build_ssh_cmd(target: TargetMachine, remote_cmd: str) -> List[str]:
    """Build an SSH command list for a target."""
    cmd = ["ssh"]
    if target.ssh_port != 22:
        cmd.extend(["-p", str(target.ssh_port)])
    if target.ssh_opts:
        cmd.extend(shlex.split(target.ssh_opts))
    cmd.append(target.ssh_host)
    cmd.append(remote_cmd)
    return cmd


def _ssh_pipe_script(target: TargetMachine, script: str,
                     args: str = "", timeout: int = 60) -> subprocess.CompletedProcess:
    """Execute a Python script on a remote machine via SSH stdin piping."""
    cmd = ["ssh"]
    if target.ssh_port != 22:
        cmd.extend(["-p", str(target.ssh_port)])
    if target.ssh_opts:
        cmd.extend(shlex.split(target.ssh_opts))
    cmd.extend([target.ssh_host, "python3", "-"])
    if args:
        cmd.extend(args.split())
    return subprocess.run(cmd, input=script, capture_output=True,
                          text=True, timeout=timeout)


def _ssh_run(target: TargetMachine, remote_cmd: str,
             timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a command on a remote machine via SSH."""
    cmd = _build_ssh_cmd(target, remote_cmd)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------------------
# GPU state checking
# ---------------------------------------------------------------------------
@dataclass
class GPUState:
    """GPU status from nvidia-smi."""
    name: str = "Unknown"
    gpu_util: float = -1
    mem_used_mb: float = -1
    mem_total_mb: float = -1
    temperature: float = -1
    ok: bool = False
    reason: str = ""


def check_gpu_state(target: TargetMachine) -> GPUState:
    """Query GPU state via nvidia-smi on the remote machine."""
    query = "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"
    try:
        result = _ssh_run(target, query, timeout=30)
        if result.returncode != 0:
            return GPUState(reason=f"nvidia-smi failed: {result.stderr.strip()}")

        line = result.stdout.strip().split("\n")[target.gpu_id]
        parts = [p.strip() for p in line.split(",")]
        state = GPUState(
            name=parts[0],
            gpu_util=float(parts[1]),
            mem_used_mb=float(parts[2]),
            mem_total_mb=float(parts[3]),
            temperature=float(parts[4]),
        )

        # Check thresholds
        issues = []
        if state.gpu_util > GPU_LOAD_THRESHOLD_PCT:
            issues.append(f"GPU load {state.gpu_util:.0f}% > {GPU_LOAD_THRESHOLD_PCT}%")
        if state.temperature > GPU_TEMP_THRESHOLD_C:
            issues.append(f"temp {state.temperature:.0f}C > {GPU_TEMP_THRESHOLD_C}C")
        mem_pct = (state.mem_used_mb / state.mem_total_mb * 100) if state.mem_total_mb > 0 else 0
        if mem_pct > GPU_MEM_USED_THRESHOLD_PCT:
            issues.append(f"mem {mem_pct:.0f}% > {GPU_MEM_USED_THRESHOLD_PCT}%")

        if issues:
            state.reason = "; ".join(issues)
        else:
            state.ok = True
            state.reason = "idle"

        return state

    except subprocess.TimeoutExpired:
        return GPUState(reason="SSH timeout (30s)")
    except Exception as e:
        return GPUState(reason=str(e))


# ---------------------------------------------------------------------------
# Image availability checking
# ---------------------------------------------------------------------------
def check_images_available(target: TargetMachine,
                           images: List[ReferenceImage]) -> Dict[str, bool]:
    """Check which reference images exist on the remote machine."""
    paths = [f"{target.images_path}/{img.filename}" for img in images]
    check_cmd = " && ".join(f'[ -f "{p}" ] && echo "OK:{p}" || echo "MISSING:{p}"'
                            for p in paths)
    try:
        result = _ssh_run(target, check_cmd, timeout=15)
        status = {}
        for line in result.stdout.strip().split("\n"):
            if line.startswith("OK:"):
                status[line[3:]] = True
            elif line.startswith("MISSING:"):
                status[line[8:]] = False
        return status
    except Exception as e:
        return {p: False for p in paths}


# ---------------------------------------------------------------------------
# Result management
# ---------------------------------------------------------------------------
def _results_key(target_name: str, instrument: str) -> str:
    """Unique key for a (target, image) benchmark result."""
    return f"{target_name}__{instrument}"


def load_existing_results(results_file: Path) -> Dict[str, dict]:
    """Load previously collected benchmark results."""
    if not results_file.exists():
        return {}
    try:
        with open(results_file) as f:
            data = json.load(f)
        return {r["key"]: r for r in data.get("results", [])}
    except Exception:
        return {}


def save_results(results_file: Path, results: Dict[str, dict],
                 metadata: dict) -> None:
    """Save benchmark results to JSON."""
    data = {
        "metadata": metadata,
        "results": list(results.values()),
    }
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Profiling execution
# ---------------------------------------------------------------------------
def run_profiling(target: TargetMachine, image: ReferenceImage,
                  warmup: int, repetitions: int,
                  verbose: bool = False) -> Optional[str]:
    """
    Execute profiling on a remote machine.

    Runs ``warmup`` iterations first (results discarded), then
    ``repetitions`` measured iterations.

    Returns the output filename of the last measured run, or None on failure.
    """
    image_path = f"{target.images_path}/{image.filename}"
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    # -- Warmup runs --
    if warmup > 0:
        print(f"    Warmup: {warmup} iteration(s)...")
        warmup_cmd = (
            f"cd /app && "
            f"CUDA_VISIBLE_DEVICES={target.gpu_id} "
            f"bash profiling_scripts/run_profiling.sh "
            f'"{image_path}" "{image.instrument}" {warmup}'
        )
        try:
            result = _ssh_run(target, warmup_cmd, timeout=PROFILE_TIMEOUT_S * warmup)
            if result.returncode != 0:
                print(f"    Warmup failed: {result.stderr.strip()[:200]}")
                return None
            if verbose:
                for line in result.stdout.strip().split("\n"):
                    print(f"      {line}")
            # Delete warmup results to avoid clutter
            _ssh_run(target, f"rm -f {target.results_path}/profile_*_run*.nsys-rep "
                     f"2>/dev/null || true", timeout=10)
        except subprocess.TimeoutExpired:
            print(f"    Warmup timed out ({PROFILE_TIMEOUT_S * warmup}s)")
            return None

    # -- Measured runs --
    print(f"    Profiling: {repetitions} measured iteration(s)...")
    profile_cmd = (
        f"cd /app && "
        f"CUDA_VISIBLE_DEVICES={target.gpu_id} "
        f"bash profiling_scripts/run_profiling.sh "
        f'"{image_path}" "{image.instrument}" {repetitions}'
    )
    try:
        result = _ssh_run(target, profile_cmd, timeout=PROFILE_TIMEOUT_S * repetitions)
        if result.returncode != 0:
            print(f"    Profiling failed: {result.stderr.strip()[:200]}")
            return None
        if verbose:
            for line in result.stdout.strip().split("\n"):
                print(f"      {line}")

        # Find the generated .nsys-rep file(s)
        ls_result = _ssh_run(target,
                             f"ls -t {target.results_path}/profile_*.nsys-rep 2>/dev/null | head -1",
                             timeout=10)
        last_file = ls_result.stdout.strip()
        if last_file:
            print(f"    Result: {Path(last_file).name}")
            return last_file
        else:
            print("    Warning: No .nsys-rep file found after profiling")
            return None

    except subprocess.TimeoutExpired:
        print(f"    Profiling timed out ({PROFILE_TIMEOUT_S * repetitions}s)")
        return None


# ---------------------------------------------------------------------------
# NVTX data extraction (reuses analyze_profiling.py logic)
# ---------------------------------------------------------------------------
EXTRACT_SCRIPT = r'''
import csv, glob, io, sqlite3, sys, socket
from pathlib import Path

profiling_dir = sys.argv[1]
writer = csv.writer(sys.stdout)
writer.writerow(["source_host", "source_file", "gpu_name", "gpu_uuid",
                 "command_args", "nvtx_name", "time_mean", "time_std",
                 "time_min", "time_max", "time_count"])

hostname = socket.gethostname()

for db_path in sorted(glob.glob(str(Path(profiling_dir) / "*.sqlite"))):
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}
        if "NVTX_EVENTS" not in tables or "StringIds" not in tables:
            conn.close(); continue
        cur.execute("SELECT COUNT(*) FROM NVTX_EVENTS WHERE eventType=59 AND end IS NOT NULL AND start IS NOT NULL AND (end-start)>0")
        if cur.fetchone()[0] < 10:
            conn.close(); continue

        gpu_name, gpu_uuid = "Unknown GPU", ""
        if "TARGET_INFO_GPU" in tables:
            cur.execute("SELECT name, uuid FROM TARGET_INFO_GPU LIMIT 1")
            row = cur.fetchone()
            if row:
                gpu_name = (row[0] or "").strip("[] ")
                gpu_uuid = row[1] or ""

        command_args = "unknown"
        for tbl, query in [
            ("ANALYSIS_DETAILS", "SELECT value FROM ANALYSIS_DETAILS WHERE LOWER(key) LIKE 'argument%' ORDER BY CAST(SUBSTR(key, 9) AS INTEGER)"),
            ("META_DATA_CAPTURE", "SELECT value FROM META_DATA_CAPTURE WHERE name LIKE 'PROCESS_%:ARGUMENT_%' ORDER BY CAST(SUBSTR(name, INSTR(name, ':ARGUMENT_') + 10) AS INTEGER)")
        ]:
            if command_args != "unknown":
                break
            if tbl not in tables:
                continue
            try:
                cur.execute(query)
                args = [r[0] for r in cur.fetchall()]
                relevant, found = [], False
                for a in args:
                    if "profile_image_processing" in str(a):
                        found = True; continue
                    if found:
                        relevant.append(str(a))
                if relevant:
                    command_args = " ".join(relevant)
            except Exception:
                pass

        cur.execute("""
            SELECT t2.value, (t1.end - t1.start) / 1e6
            FROM NVTX_EVENTS t1 JOIN StringIds t2 ON t1.textId = t2.id
            WHERE t1.eventType = 59 AND t1.end IS NOT NULL
              AND t1.start IS NOT NULL AND (t1.end - t1.start) > 0
        """)
        events = {}
        for name, dur in cur.fetchall():
            events.setdefault(name, []).append(dur)
        conn.close()

        fname = Path(db_path).name
        for name, durs in events.items():
            n = len(durs)
            mean = sum(durs) / n
            std = (sum((d - mean)**2 for d in durs) / max(n - 1, 1)) ** 0.5 if n > 1 else 0
            writer.writerow([hostname, fname, gpu_name, gpu_uuid, command_args,
                             name, f"{mean:.6f}", f"{std:.6f}",
                             f"{min(durs):.6f}", f"{max(durs):.6f}", n])
    except Exception as e:
        print(f"WARN: {Path(db_path).name}: {e}", file=sys.stderr)
'''


def extract_nvtx_results(target: TargetMachine) -> Optional[str]:
    """
    Extract aggregated NVTX data from the remote machine.

    Returns CSV string, or None on failure.
    """
    try:
        result = _ssh_pipe_script(target, EXTRACT_SCRIPT,
                                  args=target.results_path, timeout=120)
        if result.returncode != 0:
            print(f"    Extraction error: {result.stderr.strip()[:200]}")
            return None

        for line in result.stderr.strip().split("\n"):
            if line.strip():
                print(f"    [{target.name}] {line}")

        if not result.stdout.strip():
            print(f"    No NVTX data returned from {target.name}")
            return None

        return result.stdout

    except subprocess.TimeoutExpired:
        print(f"    Extraction timed out (120s)")
        return None
    except Exception as e:
        print(f"    Extraction error: {e}")
        return None


# ---------------------------------------------------------------------------
# Export to CSV (for analyze_profiling.py consumption)
# ---------------------------------------------------------------------------
def export_aggregated_csv(all_csv_data: List[str], output_path: Path) -> None:
    """Merge all extracted CSV data into a single file."""
    import pandas as pd

    dfs = []
    for csv_str in all_csv_data:
        try:
            df = pd.read_csv(io.StringIO(csv_str))
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        print("  No data to export.")
        return

    merged = pd.concat(dfs, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"\n  Exported aggregated data: {output_path}")
    print(f"  {len(merged)} rows, {merged['source_file'].nunique()} files, "
          f"GPUs: {', '.join(merged['gpu_name'].unique())}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Automated GPUPhot Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", default=True,
                      help="Execute profiling on all targets (default)")
    mode.add_argument("--analyze-only", action="store_true",
                      help="Skip profiling, only extract & analyze existing results")
    mode.add_argument("--check-only", action="store_true",
                      help="Only check GPU state on all targets")

    p.add_argument("--targets", nargs="*", default=None,
                   help=f"Target machines to use (default: all). "
                        f"Available: {', '.join(TARGETS.keys())}")
    p.add_argument("--force-rerun", action="store_true",
                   help="Force re-profiling even if results already exist")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                   help=f"Warmup iterations before measurement (default: {DEFAULT_WARMUP})")
    p.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS,
                   help=f"Measured profiling iterations (default: {DEFAULT_REPETITIONS})")
    p.add_argument("--outdir", type=str, default="benchmarks",
                   help="Output directory for aggregated CSV (default: benchmarks)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show detailed SSH output")
    p.add_argument("--skip-gpu-check", action="store_true",
                   help="Skip GPU contention check before profiling")
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
    print("=" * 70)

    # Select targets
    if args.targets:
        targets = {k: v for k, v in TARGETS.items() if k in args.targets}
        unknown = set(args.targets) - set(TARGETS.keys())
        if unknown:
            print(f"  Warning: unknown targets: {', '.join(unknown)}")
            print(f"  Available: {', '.join(TARGETS.keys())}")
    else:
        targets = {k: v for k, v in TARGETS.items() if v.enabled}

    if not targets:
        print("  No targets configured. Edit TARGETS in this script.")
        sys.exit(1)

    print(f"\n  Targets: {', '.join(targets.keys())}")
    print(f"  Reference images: {len(REFERENCE_IMAGES)}")

    # ---- Phase 1: Check GPU state ----
    print(f"\n{'─' * 70}")
    print("Phase 1: GPU State Check")
    print(f"{'─' * 70}")

    gpu_states: Dict[str, GPUState] = {}
    for name, target in targets.items():
        state = check_gpu_state(target)
        gpu_states[name] = state
        status_icon = "OK" if state.ok else "!!"
        print(f"  [{status_icon}] {name:12s}  {state.name:30s}  "
              f"util={state.gpu_util:3.0f}%  temp={state.temperature:2.0f}C  "
              f"mem={state.mem_used_mb:.0f}/{state.mem_total_mb:.0f}MB  "
              f"[{state.reason}]")

    if args.check_only:
        print("\nDone (--check-only mode).")
        return

    # ---- Phase 2: Check image availability ----
    print(f"\n{'─' * 70}")
    print("Phase 2: Reference Image Availability")
    print(f"{'─' * 70}")

    for name, target in targets.items():
        img_status = check_images_available(target, REFERENCE_IMAGES)
        for img in REFERENCE_IMAGES:
            full_path = f"{target.images_path}/{img.filename}"
            available = img_status.get(full_path, False)
            icon = "OK" if available else "!!"
            print(f"  [{icon}] {name:12s}  {img.instrument:20s}  {img.filename}")

    # ---- Phase 3: Run profiling (unless --analyze-only) ----
    if not args.analyze_only:
        print(f"\n{'─' * 70}")
        print(f"Phase 3: Profiling (warmup={args.warmup}, measured={args.repetitions})")
        print(f"{'─' * 70}")

        results_file = Path(args.outdir) / "benchmark_results.json"
        existing = load_existing_results(results_file)

        for name, target in targets.items():
            state = gpu_states.get(name)

            if not args.skip_gpu_check and state and not state.ok:
                print(f"\n  [{name}] SKIPPED: GPU contention detected ({state.reason})")
                print(f"  Use --skip-gpu-check to override.")
                continue

            for img in REFERENCE_IMAGES:
                key = _results_key(name, img.instrument)
                if key in existing and not args.force_rerun:
                    print(f"\n  [{name}] {img.instrument}: already profiled "
                          f"({existing[key].get('timestamp', 'unknown')})")
                    print(f"  Use --force-rerun to re-profile.")
                    continue

                print(f"\n  [{name}] {img.instrument}: {img.description}")

                # Check image exists
                img_status = check_images_available(target, [img])
                full_path = f"{target.images_path}/{img.filename}"
                if not img_status.get(full_path, False):
                    print(f"    SKIPPED: image not available at {full_path}")
                    continue

                # Run profiling
                t0 = time.monotonic()
                nsys_file = run_profiling(target, img, args.warmup,
                                          args.repetitions, verbose=args.verbose)
                elapsed = time.monotonic() - t0

                if nsys_file:
                    existing[key] = {
                        "key": key,
                        "target": name,
                        "instrument": img.instrument,
                        "camera": img.camera,
                        "filename": img.filename,
                        "gpu_name": state.name if state else "Unknown",
                        "nsys_file": nsys_file,
                        "warmup": args.warmup,
                        "repetitions": args.repetitions,
                        "elapsed_s": round(elapsed, 1),
                        "timestamp": now.isoformat(),
                    }

        # Save run metadata
        save_results(results_file, existing, {
            "suite_version": "1.0",
            "last_run": now.isoformat(),
            "warmup": args.warmup,
            "repetitions": args.repetitions,
        })
        print(f"\n  Results metadata saved to {results_file}")

    # ---- Phase 4: Extract NVTX data ----
    print(f"\n{'─' * 70}")
    print("Phase 4: NVTX Data Extraction")
    print(f"{'─' * 70}")

    all_csv: List[str] = []
    for name, target in targets.items():
        print(f"\n  Extracting from {name}...")
        csv_data = extract_nvtx_results(target)
        if csv_data:
            all_csv.append(csv_data)

    if all_csv:
        output_csv = Path(args.outdir) / "benchmark_nvtx_aggregated.csv"
        export_aggregated_csv(all_csv, output_csv)

        # ---- Phase 5: Run analysis ----
        print(f"\n{'─' * 70}")
        print("Phase 5: Analysis")
        print(f"{'─' * 70}")

        analyze_script = Path(__file__).resolve().parent.parent / "benchmarks" / "analyze_profiling.py"
        if analyze_script.exists():
            print(f"  Running {analyze_script.name}...")
            result = subprocess.run(
                [sys.executable, str(analyze_script),
                 "--csv", str(output_csv),
                 "--outdir", args.outdir],
                timeout=120,
            )
            if result.returncode != 0:
                print(f"  Analysis exited with code {result.returncode}")
        else:
            print(f"  Warning: {analyze_script} not found. Run manually:")
            print(f"  python benchmarks/analyze_profiling.py --csv {output_csv}")
    else:
        print("\n  No NVTX data extracted. Nothing to analyze.")

    print(f"\n{'=' * 70}")
    print("Benchmark suite complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
