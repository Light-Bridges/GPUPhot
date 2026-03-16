#!/usr/bin/env python3
"""
Analyze NVTX Profiling Data from GPUPhot Pipeline.

This script processes NVTX timing data extracted from Nsight Systems reports
(pre-aggregated CSV or raw SQLite databases) to produce publication-quality
analysis of per-stage GPU pipeline performance.

Unlike the end-to-end ``execution_time`` captured in Elasticsearch (which
includes network I/O for catalog queries and astrometry), this analysis
isolates the contribution of each computational stage and classifies them
as GPU-bound, CPU-bound, or I/O-bound.

Pipeline
--------
1. **Load** -- read pre-aggregated CSV or extract from SQLite databases.
2. **Classify stages** -- categorize each NVTX event as GPU, CPU, I/O, or overhead.
3. **Hierarchical breakdown** -- compute time contribution of each stage
   relative to the top-level ``process_image`` duration.
4. **Statistical summary** -- robust statistics per stage (median, IQR, CI).
5. **Figure generation** -- stacked bar charts, waterfall, pie charts.
6. **LaTeX table export** -- tables ready for the manuscript.

Usage
-----
::

    # From pre-aggregated CSV (output of nvtx_stats.py)
    python benchmarks/analyze_profiling.py --csv dev/profiling_results/analysis_aggregated_data.csv

    # From raw SQLite databases
    python benchmarks/analyze_profiling.py --sqlite profiling_results/profile_*.sqlite

    # Both sources combined
    python benchmarks/analyze_profiling.py \\
        --csv dev/profiling_results/analysis_aggregated_data.csv \\
        --sqlite profiling_results/profile_orin_*.sqlite
"""

from __future__ import annotations

import argparse
import glob
import io
import sqlite3
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIGURE_DIR = "figures"
TABLE_DIR = "tables"

# NVTX event classification: maps event names to categories.
# Events not listed here are classified as "other".
STAGE_CLASSIFICATION: Dict[str, Tuple[str, str]] = {
    # --- GPU-bound stages ---
    "get_local_background_fft":     ("GPU",  "Background estimation (FFT)"),
    "convolve_fft":                 ("GPU",  "FFT convolution"),
    "batch_aperture_photometry":    ("GPU",  "Batch aperture photometry"),
    "detect_sources_psf":           ("GPU",  "PSF source detection"),
    "detect_isolated_stars":        ("GPU",  "Isolated star detection"),
    "create_star_dataset":          ("GPU",  "PSF cutout extraction"),
    "get_mean_std":                 ("GPU",  "Image statistics"),
    "fill_nan_fft":                 ("GPU",  "NaN fill (FFT)"),
    "calculate_tile_percentiles":   ("GPU",  "Tile percentiles"),
    "crossmatch_sources":           ("GPU",  "Source crossmatching"),
    "crossmatch_sources_gpu_impl":  ("GPU",  "Source crossmatching (GPU)"),
    "get_aper_kernel":              ("GPU",  "Aperture kernel generation"),
    "gen_apm_filter":               ("GPU",  "Aperture mask filter"),
    "find_local_max":               ("GPU",  "Local maxima detection"),
    "find_local_centroid":          ("GPU",  "Centroid refinement"),
    "gaussian_kernel":              ("GPU",  "Gaussian kernel"),
    "stack_sigmaclip":              ("GPU",  "Sigma-clipped stacking"),
    "calculate_kernel_area":        ("GPU",  "Kernel area calculation"),
    "decompose_into_tiles":         ("GPU",  "Tile decomposition"),
    "recompose_from_percentiles":   ("GPU",  "Tile recomposition"),
    "fill_image":                   ("GPU",  "Image fill"),
    "moffat":                       ("GPU",  "Moffat profile evaluation"),

    # --- GPU/CPU stages (PCA and coefficient maps) ---
    "get_eigen_psfs":               ("GPU",  "Eigen-PSF computation (PCA)"),
    "project_all_stars_onto_eigenpsfs": ("GPU", "Star projection onto eigen-PSFs"),
    "create_coeff_map":             ("GPU",  "Coefficient map creation"),
    "calculate_tile_nanmean_sigclip": ("GPU", "Tile NaN-mean sigma clipping"),

    # --- GPU composite stages (contain sub-stages) ---
    "perform_opt_photometry":                          ("GPU",  "Optimal photometry (total)"),
    "perform_opt_photometry_optimized_gpu_crossmatch": ("GPU",  "Optimal photometry (GPU crossmatch)"),
    "create_aperture_corrections_map":                 ("GPU",  "Aperture corrections map"),
    "create_aperture_corrections_map_gpu":             ("GPU",  "Aperture corrections map (GPU)"),
    "find_aperture_corrections":                       ("GPU",  "Find aperture corrections"),
    "find_aperture_corrections_gpu":                   ("GPU",  "Find aperture corrections (GPU)"),
    "calculate_aperture_corrections":                  ("GPU",  "Calculate aperture corrections"),

    # --- CPU-bound stages ---
    "fit_moffat":                   ("CPU",  "Moffat PSF fitting"),
    "moffat_fwhm":                  ("CPU",  "FWHM calculation"),
    "group_star_dataset":           ("CPU",  "Star clustering (sklearn)"),
    "crossmatch_sources_cpu_impl":  ("CPU",  "Source crossmatching (CPU)"),
    "get_astrometry_params":        ("CPU",  "WCS parameter extraction"),
    "get_zeropoint":                ("CPU",  "Zero-point calculation"),
    "get_target_snr":               ("CPU",  "Target SNR lookup"),
    "get_maglim":                   ("CPU",  "Limiting magnitude"),
    "get_scale":                    ("CPU",  "Plate scale"),
    "get_ccw":                      ("CPU",  "Rotation angle"),
    "plate_scale_px":               ("CPU",  "Pixel plate scale"),
    "plate_scale_mm":               ("CPU",  "Physical plate scale"),
    "date_to_jd":                   ("CPU",  "Date to Julian Day"),
    "deg_to_hms":                   ("CPU",  "Coordinate conversion"),
    "delete_header_from":           ("CPU",  "FITS header cleanup"),
    "update_header_with_photometry": ("CPU", "Header update (photometry)"),
    "get_if_header_already_post_processed": ("CPU", "Header check"),

    # --- I/O and network stages ---
    "update_header_with_astrometry": ("I/O",  "Astrometry (local solver)"),
    "astrometrice2":                 ("I/O",  "Astrometry solving"),
    "catalog_results":               ("I/O",  "Catalog query"),
    "__getVizier":                   ("I/O",  "Vizier query"),
    "radec_to_altaz":                ("I/O",  "AltAz conversion (astropy)"),
    "radec_to_gal":                  ("I/O",  "Galactic coords (astropy)"),
    "radec_to_ecl":                  ("I/O",  "Ecliptic coords (astropy)"),
    "radec_to_moon_sun":             ("I/O",  "Moon/Sun distance (astropy)"),
    "initialize_solver":             ("I/O",  "Astrometry solver init"),
    "get_solver":                    ("I/O",  "Solver lookup"),
    "check_index_files_exist":       ("I/O",  "Index file check"),

    # --- Overhead / memory management ---
    "CR_filter":                    ("CPU",  "Cosmic ray filter"),
    "SP_filter":                    ("CPU",  "Salt & pepper filter"),
    "reset_cupy_allocators":        ("Overhead", "CuPy allocator reset"),
    "free_gpu_mem":                 ("Overhead", "GPU memory free"),
    "maybe_free_arrays":            ("Overhead", "Array cleanup"),
}

# Top-level container events (not leaf stages, used for hierarchy)
CONTAINER_EVENTS = {
    "process_image", "calibrate_image", "ImageProcessor.process_image",
    "create_processor",
}

# Events from CuML / CUB internals (aggregate as "GPU library internals")
CUB_CUML_PREFIXES = ("cub::", "internals.", "common.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze NVTX profiling data for GPUPhot pipeline"
    )
    p.add_argument(
        "--csv", nargs="*", default=[],
        help="Pre-aggregated CSV file(s) from nvtx_stats.py",
    )
    p.add_argument(
        "--sqlite", nargs="*", default=[],
        help="Raw SQLite database(s) from Nsight Systems. Supports wildcards.",
    )
    p.add_argument(
        "--remote", nargs="*", default=[], metavar="HOST:PATH",
        help="Remote profiling dirs via SSH. Formats: "
             "host:/path, host:port:/path, user@host:port:/path. "
             "Extracts NVTX data remotely without downloading large files. "
             "Example: --remote lenovo_slemes:2223:/mnt/vast/samueltest/gpuphot/profiling_results",
    )
    p.add_argument(
        "--ssh-opts", type=str, default="",
        help="Extra SSH options passed verbatim. "
             "Example: --ssh-opts='-o IdentitiesOnly=yes -o PasswordAuthentication=yes'",
    )
    p.add_argument(
        "--no-figures", action="store_true",
        help="Skip figure generation.",
    )
    p.add_argument(
        "--outdir", type=str, default=".",
        help="Base output directory for figures/ and tables/ subdirs.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_aggregated_csv(csv_files: List[str]) -> pd.DataFrame:
    """Load pre-aggregated CSV(s) from nvtx_stats.py."""
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = Path(f).name
            dfs.append(df)
            print(f"  Loaded {len(df)} rows from {f}")
        except Exception as e:
            print(f"  Warning: Could not read {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def extract_from_sqlite(sqlite_files: List[str]) -> pd.DataFrame:
    """Extract NVTX event data from raw SQLite databases."""
    expanded: list[str] = []
    for pattern in sqlite_files:
        found = glob.glob(pattern)
        expanded.extend(found if found else [pattern])
    expanded = sorted(set(expanded))

    all_records = []
    for db_path in expanded:
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()

            # Check required tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {r[0] for r in cursor.fetchall()}
            if "NVTX_EVENTS" not in tables or "StringIds" not in tables:
                print(f"  Skipping {db_path}: missing NVTX tables")
                conn.close()
                continue

            # Count events
            cursor.execute("SELECT COUNT(*) FROM NVTX_EVENTS WHERE eventType=59 AND end IS NOT NULL")
            n_events = cursor.fetchone()[0]
            if n_events < 10:
                conn.close()
                continue

            # Extract GPU info
            gpu_name = "Unknown GPU"
            gpu_uuid = None
            if "TARGET_INFO_GPU" in tables:
                cursor.execute("SELECT name, uuid FROM TARGET_INFO_GPU LIMIT 1")
                row = cursor.fetchone()
                if row:
                    gpu_name = row[0].strip("[] ") if row[0] else "Unknown GPU"
                    gpu_uuid = row[1]

            # Extract NVTX events
            cursor.execute("""
                SELECT t2.value AS nvtx_name,
                       (t1.end - t1.start) / 1e6 AS duration_ms
                FROM NVTX_EVENTS t1
                JOIN StringIds t2 ON t1.textId = t2.id
                WHERE t1.eventType = 59
                  AND t1.end IS NOT NULL
                  AND t1.start IS NOT NULL
                  AND (t1.end - t1.start) > 0
            """)
            events = cursor.fetchall()

            # Extract command args for identification
            command_args = _extract_command_args(cursor, tables)

            conn.close()

            # Aggregate per event name
            event_df = pd.DataFrame(events, columns=["nvtx_name", "duration_ms"])
            agg = event_df.groupby("nvtx_name")["duration_ms"].agg(
                time_mean="mean", time_std="std", time_min="min",
                time_max="max", time_count="count"
            ).reset_index()
            agg["time_std"] = agg["time_std"].fillna(0)
            agg["gpu_name"] = gpu_name
            agg["gpu_uuid"] = gpu_uuid
            agg["command_args"] = command_args
            agg["source_file"] = Path(db_path).name
            agg["report_count"] = 1

            all_records.append(agg)
            print(f"  Extracted {len(agg)} events ({n_events} raw) from {Path(db_path).name} [{gpu_name}]")

        except Exception as e:
            print(f"  Warning: Error processing {db_path}: {e}")

    return pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()


def _extract_command_args(cursor, tables: set) -> str:
    """Try to extract command arguments from various metadata tables."""
    # Try ANALYSIS_DETAILS first
    if "ANALYSIS_DETAILS" in tables:
        try:
            cursor.execute("""
                SELECT value FROM ANALYSIS_DETAILS
                WHERE LOWER(key) LIKE 'argument%'
                ORDER BY CAST(SUBSTR(key, 9) AS INTEGER)
            """)
            args = [r[0] for r in cursor.fetchall()]
            # Filter to the relevant args (after the python script)
            relevant = []
            found_script = False
            for a in args:
                if "profile_image_processing" in str(a):
                    found_script = True
                    continue
                if found_script:
                    relevant.append(str(a))
            if relevant:
                return " ".join(relevant)
        except sqlite3.Error:
            pass

    # Try META_DATA_CAPTURE
    if "META_DATA_CAPTURE" in tables:
        try:
            cursor.execute("""
                SELECT value FROM META_DATA_CAPTURE
                WHERE name LIKE 'PROCESS_%:ARGUMENT_%'
                ORDER BY CAST(SUBSTR(name, INSTR(name, ':ARGUMENT_') + 10) AS INTEGER)
            """)
            args = [r[0] for r in cursor.fetchall()]
            relevant = []
            found_script = False
            for a in args:
                if "profile_image_processing" in str(a):
                    found_script = True
                    continue
                if found_script:
                    relevant.append(str(a))
            if relevant:
                return " ".join(relevant)
        except sqlite3.Error:
            pass

    return "unknown"


def _parse_remote_spec(spec: str) -> Tuple[str, Optional[int], str]:
    """Parse ``host:/path``, ``host:port:/path``, or ``user@host:port:/path``.

    Returns (ssh_host, port_or_None, remote_path).
    """
    # Split on ":"  -- could be  host:/path  or  host:2223:/path
    parts = spec.split(":")
    if len(parts) == 2:
        return parts[0], None, parts[1]
    elif len(parts) == 3:
        # host:port:/path
        try:
            port = int(parts[1])
            return parts[0], port, parts[2]
        except ValueError:
            # Maybe host:path_with_colon -- unlikely but fallback
            return parts[0], None, ":".join(parts[1:])
    else:
        raise ValueError(f"Cannot parse remote spec: {spec}")


def extract_from_remote(remote_specs: List[str], ssh_opts: str = "") -> pd.DataFrame:
    """
    Extract NVTX profiling data from remote machines via SSH.

    Each spec can be:

    - ``ssh_host:/path``          -- standard SSH
    - ``ssh_host:port:/path``     -- custom port (e.g. Docker container)
    - ``user@host:port:/path``    -- explicit user + port

    A lightweight Python script is piped over SSH stdin and reads the
    SQLite databases in-place, outputting aggregated CSV to stdout.
    Only a few KB are transferred instead of multi-GB files.
    """
    # Self-contained extraction script executed on the remote machine.
    # Piped via stdin to avoid shell escaping issues with -c.
    # Must use only Python stdlib (sqlite3, csv, glob, pathlib, socket).
    REMOTE_SCRIPT = r'''
import csv, glob, io, sqlite3, sys
from pathlib import Path

profiling_dir = sys.argv[1]
writer = csv.writer(sys.stdout)
writer.writerow(["source_host", "source_file", "gpu_name", "gpu_uuid",
                 "command_args", "nvtx_name", "time_mean", "time_std",
                 "time_min", "time_max", "time_count"])

for db_path in sorted(glob.glob(str(Path(profiling_dir) / "*.sqlite"))):
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()

        # Check tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}
        if "NVTX_EVENTS" not in tables or "StringIds" not in tables:
            conn.close()
            continue

        # Count useful events
        cur.execute("SELECT COUNT(*) FROM NVTX_EVENTS WHERE eventType=59 AND end IS NOT NULL AND start IS NOT NULL AND (end-start)>0")
        if cur.fetchone()[0] < 10:
            conn.close()
            continue

        # GPU info
        gpu_name, gpu_uuid = "Unknown GPU", ""
        if "TARGET_INFO_GPU" in tables:
            cur.execute("SELECT name, uuid FROM TARGET_INFO_GPU LIMIT 1")
            row = cur.fetchone()
            if row:
                gpu_name = (row[0] or "").strip("[] ")
                gpu_uuid = row[1] or ""

        # Command args
        command_args = "unknown"
        if "ANALYSIS_DETAILS" in tables:
            try:
                cur.execute("SELECT value FROM ANALYSIS_DETAILS WHERE LOWER(key) LIKE 'argument%' ORDER BY CAST(SUBSTR(key, 9) AS INTEGER)")
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
        if command_args == "unknown" and "META_DATA_CAPTURE" in tables:
            try:
                cur.execute("SELECT value FROM META_DATA_CAPTURE WHERE name LIKE 'PROCESS_%:ARGUMENT_%' ORDER BY CAST(SUBSTR(name, INSTR(name, ':ARGUMENT_') + 10) AS INTEGER)")
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

        # Extract and aggregate NVTX events
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

        import socket
        hostname = socket.gethostname()
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

    all_dfs = []
    for spec in remote_specs:
        if ":" not in spec:
            print(f"  Error: invalid remote spec '{spec}' (expected host:/path or host:port:/path)")
            continue

        try:
            host, port, remote_path = _parse_remote_spec(spec)
        except ValueError as e:
            print(f"  Error: {e}")
            continue

        port_info = f" (port {port})" if port else ""
        print(f"  Connecting to {host}{port_info}:{remote_path} ...")

        try:
            # Build SSH command -- pipe script via stdin to avoid escaping issues
            ssh_cmd = ["ssh"]
            if port:
                ssh_cmd.extend(["-p", str(port)])
            # Append any extra SSH options (e.g. -o IdentitiesOnly=yes)
            if ssh_opts:
                import shlex
                ssh_cmd.extend(shlex.split(ssh_opts))
            ssh_cmd.extend([host, "python3", "-", remote_path])

            result = subprocess.run(
                ssh_cmd,
                input=REMOTE_SCRIPT,
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                print(f"  SSH error ({host}): {result.stderr.strip()}")
                continue

            # Print remote warnings
            for line in result.stderr.strip().splitlines():
                print(f"    [{host}] {line}")

            if not result.stdout.strip():
                print(f"  No data returned from {host}")
                continue

            df = pd.read_csv(io.StringIO(result.stdout))
            df["report_count"] = 1
            all_dfs.append(df)
            n_files = df["source_file"].nunique()
            n_events = df["nvtx_name"].nunique()
            gpus = df["gpu_name"].unique()
            print(f"  {host}: {n_files} files, {n_events} events, "
                  f"GPU(s): {', '.join(gpus)}")

        except subprocess.TimeoutExpired:
            print(f"  Timeout connecting to {host} (300s)")
        except FileNotFoundError:
            print(f"  Error: ssh command not found")
        except Exception as e:
            print(f"  Error with {host}: {e}")

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ---------------------------------------------------------------------------
# Classification & analysis
# ---------------------------------------------------------------------------

# Hierarchical stage definitions.
# NVTX events are nested: process_image > calibrate_image > perform_opt_photometry > ...
# To avoid double-counting, we define which stages are **direct children** of
# calibrate_image (level 1) and which are sub-stages (level 2+).
# Only level-1 stages are summed to produce the breakdown.
#
# Level 0: process_image
#   Level 1: calibrate_image  (contains all level-2 stages below)
#     Level 2 (direct children of calibrate_image -- non-overlapping):
CALIBRATE_CHILDREN = [
    # GPU stages
    "get_local_background_fft",
    "CR_filter",
    "SP_filter",
    "detect_isolated_stars",
    "create_star_dataset",
    "detect_sources_psf",
    # CPU stages
    "fit_moffat",
    # GPU composite (photometry -- choose one of the two variants)
    "perform_opt_photometry",
    "perform_opt_photometry_optimized_gpu_crossmatch",
    # I/O stages
    "update_header_with_astrometry",
    "catalog_results",
    "get_zeropoint",
    "get_target_snr",
    # PCA stages (may or may not be present)
    "get_eigen_psfs",
    "project_all_stars_onto_eigenpsfs",
    # Coefficient map (alternative to aperture corrections in some configs)
    "create_coeff_map",
    # Tile-based statistics (used in some pipeline versions)
    "calculate_tile_nanmean_sigclip",
]

# Stages that are children of perform_opt_photometry (level 3):
# Used for the detailed photometry sub-breakdown.
PHOTOMETRY_CHILDREN = [
    "batch_aperture_photometry",
    "create_aperture_corrections_map",
    "create_aperture_corrections_map_gpu",
    "find_aperture_corrections",
    "find_aperture_corrections_gpu",
    "calculate_aperture_corrections",
]


def classify_event(name: str) -> Tuple[str, str]:
    """Return (category, description) for an NVTX event name."""
    if name in STAGE_CLASSIFICATION:
        return STAGE_CLASSIFICATION[name]
    if name in CONTAINER_EVENTS:
        return ("Container", name)
    if any(name.startswith(p) for p in CUB_CUML_PREFIXES):
        return ("GPU-lib", "GPU library internals")
    return ("Other", name)


def _get_event_mean(df: pd.DataFrame, name: str) -> float:
    """Get mean time for a named NVTX event, or 0 if not found."""
    rows = df[df["nvtx_name"] == name]
    return rows["time_mean"].iloc[0] if not rows.empty else 0.0


def _get_event_row(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """Get the row for a named NVTX event."""
    rows = df[df["nvtx_name"] == name]
    return rows.iloc[0] if not rows.empty else None


def build_stage_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-stage breakdown with classification and time proportions."""
    df = df.copy()
    classifications = df["nvtx_name"].apply(classify_event)
    df["category"] = [c[0] for c in classifications]
    df["description"] = [c[1] for c in classifications]
    return df


def compute_hierarchical_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a non-overlapping time breakdown using the pipeline hierarchy.

    Uses only direct children of ``calibrate_image`` (which are siblings,
    not nested), so their times don't overlap.  The difference between
    ``process_image`` and ``calibrate_image`` is reported as pipeline
    overhead (header updates, allocator resets, etc.).

    For each configuration (command_args), returns one row per stage.
    """
    records = []

    for config, cdf in df.groupby(["command_args", "gpu_name"]):
        command_args, gpu_name = config

        pi_time = _get_event_mean(cdf, "process_image")
        cal_time = _get_event_mean(cdf, "calibrate_image")
        if pi_time <= 0:
            continue

        # --- Level 1: process_image overhead (outside calibrate_image) ---
        overhead_outside = pi_time - cal_time if cal_time > 0 else 0

        # Identify specific overhead stages
        overhead_stages = ["reset_cupy_allocators", "free_gpu_mem",
                           "update_header_with_photometry", "delete_header_from",
                           "get_if_header_already_post_processed"]
        overhead_detail_total = 0
        for stage_name in overhead_stages:
            row = _get_event_row(cdf, stage_name)
            if row is not None:
                cat, desc = classify_event(stage_name)
                stage_time = row["time_mean"]
                records.append({
                    "command_args": command_args,
                    "gpu_name": gpu_name,
                    "process_image_ms": pi_time,
                    "stage": stage_name,
                    "category": cat,
                    "description": desc,
                    "mean_ms": stage_time,
                    "std_ms": row["time_std"],
                    "pct_of_total": (stage_time / pi_time) * 100,
                    "level": "overhead",
                })
                overhead_detail_total += stage_time

        # --- Level 2: direct children of calibrate_image ---
        # Handle the two photometry variants: use whichever is present
        # (and prefer the gpu_crossmatch variant if both exist)
        phot_variant = None
        for pv in ["perform_opt_photometry_optimized_gpu_crossmatch",
                    "perform_opt_photometry"]:
            if _get_event_mean(cdf, pv) > 0:
                phot_variant = pv
                break

        children_accounted = 0
        for stage_name in CALIBRATE_CHILDREN:
            # Skip the non-active photometry variant
            if stage_name in ("perform_opt_photometry",
                              "perform_opt_photometry_optimized_gpu_crossmatch"):
                if stage_name != phot_variant:
                    continue

            row = _get_event_row(cdf, stage_name)
            if row is None:
                continue

            cat, desc = classify_event(stage_name)
            stage_time = row["time_mean"]
            children_accounted += stage_time

            records.append({
                "command_args": command_args,
                "gpu_name": gpu_name,
                "process_image_ms": pi_time,
                "stage": stage_name,
                "category": cat,
                "description": desc,
                "mean_ms": stage_time,
                "std_ms": row["time_std"],
                "pct_of_total": (stage_time / pi_time) * 100,
                "level": "main",
            })

        # Unaccounted time within calibrate_image
        cal_unaccounted = cal_time - children_accounted
        if cal_unaccounted > 1:  # > 1 ms
            records.append({
                "command_args": command_args,
                "gpu_name": gpu_name,
                "process_image_ms": pi_time,
                "stage": "(other calibration steps)",
                "category": "Other",
                "description": "Minor stages within calibrate_image",
                "mean_ms": cal_unaccounted,
                "std_ms": 0,
                "pct_of_total": (cal_unaccounted / pi_time) * 100,
                "level": "main",
            })

        # Pipeline overhead (outside calibrate_image minus detailed overhead)
        remaining_overhead = overhead_outside - overhead_detail_total
        if remaining_overhead > 1:
            records.append({
                "command_args": command_args,
                "gpu_name": gpu_name,
                "process_image_ms": pi_time,
                "stage": "(pipeline overhead)",
                "category": "Overhead",
                "description": "Time outside calibrate_image",
                "mean_ms": remaining_overhead,
                "std_ms": 0,
                "pct_of_total": (remaining_overhead / pi_time) * 100,
                "level": "overhead",
            })

    return pd.DataFrame(records)


def compute_category_summary(breakdown: pd.DataFrame) -> pd.DataFrame:
    """Aggregate breakdown by category (GPU, CPU, I/O, Overhead)."""
    # Use only 'main' level stages to avoid double-counting overhead
    main = breakdown[breakdown["level"] == "main"].copy()
    # Add overhead stages
    overhead = breakdown[breakdown["level"] == "overhead"].copy()
    combined = pd.concat([main, overhead], ignore_index=True)

    records = []
    for config, cdf in combined.groupby(["command_args", "gpu_name"]):
        command_args, gpu_name = config
        pi_time = cdf["process_image_ms"].iloc[0]

        for cat, gdf in cdf.groupby("category"):
            total = gdf["mean_ms"].sum()
            records.append({
                "command_args": command_args,
                "gpu_name": gpu_name,
                "category": cat,
                "total_ms": total,
                "pct_of_total": (total / pi_time) * 100,
                "n_stages": len(gdf),
            })

        # Compute unaccounted time
        accounted = cdf["mean_ms"].sum()
        unaccounted = pi_time - accounted
        if unaccounted > 0:
            records.append({
                "command_args": command_args,
                "gpu_name": gpu_name,
                "category": "Unaccounted",
                "total_ms": unaccounted,
                "pct_of_total": (unaccounted / pi_time) * 100,
                "n_stages": 0,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def print_stage_breakdown(breakdown: pd.DataFrame) -> None:
    """Print the hierarchical stage breakdown grouped by configuration."""
    print("\n" + "=" * 80)
    print("NVTX STAGE BREAKDOWN (Hierarchical, non-overlapping)")
    print("=" * 80)

    for config, cdf in breakdown.groupby(["command_args", "gpu_name"]):
        command_args, gpu_name = config
        pi_time = cdf["process_image_ms"].iloc[0]

        label = _short_label(command_args)
        print(f"\n--- {label} | {gpu_name} | process_image = {pi_time:.0f} ms ---")

        # Main stages (direct children of calibrate_image)
        main = cdf[cdf["level"] == "main"].sort_values("mean_ms", ascending=False)
        overhead = cdf[cdf["level"] == "overhead"].sort_values("mean_ms", ascending=False)

        print("\n  Main pipeline stages (children of calibrate_image):")
        fmt_cols = ["stage", "category", "mean_ms", "std_ms", "pct_of_total"]
        with pd.option_context("display.max_rows", None, "display.width", 160,
                               "display.float_format", "{:.1f}".format):
            print(main[fmt_cols].to_string(index=False))

        if not overhead.empty:
            print("\n  Pipeline overhead:")
            with pd.option_context("display.max_rows", None, "display.width", 160,
                                   "display.float_format", "{:.1f}".format):
                print(overhead[fmt_cols].to_string(index=False))

        accounted_pct = cdf["pct_of_total"].sum()
        print(f"\n  Total accounted: {accounted_pct:.1f}%")


def print_category_summary(cat_summary: pd.DataFrame) -> None:
    """Print category-level summary."""
    print("\n" + "=" * 80)
    print("TIME DISTRIBUTION BY CATEGORY")
    print("=" * 80)

    for config, cdf in cat_summary.groupby(["command_args", "gpu_name"]):
        command_args, gpu_name = config
        label = _short_label(command_args)
        print(f"\n--- {label} | {gpu_name} ---")

        cdf_sorted = cdf.sort_values("total_ms", ascending=False)
        for _, row in cdf_sorted.iterrows():
            bar = "#" * int(row["pct_of_total"] / 2)
            print(f"  {row['category']:<12} {row['total_ms']:>8.0f} ms  ({row['pct_of_total']:>5.1f}%)  {bar}")


def _short_label(command_args: str) -> str:
    """Create a short label from command arguments."""
    if not command_args or command_args == "unknown":
        return "unknown"
    parts = command_args.split()
    instrument = None
    fits_name = None
    filter_name = None

    for p in reversed(parts):
        if any(cam in p for cam in ("iKon", "QHY", "Atlas", "MUSCAT")):
            instrument = p
            break

    for p in parts:
        if ".fits" in p:
            fits_name = Path(p).stem
            # Try to extract filter from filename (e.g. _SDSSi, _Lum)
            for segment in fits_name.split("_"):
                if segment in ("SDSSi", "SDSSr", "SDSSg", "SDSSz", "Lum", "Ha", "R", "V", "B"):
                    filter_name = segment
                    break

    if instrument and filter_name:
        return f"{instrument} ({filter_name})"
    if instrument:
        return instrument
    if fits_name:
        return fits_name[:40]
    return command_args[:50]


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------
def generate_figures(breakdown: pd.DataFrame, cat_summary: pd.DataFrame,
                     classified_df: pd.DataFrame, outdir: str) -> None:
    """Generate publication-quality profiling figures."""
    if not HAS_PLOT:
        print("\n  [Figures skipped: matplotlib/seaborn not installed]")
        return

    figdir = Path(outdir) / FIGURE_DIR
    figdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    print(f"\n  Saving profiling figures to {figdir}/")

    _fig_category_bars(cat_summary, figdir)
    _fig_stage_waterfall(breakdown, figdir)
    _fig_category_pie(cat_summary, figdir)
    _fig_top_stages_horizontal(breakdown, figdir)


def _fig_category_bars(cat_summary: pd.DataFrame, figdir: Path) -> None:
    """Stacked bar chart: time by category for each configuration."""
    configs = cat_summary.groupby(["command_args", "gpu_name"])

    cat_order = ["GPU", "CPU", "I/O", "Overhead", "GPU-lib", "Other", "Unaccounted"]
    cat_colors = {
        "GPU": "#2196F3", "CPU": "#FF9800", "I/O": "#F44336",
        "Overhead": "#9E9E9E", "GPU-lib": "#4CAF50", "Other": "#795548",
        "Unaccounted": "#E0E0E0",
    }

    labels = []
    category_values = {c: [] for c in cat_order}

    for (cmd, gpu), cdf in configs:
        labels.append(_short_label(cmd))
        for cat in cat_order:
            row = cdf[cdf["category"] == cat]
            category_values[cat].append(row["pct_of_total"].iloc[0] if not row.empty else 0)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.5), 6))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    for cat in cat_order:
        vals = np.array(category_values[cat])
        if vals.sum() > 0:
            ax.bar(x, vals, bottom=bottom, label=cat,
                   color=cat_colors.get(cat, "#888"), width=0.6)
            # Add percentage labels for significant categories
            for i, v in enumerate(vals):
                if v >= 5:
                    ax.text(x[i], bottom[i] + v / 2, f"{v:.0f}%",
                            ha="center", va="center", fontsize=9, fontweight="bold")
            bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Percentage of Total Processing Time")
    ax.set_title("Pipeline Time Distribution by Category")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    fig.savefig(figdir / "profiling_category_bars.pdf", dpi=300)
    fig.savefig(figdir / "profiling_category_bars.png", dpi=150)
    plt.close(fig)
    print(f"    profiling_category_bars.pdf")


def _fig_stage_waterfall(breakdown: pd.DataFrame, figdir: Path) -> None:
    """Horizontal waterfall chart showing top stages for each config."""
    cat_colors = {
        "GPU": "#2196F3", "CPU": "#FF9800", "I/O": "#F44336",
        "Overhead": "#9E9E9E", "GPU-lib": "#4CAF50", "Other": "#795548",
    }

    for (cmd, gpu), cdf in breakdown.groupby(["command_args", "gpu_name"]):
        label = _short_label(cmd)
        top = cdf[cdf["pct_of_total"] >= 1.0].head(15).copy()
        if top.empty:
            continue

        top = top.sort_values("mean_ms", ascending=True)

        fig, ax = plt.subplots(figsize=(12, max(5, len(top) * 0.45)))
        colors = [cat_colors.get(c, "#888") for c in top["category"]]
        bars = ax.barh(range(len(top)), top["mean_ms"], color=colors, height=0.7)

        # Add time labels
        for i, (_, row) in enumerate(top.iterrows()):
            ax.text(row["mean_ms"] + 10, i,
                    f"{row['mean_ms']:.0f} ms ({row['pct_of_total']:.1f}%)",
                    va="center", fontsize=9)

        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["stage"], fontsize=9)
        ax.set_xlabel("Total Time (ms)")
        ax.set_title(f"Top Stages: {label} ({gpu})")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=cat)
                           for cat, c in cat_colors.items()
                           if cat in top["category"].values]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        plt.tight_layout()
        safe_label = label.replace("/", "_").replace(" ", "_")[:30]
        fig.savefig(figdir / f"profiling_waterfall_{safe_label}.pdf", dpi=300)
        fig.savefig(figdir / f"profiling_waterfall_{safe_label}.png", dpi=150)
        plt.close(fig)
        print(f"    profiling_waterfall_{safe_label}.pdf")


def _fig_category_pie(cat_summary: pd.DataFrame, figdir: Path) -> None:
    """Pie chart of time by category for each config."""
    cat_colors = {
        "GPU": "#2196F3", "CPU": "#FF9800", "I/O": "#F44336",
        "Overhead": "#9E9E9E", "GPU-lib": "#4CAF50", "Other": "#795548",
        "Unaccounted": "#E0E0E0",
    }

    for (cmd, gpu), cdf in cat_summary.groupby(["command_args", "gpu_name"]):
        label = _short_label(cmd)
        cdf = cdf[cdf["pct_of_total"] > 0.5].sort_values("total_ms", ascending=False)
        if cdf.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 8))
        colors = [cat_colors.get(c, "#888") for c in cdf["category"]]
        wedges, texts, autotexts = ax.pie(
            cdf["total_ms"], labels=cdf["category"], colors=colors,
            autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
            startangle=90, pctdistance=0.75,
        )
        for t in autotexts:
            t.set_fontsize(10)
            t.set_fontweight("bold")
        ax.set_title(f"Time Distribution: {label}\n({gpu})")

        plt.tight_layout()
        safe_label = label.replace("/", "_").replace(" ", "_")[:30]
        fig.savefig(figdir / f"profiling_pie_{safe_label}.pdf", dpi=300)
        fig.savefig(figdir / f"profiling_pie_{safe_label}.png", dpi=150)
        plt.close(fig)
        print(f"    profiling_pie_{safe_label}.pdf")


def _fig_top_stages_horizontal(breakdown: pd.DataFrame, figdir: Path) -> None:
    """Grouped horizontal bar chart comparing top stages across configurations."""
    configs = list(breakdown.groupby(["command_args", "gpu_name"]))
    if len(configs) < 2:
        return

    # Find stages that appear in multiple configs and are significant
    all_stages = set()
    for (cmd, gpu), cdf in configs:
        top = cdf[cdf["pct_of_total"] >= 2.0]["stage"]
        all_stages.update(top)

    if not all_stages:
        return

    # Build comparison matrix
    stage_list = sorted(all_stages)
    config_labels = [_short_label(cmd) for (cmd, gpu), _ in configs]

    fig, ax = plt.subplots(figsize=(12, max(6, len(stage_list) * 0.5)))
    n_configs = len(configs)
    bar_height = 0.8 / n_configs
    y_base = np.arange(len(stage_list))
    colors = plt.cm.Set2(np.linspace(0, 1, n_configs))

    for i, ((cmd, gpu), cdf) in enumerate(configs):
        vals = []
        for stage in stage_list:
            row = cdf[cdf["stage"] == stage]
            vals.append(row["pct_of_total"].iloc[0] if not row.empty else 0)
        ax.barh(y_base + i * bar_height, vals, height=bar_height,
                label=config_labels[i], color=colors[i])

    ax.set_yticks(y_base + bar_height * (n_configs - 1) / 2)
    ax.set_yticklabels(stage_list, fontsize=9)
    ax.set_xlabel("% of Total Processing Time")
    ax.set_title("Stage Comparison Across Configurations")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(figdir / "profiling_stage_comparison.pdf", dpi=300)
    fig.savefig(figdir / "profiling_stage_comparison.png", dpi=150)
    plt.close(fig)
    print(f"    profiling_stage_comparison.pdf")


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------
def export_latex_tables(breakdown: pd.DataFrame, cat_summary: pd.DataFrame,
                        outdir: str) -> None:
    """Export profiling tables in LaTeX format."""
    tbldir = Path(outdir) / TABLE_DIR
    tbldir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving profiling LaTeX tables to {tbldir}/")

    # -- Table: Category summary
    if not cat_summary.empty:
        # Pivot: rows = config, columns = category
        pivot = cat_summary.pivot_table(
            values="pct_of_total", index=["command_args", "gpu_name"],
            columns="category", aggfunc="first", fill_value=0,
        ).reset_index()
        # Add short labels
        pivot.insert(0, "Configuration",
                     pivot["command_args"].apply(_short_label))
        pivot = pivot.drop(columns=["command_args"])

        tex = pivot.to_latex(
            index=False, float_format="%.1f",
            caption="Percentage of total \\texttt{process\\_image} time by category. "
                    "GPU includes all CuPy/CUDA operations; I/O covers astrometry "
                    "solving and catalog queries; CPU covers PSF fitting and header updates.",
            label="tab:profiling_category",
        )
        outpath = tbldir / "table_profiling_category.tex"
        outpath.write_text(tex)
        print(f"    {outpath.name}")

    # -- Table: Top stages
    if not breakdown.empty:
        # Select top 15 stages across all configs
        top_stages = (
            breakdown.groupby("stage")["pct_of_total"]
            .max().sort_values(ascending=False).head(15).index
        )
        top = breakdown[breakdown["stage"].isin(top_stages)].copy()
        top["Configuration"] = top["command_args"].apply(_short_label)

        export_cols = [
            "Configuration", "stage", "category",
            "mean_ms", "std_ms", "pct_of_total",
        ]
        available = [c for c in export_cols if c in top.columns]
        top_sorted = top[available].sort_values(
            ["Configuration", "pct_of_total"], ascending=[True, False]
        )

        tex = top_sorted.to_latex(
            index=False, float_format="%.1f",
            caption="Top pipeline stages by time contribution. "
                    "\\texttt{mean\\_ms} is the mean duration of the stage; "
                    "stages are direct children of \\texttt{calibrate\\_image} "
                    "(non-overlapping).",
            label="tab:profiling_stages",
        )
        outpath = tbldir / "table_profiling_stages.tex"
        outpath.write_text(tex)
        print(f"    {outpath.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("GPUPhot NVTX Profiling Analysis")
    print("=" * 80)

    # Auto-discover local data if no explicit arguments given
    if not args.csv and not args.sqlite and not args.remote:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent

        # Look for pre-aggregated CSVs
        for candidate in [
            project_root / "dev" / "profiling_results" / "analysis_aggregated_data.csv",
        ]:
            if candidate.exists():
                args.csv = [str(candidate)]
                break

        # Look for SQLite databases in profiling_results/
        profiling_dir = project_root / "profiling_results"
        if profiling_dir.is_dir():
            sqlite_files = sorted(profiling_dir.glob("*.sqlite"))
            if sqlite_files:
                args.sqlite = [str(f) for f in sqlite_files]

        if not args.csv and not args.sqlite:
            print("\nNo data found. Searched:")
            print(f"  CSV:    {project_root / 'dev' / 'profiling_results' / 'analysis_aggregated_data.csv'}")
            print(f"  SQLite: {profiling_dir / '*.sqlite'}")
            print("\nProvide --csv, --sqlite, or --remote arguments.")
            sys.exit(1)

        print("Auto-discovered local profiling data:")

    # Load data
    dfs = []
    if args.csv:
        print("\nLoading pre-aggregated CSV data:")
        csv_df = load_aggregated_csv(args.csv)
        if not csv_df.empty:
            dfs.append(csv_df)

    if args.sqlite:
        print("\nExtracting from local SQLite databases:")
        sqlite_df = extract_from_sqlite(args.sqlite)
        if not sqlite_df.empty:
            dfs.append(sqlite_df)

    if args.remote:
        print("\nExtracting from remote machines via SSH:")
        remote_df = extract_from_remote(args.remote, ssh_opts=args.ssh_opts)
        if not remote_df.empty:
            dfs.append(remote_df)

    if not dfs:
        print("\nError: No data loaded. Provide --csv, --sqlite, or --remote arguments.")
        print("  Example: python benchmarks/analyze_profiling.py "
              "--csv dev/profiling_results/analysis_aggregated_data.csv")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total: {len(df)} event rows, "
          f"{df['nvtx_name'].nunique()} unique NVTX events, "
          f"{df['gpu_name'].nunique()} GPU(s)")

    # Classify events
    df = build_stage_breakdown(df)

    # Compute hierarchical breakdown (non-overlapping stages)
    breakdown = compute_hierarchical_breakdown(df)
    if breakdown.empty:
        print("\nError: Could not compute stage breakdown (no process_image events found).")
        sys.exit(1)

    # Category summary
    cat_summary = compute_category_summary(breakdown)

    # Print results
    print_stage_breakdown(breakdown)
    print_category_summary(cat_summary)

    # Figures
    if not args.no_figures:
        generate_figures(breakdown, cat_summary, df, args.outdir)

    # LaTeX tables
    export_latex_tables(breakdown, cat_summary, args.outdir)

    print("\n" + "=" * 80)
    print("Profiling analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
