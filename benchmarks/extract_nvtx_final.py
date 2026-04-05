#!/usr/bin/env python3
"""
Extract NVTX per-stage timing data for GPUPhot manuscript.

Processes:
1. RTX 3090 wide-format NVTX CSVs (py3.12 and py3.8)
2. RTX 3050 Ti sqlite files (py3.12 and py3.8)

Outputs:
- benchmarks/results_collected/nvtx_stage_breakdown_rtx3090.csv
- benchmarks/results_collected/nvtx_stage_breakdown_rtx3050ti.csv
- benchmarks/results_collected/nvtx_detection_vs_sep.csv

NOTE: A100 data is NOT available locally; A100 sqlite files reside at:
  /mnt/vast/samueltest/gpuphot/profiling_test_results/ (lenovo_tttserver)
"""

import glob
import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)

# ---- Configuration ----
CAMERA_MP = {
    'iKon936': 4.2, 'iKon936-1': 4.2,
    'QHY600-3': 6.8,
    'QHY600-4': 15.3,
    'QHY411-1': 37.8,
    'QHY411-3': 151.2,
}
CAMERA_SIZE = {
    'iKon936': '2048x2048', 'iKon936-1': '2048x2048',
    'QHY600-3': '3191x2129',
    'QHY600-4': '4787x3193',
    'QHY411-1': '7100x5325',
    'QHY411-3': '14200x10650',
}
CATEGORY = {
    'process_image': 'Container', 'calibrate_image': 'Container',
    'get_local_background_fft': 'GPU', 'convolve_fft': 'GPU',
    'batch_aperture_photometry': 'GPU', 'detect_sources_psf': 'GPU',
    'detect_isolated_stars': 'GPU', 'create_star_dataset': 'GPU',
    'get_mean_std': 'GPU', 'fill_nan_fft': 'GPU',
    'calculate_tile_percentiles': 'GPU', 'calculate_tile_nanmean_sigclip': 'GPU',
    'get_aper_kernel': 'GPU', 'gen_apm_filter': 'GPU',
    'find_local_max': 'GPU', 'find_local_centroid': 'GPU',
    'gaussian_kernel': 'GPU', 'stack_sigmaclip': 'GPU',
    'calculate_kernel_area': 'GPU', 'decompose_into_tiles': 'GPU',
    'recompose_from_percentiles': 'GPU', 'fill_image': 'GPU',
    'moffat': 'GPU', 'get_eigen_psfs': 'GPU',
    'project_all_stars_onto_eigenpsfs': 'GPU', 'create_coeff_map': 'GPU',
    'perform_opt_photometry': 'GPU',
    'perform_opt_photometry_optimized_gpu_crossmatch': 'GPU',
    'create_aperture_corrections_map_gpu': 'GPU',
    'find_aperture_corrections_gpu': 'GPU',
    'calculate_aperture_corrections_gpu': 'GPU',
    'crossmatch_sources': 'GPU', 'crossmatch_sources_gpu_impl': 'GPU',
    'crossmatch_sources_cpu_impl': 'CPU',
    'CR_filter': 'CPU', 'SP_filter': 'CPU',
    'fit_moffat': 'CPU', 'moffat_fwhm': 'CPU',
    'group_star_dataset': 'CPU', 'get_zeropoint': 'CPU',
    'get_target_snr': 'CPU', 'get_maglim': 'CPU',
    'update_header_with_astrometry': 'I/O', 'astrometrice2': 'I/O',
    'catalog_results': 'I/O', '__getVizier': 'I/O',
    'radec_to_altaz': 'I/O', 'radec_to_moon_sun': 'I/O',
    'radec_to_gal': 'I/O', 'radec_to_ecl': 'I/O',
    'reset_cupy_allocators': 'Overhead', 'free_gpu_mem': 'Overhead',
}

# Stages for scope-equivalent detection+photometry comparison with sep
# sep does: detection (thresholding) + circular aperture photometry
# GPUPhot equivalent: detect_sources_psf + batch_aperture_photometry
DETECTION_PHOTOMETRY_STAGES = ['detect_sources_psf', 'batch_aperture_photometry']

# All reportable stages (skip noisy micro-stages)
ALL_STAGES = list(CATEGORY.keys())


# ===================== RTX 3090 wide-format CSVs =====================

def extract_rtx3090():
    """Extract from RTX 3090 wide-format NVTX CSVs."""
    nvtx_dir = 'benchmarks/results_collected/run2_v2_20260320_1917/nvtx'
    files = sorted(glob.glob(f'{nvtx_dir}/nvtx_*.csv'))
    print(f"\n[RTX 3090] Found {len(files)} NVTX CSV files")

    records = []
    for fpath in files:
        fname = Path(fpath).name
        # Parse: nvtx_<pyver>_<camera...>_<run_type>.csv
        m = re.match(r'nvtx_(py\d+)_(.+?)_(nsys\d+|rep\d+)\.csv', fname)
        if not m:
            continue
        pyver = m.group(1)
        middle = m.group(2)
        run_type = m.group(3)

        # Identify camera
        camera = None
        for cam in ['iKon936-1', 'QHY600-3', 'QHY600-4', 'QHY411-1', 'QHY411-3', 'iKon936']:
            if middle.startswith(cam.replace('-', '-')) or middle.startswith(cam):
                camera = cam
                break
        if not camera:
            continue
        if camera not in CAMERA_MP:
            continue

        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        if len(df) == 0:
            continue

        row = df.iloc[0]
        gpu_name = row.get('gpu_name', 'Unknown')
        mp = CAMERA_MP[camera]

        for stage in ALL_STAGES:
            col = f'nvtx_{stage}_mean_ms'
            if col in row and pd.notna(row[col]):
                val = float(row[col])
                if val > 0:
                    records.append({
                        'gpu': gpu_name,
                        'python_ver': pyver,
                        'camera': camera,
                        'mp': mp,
                        'image_size': CAMERA_SIZE.get(camera, ''),
                        'stage': stage,
                        'category': CATEGORY.get(stage, 'Other'),
                        'duration_ms': val,
                        'run_type': run_type,
                        'source_file': fname,
                    })

    df_all = pd.DataFrame(records)
    if df_all.empty:
        print("[RTX 3090] No data extracted!")
        return None

    # Filter to only runs with "NVIDIA GeForce RTX 3090" GPU
    # (some files may be from other GPUs)
    rtx3090_df = df_all[df_all['gpu'].str.contains('3090', na=False)]
    if rtx3090_df.empty:
        print("[RTX 3090] No RTX 3090 data found, using all")
        rtx3090_df = df_all

    print(f"[RTX 3090] {len(rtx3090_df)} records, "
          f"GPUs: {rtx3090_df['gpu'].unique()}, "
          f"py versions: {sorted(rtx3090_df['python_ver'].unique())}")
    return rtx3090_df


# ===================== RTX 3050 Ti sqlite =====================

def extract_rtx3050ti():
    """Extract from local RTX 3050 Ti sqlite files."""
    sqlite_files = sorted(glob.glob('benchmarks/benchmark_results/*.sqlite'))
    print(f"\n[RTX 3050 Ti] Found {len(sqlite_files)} sqlite files")

    # Container -> Python version mapping
    py_map = {}
    mem_csv = 'benchmarks/results_collected/profiler_nsys_memory_20260326.csv'
    if os.path.exists(mem_csv):
        mem_df = pd.read_csv(mem_csv)
        local_mem = mem_df[mem_df['machine'] == 'local']
        for _, row in local_mem.drop_duplicates('container_id').iterrows():
            # Normalize to pyXYZ format (e.g., py312, py38)
            ver_str = str(row['python_ver']).replace('.', '')
            py_map[row['container_id']] = f"py{ver_str}"

    records = []
    for db_path in sqlite_files:
        fname = Path(db_path).name
        parts = fname.replace('.sqlite', '').split('_')
        container_id = parts[1]

        camera = None
        for cam in ['iKon936-1', 'QHY600-3', 'QHY600-4', 'QHY411-1', 'QHY411-3', 'iKon936']:
            if cam in fname:
                camera = cam
                break
        if not camera or camera not in CAMERA_MP:
            continue

        pyver = py_map.get(container_id, 'unknown')

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cur = conn.cursor()
            cur.execute("SELECT name FROM TARGET_INFO_GPU LIMIT 1")
            gpu_row = cur.fetchone()
            gpu_name = gpu_row[0].strip('[] ') if gpu_row else 'Unknown'

            cur.execute("""
                SELECT t2.value, (t1.end - t1.start) / 1e6
                FROM NVTX_EVENTS t1 JOIN StringIds t2 ON t1.textId = t2.id
                WHERE t1.eventType = 59 AND t1.end IS NOT NULL
                  AND t1.start IS NOT NULL AND (t1.end - t1.start) > 0
            """)
            events = defaultdict(list)
            for name, dur in cur.fetchall():
                events[name].append(dur)
            conn.close()

            if 'process_image' not in events:
                continue

            mp = CAMERA_MP[camera]
            for stage_name in ALL_STAGES:
                if stage_name in events:
                    for dur in events[stage_name]:
                        records.append({
                            'gpu': gpu_name,
                            'python_ver': pyver,
                            'camera': camera,
                            'mp': mp,
                            'image_size': CAMERA_SIZE.get(camera, ''),
                            'stage': stage_name,
                            'category': CATEGORY.get(stage_name, 'Other'),
                            'duration_ms': dur,
                            'run_type': 'nsys',
                            'source_file': fname,
                        })
        except Exception as e:
            print(f"  Error: {db_path}: {e}", file=sys.stderr)

    df_all = pd.DataFrame(records)
    if df_all.empty:
        print("[RTX 3050 Ti] No data extracted!")
        return None

    print(f"[RTX 3050 Ti] {len(df_all)} records, "
          f"py versions: {sorted(df_all['python_ver'].unique())}")
    return df_all


# ===================== Summary computation =====================

def compute_summary(df, gpu_label, pyver='py312'):
    """Compute per-camera, per-stage median and IQR."""
    df_py = df[df['python_ver'] == pyver].copy()
    if df_py.empty:
        print(f"  No {pyver} data for {gpu_label}")
        return pd.DataFrame()

    summary = df_py.groupby(['camera', 'mp', 'image_size', 'stage', 'category']
                            )['duration_ms'].agg(
        median='median',
        q25=lambda x: np.percentile(x, 25),
        q75=lambda x: np.percentile(x, 75),
        count='count',
    ).reset_index()
    summary['iqr'] = summary['q75'] - summary['q25']
    summary['median_s'] = summary['median'] / 1000.0
    summary['iqr_s'] = summary['iqr'] / 1000.0

    return summary


def print_breakdown(summary, gpu_label, pyver):
    """Print formatted stage breakdown."""
    print(f"\n{'=' * 110}")
    print(f"NVTX STAGE BREAKDOWN -- {gpu_label} ({pyver})")
    print(f"{'=' * 110}")

    for camera in sorted(summary['camera'].unique(), key=lambda c: CAMERA_MP.get(c, 0)):
        cam_data = summary[summary['camera'] == camera].sort_values('median', ascending=False)
        mp = CAMERA_MP[camera]
        size = CAMERA_SIZE[camera]

        pi_row = cam_data[cam_data['stage'] == 'process_image']
        pi_ms = pi_row['median'].iloc[0] if not pi_row.empty else 0

        print(f"\n--- {camera} ({size}, {mp} MP) | process_image = {pi_ms:.0f} ms ({pi_ms/1000:.1f} s) ---")
        print(f"{'Stage':<50} {'Cat':>4} {'Median(s)':>10} {'IQR(s)':>8} {'%total':>7} {'N':>4}")
        print("-" * 88)

        for _, row in cam_data.iterrows():
            if row['stage'] in ('process_image', 'calibrate_image'):
                continue
            if row['median_s'] < 0.001:
                continue
            pct = (row['median'] / pi_ms * 100) if pi_ms > 0 else 0
            cat_short = row['category'][:4]
            print(f"{row['stage']:<50} {cat_short:>4} {row['median_s']:>10.3f} {row['iqr_s']:>8.3f} {pct:>6.1f}% {int(row['count']):>4}")


def save_breakdown(summary, out_path):
    """Save breakdown CSV."""
    export = summary[['camera', 'mp', 'image_size', 'stage', 'category',
                       'median_s', 'iqr_s', 'count']].copy()
    export.columns = ['Camera', 'MP', 'Image_Size', 'Stage', 'Category',
                       'Median_s', 'IQR_s', 'N']
    export = export.sort_values(['MP', 'Median_s'], ascending=[True, False])
    export.to_csv(out_path, index=False, float_format='%.4f')
    print(f"\nSaved: {out_path}")


# ===================== Detection vs sep =====================

def detection_vs_sep(summary_3090, summary_3050ti):
    """Compare scope-equivalent detection+photometry with sep."""
    cpu_csv = 'benchmarks/cpu_baseline_results.csv'
    if not os.path.exists(cpu_csv):
        print(f"\nCPU baseline not found: {cpu_csv}")
        return

    cpu_df = pd.read_csv(cpu_csv)

    # sep by megapixels (median across images of same size)
    sep_agg = cpu_df.groupby('megapixels').agg(
        sep_detect_s=('sep_median_s', 'median'),
        photutils_detect_s=('photutils_median_s', 'median'),
    ).reset_index()

    print(f"\n{'=' * 110}")
    print("DETECTION+PHOTOMETRY: GPUPhot (NVTX stages) vs sep vs Photutils (CPU)")
    print("Scope-equivalent comparison: detect_sources_psf + batch_aperture_photometry")
    print(f"{'=' * 110}")

    results = []

    for label, summary in [('RTX 3090', summary_3090), ('RTX 3050 Ti', summary_3050ti)]:
        if summary is None or summary.empty:
            continue

        det_phot = summary[summary['stage'].isin(DETECTION_PHOTOMETRY_STAGES)].copy()
        if det_phot.empty:
            continue

        # Sum detection + photometry per camera
        det_agg = det_phot.groupby(['camera', 'mp', 'image_size']).agg(
            gpuphot_det_phot_s=('median_s', 'sum')
        ).reset_index()

        merged = det_agg.merge(sep_agg, left_on='mp', right_on='megapixels', how='left')
        merged['gpu'] = label
        merged['speedup_vs_sep'] = merged['sep_detect_s'] / merged['gpuphot_det_phot_s']
        merged['speedup_vs_photutils'] = merged['photutils_detect_s'] / merged['gpuphot_det_phot_s']
        results.append(merged)

        print(f"\n--- {label} (py3.12) ---")
        print(f"{'Camera':<12} {'MP':>6} {'GPUPhot(s)':>11} {'sep(s)':>8} {'Phot(s)':>8} {'vs sep':>8} {'vs Phot':>9}")
        print("-" * 70)
        for _, row in merged.sort_values('mp').iterrows():
            vs_sep = f"{row['speedup_vs_sep']:.1f}x" if pd.notna(row['speedup_vs_sep']) else "N/A"
            vs_phot = f"{row['speedup_vs_photutils']:.1f}x" if pd.notna(row['speedup_vs_photutils']) else "N/A"
            print(f"{row['camera']:<12} {row['mp']:>6.1f} {row['gpuphot_det_phot_s']:>11.3f} "
                  f"{row.get('sep_detect_s', float('nan')):>8.3f} "
                  f"{row.get('photutils_detect_s', float('nan')):>8.3f} "
                  f"{vs_sep:>8} {vs_phot:>9}")

    if results:
        combined = pd.concat(results, ignore_index=True)
        out_path = 'benchmarks/results_collected/nvtx_detection_vs_sep.csv'
        combined.to_csv(out_path, index=False, float_format='%.4f')
        print(f"\nSaved: {out_path}")


# ===================== Main =====================

def main():
    # Extract data
    df_3090 = extract_rtx3090()
    df_3050ti = extract_rtx3050ti()

    # Compute summaries (py3.12)
    summary_3090 = compute_summary(df_3090, 'RTX 3090', 'py312') if df_3090 is not None else None
    summary_3050ti = compute_summary(df_3050ti, 'RTX 3050 Ti', 'py312') if df_3050ti is not None else None

    # Print and save breakdowns
    if summary_3090 is not None and not summary_3090.empty:
        print_breakdown(summary_3090, 'RTX 3090', 'py3.12')
        save_breakdown(summary_3090, 'benchmarks/results_collected/nvtx_stage_breakdown_rtx3090.csv')

    if summary_3050ti is not None and not summary_3050ti.empty:
        print_breakdown(summary_3050ti, 'RTX 3050 Ti', 'py3.12')
        save_breakdown(summary_3050ti, 'benchmarks/results_collected/nvtx_stage_breakdown_rtx3050ti.csv')

    # Detection vs sep comparison
    detection_vs_sep(summary_3090, summary_3050ti)

    # ---- Key finding summary ----
    print(f"\n{'=' * 110}")
    print("SUMMARY")
    print(f"{'=' * 110}")
    print()
    print("Data available locally:")
    print("  - RTX 3090: NVTX per-stage timing from run2 (20 Mar 2026)")
    print("  - RTX 3050 Ti: NVTX per-stage from sqlite files (24-26 Mar 2026)")
    print()
    print("Data NOT available locally:")
    print("  - A100 NVTX per-stage timing (sqlite files on remote VAST)")
    print("  - H100 NVTX per-stage timing (sqlite files on remote VAST)")
    print()
    print("To extract A100/H100 NVTX data, run:")
    print("  python benchmarks/analyze_profiling.py \\")
    print("    --remote lenovo_slemes:2223:/mnt/vast/samueltest/gpuphot/profiling_test_results/")
    print()
    print("NOTE: The run2 data (20 Mar 2026) may use an older pipeline version.")
    print("The authoritative profiler benchmark is from 25-26 Mar 2026.")
    print("For manuscript, prefer the 26 Mar sqlite data (RTX 3050 Ti available locally).")
    print("A100 data needs to be extracted from the remote server.")


if __name__ == '__main__':
    main()
