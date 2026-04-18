#!/usr/bin/env python3
"""
Extract NVTX stage breakdown for QHY411-1_Lum_full (151.2 MP, image 8)
from existing PASO 2 nsys sqlite files (container a80f7e35e9e0).

Adds rows with camera=QHY411-1, MP=151.2 to nvtx_stage_breakdown.csv.

Usage (inside Docker container on lenovo_tttserver):
    python3 /tmp/extract_stage_breakdown_qhy411_151mp.py \
        --sqlite-dir /app/profiling_results \
        --container-id a80f7e35e9e0 \
        --nsys-reps 4 \
        --output /app/profiling_results/stage_breakdown_qhy411_151mp.csv

The script identifies the QHY411-1 sqlite files and assigns image 8
(QHY411-1_Lum_full, the 3rd group of 4 QHY411-1 files by timestamp).
Images 6=Lum_bin2 → files [0:4], 7=SDSSi_bin2 → files [4:8], 8=Lum_full → files [8:12]
"""

import argparse
import csv
import glob
import os
import re
import sqlite3
import sys
from collections import defaultdict
from statistics import median

# Image index 8 in BENCHMARK_IMAGES: QHY411-1_Lum_full, 151.2 MP
# QHY411-1 images: 6=Lum_bin2, 7=SDSSi_bin2, 8=Lum_full
# So image 8 = QHY411-1 sqlite files at indices [8:12] within the camera group
TARGET_CAMERA = 'QHY411-1'
TARGET_IMAGE_GROUP_OFFSET = 8  # Skip first 8 QHY411-1 files (images 6 and 7, 4 reps each)
TARGET_MP = 151.2
TARGET_LABEL = 'QHY411-1_Lum_full'

# Human-readable stage names (from generate_manuscript_tables.py STAGE_ORDER)
# Maps CSV stage key → display label (column name in nvtx_stage_breakdown.csv)
STAGE_DISPLAY = {
    'background_fft':          'Background (FFT)',
    'star_detection':          'Star detection',
    'detect_sources_psf':      'Source detection (PSF)',
    'psf_cutout_extraction':   'PSF cutout extraction',
    'eigen_psf_pca':           'Eigen-PSF (PCA)',
    'coeff_map':               'Coeff map',
    'tile_statistics':         'Tile statistics',
    'batch_aperture_photometry': 'Aperture photometry',
    'optimal_photometry':      'Optimal photometry',
    'fft_convolution':         'FFT convolution',
    'crossmatch':              'Crossmatch',
    'moffat_fitting':          'Moffat fitting (CPU)',
    'salt_pepper_filter':      'Salt-pepper filter',
    'cosmic_ray_filter':       'Cosmic ray filter',
    'zero_point':              'Zero-point (CPU)',
    'astrometry_solver':       'Astrometry (solver)',
    'astrometry_cpu':          'Astrometry (CPU)',
    'catalog_query':           'Catalog query (I/O)',
    'star_projection':         'Star projection',
}

# Also accept raw NVTX names (as they appear in sqlite) directly
# Map NVTX text → display name used in nvtx_stage_breakdown.csv
NVTX_TO_DISPLAY = {
    # GPU stages — real NVTX function names from sqlite
    'get_local_background_fft':          'Background (FFT)',
    'detect_isolated_stars':             'Star detection',
    'detect_sources_psf':                'Source detection (PSF)',
    'create_star_dataset':               'PSF cutout extraction',
    'get_eigen_psfs':                    'Eigen-PSF (PCA)',
    'create_coeff_map':                  'Coeff map',
    'calculate_tile_nanmean_sigclip':    'Tile statistics',
    'batch_aperture_photometry':         'Aperture photometry',
    'perform_opt_photometry':            'Optimal photometry',
    'perform_opt_photometry_optimized_gpu_crossmatch': 'Optimal photometry',
    'convolve_fft':                      'FFT convolution',
    'crossmatch_sources':                'Crossmatch',
    'project_all_stars_onto_eigenpsfs':  'Star projection',
    # CPU stages
    'fit_moffat':                        'Moffat fitting (CPU)',
    'SP_filter':                         'Salt-pepper filter',
    'CR_filter':                         'Cosmic ray filter',
    'get_zeropoint':                     'Zero-point (CPU)',
    # I/O / CPU stages
    'astrometrice2':                     'Astrometry (solver)',
    'update_header_with_astrometry':     'Astrometry (CPU)',
    'catalog_results':                   'Catalog query (I/O)',
    '__getVizier':                       'Catalog query (I/O)',
}


def extract_all_stages(db_path):
    """
    Extract all NVTX stage durations from a single nsys sqlite file.

    Returns dict: nvtx_name → list of durations in seconds.
    Returns None if unreadable.
    """
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        cur = conn.cursor()
        cur.execute("""
            SELECT t2.value, (t1.end - t1.start) / 1e9
            FROM NVTX_EVENTS t1 JOIN StringIds t2 ON t1.textId = t2.id
            WHERE t1.eventType = 59
              AND t1.end IS NOT NULL AND t1.start IS NOT NULL
              AND (t1.end - t1.start) > 0
        """)
        events = defaultdict(list)
        for name, dur_s in cur.fetchall():
            events[name].append(dur_s)
        conn.close()
    except Exception as e:
        print(f'  WARNING: error reading {os.path.basename(db_path)}: {e}', file=sys.stderr)
        return None

    if 'process_image' not in events:
        print(f'  WARNING: no process_image in {os.path.basename(db_path)}', file=sys.stderr)
        return None

    print(f'  OK: {os.path.basename(db_path)} — {len(events)} unique NVTX stages', file=sys.stderr)
    return dict(events)


def load_sqlite_files_for_camera(sqlite_dir, container_id, camera, offset, n_reps):
    """
    Load sqlite files for a specific camera group, selecting files [offset:offset+n_reps].

    Filename format: benchmark_<container>_<camera>_<YYYYMMDD>_<HHMMSS>_run<N>.sqlite
    """
    pattern = os.path.join(sqlite_dir, f'benchmark_{container_id}_{camera}_*.sqlite')
    files = sorted(glob.glob(pattern))
    print(f'Found {len(files)} sqlite files for camera {camera} (container {container_id})',
          file=sys.stderr)

    if len(files) < offset + n_reps:
        print(f'  WARNING: need files [{offset}:{offset+n_reps}] but only {len(files)} found',
              file=sys.stderr)

    selected = files[offset:offset + n_reps]
    print(f'  Using files [{offset}:{offset+n_reps}]: {[os.path.basename(f) for f in selected]}',
          file=sys.stderr)
    return selected


def compute_stage_stats(all_runs_events):
    """
    Given list of dicts (nvtx_name → [dur_s, ...]) from multiple reps,
    compute per-stage: median_s, iqr_s, count across all events in all reps.

    Returns dict: display_name → (median_s, iqr_s, count)
    """
    # Aggregate all durations per display name across reps
    aggregated = defaultdict(list)
    for run_events in all_runs_events:
        for nvtx_name, durs in run_events.items():
            display = NVTX_TO_DISPLAY.get(nvtx_name)
            if display is None:
                # Not a known stage, skip (process_image, etc.)
                continue
            aggregated[display].extend(durs)

    stats = {}
    for display_name, durs in aggregated.items():
        n = len(durs)
        med = median(durs)
        # IQR: Q3 - Q1
        sorted_durs = sorted(durs)
        q1_idx = n // 4
        q3_idx = (3 * n) // 4
        q1 = sorted_durs[q1_idx] if q1_idx < n else sorted_durs[-1]
        q3 = sorted_durs[q3_idx] if q3_idx < n else sorted_durs[-1]
        iqr = q3 - q1
        stats[display_name] = (med, iqr, n)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Extract NVTX stage breakdown for QHY411-1_Lum_full (151.2 MP) from PASO 2 sqlite files'
    )
    parser.add_argument('--sqlite-dir', default='/app/profiling_results',
                        help='Directory containing nsys sqlite files')
    parser.add_argument('--container-id', default='a80f7e35e9e0',
                        help='Docker container short ID (12 hex chars)')
    parser.add_argument('--nsys-reps', type=int, default=4,
                        help='Number of nsys reps per image')
    parser.add_argument('--output', default='/app/profiling_results/stage_breakdown_qhy411_151mp.csv',
                        help='Output CSV path')
    parser.add_argument('--dump-nvtx-names', action='store_true',
                        help='Print all unique NVTX names found in sqlite files (for debugging)')
    args = parser.parse_args()

    # Get the 4 sqlite files for image 8 (QHY411-1_Lum_full)
    # QHY411-1 group: images 6 (Lum_bin2, offset 0), 7 (SDSSi_bin2, offset 4), 8 (Lum_full, offset 8)
    selected_files = load_sqlite_files_for_camera(
        args.sqlite_dir, args.container_id,
        TARGET_CAMERA, TARGET_IMAGE_GROUP_OFFSET, args.nsys_reps
    )

    if not selected_files:
        print('ERROR: no sqlite files found', file=sys.stderr)
        sys.exit(1)

    # Extract stages from each rep
    all_runs_events = []
    for db_path in selected_files:
        events = extract_all_stages(db_path)
        if events is not None:
            all_runs_events.append(events)

    if args.dump_nvtx_names:
        all_names = set()
        for run_events in all_runs_events:
            all_names.update(run_events.keys())
        print('\nAll NVTX stage names found:')
        for name in sorted(all_names):
            print(f'  {name}')
        print()

    if not all_runs_events:
        print('ERROR: all sqlite reads failed', file=sys.stderr)
        sys.exit(1)

    print(f'\nSuccessfully read {len(all_runs_events)}/{len(selected_files)} sqlite files',
          file=sys.stderr)

    # Compute stats per stage
    stats = compute_stage_stats(all_runs_events)

    # Save CSV
    fieldnames = ['camera', 'stage', 'median_s', 'iqr_s', 'count', 'MP']
    rows = []
    for display_name, (med_s, iqr_s, count) in stats.items():
        rows.append({
            'camera': TARGET_CAMERA,
            'stage': display_name,
            'median_s': round(med_s, 10),
            'iqr_s': round(iqr_s, 10),
            'count': count,
            'MP': TARGET_MP,
        })

    # Sort by median descending (same convention as existing data)
    rows.sort(key=lambda r: r['median_s'], reverse=True)

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nSaved {len(rows)} stage rows to {args.output}')

    # Summary table
    print(f'\n{"Stage":<35} {"Median (s)":>12} {"IQR (s)":>10} {"Count":>6}')
    print('-' * 70)
    for r in rows:
        print(f'{r["stage"]:<35} {r["median_s"]:>12.4f} {r["iqr_s"]:>10.4f} {r["count"]:>6}')


if __name__ == '__main__':
    main()
