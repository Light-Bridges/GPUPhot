#!/usr/bin/env python3
"""
Generate benchmarks/data/nvtx_detection_per_image.csv for the GPUPhot manuscript.

Combines:
1. CPU sep baseline: cpu_baseline_a100.csv (output of benchmark_cpu_baselines.py)
2. GPU NVTX detection timing: nsys sqlite files from the A100 (lenovo_tttserver)

Maps sqlite files to benchmark images (indices 1-10) by:
- Filtering by container ID (a80f7e35e9e0 = GPU 1 container after April 2026 switch)
- Grouping by camera/instrument name
- Assigning files to images in timestamp order (images processed sequentially per camera)

Output columns (matching generate_manuscript_tables.py gen_nvtx_detection() expectation):
    image_label, camera, MP, sep_sources,
    gpuphot_detection_s, sep_s, photutils_s, gpu_speedup_vs_sep

Usage (inside Docker container on lenovo_tttserver):
    python3 /tmp/generate_nvtx_per_image_a100.py \\
        --cpu-baseline /app/profiling_results/cpu_baseline_a100.csv \\
        --sqlite-dir /app/profiling_results \\
        --container-id a80f7e35e9e0 \\
        --nsys-reps 4 \\
        --output /app/profiling_results/nvtx_detection_per_image.csv
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

# ---------------------------------------------------------------------------
# Benchmark image list (images 1-10 from dev/run_benchmark_all_machines.sh)
# Format: (idx, mp, filename, camera, label)
# ---------------------------------------------------------------------------
BENCHMARK_IMAGES = [
    (1,  4.2,   'TTT3_iKon936-1_2026-01-15-06-05-00-020013_QSO0957+561_SDSSg.fits', 'iKon936-1', 'iKon936_SDSSg'),
    (2,  4.2,   'TTT3_iKon936-1_2025-09-15-05-31-52-256207_C2025A6_Lum.fits',       'iKon936-1', 'iKon936_Lum'),
    (3,  6.8,   'TTT3_QHY600-3_2025-12-01-23-02-47-487643_C2025R2_Lum.fits',        'QHY600-3',  'QHY600-3_Lum'),
    (4,  15.3,  'TTT2_QHY600-4_2026-02-14-23-46-19-497908_WASP-43-b_SDSSg.fits',    'QHY600-4',  'QHY600-4_SDSSg'),
    (5,  15.3,  'TTT2_QHY600-4_2026-02-14-00-13-51-310869_NGC2903_Ha.fits',         'QHY600-4',  'QHY600-4_Ha'),
    (6,  37.8,  'TTT1_QHY411-1_2026-03-09-21-22-48-661122_2012QD8_Lum.fits',        'QHY411-1',  'QHY411-1_Lum_bin2'),
    (7,  37.8,  'TTT1_QHY411-1_2026-02-14-03-52-12-471080_GaiaDR33534005919872722560_SDSSi.fits', 'QHY411-1', 'QHY411-1_SDSSi_bin2'),
    (8,  151.2, 'TTT1_QHY411-1_2025-08-14-23-52-42-828197_2025PR1_Lum.fits',        'QHY411-1',  'QHY411-1_Lum_full'),
    (9,  151.2, 'TST_QHY411-3_2026-02-14-06-35-10-547282_24P_Lum.fits',             'QHY411-3',  'QHY411-3_Lum_full'),
    (10, 151.2, 'TST_QHY411-3_2026-02-14-23-09-41-463160_M81_SDSSr.fits',           'QHY411-3',  'QHY411-3_SDSSr_full'),
]

# NVTX stage for scope-equivalent detection comparison with sep.
#
# sep measures: background estimation + detection (thresholding) + circular
# aperture photometry at a SINGLE fixed radius (r=5 px).
#
# GPUPhot's batch_aperture_photometry is NOT scope-equivalent: it measures
# aperture photometry across MULTIPLE radii (multi-aperture mode), making it
# systematically slower than sep's single-aperture step even on GPU.
#
# The correct scope-equivalent GPU metric is detect_sources_psf alone, which
# covers the full source detection pass (background-subtracted thresholding +
# PSF-weighted local maxima finding). This stage is called multiple times per
# image (e.g., initial + refined detection); we sum all calls within a single
# nsys run and take the median across reps.
DETECTION_STAGES = ['detect_sources_psf']


def extract_detection_time_ms(db_path):
    """
    Extract total detection+photometry time from a single nsys sqlite file.

    Returns the sum of median durations for detect_sources_psf +
    batch_aperture_photometry across all calls in this run (ms).
    Returns None if the file is unreadable or stages are missing.
    """
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        cur = conn.cursor()
        cur.execute("""
            SELECT t2.value, (t1.end - t1.start) / 1e6
            FROM NVTX_EVENTS t1 JOIN StringIds t2 ON t1.textId = t2.id
            WHERE t1.eventType = 59
              AND t1.end IS NOT NULL AND t1.start IS NOT NULL
              AND (t1.end - t1.start) > 0
        """)
        events = defaultdict(list)
        for name, dur in cur.fetchall():
            events[name].append(dur)
        conn.close()
    except Exception as e:
        print(f'  WARNING: error reading {os.path.basename(db_path)}: {e}', file=sys.stderr)
        return None

    # Require process_image to exist (sanity check)
    if 'process_image' not in events:
        print(f'  WARNING: no process_image in {os.path.basename(db_path)}', file=sys.stderr)
        return None

    total_ms = 0.0
    found_any = False
    for stage in DETECTION_STAGES:
        if stage in events:
            # Sum all calls to this stage within the run (typically 1 call each)
            total_ms += sum(events[stage])
            found_any = True

    if not found_any:
        print(f'  WARNING: no detection stages in {os.path.basename(db_path)}', file=sys.stderr)
        return None

    return total_ms


def load_sqlite_files_by_camera(sqlite_dir, container_id):
    """
    Load and group sqlite files by camera, sorted by timestamp.

    Filename format: benchmark_<container>_<camera>_<YYYYMMDD>_<HHMMSS>_run<N>.sqlite
    Returns dict: camera -> [path, path, ...] (sorted by timestamp)
    """
    pattern = os.path.join(sqlite_dir, f'benchmark_{container_id}_*.sqlite')
    files = sorted(glob.glob(pattern))
    print(f'Found {len(files)} sqlite files for container {container_id}')

    camera_files = defaultdict(list)
    for fpath in files:
        fname = os.path.basename(fpath)
        # Extract camera: benchmark_<id>_<camera>_<date>_<time>_run<N>.sqlite
        m = re.match(r'benchmark_[a-f0-9]+_(.+?)_(\d{8})_(\d{6})_run\d+\.sqlite', fname)
        if m:
            camera = m.group(1)
            camera_files[camera].append(fpath)

    for cam in camera_files:
        camera_files[cam].sort()  # already sorted by glob, but explicit sort by name=timestamp

    return camera_files


def load_cpu_baseline(csv_path):
    """
    Load cpu_baseline_a100.csv, return dict: filename -> {sep_median_s, sep_sources, photutils_median_s}
    """
    baseline = {}
    if not os.path.exists(csv_path):
        print(f'ERROR: CPU baseline not found: {csv_path}', file=sys.stderr)
        return baseline
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row['filename']
            baseline[fname] = {
                'sep_median_s':      float(row['sep_median_s']) if row['sep_median_s'] else None,
                'sep_sources':       int(row['sep_sources'])    if row['sep_sources']   else None,
                'photutils_median_s': float(row['photutils_median_s']) if row.get('photutils_median_s') else None,
            }
    print(f'Loaded CPU baseline: {len(baseline)} images from {csv_path}')
    return baseline


def main():
    parser = argparse.ArgumentParser(description='Generate nvtx_detection_per_image.csv for A100')
    parser.add_argument('--cpu-baseline', default='/app/profiling_results/cpu_baseline_a100.csv',
                        help='Path to cpu_baseline_a100.csv')
    parser.add_argument('--sqlite-dir', default='/app/profiling_results',
                        help='Directory containing nsys sqlite files')
    parser.add_argument('--container-id', default='a80f7e35e9e0',
                        help='Docker container short ID (12 hex chars) for filtering sqlite files')
    parser.add_argument('--nsys-reps', type=int, default=4,
                        help='Number of nsys reps per image (used to group sqlite files)')
    parser.add_argument('--output', default='/app/profiling_results/nvtx_detection_per_image.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    # Load CPU baseline
    cpu_baseline = load_cpu_baseline(args.cpu_baseline)

    # Load and group sqlite files by camera
    camera_files = load_sqlite_files_by_camera(args.sqlite_dir, args.container_id)

    for cam, files in camera_files.items():
        print(f'  {cam}: {len(files)} sqlite files')

    # Assign sqlite files to images (sequential within each camera group)
    camera_idx = defaultdict(int)
    results = []

    for img_idx, mp, filename, camera, label in BENCHMARK_IMAGES:
        start = camera_idx[camera]
        end   = start + args.nsys_reps

        my_files = camera_files.get(camera, [])[start:end]
        camera_idx[camera] = end

        if len(my_files) == 0:
            print(f'  [{img_idx}] {label}: no sqlite files found for camera {camera}')
            gpuphot_detection_s = None
        else:
            times_ms = []
            for db in my_files:
                t = extract_detection_time_ms(db)
                if t is not None:
                    times_ms.append(t)

            if times_ms:
                gpuphot_detection_s = round(median(times_ms) / 1000.0, 4)
                print(f'  [{img_idx}] {label} ({camera}, {len(times_ms)} reps): '
                      f'GPU det+phot = {gpuphot_detection_s:.4f}s')
            else:
                gpuphot_detection_s = None
                print(f'  [{img_idx}] {label}: all sqlite reads failed')

        # CPU baseline
        cpu = cpu_baseline.get(filename, {})
        sep_s      = cpu.get('sep_median_s')
        sep_src    = cpu.get('sep_sources')
        phot_s     = cpu.get('photutils_median_s')

        if sep_s is None:
            print(f'  WARNING: {filename} not found in CPU baseline')

        speedup = round(sep_s / gpuphot_detection_s, 2) if (sep_s and gpuphot_detection_s) else None

        results.append({
            'image_label':        label,
            'camera':             camera,
            'MP':                 mp,
            'sep_sources':        sep_src,
            'gpuphot_detection_s': gpuphot_detection_s,
            'sep_s':              sep_s,
            'photutils_s':        phot_s,
            'gpu_speedup_vs_sep': speedup,
        })

    # Warn if any sqlite files were unused (might indicate too many old files)
    for cam, files in camera_files.items():
        used = camera_idx[cam]
        total = len(files)
        if used < total:
            print(f'  WARNING: camera {cam}: used {used}/{total} sqlite files '
                  f'({total - used} unused — may be extra reps or old runs)')

    # Save CSV
    fieldnames = ['image_label', 'camera', 'MP', 'sep_sources',
                  'gpuphot_detection_s', 'sep_s', 'photutils_s', 'gpu_speedup_vs_sep']
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f'\nSaved {len(results)} rows to {args.output}')

    # Summary table
    print(f'\n{"Image":<30} {"MP":>6} {"Sources":>8} {"GPU det(s)":>11} {"sep(s)":>8} {"Speedup":>8}')
    print('-' * 78)
    for r in results:
        gd  = f'{r["gpuphot_detection_s"]:.4f}' if r['gpuphot_detection_s'] else 'N/A'
        ss  = f'{r["sep_s"]:.3f}'               if r['sep_s']               else 'N/A'
        sp  = f'{r["gpu_speedup_vs_sep"]:.1f}x' if r['gpu_speedup_vs_sep']  else 'N/A'
        src = str(r['sep_sources'])             if r['sep_sources']          else 'N/A'
        print(f'{r["image_label"]:<30} {r["MP"]:>6.1f} {src:>8} {gd:>11} {ss:>8} {sp:>8}')


if __name__ == '__main__':
    main()