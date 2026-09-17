#!/usr/bin/env python3
"""
Extract NVTX stage breakdown from post-PCA-fix py3.12 profiling campaign (2026-06-07).

Input:  benchmarks/benchmark_results/benchmark_<cid>_<camera>_20260607_*.sqlite
Output: benchmarks/data/nvtx_pca_stage_postfix.csv

Schema matches the per-stage CSVs in data/:
  camera, stage, median_s, iqr_s, count, MP
  + extra columns: gpu, profiler_label  (for comparison / merging)

Container → GPU mapping (campaign 2026-06-07):
  18a4d753caa0  →  H100 PCIe (azken)
  ce4950a22daa  →  L40S (hp3)
  4d6efd00b9d2  →  A100-SXM4 (lenovo_tttserver)
"""

import csv
import glob
import os
import re
import sqlite3
import sys
from collections import defaultdict
from statistics import median

# ── Config ────────────────────────────────────────────────────────────────────

SQLITE_DIR  = os.path.join(os.path.dirname(__file__), 'benchmark_results')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'nvtx_pca_stage_postfix.csv')

DATE_FILTER = '20260607'  # only files from this campaign

CONTAINER_GPU = {
    '18a4d753caa0': 'H100 (80 GB)',
    'ce4950a22daa': 'L40S (48 GB)',
    '4d6efd00b9d2': 'A100 (80 GB)',
}

CAMERA_MP = {
    'iKon936-1': 4.2,
    'QHY411-1':  151.2,
    'QHY411-3':  151.2,
}

# NVTX function name → human-readable stage (same mapping as existing data)
NVTX_TO_DISPLAY = {
    'get_local_background_fft':                           'Background estimation (FFT)',
    'detect_isolated_stars':                              'Star detection',
    'detect_sources_psf':                                 'PSF source detection',
    'create_star_dataset':                                'PSF cutout extraction',
    'get_eigen_psfs':                                     'Eigen-PSF (PCA)',
    'create_coeff_map':                                   'PSF coefficient map',
    'calculate_tile_nanmean_sigclip':                     'Tile statistics',
    'batch_aperture_photometry':                          'Aperture photometry',
    'perform_opt_photometry':                             'Optimal photometry',
    'perform_opt_photometry_optimized_gpu_crossmatch':    'Optimal photometry',
    'convolve_fft':                                       'FFT convolution',
    'crossmatch_sources':                                 'Source crossmatch',
    'project_all_stars_onto_eigenpsfs':                   'Star projection',
    'fit_moffat':                                         'Moffat PSF fitting',
    'SP_filter':                                          'Salt-and-pepper filter',
    'CR_filter':                                          'Cosmic-ray filter',
    'get_zeropoint':                                      'Zero-point calibration',
    'astrometrice2':                                      'Astrometry.net solver',
    'update_header_with_astrometry':                      'WCS fitting & header update',
    'catalog_results':                                    'Catalog query (network I/O)',
    '__getVizier':                                        'Catalog query (network I/O)',
}

STAGE_TYPE = {
    'Background estimation (FFT)':  'GPU',
    'Star detection':               'GPU',
    'PSF source detection':         'GPU',
    'PSF cutout extraction':        'GPU',
    'Eigen-PSF (PCA)':              'GPU',
    'PSF coefficient map':          'GPU',
    'Tile statistics':              'GPU',
    'Aperture photometry':          'GPU',
    'Optimal photometry':           'GPU',
    'FFT convolution':              'GPU',
    'Source crossmatch':            'GPU',
    'Star projection':              'GPU',
    'Moffat PSF fitting':           'CPU',
    'Salt-and-pepper filter':       'CPU',
    'Cosmic-ray filter':            'CPU',
    'Zero-point calibration':       'CPU',
    'Astrometry.net solver':        'CPU',
    'WCS fitting & header update':  'CPU',
    'Catalog query (network I/O)':  'I/O',
}


def extract_stages(db_path):
    """Extract NVTX stage durations (seconds) from one sqlite file."""
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        cur  = conn.cursor()
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
        if 'process_image' not in events:
            print(f'  WARN: no process_image in {os.path.basename(db_path)}',
                  file=sys.stderr)
            return None
        return dict(events)
    except Exception as e:
        print(f'  ERROR {os.path.basename(db_path)}: {e}', file=sys.stderr)
        return None


def iqr(values):
    s = sorted(values)
    n = len(s)
    q1 = s[n // 4]
    q3 = s[(3 * n) // 4] if (3 * n) // 4 < n else s[-1]
    return q3 - q1


def main():
    pattern = os.path.join(SQLITE_DIR, f'benchmark_*_{DATE_FILTER}_*.sqlite')
    files   = sorted(glob.glob(pattern))
    print(f'Found {len(files)} sqlite files for {DATE_FILTER}')

    # Group by (container_id, camera) → list of sqlite paths
    groups = defaultdict(list)
    for fpath in files:
        fname = os.path.basename(fpath)
        # benchmark_<cid12>_<camera>_<YYYYMMDD>_<HHMMSS>_run<N>.sqlite
        m = re.match(
            r'benchmark_([0-9a-f]{12})_(iKon936-1|QHY411-1|QHY411-3)_'
            r'(\d{8})_(\d{6})_run(\d+)\.sqlite', fname)
        if not m:
            print(f'  SKIP (unrecognised): {fname}', file=sys.stderr)
            continue
        cid, camera = m.group(1), m.group(2)
        groups[(cid, camera)].append(fpath)

    print(f'Groups: {len(groups)}  ({sorted(groups.keys())})')

    # Aggregate per group
    all_rows = []
    for (cid, camera), sqlite_paths in sorted(groups.items()):
        gpu = CONTAINER_GPU.get(cid, f'unknown({cid})')
        mp  = CAMERA_MP.get(camera, 0)
        print(f'\n  [{gpu}] {camera} ({len(sqlite_paths)} files)')

        # Collect stage durations across all reps
        per_display = defaultdict(list)
        for sp in sqlite_paths:
            ev = extract_stages(sp)
            if ev is None:
                continue
            for nvtx_name, durs in ev.items():
                display = NVTX_TO_DISPLAY.get(nvtx_name)
                if display:
                    per_display[display].extend(durs)

        # Print summary and build output rows
        print(f'  {"Stage":<40} {"Median(s)":>10} {"IQR(s)":>8} {"N":>5}')
        print(f'  {"-"*65}')
        for display_name, durs in sorted(
                per_display.items(), key=lambda x: -median(x[1])):
            med = median(durs)
            iq  = iqr(durs)
            n   = len(durs)
            print(f'  {display_name:<40} {med:>10.4f} {iq:>8.4f} {n:>5}')
            all_rows.append({
                'camera':         camera,
                'stage':          display_name,
                'median_s':       round(med, 6),
                'iqr_s':          round(iq,  6),
                'count':          n,
                'MP':             mp,
                'gpu':            gpu,
                'profiler_label': 'py312_cuml_adaptive',
            })

    # Write CSV
    fieldnames = ['camera', 'stage', 'median_s', 'iqr_s', 'count',
                  'MP', 'gpu', 'profiler_label']
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sorted(all_rows, key=lambda r: (r['camera'], r['gpu'], -r['median_s'])))

    print(f'\nWritten {len(all_rows)} rows → {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
