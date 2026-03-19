#!/usr/bin/env python3
"""
Extract benchmark results from nsys .sqlite files into a structured CSV.

Reads all .sqlite files in a directory and produces a single CSV with:
- Machine/GPU metadata
- Per-run process_image timing
- NVTX phase breakdown (top-level functions)
- CUDA memory transfer stats
- GPU hardware info

Usage:
    python extract_benchmark_csv.py <results_dir> [--output results.csv]
"""

import argparse
import csv
import sqlite3
import sys
from pathlib import Path


# Top-level NVTX phases we care about for the manuscript
KEY_PHASES = [
    'process_image',
    'calibrate_image',
    'get_local_background_fft',
    'get_mean_std',
    'CR_filter',
    'SP_filter',
    'detect_gpu',
    'detect_sources_psf',
    'detect_isolated_stars',
    'create_star_dataset',
    'group_star_dataset',
    'get_eigen_psfs',
    'create_coeff_map',
    'fit_moffat',
    'perform_opt_photometry',
    'batch_aperture_photometry',
    'calculate_aperture_corrections_gpu',
    'create_aperture_corrections_map_gpu',
    'find_aperture_corrections_gpu',
    'astrometrice2',
    '_attempt_local_solve',
    '_attempt_online_solve',
    'crossmatch_sources',
    'crossmatch_sources_gpu_impl',
    'crossmatch_sources_cpu_impl',
    '__getVizier',
    'catalog_results',
    'get_zeropoint',
    'get_target_snr',
    'get_maglim',
    'update_header_with_astrometry',
    'update_header_with_photometry',
    'convolve_fft',
    'fill_nan_fft',
    'gen_apm_filter',
    'batch_aper_kernel',
    'gaussian_kernel',
    'gen_moff_filter',
    'gen_moff_filter2',
    'get_sky',
    'free_gpu_mem',
    'reset_cupy_allocators',
    'adaptive_memory_management',
    'stack_sigmaclip',
    'radec_to_moon_sun',
    'radec_to_altaz',
]


def extract_from_sqlite(db_path: str) -> dict:
    """Extract benchmark data from a single nsys .sqlite file."""
    result = {
        'source_file': Path(db_path).stem,
        'gpu_name': '',
        'gpu_uuid': '',
    }

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()

        # Check tables exist
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}

        if 'NVTX_EVENTS' not in tables or 'StringIds' not in tables:
            conn.close()
            return None

        # GPU info
        if 'TARGET_INFO_GPU' in tables:
            cur.execute("SELECT name, uuid FROM TARGET_INFO_GPU LIMIT 1")
            row = cur.fetchone()
            if row:
                result['gpu_name'] = (row[0] or '').strip('[] ')
                result['gpu_uuid'] = (row[1] or '').strip()

        # Command line args (to identify image and instrument)
        result['command_args'] = ''
        for tbl in ['ANALYSIS_DETAILS', 'META_DATA_CAPTURE']:
            if tbl not in tables:
                continue
            try:
                if tbl == 'ANALYSIS_DETAILS':
                    cur.execute("SELECT value FROM ANALYSIS_DETAILS WHERE LOWER(key) LIKE 'argument%' ORDER BY CAST(SUBSTR(key, 9) AS INTEGER)")
                else:
                    cur.execute("SELECT value FROM META_DATA_CAPTURE WHERE name LIKE 'PROCESS_%:ARGUMENT_%' ORDER BY CAST(SUBSTR(name, INSTR(name, ':ARGUMENT_') + 10) AS INTEGER)")
                args = [r[0] for r in cur.fetchall()]
                relevant, found = [], False
                for a in args:
                    if 'profile_process_image' in str(a):
                        found = True
                        continue
                    if found:
                        relevant.append(str(a))
                if relevant:
                    result['command_args'] = ' '.join(relevant)
                    break
            except Exception:
                pass

        # NVTX events — all phases
        cur.execute("""
            SELECT t2.value,
                   COUNT(*),
                   SUM(t1.end - t1.start) / 1e9,
                   AVG(t1.end - t1.start) / 1e6,
                   MIN(t1.end - t1.start) / 1e6,
                   MAX(t1.end - t1.start) / 1e6
            FROM NVTX_EVENTS t1 JOIN StringIds t2 ON t1.textId = t2.id
            WHERE t1.eventType = 59 AND t1.end IS NOT NULL
              AND t1.start IS NOT NULL AND (t1.end - t1.start) > 0
            GROUP BY t2.value
        """)

        for name, calls, total_s, mean_ms, min_ms, max_ms in cur.fetchall():
            if name in KEY_PHASES:
                result[f'nvtx_{name}_total_s'] = round(total_s, 6)
                result[f'nvtx_{name}_mean_ms'] = round(mean_ms, 3)
                result[f'nvtx_{name}_calls'] = calls

        # CUDA memory transfers
        if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
            cur.execute("""
                SELECT copyKind, COUNT(*), SUM(bytes)/1e6, SUM(end-start)/1e6
                FROM CUPTI_ACTIVITY_KIND_MEMCPY
                GROUP BY copyKind
            """)
            kind_map = {1: 'h2d', 2: 'd2h', 8: 'd2d'}
            for kind, ops, total_mb, total_ms in cur.fetchall():
                label = kind_map.get(kind, f'kind{kind}')
                result[f'cuda_memcpy_{label}_ops'] = ops
                result[f'cuda_memcpy_{label}_mb'] = round(total_mb, 1)
                result[f'cuda_memcpy_{label}_ms'] = round(total_ms, 1)

        # CUDA memset
        if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables:
            cur.execute("SELECT COUNT(*), SUM(bytes)/1e6 FROM CUPTI_ACTIVITY_KIND_MEMSET")
            row = cur.fetchone()
            if row:
                result['cuda_memset_ops'] = row[0]
                result['cuda_memset_mb'] = round(row[1] or 0, 1)

        conn.close()
        return result

    except Exception as e:
        print(f"  WARN: {Path(db_path).name}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description='Extract benchmark CSV from nsys sqlite files')
    parser.add_argument('results_dir', help='Directory with .sqlite files')
    parser.add_argument('--output', '-o', default=None, help='Output CSV path (default: <results_dir>/benchmark_nvtx.csv)')
    parser.add_argument('--machine', '-m', default='unknown', help='Machine name to include in CSV')
    parser.add_argument('--profiler', '-p', default='unknown', help='Profiler label (e.g. py312, py38)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output = Path(args.output) if args.output else results_dir / 'benchmark_nvtx.csv'

    sqlite_files = sorted(results_dir.glob('*.sqlite'))
    if not sqlite_files:
        print(f"No .sqlite files found in {results_dir}")
        sys.exit(1)

    print(f"Extracting from {len(sqlite_files)} sqlite files in {results_dir}")

    rows = []
    for db in sqlite_files:
        data = extract_from_sqlite(str(db))
        if data:
            data['machine'] = args.machine
            data['profiler'] = args.profiler
            rows.append(data)

    if not rows:
        print("No data extracted")
        sys.exit(1)

    # Collect all column names
    all_cols = ['machine', 'profiler', 'source_file', 'gpu_name', 'gpu_uuid', 'command_args']
    nvtx_cols = sorted(set(k for r in rows for k in r if k.startswith('nvtx_')))
    cuda_cols = sorted(set(k for r in rows for k in r if k.startswith('cuda_')))
    all_cols.extend(nvtx_cols)
    all_cols.extend(cuda_cols)

    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {output}")
    print(f"Columns: {len(all_cols)} ({len(nvtx_cols)} NVTX phases, {len(cuda_cols)} CUDA metrics)")


if __name__ == '__main__':
    main()
