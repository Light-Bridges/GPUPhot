#!/usr/bin/env python3
"""
build_memory_csv.py
Build the peak-VRAM CSVs for the August-2026 nsys campaign in the two shapes the
manuscript pipeline consumes: a raw per-run file for generate_manuscript_tables.py
and a per-cell summary for generate_manuscript_figures.py.

Input  : benchmarks/results_collected/latency_campaign/profiler_nsys_memory_FINAL.csv
Outputs: benchmarks/data/profiler_memory_raw.csv
         benchmarks/data/profiler_memory_summary.csv

Two corrections are applied to the export:

  Megapixels.  The extractor mapped camera name to image size, but QHY411-1 shoots
  both binned (37.8 MP) and full frame (151.2 MP), so its full-frame runs are
  labelled 37.8 MP.  They are unambiguous by footprint: a 37.8 MP frame peaks near
  8.5 GB and a 151.2 MP frame near 27 GB, so QHY411-1 rows above 20 GB are
  relabelled 151.2 MP.  Left uncorrected they would drag the 37.8 MP row of the
  VRAM table upward.

  capture_complete.  The v3/v4 export carries no completion flag, so it is taken
  per run from nsys_completion_flags.csv, extracted from the .sqlite traces of all
  six hosts.  The criterion there is the pipeline milestones, not whether the
  process_image NVTX range closed: the decorator closes that range on the way out
  of an exception too, which marks every trace complete.  A run counts as complete
  only if get_zeropoint and update_header_with_astrometry both closed, which an
  OOM in the FFT never reaches.  The `completion` column keeps how far each run
  got.

  The cell-level rule this replaced is kept as an independent cross-check: a run
  whose machine, stack and image size never produced a successful pipeline run in
  the latency campaign cannot be a complete capture.  The two disagreeing is a
  hard error, the same way C4 checks C2 in the latency builder.

  Without this the peak recorded just before an OOM crash is published as if it
  were a working set, and it is a low number precisely because the card is small:
  the export credits the RTX 3050 Ti with 3465 MB for a 151.2 MP frame that the
  H100, A100 and L40S all measure at 26668 MB.  The physical guard already in
  vram_median(), peak <= gpu_total, cannot catch this because a pre-crash peak is
  by construction below the card's capacity.

peak_gpu_memory_pct is simply recomputed from peak / total.
"""

import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from build_unified_csv import RAW_CSV, MACHINE_MAP   # noqa: E402

# The nsys extractor names the 4.2 MP camera with its instrument suffix; the
# latency campaign labels it without.
CAMERA_ALIAS = {'iKon936-1': 'iKon936'}
SRC  = os.path.join(BASE, 'data', 'raw', 'nsys_memory_peaks.csv')
FLAGS = os.path.join(BASE, 'data', 'raw', 'nsys_completion_flags.csv')
RAW_OUT = os.path.join(BASE, 'data', 'profiler_memory_raw.csv')
SUM_OUT = os.path.join(BASE, 'data', 'profiler_memory_summary.csv')

df = pd.read_csv(SRC)

# One ttt_server run reports gpu_total_MB = 0 (bad NVML read); the percentage is
# then infinite and the row cannot be checked against the card's capacity.
_bad_total = df['gpu_total_MB'] <= 0
if _bad_total.any():
    print(f'dropped {int(_bad_total.sum())} rows with gpu_total_MB <= 0 (bad NVML read)')
    df = df[~_bad_total].copy()

# QHY411-1 full-frame runs mislabelled as its binned size (see module docstring).
_mislabelled = (df['camera'] == 'QHY411-1') & (df['peak_gpu_memory_MB'] > 20000)
df.loc[_mislabelled, 'megapixels'] = 151.2
print(f'relabelled {int(_mislabelled.sum())} QHY411-1 full-frame rows to 151.2 MP')

df['peak_gpu_memory_pct'] = (100.0 * df['peak_gpu_memory_MB'] / df['gpu_total_MB']).round(2)
df = df.rename(columns={'file': 'sqlite_file'})

# The April pipeline stored gpu_name without the vendor prefix, and the figures filter
# on the exact string (figure3 does gpu_name == 'A100-SXM4-80GB').  Keeping the prefix
# silently empties that figure, so normalise to the same form.
df['gpu_name'] = df['gpu_name'].str.replace('NVIDIA ', '', regex=False)


def _completed_cells():
    """(machine, python_ver, camera, MP) that produced a real result table.

    Cell-level fallback, kept only as a cross-check on the per-run flags.
    """
    lat = pd.read_csv(RAW_CSV)
    lat = lat[lat['campana'] != 'nsys']
    lat = lat[lat['n_sources_detected'].notna()]
    lat['machine'] = lat['machine'].replace(MACHINE_MAP)
    py = {'py38_baseline': 3.8, 'py312_cuml_adaptive': 3.12, 'py312_cuml_always': 3.12}
    return set(zip(lat['machine'], lat['profiler_label'].map(py),
                   lat['image_label'].str.split('_').str[0], lat['mp']))


flags = pd.read_csv(FLAGS).set_index('sqlite_file')['process_image_cerrado'].astype(str)
_done = _completed_cells()
_cell_ok = pd.Series([k in _done for k in zip(
    df['machine'], pd.to_numeric(df['python_ver']),
    df['camera'].replace(CAMERA_ALIAS), df['megapixels'])], index=df.index)

_per_run = df['sqlite_file'].map(flags)
_uncovered = int(_per_run.isna().sum())
if _uncovered:
    raise AssertionError(f'{_uncovered} profiled runs have no completion flag')

df['completion'] = _per_run
df['capture_complete'] = df['completion'] == '1'

_n_bad = int((~df['capture_complete']).sum())
print(f'capture_complete=False on {_n_bad} of {len(df)} runs')
for k, n in df[~df['capture_complete']].groupby(
        ['machine', 'python_ver', 'camera', 'megapixels', 'completion']).size().items():
    print(f'    {k[0]:<9} py{k[1]:<5} {k[2]:<10} {k[3]:>6} MP  x{n}  {k[4]}')

# Cross-check against the latency campaign: a cell that never produced a result
# table cannot have yielded a complete capture.
_dis = df[df['capture_complete'] & ~_cell_ok]
if len(_dis):
    raise AssertionError(
        'trace flag says complete for runs in cells with no successful latency run:\n'
        + _dis.groupby(['machine', 'python_ver', 'camera', 'megapixels'])
              .size().to_string())
print(f'cross-check: no run flagged complete in a cell that never produced a result')

raw = df[['machine', 'gpu_name', 'gpu_total_MB', 'python_ver', 'camera', 'megapixels',
          'peak_gpu_memory_MB', 'peak_gpu_memory_pct', 'capture_complete',
          'completion', 'sqlite_file']]
raw.to_csv(RAW_OUT, index=False)

# The summary drives the figures, which have no completeness filter of their own.
df_ok = df[df['capture_complete']]

g = df_ok.groupby(['machine', 'gpu_name', 'gpu_total_MB', 'python_ver', 'camera', 'megapixels'])
summary = g['peak_gpu_memory_MB'].agg(
    count='size', median_peak_MB='median',
    q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)).reset_index()
summary['peak_pct'] = (100.0 * summary['median_peak_MB'] / summary['gpu_total_MB']).round(1)
summary.to_csv(SUM_OUT, index=False)

print(f'raw     : {len(raw):4d} rows -> {RAW_OUT}')
print(f'summary : {len(summary):4d} rows -> {SUM_OUT}')
