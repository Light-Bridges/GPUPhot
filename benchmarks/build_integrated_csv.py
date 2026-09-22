#!/usr/bin/env python3
"""build_integrated_csv.py — Merge 5E data into master benchmark CSV.

Writes: data/benchmark_latency_integrated.csv

Policy:
  - historical rows get catalog_backend='vizier' (assumption: Vizier was the
    only catalog before the 5E local-catalog campaign)
  - NaN machines for datacenter GPUs are resolved using the known GPU→machine
    mapping (one machine per GPU in this project)
  - 5E rows replace historical rows for the same
    (machine, gpu_label, profiler_label, image_label, catalog_backend)
  - 5E source_round column is dropped (5E-internal, not part of master schema)
"""
import os, sys, pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from generate_manuscript_tables import GPU_LABEL_MAP

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

df_master = pd.read_csv(os.path.join(DATA_DIR, 'benchmark_latency.csv'))
df_5e     = pd.read_csv(os.path.join(DATA_DIR, 'measure_151mp_datacenter_5E_unified.csv'))

# ── Prepare master ──
def add_gpu_label(df):
    df = df.copy()
    df['_gpu_label'] = (df['gpu_name'].str.replace('NVIDIA ', '', regex=False)
                        .map(GPU_LABEL_MAP)
                        .fillna(df['gpu_name'].str.replace('NVIDIA ', '', regex=False)))
    return df

df_master = add_gpu_label(df_master)
df_master['catalog_backend'] = 'vizier'

GPU_TO_MACHINE = {
    'H100 (80 GB)':  'azken',
    'A100 (80 GB)':  'lenovo_tttserver',
    'L40S (48 GB)':  'hp3',
}
nan_mask = df_master['machine'].isna()
df_master.loc[nan_mask, 'machine'] = df_master.loc[nan_mask, '_gpu_label'].map(GPU_TO_MACHINE)
print(f'NaN machines resolved: {nan_mask.sum()}, remaining: {df_master["machine"].isna().sum()}')

# ── Prepare 5E ──
df_5e = add_gpu_label(df_5e)
df_5e_clean = df_5e.drop(columns=['source_round', '_gpu_label'], errors='ignore')

KEY = ['machine', '_gpu_label', 'profiler_label', 'image_label', 'catalog_backend']
e5_cells = df_5e[KEY].drop_duplicates()
print(f'5E unique cells: {len(e5_cells)}')

# ── Dedup: remove master rows that will be replaced by 5E ──
merged = df_master.merge(e5_cells, on=KEY, how='left', indicator=True)
keep_mask = (merged['_merge'] == 'left_only').values
df_master_kept = df_master[keep_mask].drop(columns=['_gpu_label'])
removed_count = (~keep_mask).sum()
print(f'Master rows kept: {len(df_master_kept)}, removed: {removed_count}')
print(f'5E rows to add:   {len(df_5e_clean)}')

# ── Align schemas ──
master_cols = list(df_master_kept.columns)
extra_e5 = [c for c in df_5e_clean.columns if c not in master_cols]
if extra_e5:
    print(f'WARNING: 5E-only columns dropped: {extra_e5}')

df_integrated = pd.concat(
    [df_master_kept, df_5e_clean.reindex(columns=master_cols)],
    ignore_index=True
)
print(f'Integrated CSV:  {len(df_integrated)} rows  (was {len(df_master)} master rows + {len(df_5e_clean)} 5E rows - {removed_count} removed)')

# ── Write ──
outpath = os.path.join(DATA_DIR, 'benchmark_latency_integrated.csv')
df_integrated.to_csv(outpath, index=False)
print(f'Written: {outpath}')

# ── Sanity: 5E cell counts ──
sub5e = df_integrated[df_integrated['remeasure_round'] == '5E'].copy()
sub5e['_gl'] = (sub5e['gpu_name'].str.replace('NVIDIA ','',regex=False)
                .map(GPU_LABEL_MAP)
                .fillna(sub5e['gpu_name'].str.replace('NVIDIA ','',regex=False)))
print('\n5E rows in integrated CSV per cell:')
ct = (sub5e.groupby(['machine','profiler_label','image_label','catalog_backend'])
      ['execution_time'].count().reset_index()
      .rename(columns={'execution_time':'N'})
      .sort_values(['machine','profiler_label','image_label','catalog_backend']))
print(ct.to_string(index=False))
