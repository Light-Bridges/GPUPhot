#!/usr/bin/env python3
"""Render the manuscript's tables and figures as they would look with the fix.

This is a preview, not a decision.  The campaign CSV keeps carrying the system as
deployed; here its cells are replaced, where and only where the fix1 campaign
measured them, by the post-fix numbers, and the whole set is rendered into a
separate directory so the manuscript's current outputs are untouched.

Coverage is partial by construction: fix1 measured three of the seven GPUs, so the
RTX 3090, RTX 3060, RTX 3050 Ti and the two Jetson modules keep their campaign
values.  A table mixing both is not publishable as it stands -- it is here to show
which numbers move and by how much.

azken contributes only its clean night session: its daytime block ran under a
weather model whose epoch reached x3.
"""
import os
import shutil
import sys

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

V4 = os.path.join(BASE, 'data', 'benchmark_latency.csv')
FIX1 = os.path.join(BASE, 'data', 'benchmark_latency_allocator_fix.csv')
MERGED = os.path.join(BASE, 'data', 'benchmark_latency_fix1_preview.csv')
AZKEN_NIGHT_FROM = '2026-08-31T19:00:00Z'

PREVIEW = os.path.join(os.path.dirname(BASE), 'GPUPHOT_manuscript', 'preview_fix1')
FIG_PREVIEW = os.path.join(os.path.dirname(BASE), 'figures_profiler_fix1')


def merge():
    v4 = pd.read_csv(V4)
    f1 = pd.read_csv(FIX1)
    ts = pd.to_datetime(f1['timestamp'], format='mixed', utc=True)
    f1 = f1[(f1['machine'] != 'azken') | (ts >= AZKEN_NIGHT_FROM)]
    f1 = f1[v4.columns]                       # drop zp/ezp/catnstar, keep the schema

    covered = set(map(tuple, f1[['machine', 'profiler_label', 'image_label']]
                      .drop_duplicates().to_numpy()))
    keep = [k not in covered for k in
            zip(v4.machine, v4.profiler_label, v4.image_label)]
    out = pd.concat([v4[keep], f1], ignore_index=True).sort_values('timestamp')
    out.to_csv(MERGED, index=False)
    print(f'celdas sustituidas por fix1 : {len(covered)}')
    print(f'filas v4 conservadas        : {sum(keep)}')
    print(f'filas fix1 añadidas         : {len(f1)}')
    print(f'escrito                     : {MERGED}\n')


def render():
    os.makedirs(PREVIEW, exist_ok=True)
    os.makedirs(FIG_PREVIEW, exist_ok=True)

    # The generators run their whole batch under a __main__ guard, so the
    # individual builders are called here after redirecting their constants.
    import generate_manuscript_tables as T
    T.BENCHMARK_CSV = MERGED
    T.OUT_DIR = PREVIEW
    # The two catalogue-comparison tables read only the April CSV, so fix1 cannot
    # move them; they are skipped rather than regenerated identical.
    for fn in ('gen_latency_py312', 'gen_latency_py38', 'gen_vram_merged',
               'gen_concurrency', 'gen_cuml_ablation', 'gen_cpu_baseline',
               'gen_nvtx_detection', 'gen_stage_breakdown'):
        getattr(T, fn)()

    import generate_manuscript_figures as F
    F.BENCHMARK_CSV = MERGED
    F.OUT_DIR = FIG_PREVIEW
    F.MANUSCRIPT_FIGURES_DIR = os.path.join(PREVIEW, 'figures')
    os.makedirs(F.MANUSCRIPT_FIGURES_DIR, exist_ok=True)
    for fn in ('figure2', 'figure4', 'figure7', 'figure8'):
        getattr(F, fn)()


if __name__ == '__main__':
    merge()
    render()
