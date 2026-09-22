#!/usr/bin/env python3
"""Regenerate figures 7 and 8 from the post-fix campaign, arms paired in time.

Only these two figures can be rebuilt: both restrict themselves to the three
datacenter GPUs, and those are exactly the three hosts the fix1 campaign covered.
Every other table and figure needs the four cards nobody re-measured, so they stay
on the campaign data.

Figure 7 is written into the manuscript itself; figure 8 cannot be rebuilt at all,
because it needs adaptive against forced and the post-fix campaign only ran forced
mode on three images per host, not nineteen.

That leaves the manuscript with figure 7 post-fix and everything else pre-fix, so
the caption has to say so: the tables and the heatmap need the four cards nobody
re-measured, and figure 8 needs an arm nobody measured.
"""
import os
import shutil
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

PAIRED = os.path.join(BASE, 'data', 'benchmark_latency_fig78_paired.csv')
OUT = os.path.join(os.path.dirname(BASE), 'figures_profiler_fix1')

import generate_manuscript_tables as T                                    # noqa: E402
import generate_manuscript_figures as F                                   # noqa: E402

os.makedirs(OUT, exist_ok=True)
T.BENCHMARK_CSV = PAIRED          # figures read the benchmark through the tables module
F.OUT_DIR = OUT
MANUSCRIPT = os.path.join(os.path.dirname(BASE), 'GPUPHOT_manuscript', 'figures')

F.figure7()
for ext in ('pdf', 'png'):
    src = os.path.join(OUT, f'fig7_py38_vs_adaptive.{ext}')
    shutil.copy2(src, os.path.join(MANUSCRIPT, f'fig7_py38_vs_adaptive.{ext}'))
print(f'\nfigura 7 post-arreglo copiada a {MANUSCRIPT}')
print('figura 8 NO regenerada: el modo forzado solo se midió en 3 imágenes por máquina')
