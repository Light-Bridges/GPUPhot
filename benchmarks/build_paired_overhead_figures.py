#!/usr/bin/env python3
"""Figures 7 and 8 rebuilt from the post-fix campaign, arms paired in time.

Both figures use only the three datacenter GPUs, and those are exactly the three
hosts the fix1 campaign measured, so there is no coverage gap here -- unlike the
tables, which need the four cards nobody re-measured.

What there could be is a pairing gap, and it is the error this project already
diagnosed once: comparing two arms measured in different windows attributes the
calendar to the variable under study.  So a cell counts only when its two arms
were measured close together -- the same criterion the figure-8 audit used, where
the effect tracked the separation at r = 0.50 and every large bar turned out to be
a badly paired one.

Closeness is measured between the arms' central times, not as an overlap of their
spans: the campaign interleaves arm by arm per image, so within a cell the two
blocks are adjacent rather than simultaneous and their spans barely intersect.

And it is measured session by session, not over the whole arm.  A cell can carry
blocks from several nights, and a median taken over all of them lands between two
distant sessions and matches neither -- which would throw away pairs that were in
fact measured minutes apart.

That rule does the right thing on its own for the host that ran a weather model
during part of its campaign: the sequential block, measured before the paired one
began, falls outside the overlap and never reaches the median.  The check that it
works is internal -- on that host the paired daytime contrast under the model and
the clean night contrast agree to a tenth of a point.

Two corrections to the paragraph above, both from 2026-09-11.

The contamination filter is switched off here on purpose.  It is the figure's own
premise: a cell counts when its two arms share a session, and then whatever slowed
the host slowed both of them.  Leaving the filter on would drop sixteen of azken's
nineteen paired cells for being inside a window the tables exclude, which is the
right call for a table that reports one arm and the wrong one for a ratio of two.
Saying it here also makes the builder idempotent: until today the committed CSV
had been written before that window was registered, so a rebuild silently produced
a different figure from the published one.

And the cancellation is a premise, not a fact, so PAIR_EXCLUSIONS below carries the
cells where it was tested and failed.  The test is per cell: compare how much each
arm was degraded against its own cleanest reference.  In sixteen of azken's
nineteen cells py3.8 is degraded as much as py3.12 or more, which is what the
premise predicts; the entry below is the one cell that goes the other way, and it
goes there by a wide margin.  The test is too noisy to act on at ten points -- the
two arms' clean references come from different nights -- so nothing else is removed
on its strength alone.
"""

# Discarded pairs: within-session cancellation was checked and does NOT hold.
PAIR_EXCLUSIONS = [
    dict(
        machine='azken',
        image_label='QHY411-3_Lum_full',
        # Only the figure-7 pair. The figure-8 one (adaptive vs forced) is a
        # different pair: it runs on 03-Sep, both arms at 220 W and 67 C, and
        # does not have this problem.
        pair=('py38_baseline', 'py312_cuml_adaptive'),
        reason=(
            'The two arms run in the same 31-Aug daytime session but degrade '
            'unevenly (py3.12 2.41x its best reference vs. py3.8 1.91x), '
            'giving +77.8% when each arm\'s best reference would give +41%. '
            'It is the only azken cell where py3.12 degrades more than '
            'py3.8 (16 of 19 go the other way). The two blocks are also '
            'sequential, not interleaved, sampling different parts of a '
            'varying interference. Same image whose 21:17 block that night '
            'is separately excluded as anomalous. '
            'Evidence: benchmarks/data/azken_h100_lum_full_block_anomaly.csv, control E.'),
    ),
    dict(
        machine='lenovo_tttserver',
        image_label='QHY411-3_SDSSg_10k',
        pair=('py38_baseline', 'py312_cuml_adaptive'),
        reason=(
            'Not a new judgment: its py3.8 arm IS the contaminated 02-Sep '
            'block the tables replace elsewhere; using it here while '
            'excluding it there would be inconsistent. The figure\'s premise '
            'also fails outright: the two arms run back to back, but the '
            'interference ends between them, so py3.8 comes out 1.64x its '
            'clean reference and py3.12 0.94x (nothing degraded). The bar '
            'would show -44.1% when each arm\'s best reference gives -2.1%: '
            'a 42-point error, with the wrong sign. '
            'Evidence: benchmarks/data/measure_lenovo_sdssg10k_coldblock_20260911_115622.csv '
            'and this cell\'s CONTAMINATED_SESSIONS entry.'),
    ),
]


def drop_excluded(d, pair):
    """Drop the PAIR_EXCLUSIONS entries that apply to THIS contrast.

    An entry without 'pair' applies to both; with 'pair', only to the one it names.  The
    distinción importa: una celda puede estar mal emparejada en la figura 7 y bien en la 8,
    porque son contrastes distintos medidos en sesiones distintas.
    """
    for x in PAIR_EXCLUSIONS:
        if x.get('pair') and tuple(x['pair']) != tuple(pair):
            continue
        m = (d['machine'] == x['machine']) & (d['image_label'] == x['image_label'])
        if not m.any():
            print(f"  AVISO: la exclusión {x['machine']}/{x['image_label']} no casa con ninguna "
                  f'fila; si la celda ya no existe, sobra la entrada')
        d = d[~m]
    return d
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import generate_manuscript_tables as T                                    # noqa: E402

FIX1 = os.path.join(BASE, 'data', 'benchmark_latency_allocator_fix.csv')
V4 = os.path.join(BASE, 'data', 'benchmark_latency.csv')
OUT = os.path.join(BASE, 'data', 'benchmark_latency_fig78_paired.csv')      # fig 7
OUT8 = os.path.join(BASE, 'data', 'benchmark_latency_fig8_paired.csv')      # fig 8
ARMS = ['py38_baseline', 'py312_cuml_adaptive', 'py312_cuml_always']
HOSTS = ['azken', 'hp3', 'lenovo_tttserver']
MAX_SEPARATION_H = 3.0   # the figure-8 audit's own threshold for 'same window'


def paired(path, arms, pair=None):
    """Keep, per cell, only what both arms measured while both were running."""
    d = T.load_benchmark(arms, csv_path=path, drop_contaminated=False)
    d = d[d['machine'].isin(HOSTS)].copy()
    d['ts'] = pd.to_datetime(d['timestamp'], format='mixed', utc=True)
    # Which two arms this figure contrasts; anything else in `arms` rides along but
    # does not decide whether a session counts as paired.
    pair = pair or ['py38_baseline', 'py312_cuml_adaptive']
    keep = []
    for (mach, img), g in d.groupby(['machine', 'image_label']):
        blocks = {}
        for arm in pair:
            h = g[g['profiler_label'] == arm].sort_values('ts')
            if h.empty:
                continue
            sess = (h['ts'].diff().dt.total_seconds().fillna(0) > 1800).cumsum()
            blocks[arm] = [(b['ts'].median(), b) for _, b in h.groupby(sess)]
        if len(blocks) < 2:
            continue
        a, b = pair
        for ta, ba in blocks[a]:
            near = [(abs((ta - tb).total_seconds()), bb) for tb, bb in blocks[b]]
            gap, bb = min(near, key=lambda x: x[0])
            if gap <= MAX_SEPARATION_H * 3600:
                keep += [ba, bb]
    if not keep:
        return d.iloc[0:0]
    return drop_excluded(pd.concat(keep, ignore_index=True).drop_duplicates(), pair)


def overhead(d, a='py312_cuml_adaptive', b='py38_baseline'):
    m = d.groupby(['machine', 'profiler_label', 'image_label'])['execution_time'] \
         .median().unstack('profiler_label')
    if a not in m.columns or b not in m.columns:
        return pd.Series(dtype=float)
    m = m[[a, b]].dropna()
    return 100 * (m[a] - m[b]) / m[b]


def report():
    for lab, path in [('campaña (pre-arreglo)', V4), ('fix1 (post-arreglo)', FIX1)]:
        d = paired(path, ARMS)
        ov = overhead(d)
        print(f'--- {lab} ---')
        for mach, s in ov.groupby(level=0):
            print(f'    {mach:<18} n={len(s):2d}  mediana {s.median():+6.1f}%  '
                  f'rango {s.min():+.0f} a {s.max():+.0f}')
        if len(ov):
            print(f'    {"TODAS":<18} n={len(ov):2d}  mediana {ov.median():+6.1f}%')
        print()
    d = paired(FIX1, ARMS)
    d.to_csv(OUT, index=False)
    print(f'escrito: {OUT}  ({len(d)} filas)   [fig 7: py3.8 vs adaptativo]')

    d8 = paired(FIX1, ARMS, pair=['py312_cuml_always', 'py312_cuml_adaptive'])
    d8.to_csv(OUT8, index=False)
    ov8 = overhead(d8, a='py312_cuml_adaptive', b='py312_cuml_always')
    print(f'escrito: {OUT8}  ({len(d8)} filas)  [fig 8: forzado vs adaptativo]')
    for mach, s in ov8.groupby(level=0):
        print(f'    {mach:<18} n={len(s):2d}  mediana {-s.median():+6.1f}% a favor del adaptativo')


if __name__ == '__main__':
    report()
