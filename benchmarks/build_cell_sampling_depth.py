#!/usr/bin/env python3
"""How many repetitions and how many sessions support EACH published cell.

WHY THIS EXISTS. The methodology text claimed that "most cells were measured in
two sessions on different days, with a median of 50 raw repetitions per cell."
Checking this in order to write a table caption showed it does not hold across
the whole set, and that the difference between the two stacks is large: the
py3.12 arm has cells with two and three sessions, while the py3.8 arm, on all
three datacenter cards, rests entirely on ONE session per cell. This does not
invalidate any number, but it is a sampling limitation that has to be stated
where it belongs, and what gets stated has to come out of a file.

WHAT IS COUNTED, and the distinction matters:
  raw / raw_sessions   = what is in the CSV of the epoch assigned to that cell.
  clean / clean_blocks = what SURVIVES load_benchmark (first 2 reps of each
                          session dropped, warm-up detector, 3-MAD filter,
                          contaminated windows dropped). This is what feeds
                          the published median.

A cell can have two raw sessions and one clean one: the 19 py3.8 cells of the
H100 are like this, because one of their two sessions is the daytime WRF
window of 31 Aug.

The epoch of each (arm, card) pair is decided by SESSION_BY_ARM, not by this file.

Output: benchmarks/data/cell_sampling_depth.csv
"""
import os
import sys
import warnings

import pandas as pd

# ANALYSIS date, fixed on purpose: writing execution date would produce spurious diffs.

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import generate_manuscript_tables as T                                    # noqa: E402

OUT = os.path.join(BASE, 'data', 'cell_sampling_depth.csv')
ARMS = ['py312_cuml_adaptive', 'py38_baseline']


def n_sesiones(g):
    g = g.sort_values('ts')
    return int(((g['ts'].diff().dt.total_seconds() / 60) > 30).cumsum().nunique())


def main():
    filas = []
    for arm in ARMS:
        limpio = T.load_benchmark_by_epoch([arm])
        limpio['ts'] = pd.to_datetime(limpio['timestamp'], utc=True, errors='coerce')
        for gpu, ep in T.SESSION_BY_ARM[arm].items():
            path = T.LATENCY_POSTFIX_CSV if ep == 'postfix' else T.BENCHMARK_CSV
            raw = pd.read_csv(path, comment='#', low_memory=False)
            raw['gpu_name'] = raw['gpu_name'].str.replace('NVIDIA ', '', regex=False)
            raw['gpu_label'] = raw['gpu_name'].map(T.GPU_LABEL_MAP).fillna(raw['gpu_name'])
            raw = raw[(raw['gpu_label'] == gpu) & (raw['profiler_label'] == arm)].copy()
            if not len(raw):
                continue
            raw['ts'] = pd.to_datetime(raw['timestamp'], utc=True, errors='coerce')
            for img, g in raw.groupby('image_label'):
                l = limpio[(limpio['gpu_label'] == gpu) & (limpio['image_label'] == img)]
                filas.append(dict(
                    arm=arm, gpu=gpu, epoch=ep, image_label=img,
                    raw_runs=len(g), raw_sessions=n_sesiones(g),
                    clean_runs=len(l), clean_blocks=n_sesiones(l) if len(l) else 0,
                    median_s=round(l['execution_time'].median(), 2) if len(l) else ''))
    f = pd.DataFrame(filas)
    dc = ['H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)']
    res = []
    for arm in ARMS:
        a = f[f.arm == arm]
        for nombre, sub in [('datacenter', a[a.gpu.isin(dc)]), ('all', a)]:
            res.append(f'#   {arm:20} {nombre:13} {len(sub):3d} cells | median raw '
                       f'{sub.raw_runs.median():5.0f} | median clean {sub.clean_runs.median():4.0f} | '
                       f'with >=2 raw sessions {(sub.raw_sessions >= 2).sum():3d} | '
                       f'with >=2 clean blocks {(sub.clean_blocks >= 2).sum():3d}')
    res.append(f'#   {"COMBINED":20} {"both stacks":13} {len(f):3d} cells | median raw '
               f'{f.raw_runs.median():5.0f} | median clean {f.clean_runs.median():4.0f} | '
               f'with >=2 raw sessions {(f.raw_sessions >= 2).sum():3d} | '
               f'with >=2 clean blocks {(f.clean_blocks >= 2).sum():3d}')
    cab = ('# Sampling depth of every published cell, {}.\n'
           '# raw/raw_sessions = what the epoch CSV for that cell contains. clean/clean_blocks = what\n'
           '#   survives the screening and enters the published median: first two repetitions of each\n'
           '#   session dropped, adaptive warm-up detector, 3-MAD upper-tail filter, excluded windows.\n'
           '# A cell can have two raw sessions and one clean block: that is not a defect, it is an\n'
           '#   excluded window doing its job, and the distinction is why both counts are given.\n'
           '# WHY IT EXISTS: a statement that most cells were measured in two sessions did not survive\n'
           '#   checking. Over both stacks the median is well below that, and the two stacks differ\n'
           '#   sharply: the py3.12 datacenter cells have two sessions or more, while every py3.8\n'
           '#   datacenter cell rests on a single clean block. This invalidates no number, but it is a\n'
           '#   sampling limit that belongs next to the numbers rather than in a footnote.\n'
           '# The epoch of each (stack, card) pair is decided by SESSION_BY_ARM, not by this file.\n'
           '# Reproduce: python3 benchmarks/build_cell_sampling_depth.py\n').format('2026-09-11', '\n'.join(res))
    with open(OUT, 'w') as fh:
        fh.write(cab)
        f.to_csv(fh, index=False)
    print(f'written {OUT}: {len(f)} cells')
    print('\n'.join(r[2:] for r in res))


if __name__ == '__main__':
    main()
