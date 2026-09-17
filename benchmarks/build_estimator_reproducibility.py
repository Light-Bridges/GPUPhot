#!/usr/bin/env python3
"""Which cell estimator repeats better across blocks: the median or a low quantile?

WHY THIS EXISTS. In the two long azken runs the 25th percentile came out almost
identical (32.0 and 32.7 s, 2%) while the median moved (36.1 and 34.8 s), and
the measuring agent left it as an observation without proposing any change,
which was the right call. Evaluating it by looking only at that cell would be
choosing the estimator by its result, so it is evaluated BLIND: over every
campaign cell that has two or more long blocks.

RESULT: yes, low quantiles repeat better, and it is not a coincidence of that
one cell.

AND THE TRAP, which is what stops this from being used as a criterion:
reproducibility is MONOTONIC as the quantile goes down. p10 repeats better
than p25, and p25 better than the median; following the argument to its end,
the best estimator would be the MINIMUM, which obviously does not describe
how long a reduction takes. So "repeats better" CANNOT by itself decide which
statistic to publish: it only says that the episodes live in the upper half
of the distribution.

WHAT WOULD DECIDE IT: knowing whether the episodes are EXTERNAL to the
pipeline. If they are, publishing the interference-free regime is justified;
if they are not, the median is the right choice, because it answers "how
long does a typical reduction take," which is what the paper claims. We do
not know today, so nothing changes.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

# ANALYSIS date, fixed on purpose: if the execution date were written instead,
# every run would produce a diff, and "nothing changed" could not be told apart
# from "something changed".

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import generate_manuscript_tables as T                                    # noqa: E402

OUT = os.path.join(BASE, 'data', 'estimator_reproducibility.csv')
N_MIN = 20
EST = {'mean': np.mean, 'median': np.median,
       'p25': lambda v: np.percentile(v, 25), 'p10': lambda v: np.percentile(v, 10),
       'min': np.min}


def celdas():
    out = {}
    for path in (T.LATENCY_POSTFIX_CSV, T.BENCHMARK_CSV):
        d = pd.read_csv(path, comment='#', low_memory=False)
        d['ts'] = pd.to_datetime(d['timestamp'], utc=True, errors='coerce', format='mixed')
        d = d.dropna(subset=['ts', 'execution_time']).sort_values('ts')
        for k, g in d.groupby(['machine', 'profiler_label', 'image_label']):
            g = g.sort_values('ts')
            sess = ((g['ts'].diff().dt.total_seconds() / 60) > 30).cumsum()
            bs = [b['execution_time'].values for _, b in g.groupby(sess) if len(b) >= N_MIN]
            if len(bs) >= 2:
                out[k] = bs
    return out


def main():
    cel = celdas()
    filas, disp = [], {k: [] for k in EST}
    for k, bs in cel.items():
        for nombre, f in EST.items():
            vals = [float(f(b)) for b in bs]
            disp[nombre].append((max(vals) - min(vals)) / np.median(vals))
    med = np.array(disp['median'])
    for nombre in EST:
        a = np.array(disp[nombre])
        filas.append(dict(estimator=nombre, n_cells=len(a),
                          median_dispersion_pct=round(float(np.median(a)) * 100, 1),
                          p75_dispersion_pct=round(float(np.percentile(a, 75)) * 100, 1),
                          max_dispersion_pct=round(float(a.max()) * 100, 1),
                          better_than_median_count=int((a < med).sum())))
    df = pd.DataFrame(filas)
    cab = f"""# Which cell estimator reproduces best between blocks of the SAME cell. 2026-09-11.
# dispersion = (max - min) / median of the values that estimator gives across the blocks of one cell.
# Lower is more reproducible. Only cells with two or more blocks of at least {N_MIN} repetitions.
#
# WHY IT IS EVALUATED BLIND: the observation came from ONE cell, where the 25th percentile was almost
# identical across two long runs while the median moved. Judging it on that cell would be choosing the
# estimator by the answer it gives. It is therefore computed over every cell that qualifies.
#
# RESULT: yes, low quantiles do reproduce better, and it is not a peculiarity of that cell.
#
# AND THE TRAP, which is why this changes nothing: reproducibility is MONOTONIC as the quantile falls.
# The minimum reproduces best of all, and the minimum plainly does not describe how long a reduction
# takes. "Reproduces better" therefore cannot decide which statistic to publish on its own; it only
# says that the episodes live in the upper half of the distribution.
#
# WHAT WOULD DECIDE IT: knowing whether the episodes are EXTERNAL to the pipeline. If they are,
# publishing the uninterfered regime is justified; if they are not, the median answers the question the
# paper asks, which is how long a typical reduction takes. That is not known, so the median is
# published, as everywhere else in this work.
#
# Reproduce: python3 benchmarks/build_estimator_reproducibility.py
"""
    with open(OUT, 'w') as fh:
        fh.write(cab)
        df.to_csv(fh, index=False)
    print(f'escrito {OUT}  ({len(cel)} celdas)')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
