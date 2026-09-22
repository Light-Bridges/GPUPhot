#!/usr/bin/env python3
"""Do slow repetitions cluster into EPISODES, or are they scattered?

WHY THIS EXISTS. The long run of 11 Sep on azken showed five slow stretches
separated by fast stretches, so the phenomenon looked episodic rather than a
warm-up. But that was ONE run, and "looks episodic" from eyeballing a series
is exactly the kind of thing the eye invents. This file tests it against the
blocks we ALREADY had and, above all, against chance.

THE TEST, which is what makes it honest: a repetition is marked "slow" when
it exceeds its own block's 25th percentile by FACTOR, and **the longest
contiguous run of slow repetitions** is measured. The same block is then
SHUFFLED many times and the same thing is measured. Shuffling preserves the
exact distribution of times and destroys only the ORDER, so the comparison
isolates temporal clustering and **does not depend on where the threshold is
set**: observed and shuffled use the same one.

WHAT THIS ESTABLISHES: that slow repetitions come in runs, not scattered.
WHAT THIS DOES NOT ESTABLISH: the cause, or whether the mechanism is external
to the pipeline or internal to it.

The RAW data is read on purpose: load_benchmark() removes the warm-up and
trims the upper tail with 3 MAD, which would erase exactly what is being
looked for.
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

OUT = os.path.join(BASE, 'data', 'episode_clustering_test.csv')
SOAK = os.path.join(BASE, 'data', 'measure_azken_lumfull_soak_run1_20260911_123758.csv')
N_MIN = 20          # bloques más cortos no dan para medir rachas
MIN_LENTAS = 4      # con menos de 4 lentas, la racha más larga es ruido
N_BARAJAS = 300


def bloques(path):
    d = pd.read_csv(path, comment='#', low_memory=False)
    d['ts'] = pd.to_datetime(d['timestamp'], utc=True, errors='coerce', format='mixed')
    d = d.dropna(subset=['ts', 'execution_time']).sort_values('ts')
    out = []
    for _, g in d.groupby(['machine', 'profiler_label', 'image_label']):
        g = g.sort_values('ts')
        sess = ((g['ts'].diff().dt.total_seconds() / 60) > 30).cumsum()
        for _, b in g.groupby(sess):
            if len(b) >= N_MIN:
                out.append(b['execution_time'].values)
    return out


def max_streak(mask):
    m = c = 0
    for x in mask:
        c = c + 1 if x else 0
        m = max(m, c)
    return m


def evaluate_clustering(series, factor, rng, label):
    sel = []
    for v in series:
        l = v > np.percentile(v, 25) * factor
        if l.sum() >= MIN_LENTAS:
            sel.append(l)
    if not sel:
        return None
    obs = np.array([max_streak(l) for l in sel])
    sh = np.array([[max_streak(rng.permutation(l)) for l in sel] for _ in range(N_BARAJAS)])
    dist = sh.mean(axis=1)
    return dict(dataset=label, factor=factor, n_blocks=len(sel),
                mean_observed_streak=round(float(obs.mean()), 2),
                mean_shuffled_streak=round(float(dist.mean()), 2),
                z=round(float((obs.mean() - dist.mean()) / dist.std()), 1),
                blocks_streak_ge4_observed=int((obs >= 4).sum()),
                blocks_streak_ge4_shuffled=round(float((sh >= 4).sum(axis=1).mean()), 1))


def main():
    rng = np.random.default_rng(0)
    campana = []
    for p in (T.LATENCY_POSTFIX_CSV, T.BENCHMARK_CSV):
        campana += bloques(p)
    filas = []
    for f in (1.20, 1.35, 1.50):
        r = evaluate_clustering(campana, f, rng, 'full_campaign')
        if r:
            filas.append(r)
    if os.path.exists(SOAK):
        d = pd.read_csv(SOAK, comment='#', low_memory=False)
        s = d[(d['role'] == 'soak') & (d['is_warmup'] == 0)].sort_values('timestamp')
        v = s['execution_time'].dropna().values
        for f in (1.20, 1.35, 1.50):
            r = evaluate_clustering([v], f, rng, 'soak_azken_run1')
            if r:
                filas.append(r)
    df = pd.DataFrame(filas)
    cab = f"""# Do slow repetitions arrive in runs? A test against chance. 2026-09-11.
# A block is the consecutive repetitions of one image under one stack; a gap over 30 min starts a new
# one. A repetition is called slow when it exceeds FACTOR times the 25th percentile of its own block,
# and the statistic is the longest contiguous run of slow repetitions.
#
# THE CONTROL, which is what makes this a result rather than an impression: the same block is SHUFFLED
# {N_BARAJAS} times and the statistic recomputed. Shuffling preserves the distribution exactly and
# destroys only the order, so the comparison isolates temporal clustering and does not depend on where
# the threshold is placed, since observed and shuffled use the same one. Three thresholds are reported
# for that reason.
#
# Only blocks of at least {N_MIN} repetitions with at least {MIN_LENTAS} slow ones enter: with fewer,
# the longest run is noise. Read from the RAW exports on purpose, because the publication pipeline
# removes warm-up and trims the upper tail, which would erase exactly what is being looked for.
#
# WHAT THIS ESTABLISHES: slow repetitions come in runs, not scattered, at all three thresholds.
# WHAT IT DOES NOT: the cause, nor whether the mechanism is external to the pipeline or inside it.
#
# Reproduce: python3 benchmarks/build_episode_clustering_test.py
"""
    with open(OUT, 'w') as fh:
        fh.write(cab)
        df.to_csv(fh, index=False)
    print(f'escrito {OUT}')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
