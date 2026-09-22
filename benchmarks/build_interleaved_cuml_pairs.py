#!/usr/bin/env python3
"""Paired adaptive-vs-forced comparison, 2026-08-30.

Figure 8 compares the two cuML modes cell by cell, but the campaign ran the two
profiles in blocks, so no pair had its arms measured within three hours of each
other.  The separation predicted the effect: the six pairs whose arms were ten
days apart showed a median effect of 15.3% against 3.3% for the fifty measured
within thirty hours.  On a host whose catalogue phase ranges over a factor of
4.6 with production load, that is a statement about the calendar.

This experiment re-measures those six pairs with the two arms interleaved shot by
shot inside one session, alternating which arm starts each pair.  The alternation
does two things: it cancels the advantage the second shot of a pair gets from
finding the catalogue pages of that same field already warm, and it measures it,
because the difference between pairs that start with one arm and pairs that start
with the other is exactly that benefit.

Reported per cell:
  delta_paired_pct   median over pairs of (forced - adaptive) / forced, in %.
                     The paired statistic, and the one figure 8 uses.
  delta_start_a/f    the same split by which arm started; they should agree.
  cache_s            median of (second shot - first shot) within a pair, in
                     seconds.  Negative means the second shot is faster, which is
                     the warm-cache benefit.

Absolute levels from this session are NOT comparable with the campaign (it caught
the catalogue disturbed), which is why C9 keeps these repetitions out of the
latency CSV.  The differences are protected by the interleaving; the levels are
not.
"""
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from build_unified_csv import (RAW_CSV, INTERLEAVE_WINDOW,        # noqa: E402
                                        INTERLEAVE_HOSTS)
from generate_manuscript_tables import GPU_LABEL_MAP                       # noqa: E402

OUT_CSV = os.path.join(BASE, 'data', 'interleaved_cuml_pairs.csv')
ARMS = {'py312_cuml_adaptive': 'A', 'py312_cuml_always': 'F'}
WARMUP_PAIRS = 2


def _pairs(g):
    """Consecutive (adaptive, forced) shots, in order, with which arm started."""
    g = g.sort_values('ts').reset_index(drop=True)
    arm = g['profiler_label'].map(ARMS).tolist()
    t = g['execution_time'].tolist()
    out, starter, i = [], [], 0
    while i + 1 < len(g):
        if arm[i] != arm[i + 1]:
            a = t[i] if arm[i] == 'A' else t[i + 1]
            f = t[i + 1] if arm[i] == 'A' else t[i]
            out.append((a, f))
            starter.append(arm[i])
            i += 2
        else:                      # a shot lost its partner; skip it
            i += 1
    return out[WARMUP_PAIRS:], starter[WARMUP_PAIRS:]


def build():
    df = pd.read_csv(RAW_CSV)
    df = df[df['n_sources_detected'].notna()].copy()
    df['ts'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    df = df[(df['ts'] >= INTERLEAVE_WINDOW[0]) & (df['ts'] <= INTERLEAVE_WINDOW[1])
            & df['machine'].isin(INTERLEAVE_HOSTS)
            & df['profiler_label'].isin(ARMS)]

    rows = []
    for (mach, img), g in df.groupby(['machine', 'image_label']):
        pairs, starter = _pairs(g)
        if not pairs:
            continue
        a = np.array([p[0] for p in pairs])
        f = np.array([p[1] for p in pairs])
        st = np.array(starter)
        delta = 100 * (f - a) / f
        second = np.where(st == 'A', f, a)
        first = np.where(st == 'A', a, f)
        gpu = GPU_LABEL_MAP.get(g['gpu_name'].iloc[0].replace('NVIDIA ', ''),
                                g['gpu_name'].iloc[0])
        rows.append(dict(
            machine=mach, gpu_label=gpu, image_label=img, n_pairs=len(pairs),
            n_start_a=int((st == 'A').sum()), n_start_f=int((st == 'F').sum()),
            delta_paired_pct=round(float(np.median(delta)), 2),
            delta_start_a=round(float(np.median(delta[st == 'A'])), 2),
            delta_start_f=round(float(np.median(delta[st == 'F'])), 2),
            cache_s=round(float(np.median(second - first)), 2),
            med_adaptive=round(float(np.median(a)), 2),
            med_forced=round(float(np.median(f)), 2)))

    out = pd.DataFrame(rows).sort_values(['gpu_label', 'image_label'])
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False))
    print(f'\nwritten: {OUT_CSV}')
    return out


if __name__ == '__main__':
    build()
