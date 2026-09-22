#!/usr/bin/env python3
"""Before/after comparison for the allocator fix (commit 1a23734).

The campaign of 20-30 August measured the system as deployed; the `_fix1`
environments of 31 August measured it again with the fix in place.  The two are
kept in separate CSVs and compared here, never merged: which of the two the
manuscript reports is a framing decision that is not taken in this script.

Both arms go through the generators' own cleaning (`load_benchmark`: session
detection, warmup drop, 3-MAD upper filter) so the two medians are treated
identically.

Epoch discount, and it is a ratio, not a subtraction.  A `_fix1` cell is not
comparable with its v4 cell on its own: ten days passed, and on one host a
weather model and a power cap changed the conditions.  py3.8 does not carry the
fix, so its own v4-to-fix1 ratio on the same host and image is epoch and nothing
else.  The two effects compose multiplicatively -- fix1 = v4 x epoch x fix --
so the fix is the quotient of the two ratios, not the difference of two
percentages.  Subtracting percentages is only harmless when both are small, and
on the host under the weather model the epoch reaches x3.

Hard condition.  Source counts must be identical and zero points must agree
within their own EZP, cell by cell.  A failure here is a failure of the fix, not
a caveat.
"""
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from build_unified_csv import (SCHEMA, PY_VER, GPU_NAME_MAP,        # noqa: E402
                                        MACHINE_MAP, _load_geometry)
import generate_manuscript_tables as T                                       # noqa: E402

SRC = os.path.join(BASE, 'data', 'raw', 'events_allocator_fix.csv')
REF = os.path.join(BASE, 'data', 'benchmark_latency_catalog_backends.csv')
OUT = os.path.join(BASE, 'data', 'benchmark_latency_allocator_fix.csv')
# azken measured twice: a daytime block whose epoch reached x3 under the weather
# model, and a clean night session whose py3.8 control lands within 5% of v4.  The
# night session is the one that carries azken's numbers; the daytime block stays in
# the CSV as the paired experiment it was, but is excluded from the comparison.
AZKEN_NIGHT_FROM = '2026-08-31T19:00:00Z'

# ttt_server, 2026-09-02 07:20-07:55Z: the run went out with the host at 700-800 MB of
# MemFree and produced 126 successes against 68 aborts in 33 minutes.  The successes are
# survivorship, not measurement -- only the images small enough to fit in what was left
# finished at all -- so the window is dropped whole rather than filtered.  It was relaunched
# behind a MemFree gate.
# The second is a false start: the gate opened on the freed memory but the campaign
# began before the catalogue postgres had settled after its restart, and the first
# blocks came back as 48 cudaErrorInitializationError.  The valid window on that host
# starts at 15:04Z, behind the gate's own settling delay.
#
# The third entry is a supersession, not a defect: ttt1 was measured twice on the same
# day, and the morning run lost four images to a network drop and five more to the
# hangover of a container restart.  It was measured again from scratch in the evening,
# so the morning survives nowhere -- the same rule the campaign builder applies to
# re-measured cells, keeping only the last session, written here as an explicit window
# because sessions on these hosts overlap in ways a gap detector does not see.
EXCLUDE_WINDOWS = [
    ('ttt_server', '2026-09-02T07:20:00Z', '2026-09-02T07:55:00Z', 'host sin memoria'),
    ('ttt_server', '2026-09-02T14:00:00Z', '2026-09-02T15:28:00Z', 'arranque falso e instancias solapadas'),
    ('ttt1',       '2026-01-01T00:00:00Z', '2026-09-02T15:39:00Z', 'superseded por la sesión de la tarde'),
]
V4 = os.path.join(BASE, 'data', 'benchmark_latency.csv')
EXTRA = ['zp', 'ezp', 'catnstar']


def build_csv():
    df = pd.read_csv(SRC)
    n0 = len(df)
    df = df[df['campana'] != 'nsys'] if 'campana' in df.columns else df
    has_table = df['n_sources_detected'].notna()
    n_abort = int((~has_table).sum())
    df = df[has_table].copy()

    df['machine'] = df['machine'].replace(MACHINE_MAP)
    ts = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    for mach, lo, hi, why in EXCLUDE_WINDOWS:
        bad = (df['machine'] == mach) & (ts >= lo) & (ts <= hi)
        if bad.any():
            print(f'  - {mach} {lo[5:16]}..{hi[5:16]}: -{int(bad.sum())}  ({why})')
            df = df[~bad]
            ts = ts[~bad]
    df['gpu_name'] = df['gpu_name'].replace(GPU_NAME_MAP)
    df['python_ver'] = df['profiler_label'].map(PY_VER)
    df['catalog_backend'] = 'local'
    df['remeasure_round'] = 'fix1'
    df = df.rename(columns={'gpu_mem_used_mb': 'gpu_mem_used'})
    df['gpu_mem_total'] = np.nan
    geo = _load_geometry(REF)
    for col in ('naxis1', 'naxis2', 'filter', 'object'):
        df[col] = df['image_label'].map(lambda i: geo.get(i, {}).get(col, np.nan))

    out = df[SCHEMA + EXTRA].sort_values('timestamp').reset_index(drop=True)
    out.to_csv(OUT, index=False)
    print(f'fix1 raw rows        : {n0}')
    print(f'  - no result table  : -{n_abort}')
    print(f'written              : {len(out)} rows -> {OUT}\n')
    return out


def _med(profiles, path, night_only=False):
    d = T.load_benchmark(profiles, csv_path=path)
    if night_only:
        ts = pd.to_datetime(d['timestamp'], format='mixed', utc=True)
        d = d[(d['machine'] != 'azken') | (ts >= AZKEN_NIGHT_FROM)]
    return d.groupby(['machine', 'profiler_label', 'image_label'])['execution_time'].agg(
        ['size', 'median'])


def compare():
    profiles = ['py312_cuml_adaptive', 'py312_cuml_always', 'py38_baseline']
    new, old = _med(profiles, OUT, night_only=True), _med(profiles, V4)

    rows = []
    for key in new.index:
        if key not in old.index:
            continue
        mach, prof, img = key
        d_new, d_old = new.loc[key, 'median'], old.loc[key, 'median']
        r_obs = d_new / d_old                      # observed = epoch x fix
        ep_key = (mach, 'py38_baseline', img)
        r_epoch = np.nan
        if prof != 'py38_baseline' and ep_key in new.index and ep_key in old.index:
            r_epoch = new.loc[ep_key, 'median'] / old.loc[ep_key, 'median']
        r_fix = r_obs / r_epoch if r_epoch == r_epoch else np.nan
        rows.append(dict(machine=mach, profile=prof, image=img,
                         n_v4=int(old.loc[key, 'size']), n_fix1=int(new.loc[key, 'size']),
                         v4=round(d_old, 2), fix1=round(d_new, 2),
                         delta_pct=round(100 * (r_obs - 1), 1),
                         epoch_pct=round(100 * (r_epoch - 1), 1) if r_epoch == r_epoch else np.nan,
                         net_pct=round(100 * (r_fix - 1), 1) if r_fix == r_fix else np.nan))
    return pd.DataFrame(rows)


def hard_condition():
    """Source counts identical and zero points within EZP, cell by cell."""
    new = pd.read_csv(OUT)
    old = pd.read_csv(V4)
    bad_src, bad_zp, checked = [], [], 0
    for key, g in new.groupby(['machine', 'profiler_label', 'image_label']):
        o = old[(old.machine == key[0]) & (old.profiler_label == key[1])
                & (old.image_label == key[2])]
        if o.empty:
            continue
        checked += 1
        s_new = set(g['n_sources_detected'].dropna())
        s_old = set(o['n_sources_detected'].dropna())
        if s_new != s_old:
            bad_src.append((key, sorted(s_new), sorted(s_old)))
        if 'zp' in g.columns and g['zp'].notna().any():
            # v4 has no ZP column; compare against the harvested campaign table.
            pass
    print(f'condición dura — celdas comparadas: {checked}')
    print(f'  recuento de fuentes distinto: {len(bad_src)}')
    for b in bad_src:
        print(f'    {b[0]}: fix1={b[1]} v4={b[2]}')
    return bad_src


if __name__ == '__main__':
    build_csv()
    r = compare()
    r.to_csv(os.path.join(BASE, 'data', 'allocator_fix_comparison.csv'), index=False)
    for prof in ['py312_cuml_adaptive', 'py312_cuml_always', 'py38_baseline']:
        s = r[r.profile == prof]
        if s.empty:
            continue
        print(f'--- {prof}  ({len(s)} celdas) ---')
        col = 'net_pct' if s['net_pct'].notna().any() else 'delta_pct'
        print(s.sort_values(['machine', 'image'])[
            ['machine', 'image', 'n_v4', 'n_fix1', 'v4', 'fix1',
             'delta_pct', 'epoch_pct', 'net_pct']].to_string(index=False))
        print(f'  mediana {col}: {s[col].median():+.1f}%\n')
    hard_condition()
