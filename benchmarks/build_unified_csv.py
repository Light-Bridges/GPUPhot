#!/usr/bin/env python3
"""
build_unified_csv.py
Build the unified latency CSV for the August-2026 re-measurement campaign (v3/v4)
from the raw per-repetition export, in the exact schema consumed by
generate_manuscript_tables.py / generate_manuscript_figures.py.

Input  : benchmarks/results_collected/latency_campaign/eventos_crudos_v3v4_completo_v2.csv
Output : benchmarks/data/benchmark_latency.csv

The raw export carries one row per repetition with the source count and the GPU
telemetry sampled at that instant.  It does NOT carry python_ver, image geometry,
or a catalog flag, and it labels the two Jetson hosts by what nvidia-smi could
report rather than by host.  This script fills those in and applies the
campaign-level culling described below.  It performs no warmup removal and no
outlier rejection: that lives in the generators, exactly as the April pipeline did.

Culling applied here (documented, auditable, each switchable):
  C1  nsys rows dropped         — nsys adds 3-6x overhead, never a latency datum.
  C2  runs with no result table dropped — a run is only a measurement if it produced
                                  one.  The pipeline logs the repr of the result
                                  DataFrame in extra.return_value, so a run whose event
                                  carries no "[N rows x M columns]" never got that far,
                                  whatever its wall time says.  This is provenance, not
                                  a threshold, and it strictly contains the wall-time
                                  rule it replaces: every one of the 855 sub-2 s runs is
                                  in this set, plus 274 more that look like completions
                                  and are not, with times up to 642 s.
  C3  protocol-validity windows  — two hosts measured part of the campaign under
                                  conditions the protocol does not admit, so each is
                                  restricted to the window that is comparable with
                                  the rest of the fleet.  See PROTOCOL_WINDOWS.

  C5  bimodal cells dropped      — see below.

  C4  truncated cells dropped   — hosts whose GPU cannot hold the working set do
                                  not abort instantly: the run advances as far as
                                  memory allows and then fails, at a wall time that
                                  is reproducible and therefore looks like a clean
                                  measurement.  The 4 GB RTX 3050 Ti completes
                                  nothing above 4.2 MP, and 151.2 MP needs ~27 GB so
                                  the 12 GB RTX 3060 completes nothing there.  GPU
                                  telemetry in both cases sits pinned at the card's
                                  capacity.  These cells are excluded by name rather
                                  than left to the generators' warmup rule, which
                                  happens to remove them but does not mean to.

C5 implements the project's own quality rule (BENCHMARK_MASTER section 6, rule 1:
a bimodal distribution is re-run, not published).  A cell whose repetitions split
into two well-separated modes has no meaningful median, so publishing one would be
publishing a number nobody can stand behind.  The detector is uniform across every
cell in the campaign, not aimed at any host: it finds the 1-D split that best
separates the cell's post-warmup times and drops the cell when the two medians
differ by more than BIMODAL_GAP and the smaller mode still holds at least
BIMODAL_MIN_FRAC of the points.  It catches 8 of 261 cells, on three different
machines, almost all at 151.2 MP; those eight are queued for re-measurement.

There is one judgement call worth stating plainly.  C3 discards repetitions that
completed normally: a run that finished, finished, and those are valid measurements.
Dropping them buys homogeneity, not correctness.

Usage:
  python3 benchmarks/build_unified_csv.py [--out PATH] [--no-cull]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE    = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(BASE, 'data', 'raw', 'events_latency_campaign.csv')
REF_CSV = os.path.join(BASE, 'data', 'benchmark_latency_catalog_backends.csv')  # April CSV, for geometry
OUT_CSV = os.path.join(BASE, 'data', 'benchmark_latency.csv')

# Schema expected by the generators, in order.
SCHEMA = ['machine', 'gpu_name', 'python_ver', 'profiler_label', 'environment',
          'image_label', 'mp', 'execution_time', 'n_sources_detected', 'timestamp',
          'naxis1', 'naxis2', 'filter', 'object', 'gpu_mem_total', 'gpu_mem_used',
          'gpu_temp', 'remeasure_round', 'catalog_backend']

# The raw export labels the ARM hosts by what nvidia-smi could report, not by host.
# '?' / 'Error retrieving GPU info' is the Orin NX (JetPack 5.x, py3.8 only);
# 'jetson' / 'Orin (nvgpu)' is the Orin Super (JetPack 6.x, py3.8 + py3.12).
MACHINE_MAP = {'?': 'jetson_orin', 'jetson': 'jetson_local'}
GPU_NAME_MAP = {'Error retrieving GPU info': 'Orin NX 8GB (nvgpu)',
                'Orin (nvgpu)':              'Orin Super 8GB (nvgpu)'}

# python_ver strings must match the April CSV so figures' string comparisons hold.
PY_VER = {
    'py38_baseline':       '3.8.10 (default, Nov 22 2023, 10:22:35) \n[GCC 9.4.0]',
    'py312_cuml_adaptive': '3.12.3 (main, Mar  3 2026, 12:15:18) [GCC 13.3.0]',
    'py312_cuml_always':   '3.12.3 (main, Mar  3 2026, 12:15:18) [GCC 13.3.0]',
}

# C2 — a run counts only if its event carried the result DataFrame.  The wall-time
# floor below is kept for reference: the bimodal distribution's valley sits at ~2 s and
# every run under it also lacks a result table, so the provenance rule subsumes it.
ABORT_FLOOR_S = 2.0

# C5 — a cell is unpublishable when its post-warmup times split into two modes whose
# medians differ by more than BIMODAL_GAP and the smaller mode still holds at least
# BIMODAL_MIN_FRAC of the repetitions.
BIMODAL_GAP      = 0.40
BIMODAL_MIN_FRAC = 0.25
BIMODAL_MIN_N    = 8
# The split must fall on an actual empty interval, at least BIMODAL_GAP of the lower
# mode wide.  Without it the rule fires on any sufficiently broad unimodal cell: the
# best split of a smooth spread still puts the two halves' medians far apart, which
# is a statement about width, not about modes.  Same constant, applied to the gap the
# rule is named after instead of to the distance between the halves' medians.
# C3 — per-host windows outside which the measurement conditions are not comparable.
#
# ttt_server: the only session in which the RTX 3090 host logged no aborted
#   repetitions.  Its abort rate ran 6-100% per hour across every session from 20 to
#   24 Aug and 0-17% on 28 Aug.  The host suffers chronic pinned-memory starvation
#   because a catalog postgres with 24 GB of shared buffers leaves MemFree near zero,
#   making every pinned CUDA allocation a lottery; that container was restarted on
#   26 Aug and postgres refilled its buffers lazily over the following ~14 hours,
#   which is exactly this window.  Inside it the host ran ~1,900 repetitions over all
#   19 images with no aborts.
#
# jetson_local: on 26 Aug the container reached the central catalog over a slow
#   non-standard network path, and that I/O dominated the measurement: 79-84 s per
#   image with an interquartile range of ~0.3 s, against 22-27 s on 28 Aug once the
#   path was fixed, and 21-30 s in the April campaign.  The two regimes do not mix.
#   The 28 Aug passes carry twice the repetitions, so the medians were already
#   landing in the correct regime, but every dispersion statistic over these cells
#   was contaminated.
#
# azken / py312_cuml_adaptive: this profile is the only one on the H100 host that ran
#   outside the night.  Its 20 Aug afternoon block is 1.41x slower than the 21 Aug
#   night block, image for image, on the same host and code, because the GPU fans
#   were governed off the CPU and a service cycled the power limit down; the fans
#   were fixed on 27 Aug.  Its 28 Aug block is worse still, 2.11x, at 62 C, and it
#   covers only the three densest 151.2 MP images, which are exactly the cells that
#   carry the headline numbers.  The 21 Aug night block covers all 19 images on its
#   own.  The host's other two profiles ran at night throughout and are unaffected:
#   py312_cuml_always measures 0.93x over its 20 Aug evening block, so it is not
#   thermally penalised despite sampling a few sub-350 W readings, and py38_baseline
#   has no daytime block at all.
#
# The per-event gpu_power_limit_w readings are a weak witness on their own: only 37
# of 3,453 azken rows sample below 350 W, and some of those sit inside clean night
# blocks.  The capping was duty-cycled, so a point sample per event undersamples it.
# The wall-time ratio between blocks is the reliable signal, and it is unambiguous.
# C6 — cells retired for contamination and then re-measured in a dedicated session.
# The rule is declared up front and applied symmetrically to every retired cell,
# whatever the re-measurement turns out to say: the dedicated session REPLACES the
# contaminated ones instead of being added to them.
#
# Merging sessions taken in different machine states does not average noise away, it
# manufactures extra modes.  Measured case: one lenovo cell has session medians of
# 9.15, 5.69 and 7.49 s; fused they give a 33.6% interquartile range, at the campaign's
# 99th percentile, yet the two-mode detector waves them through because with three
# modes there is no clean two-way split.  Substitution removes that failure mode by
# construction, because each of these cells then holds a single session.
#
# Two consequences to declare rather than hide: these cells become single-session
# (n=20), so they do not meet the two-pass criterion the rest of the table meets; and
# the strength of the substitution differs per cell.  On azken the contaminating cause
# is identified and demonstrated by a three-point experiment (production workers
# resident on the GPU, then GPU empty); elsewhere what it buys is "the most recent
# clean session", not "a cell decontaminated of a known cause".
# The dedicated re-measurement is by construction the cell's LAST session, so the rule
# is stated that way rather than as a date.  It matters: azken was attempted twice on
# the same day, once by daylight with production workers resident on the GPU (medians
# 49.3 and 41.9 s) and once at night with the GPU empty (30.2 and 29.6 s).  A date
# cutoff would have merged the two attempts and undone the point of re-measuring.
REMEASURE_SESSION_GAP_S = 1800.0
REMEASURED_CELLS = {
    ('azken',            'py312_cuml_adaptive', 'QHY411-3_Lum_full'),
    ('azken',            'py312_cuml_adaptive', 'QHY411-3_SDSSr_full'),
    ('hp3',              'py312_cuml_adaptive', 'QHY411-3_Lum_131k'),
    # hp3 / cuML forced, 2026-08-30: 3 interleaved rounds, 36 clean reps per cell,
    # GPU-0 verified empty in all four snapshots.  They stay dispersed (IQR 40-44%),
    # but that is now a measured property of the cell rather than a dirty session:
    # the two images drift in opposite directions within the same afternoon while
    # interleaved, which rules out anything global to the host or the hour.
    ('hp3',              'py312_cuml_always',   'QHY411-3_Lum_full'),
    ('hp3',              'py312_cuml_always',   'QHY411-3_SDSSr_full'),
    ('lenovo_tttserver', 'py312_cuml_always',   'iKon936_SDSSg'),
    ('lenovo_tttserver', 'py38_baseline',       'QHY411-3_Lum_full'),
    ('lenovo_tttserver', 'py38_baseline',       'QHY411-3_SDSSr_19k'),
    # RTX 3050 Ti, 2026-08-30: the whole 4.2 MP block re-measured from a clean
    # state, restarting the container before every shot and rotating the profile
    # order.  Five of the six cells were running about 2x slow in the campaign
    # sessions; the sixth, the one that looked suspiciously fast, is the one that
    # reproduces.  Session medians move 35.7 -> 17.3, 30.5 -> 14.6, 45.8 -> 20.9,
    # 45.2 -> 20.6, 40.0 -> 17.8, and 16.0 -> 17.1.
    ('local',            'py38_baseline',       'iKon936_Lum'),
    ('local',            'py38_baseline',       'iKon936_SDSSg'),
    ('local',            'py312_cuml_adaptive', 'iKon936_Lum'),
    ('local',            'py312_cuml_adaptive', 'iKon936_SDSSg'),
    ('local',            'py312_cuml_always',   'iKon936_Lum'),
    ('local',            'py312_cuml_always',   'iKon936_SDSSg'),
}

# C7 — cells whose dedicated re-measurement did not settle them either, and for which
# a further one has been requested.  They are held out by name rather than by a
# threshold because no statistic separated them cleanly: the campaign-wide reference
# for every measure tried (interquartile range, spread between session medians,
# fraction the MAD filter must discard) is itself polluted by the multi-session
# contamination being looked for.  What each of them shows, inside a single dedicated
# session and before any filtering, is visible without a statistic:
#
#   hp3 / always   / QHY411-3_Lum_full   sustained drift: last 7 reps climb 67->90 s
#                                        from a 44 s base (this one the detector does
#                                        catch, and it is listed here for the record)
#   hp3 / always   / QHY411-3_SDSSr_full repeated excursions to 63-71 s and 51-60 s
#   hp3 / adaptive / QHY411-3_Lum_131k   regime change: 17 reps near 200 s, then
#                                        90/95/86/66/130/77
#   lenovo / py38  / QHY411-3_Lum_full   regime change: ~30 s base, last 4 at 62-69 s
#
# Holding them is not a preference for nicer numbers: the same substitution rule that
# readmits four cells leaves these out, and the raw sequences are the justification.
# Only hp3's three remain.  Their dedicated re-measurement on 2026-08-30 ran with GPU-0
# verified empty by snapshot and still could not produce a stable session: 33.9%, 28.2%
# and 24.1% interquartile range, with a regime change inside the session in two of them.
# They are not cells we failed to measure well; on that host that work is not stable.
# Every other cell that went through the same protocol came back clean, including the
# same 134,206-source image on lenovo at 2.2%.
# hp3 / adaptive / Lum_131k came off this list on 2026-08-30: the hold was declared as
# "pending a further re-measurement", that re-measurement arrived, and keeping it here
# afterwards would be tailoring the rule to the cell.  Its fate is now decided by C5
# and C8 with their thresholds untouched, whatever they say.
AWAITING_REMEASURE = set()

# C8 — cross-machine coherence, the project's own control (analiza_final_v34.py, block
# C): flag any cell where a lower-tier GPU beats a higher-tier one by more than 20%.
# It catches what no dispersion measure can — a cell that is internally tidy but sits
# at the wrong level, which is exactly what "keep the last session" produces when the
# last session lands on the wrong mode of a cell that is bimodal ACROSS sessions.
# Over the whole campaign it fires twice, and both are in AWAITING_REMEASURE above.
GPU_TIER_ORDER = ['H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)',
                  'RTX 3090 (24 GB)', 'RTX 3060 (12 GB)']
COHERENCE_TOL = 1.20

# C9 — the paired experiment of 2026-08-30 is a controlled comparison, not a
# latency measurement, and its absolute levels are not comparable with the rest of
# the campaign: it caught the catalogue disturbed, and every 151.2 MP cell it
# touched sits well above its own campaign level (hp3 / Lum_full / adaptive
# 45.3 -> 78.7 s, azken / SDSSr_full / adaptive 33.3 -> 50.2 s).  What it measures
# is the difference between the two arms, which is protected by the interleaving;
# that difference is used by figure 8 through interleaved_cuml_pairs.csv.  Feeding
# these repetitions into the medians would import a bad afternoon into six cells.
INTERLEAVE_WINDOW = ('2026-08-30T16:00:00Z', '2026-08-30T20:00:00Z')
INTERLEAVE_HOSTS = ('hp3', 'azken', 'lenovo_tttserver')

PROTOCOL_WINDOWS = {
    ('ttt_server',   None):                  ('2026-08-26T18:00:00Z', '2026-08-27T08:00:00Z'),
    ('jetson_local', None):                  ('2026-08-27T12:00:00Z', '2026-08-29T00:00:00Z'),
    ('azken',        'py312_cuml_adaptive'): ('2026-08-21T00:00:00Z', '2026-08-23T00:00:00Z'),
}

# C4 — (machine, minimum megapixels) above which the host completes nothing.
TRUNCATED_ABOVE_MP = {
    'local':        4.2,   # 4 GB RTX 3050 Ti: nothing above the 4.2 MP frames completes
    'ttt1':        37.8,   # 12 GB RTX 3060: 151.2 MP needs ~27 GB
    'jetson_local': 4.2,   # 8 GB unified Orin Super: 6.8 MP no longer fits (see below)
}


def _load_geometry(ref_csv):
    """Per-image naxis1/naxis2/filter/object from the April CSV.

    Geometry, filter and target are properties of the image file, so they carry
    over unchanged.  n_sources_detected is NOT taken from here: the v3/v4 export
    carries the real per-repetition count.
    """
    if not os.path.exists(ref_csv):
        return {}
    ref = pd.read_csv(ref_csv)
    geo = {}
    for img, g in ref.groupby('image_label'):
        def _mode(col):
            s = g[col].dropna()
            return s.mode().iloc[0] if len(s) else np.nan
        geo[img] = dict(naxis1=_mode('naxis1'), naxis2=_mode('naxis2'),
                        filter=_mode('filter'), object=_mode('object'))
    return geo


def _drop_runs_without_results(df):
    """C2 — drop runs whose event carries no result DataFrame.

    A run is a measurement only if it produced something to measure.  The pipeline
    returns a (DataFrame, FITS header) tuple, and the collector records the repr of
    both plus any exception, so validity has three independent witnesses: the result
    table, the header, and the absence of an exception.

    Wall time cannot substitute for any of them: 274 failed runs lasted 2 s to 642 s
    and would pass any plausible threshold, and 47 whole cells never produced a table
    in any repetition — including every 151.2 MP run on the RTX 3090, whose clean-
    session times of 6-45 s look entirely reasonable and mean nothing.
    """
    has_table  = df['n_sources_detected'].notna()
    has_header = df['tiene_header'].fillna(0).astype(int) == 1
    no_except  = df['excepcion'].isna() | (df['excepcion'].astype(str).str.strip() == '')

    # The three conditions are equivalent on this census: 19,037 events have all three
    # and 1,206 have none, with not one mixed state.  They are checked together anyway,
    # so that an export where they ever disagree fails loudly instead of quietly.
    disagree = int(((has_table != has_header) | (has_table == ~no_except)).sum())
    if disagree:
        raise AssertionError(
            f'{disagree} events disagree on validity (table / header / exception). '
            'The three conditions are meant to be equivalent; investigate the export.')

    return df[has_table], int((~has_table).sum())


def _substitute_remeasured(df):
    """C6 — for a retired cell, keep only its last session."""
    ts = pd.to_datetime(df['timestamp'], format='ISO8601')
    keep = pd.Series(True, index=df.index)
    for key in REMEASURED_CELLS:
        sel = ((df['machine'] == key[0]) & (df['profiler_label'] == key[1])
               & (df['image_label'] == key[2]))
        if not sel.any():
            continue
        t = ts[sel].sort_values()
        gaps = t.diff().dt.total_seconds().fillna(0) > REMEASURE_SESSION_GAP_S
        last_start = t[gaps.cumsum() == gaps.cumsum().max()].min()
        keep &= ~(sel & (ts < last_start))
    return df[keep], int((~keep).sum())


def _apply_protocol_windows(df, exempt):
    """C3 — restrict the hosts in PROTOCOL_WINDOWS to their comparable window.

    Cells already reduced to their re-measurement session by C6 are exempt: their
    window was chosen for them, and the campaign-era window would delete them.

    Worth stating plainly: every repetition discarded here completed normally.  A run
    that finished, finished, and those are valid measurements.  Dropping them buys
    homogeneity, not correctness.  What each window removes is a block measured under
    conditions that are not the pipeline: a memory-starved host on ttt_server, a slow
    catalog network path on jetson_local, and a thermally capped GPU on azken.
    """
    ts = pd.to_datetime(df['timestamp'], format='ISO8601')
    bad = pd.Series(False, index=df.index)
    for (machine, profile), (lo, hi) in PROTOCOL_WINDOWS.items():
        sel = df['machine'] == machine
        if profile is not None:
            sel &= df['profiler_label'] == profile
        bad |= sel & ~ts.between(pd.Timestamp(lo), pd.Timestamp(hi))
    bad &= ~pd.Series(exempt, index=df.index)
    return df[~bad], int(bad.sum())


def _report_incoherent_cells(df):
    """C8 — warn about cells that invert the hardware hierarchy.

    Reported rather than filtered: the cells it finds are withheld by name in
    AWAITING_REMEASURE, so that the reason a cell is out is written down instead of
    being an emergent property of a threshold.  If this ever prints something that is
    not already on that list, a new anomaly has appeared and wants looking at.
    """
    from generate_manuscript_tables import GPU_LABEL_MAP
    d = df.copy()
    d['gpu_label'] = (d['gpu_name'].str.replace('NVIDIA ', '', regex=False)
                      .map(GPU_LABEL_MAP).fillna(d['gpu_name']))
    med = d.groupby(['profiler_label', 'image_label', 'gpu_label'])['execution_time'].median()
    found = []
    for prof, sub in med.groupby(level=0):
        m = sub.droplevel(0).unstack()
        for i, a in enumerate(GPU_TIER_ORDER):
            for b in GPU_TIER_ORDER[i + 1:]:
                if a not in m or b not in m:
                    continue
                ratio = (m[a] / m[b]).dropna()
                for img, val in ratio[ratio > COHERENCE_TOL].items():
                    found.append((prof, img, a, b, val))
    return found


def _hold_awaiting_remeasure(df):
    """C7 — withhold cells whose re-measurement did not settle them."""
    key = list(zip(df.machine, df.profiler_label, df.image_label))
    bad = np.array([k in AWAITING_REMEASURE for k in key])
    return df[~bad], int(bad.sum())


def _drop_bimodal_cells(df):
    """C5 — drop cells whose repetitions split into two well-separated modes.

    Warmup is removed first, with the generators' own session-aware detector, so a
    cold first repetition cannot masquerade as a second mode.  What survives is
    genuine intra-session instability: the same host running the same image twice a
    minute apart and taking 29 s one time and 62 s the next.

    Three conditions, all required: the two halves' medians differ by more than
    BIMODAL_GAP, the smaller half holds at least BIMODAL_MIN_FRAC of the points, and
    the split falls on an empty interval at least BIMODAL_GAP wide.  The third is what
    separates two modes from one broad one.  On this campaign the genuinely bimodal
    cells clear it by a factor of five (empty intervals of 2.0-2.3 medians on
    jetson_local) while a wide unimodal cell sits near the campaign median of 0.06.

    The affected cells cluster at 151.2 MP across three different machines and both
    Python stacks, which is why they are reported as a finding rather than blamed on
    any one host.  The cause is not established; the cells are queued for re-measure.
    """
    from generate_manuscript_tables import _drop_warmup_session_aware

    drop = set()
    for key, g in df.groupby(['machine', 'profiler_label', 'image_label'], sort=False):
        clean = _drop_warmup_session_aware(g)
        v = np.sort(pd.to_numeric(clean['execution_time'], errors='coerce').dropna().values)
        if len(v) < BIMODAL_MIN_N:
            continue
        best = None
        for i in range(2, len(v) - 1):   # noqa: E501 - split index also gives the empty interval
            lo, hi = v[:i], v[i:]
            sep = (hi.mean() - lo.mean()) / np.sqrt((lo.var() + hi.var()) / 2 + 1e-9)
            if best is None or sep > best[0]:
                best = (sep, lo, hi)
        _, lo, hi = best
        gap   = (np.median(hi) - np.median(lo)) / np.median(lo)
        frac  = min(len(lo), len(hi)) / len(v)
        empty = (hi[0] - lo[-1]) / np.median(lo)
        if gap > BIMODAL_GAP and frac >= BIMODAL_MIN_FRAC and empty >= BIMODAL_GAP:
            drop.add(key)
    if not drop:
        return df, 0, drop
    key = list(zip(df.machine, df.profiler_label, df.image_label))
    mask = np.array([k not in drop for k in key])
    return df[mask], int((~mask).sum()), drop


def _drop_truncated_cells(df):
    """C4 — drop image sizes a host's GPU cannot hold.

    These runs do not abort at once, so C2 does not catch them: the pipeline gets
    as far as memory allows and fails there, giving a repeatable wall time that
    reads like a real measurement.  The tell is an inversion: the larger frame comes
    out faster than the smaller one, with an unusually tight spread.

    Since C2 became provenance-based this rule is a CROSS-CHECK, not a filter: it
    removes nothing, because every cell it names turns out to have produced no result
    table in any repetition, so C2 has already dropped them.  Two rules derived
    independently — one from wall-time inversions, one from whether a photometric
    table exists — agreeing exactly on 47 cells is the strongest evidence either of
    them has.  It stays in place so that a future export that breaks the agreement
    says so out loud.
    """
    bad = pd.Series(False, index=df.index)
    for machine, max_mp in TRUNCATED_ABOVE_MP.items():
        bad |= (df['machine'] == machine) & (df['mp'] > max_mp)
    n = int(bad.sum())
    if n:
        print(f'  ! C4 removed {n} rows that C2 did not — the two rules disagree, '
              f'which they should not; check the provenance export.')
    return df[~bad], n


def build(cull=True, out_csv=OUT_CSV, raw_csv=RAW_CSV, ref_csv=REF_CSV):
    raw = pd.read_csv(raw_csv)
    n0 = len(raw)

    # C1 — nsys repetitions are for stage breakdown only, never for latency.
    df = raw[raw['campana'] != 'nsys'].copy()
    n_nsys = n0 - len(df)

    df['machine']  = df['machine'].replace(MACHINE_MAP)
    df['gpu_name'] = df['gpu_name'].replace(GPU_NAME_MAP)

    n_abort = n_cell = n_trunc = n_bimod = n_subst = n_hold = n_pair = 0
    bimodal_cells = set()
    if cull:
        # C9 first: the paired repetitions must not look like the latest session of
        # a re-measured cell, or C6 would keep them and drop everything else.
        _ts = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        _in = ((_ts >= INTERLEAVE_WINDOW[0]) & (_ts <= INTERLEAVE_WINDOW[1])
               & df['machine'].isin(INTERLEAVE_HOSTS))
        n_pair = int(_in.sum())
        df = df[~_in]
        df, n_abort = _drop_runs_without_results(df)
        df, n_subst = _substitute_remeasured(df)
        df, n_cell = _apply_protocol_windows(df, _exempt[df.index.isin(df.index)] if False else
                                             np.array([k in REMEASURED_CELLS for k in
                                                       zip(df.machine, df.profiler_label, df.image_label)]))
        df, n_trunc = _drop_truncated_cells(df)
        df, n_bimod, bimodal_cells = _drop_bimodal_cells(df)
        df, n_hold = _hold_awaiting_remeasure(df)
        for prof, img, a, b, val in _report_incoherent_cells(df):
            print(f'  ! coherence: {prof} / {img}: {a} is {val:.2f}x {b}')

    df['python_ver']      = df['profiler_label'].map(PY_VER)
    df['catalog_backend'] = 'local'
    df['remeasure_round'] = df['campana']          # 'v3' | 'v4'
    df = df.rename(columns={'gpu_mem_used_mb': 'gpu_mem_used'})
    df['gpu_mem_total']   = np.nan                 # not sampled by the v3/v4 export

    geo = _load_geometry(ref_csv)
    for col in ('naxis1', 'naxis2', 'filter', 'object'):
        df[col] = df['image_label'].map(lambda i: geo.get(i, {}).get(col, np.nan))

    df = df[SCHEMA].sort_values('timestamp').reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    print(f'raw rows              : {n0}')
    print(f'  - nsys              : -{n_nsys}')
    if cull:
        print(f'  - runs with no result table: -{n_abort}')
        print(f'  - superseded by re-measurement: -{n_subst}')
        print(f'  - outside protocol windows: -{n_cell}')
        print(f'  - truncated cells   : -{n_trunc}')
        print(f'  - bimodal cells     : -{n_bimod}  ({len(bimodal_cells)} cells, queued for re-measure)')
        for c in sorted(bimodal_cells):
            print(f'      {c[0]:<17} {c[1]:<20} {c[2]}')
        print(f'  - awaiting re-measurement: -{n_hold}  ({len(AWAITING_REMEASURE)} cells)')
        print(f'  - paired fig-8 experiment: -{n_pair}  (controlled comparison, see C9)')
    print(f'written               : {len(df)} rows -> {out_csv}')
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT_CSV)
    ap.add_argument('--no-cull', action='store_true')
    a = ap.parse_args()
    build(cull=not a.no_cull, out_csv=a.out)
