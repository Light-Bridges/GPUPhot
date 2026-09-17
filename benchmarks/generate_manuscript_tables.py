#!/usr/bin/env python3
"""
generate_manuscript_tables.py
Generate all data-driven LaTeX table bodies for the GPUPhot manuscript.

Run this script whenever benchmark data changes to regenerate tables.
Output .tex files are written to GPUPHOT_manuscript/tables_generated/
and committed alongside the manuscript so that LaTeX compilation never
requires re-running this script.

Tables generated (tabular body: \\begin{tabular}...\\end{tabular}):
  body_latency_py312.tex        — tab:latency_py312          median latency, py3.12+adaptive
  body_latency_py38.tex         — tab:latency_py38           median latency, py3.8 baseline
  body_speedup_local_vizier.tex — tab:speedup_local_vizier   speedup local/Vizier catalog
  body_concurrency.tex          — tab:concurrency            concurrent 4.2 MP images per GPU
  body_cuml_ablation.tex        — tab:cuml_ablation          cuML vs cKDTree ablation (A100)
  body_cpu_baseline.tex         — tab:cpu_baseline           sep / Photutils / GPUPhot (A100 py3.12)
  body_nvtx_detection.tex       — tab:nvtx_detection         stage-level GPU detection vs sep (A100)
  body_stage_breakdown.tex      — tab:stage_breakdown        full per-stage timing (4.2 MP & 151.2 MP)

Data sources (relative paths from this file):
  data/benchmark_latency.csv        → latency tables, speedup_local_vizier, iqr_collapse,
                                       cpu_baseline
  data/profiler_memory_raw.csv      → VRAM tables
  data/cuml_ablation.csv            → body_cuml_ablation
  data/detection_stage_vs_cpu_tools.csv → body_nvtx_detection
  data/stage_breakdown.csv          → body_stage_breakdown
  data/cpu_baseline.csv             → body_cpu_baseline (sep/Photutils)

Usage:
  cd benchmarks/
  python generate_manuscript_tables.py

Then compile the manuscript normally: pdflatex main.tex (performance.tex inputs
the generated files via \\input{tables_generated/body_*.tex}).
"""

import os
import re
import math
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
PROJECT  = os.path.dirname(BASE)
DATA_DIR = os.path.join(BASE, 'data')
OUT_DIR  = os.path.join(PROJECT, 'GPUPHOT_manuscript', 'tables_generated')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Source CSVs ────────────────────────────────────────────────────────────────
BENCHMARK_CSV      = os.path.join(DATA_DIR, 'benchmark_latency.csv')
# The only campaign that measured both catalogue backends on the same code, so the
# two backend-comparison tables read it instead of the current campaign.
CATALOG_BACKENDS_CSV = os.path.join(DATA_DIR, 'benchmark_latency_catalog_backends.csv')

# ── Post-allocator-fix campaign (path B) ───────────────────────────────────────
# Canonical file of the campaign after commit 1a23734 (2026-08-31 08:28). It is
# built from the blocks the measuring agent delivers, each in its own
# measure_*.csv file, and is NEVER mixed with benchmark_latency.csv: the August
# campaign is kept intact as evidence of what the paper published before.
LATENCY_POSTFIX_CSV = os.path.join(DATA_DIR, 'benchmark_latency_postfix.csv')

# Single switch. While it is False, EVERYTHING still comes from the August
# campaign and the .tex files reproduce byte for byte. Setting it to True (or
# exporting GPUPHOT_TABLES_POSTFIX=1) flips all six call sites that today use
# the default file, at once. Changing ONE constant is the difference between
# an auditable switch-over and six loose edits that fall out of sync.
USE_POSTFIX_CAMPAIGN = os.environ.get('GPUPHOT_TABLES_POSTFIX', '1') == '1'


def latency_csv():
    """Currently active latency file. The single point that decides the campaign."""
    if USE_POSTFIX_CAMPAIGN:
        if not os.path.exists(LATENCY_POSTFIX_CSV):
            raise FileNotFoundError(
                f'USE_POSTFIX_CAMPAIGN is on but {LATENCY_POSTFIX_CSV} is missing. '
                'Build it with build_postfix_latency_csv.py from the measure_*.csv files.')
        return LATENCY_POSTFIX_CSV
    return BENCHMARK_CSV


# ── Contaminated sessions, excluded by a written rule ──────────────────────────
# A session is defined by a gap of more than 30 min between repetitions
# (session_gap_s below). These windows are ALWAYS discarded, in any file,
# because what invalidates them is a fact about the host, not about the
# campaign. Every entry carries its evidence: without it this would be a
# marginal note, not a reproducible filter.
# Every entry is bounded by machine, window and OPTIONALLY by arm
# (profiler_label). The arm is not a whim: CPU contention sinks the arm that
# clusters on CPU (py3.8) and leaves the one that does it on GPU (py3.12)
# untouched, so dropping the whole session would destroy good measurements.
# The asymmetry is justified by the check, not by a hunch.
CONTAMINATED_SESSIONS = [
    dict(
        machine='azken',
        start='2026-08-31T07:00:00Z',
        end='2026-08-31T18:00:00Z',
        reason=(
            'Daytime session on 2026-08-31: host CPU occupied by the WRF weather '
            'model. Median cell latency +48.2% vs. that night\'s session (n=19). '
            'The same day/night check on lenovo_tttserver and hp3 shows only ±3% '
            '(noise), confirming the effect is azken-specific and CPU-bound, not '
            'a real GPU signal.'),
    ),
    dict(
        machine='ttt_server',
        profiler_label='py38_baseline',
        start='2026-09-02T17:00:00Z',
        end='2026-09-02T18:00:00Z',
        reason=(
            'Sustained host CPU contention: ttt_server hosts the production '
            'catalog, and several Postgres processes pinned the host CPU to '
            '99% (load 46.8) in September. py3.8 (CPU-side clustering) runs '
            '31-41% slower from the first repetition; py3.12 (GPU-side) runs '
            '6% faster in the same window, confirming the contention hits '
            'CPU work only. Left in, it would flip the sign of the '
            'py3.8-vs-py3.12 ratio for this card (0.87 vs. true 1.31). '
            'Window-specific: a later, clean measurement of ttt_server is not '
            'covered by this rule.'),
    ),
    dict(
        machine='local',
        profiler_label='py38_baseline',
        start='2026-09-10T10:00:00Z',
        end='2026-09-10T11:00:00Z',
        reason=(
            'Thermal throttling on the laptop: the first 10 repetitions match '
            'the campaign (18.93 vs 17.81 s), then at minute 4.3 the time '
            'jumps to ~42 s and stays there as temperature rises 64->68 C and '
            'the CPU clocks down to 600-800 MHz (from 4600). The session '
            'median mixes both regimes and represents neither; the campaign '
            'value is the clean regime.'),
    ),
    dict(
        machine='azken',
        profiler_label='py312_cuml_adaptive',
        image_label='QHY411-3_Lum_full',
        start='2026-08-31T21:17:00Z',
        end='2026-08-31T21:40:30Z',
        reason=(
            'Single anomalous block, not the whole session: this image\'s 25 '
            'repetitions give a median of 53.19 s, while the other 18 images '
            'in that SAME session are all faster than in any other session '
            '(17 by 14-34%); this is the only one going the other way (1.60x '
            'slower). The same image gives 30.19-33.60 s on three other dates. '
            'Within the block the last three repetitions drop to '
            '37.2/25.4/24.1 s and the next block starts clean, so the '
            'interference starts and ends inside this window. Not explained '
            'by the work done (identical zp/ezp/source count throughout), '
            'power (steady 220 W) or temperature (61-68 C, below throttle). '
            'CAUSE NOT ESTABLISHED -- only the anomaly. Remaining data for '
            'this cell (12 reps from a session itself ~35% slower) is still '
            'not comparable to its column neighbors; needs remeasurement in '
            'a clean window. '
            'Evidence: benchmarks/data/azken_h100_lum_full_block_anomaly.csv'),
    ),
    dict(
        machine='lenovo_tttserver',
        profiler_label='py38_baseline',
        image_label='QHY411-3_SDSSg_10k',
        start='2026-09-02T00:00:00Z',
        end='2026-09-02T00:35:00Z',
        reason=(
            'Block SUBSTITUTED, not simply dropped: a remeasurement that '
            'passes the acceptance criteria replaces it. The original 02-Sep '
            'block (46.58 s) matched the excluded-block signature '
            '(median/tail ratio 1.66 vs. 0.94-1.07 for the other 20 blocks '
            'that session), flagged automatically by an 883-block sweep. The '
            '11-Sep remeasurement passes all three pre-registered conditions '
            'in scan_block_signature.py (ratio 0.958, control 1% faster than '
            'August, half-to-half drift 2%). The two blocks are not pooled '
            '(that would give 29.10 s, representing neither). Open '
            'observation: the new value is 14.4% off its control relative to '
            'August, unexplained. '
            'Evidence: benchmarks/data/measure_lenovo_sdssg10k_coldblock_20260911_115622.csv'),
    ),
]


def _drop_contaminated_sessions(df):
    """Drop the rows that fall inside the windows of CONTAMINATED_SESSIONS.

    It does nothing to the earlier campaign file, whose measurements end before the first
    registered window, so enabling it alters no previously published table; it only bites
    on the later campaign.
    """
    if 'machine' not in df.columns or 'timestamp' not in df.columns or not len(df):
        return df
    ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    drop = pd.Series(False, index=df.index)
    for w in CONTAMINATED_SESSIONS:
        m = ((df['machine'] == w['machine'])
             & ts.ge(pd.Timestamp(w['start'])) & ts.le(pd.Timestamp(w['end'])))
        if w.get('profiler_label') and 'profiler_label' in df.columns:
            m &= df['profiler_label'] == w['profiler_label']
        if w.get('image_label') and 'image_label' in df.columns:
            m &= df['image_label'] == w['image_label']
        drop |= m
    return df[~drop]

# Remeasure rounds excluded from all statistical analysis.
# '5C' data was collected with py38 contamination (bimodal intra-session on H100
# and A100); replaced by clean '5E' measurements. Excluded here so they never
# reach the pipeline even if still present in the CSV for audit purposes.
_EXCLUDED_ROUNDS = {'5C'}
MEMORY_CSV         = os.path.join(DATA_DIR, 'profiler_memory_raw.csv')
MEMORY_POSTFIX_CSV = os.path.join(DATA_DIR, 'profiler_memory_postfix.csv')
MEMORY_ALLOC_FIX_CSV = os.path.join(DATA_DIR, 'memory_allocator_fix.csv')
CUML_ABLATION_CSV  = os.path.join(DATA_DIR, 'cuml_ablation.csv')
ALLOCATOR_ABLATION_CSV = os.path.join(DATA_DIR, 'allocator_fix_comparison.csv')
DETECTION_CSV              = os.path.join(DATA_DIR, 'detection_stage_vs_cpu_tools.csv')
STAGE_BREAKDOWN_CSV = os.path.join(DATA_DIR, 'stage_breakdown.csv')
CPU_BASELINE_CSV      = os.path.join(DATA_DIR, 'cpu_baseline.csv')
CPU_BASELINE_A100_CSV = os.path.join(DATA_DIR, 'cpu_baseline_a100.csv')

# ── Instrument naming ─────────────────────────────────────────────────────────
# The CSVs key their rows by the camera model, which is what the raw exports carry and
# what every culling rule in the builders matches on, so the identifiers stay.  What the
# manuscript prints is the instrument, and the discriminant is the telescope, read from
# TELESCOP in each frame's own header rather than assumed from the file name: the same
# camera can move, and the suffix in the identifier is a serial number, not a position.
#
#   QHY600 -> FERVOR-M      QHY411 -> FERVOR-L      iKon936 -> COLORS
#
# The telescope matters beyond nomenclature.  FERVOR-M sits on TTT2 with the PCA path off
# and on TTT3 with it on; FERVOR-L sits on TTT1, where the long focus gives 19,881-pixel
# stamps, and on TST, where it gives 1,089.  Collapsing either pair would merge cameras
# the results distinguish.
INSTRUMENT_OF = {'QHY600': 'FERVOR-M', 'QHY411': 'FERVOR-L', 'iKon936': 'COLORS'}
TELESCOPE_OF  = {'QHY411-1': 'TTT1', 'QHY411-3': 'TST',
                 'QHY600-3': 'TTT3', 'QHY600-4': 'TTT2', 'iKon936': 'TTT3'}


def instrument_label(image_label):
    """'QHY411-1_Lum_full' -> 'FERVOR-L@TTT1'.  Raises on an unknown camera."""
    cam = image_label.split('_')[0]
    base = cam.rsplit('-', 1)[0] if cam.startswith('QHY') else cam.rstrip('-1')
    if base not in INSTRUMENT_OF or cam not in TELESCOPE_OF:
        raise KeyError(f'no instrument mapping for {image_label!r}')
    return f'{INSTRUMENT_OF[base]}@{TELESCOPE_OF[cam]}'


# ── GPU label normalisation ────────────────────────────────────────────────────
GPU_LABEL_MAP = {
    'H100 PCIe':                      'H100 (80 GB)',
    'A100-SXM4-80GB':                 'A100 (80 GB)',
    'L40S':                           'L40S (48 GB)',
    'GeForce RTX 3090':               'RTX 3090 (24 GB)',
    'GeForce RTX 3060':               'RTX 3060 (12 GB)',
    'GeForce RTX 3050 Ti Laptop GPU': 'RTX 3050 Ti (4 GB)',
    'Orin Super 8GB (nvgpu)':         'Orin Super (8 GB)',
    # NOTE: the KEY is the label the CSVs carry from the campaign; the VALUE
    # is how it is presented. The actual module is an Orin Nano 8GB (part
    # number p3767-0003 from the device tree, read 2026-09-11); the key
    # keeps 'Orin NX' for continuity with the data and must not be renamed
    # without rewriting every CSV. See benchmarks/data/raw/README.md.
    'Orin NX 8GB (nvgpu)':            'Orin Nano (8 GB)',
}

GPU_TEX_HEADER = {
    'H100 (80 GB)':       r'\textbf{H100}',
    'A100 (80 GB)':       r'\textbf{A100}',
    'L40S (48 GB)':       r'\textbf{L40S}',
    'RTX 3090 (24 GB)':   r'\textbf{3090}',
    'RTX 3060 (12 GB)':   r'\textbf{3060}',
    'RTX 3050 Ti (4 GB)': r'\textbf{3050\,Ti}',
    'Orin Super (8 GB)':  r'\textbf{Orin Super}$^*$',
    'Orin Nano (8 GB)':     r'\textbf{Orin Nano}$^\dagger$',
}

# Total VRAM in MB for concurrency calculation
GPU_VRAM_MB = {
    'H100 (80 GB)':       80 * 1024,
    'A100 (80 GB)':       80 * 1024,
    'L40S (48 GB)':       48 * 1024,
    'RTX 3090 (24 GB)':   24 * 1024,
    'RTX 3060 (12 GB)':   12 * 1024,
    'RTX 3050 Ti (4 GB)':  4 * 1024,
}
GPU_VRAM_GB = {k: v // 1024 for k, v in GPU_VRAM_MB.items()}

# ── Image definitions ─────────────────────────────────────────────────────────
# (image_label, MP, source count)
# Tables and figures identify images by MP + source count only.
# Camera, filter, and target names are implementation details not relevant to
# pipeline performance; removing them keeps all tables consistent.
IMAGE_DEFS = [
    # Ordered by MP ascending, then source count ascending within each MP group.
    # Source counts are the median of n_sources_detected over the campaign in
    # data/benchmark_latency.csv.  They differ from the April counts on every camera
    # except QHY411-3 because the instrument configurations were unified: the images
    # are the same files, the workload they represent is not.
    ('iKon936_Lum',           4.2,    228),
    ('iKon936_SDSSg',         4.2,    279),
    ('QHY600-3_SDSSi_2k',     6.8,   3534),
    ('QHY600-3_Lum',          6.8,   8093),
    ('QHY600-4_SDSSg',       15.3,    307),
    ('QHY600-4_Ha',          15.3,    601),
    ('QHY600-4_Lum_2k',      15.3,   2938),
    ('QHY600-4_SDSSg_4k',    15.3,   8294),
    ('QHY600-4_SDSSi_10k',   15.3,  15759),
    ('QHY411-1_Lum_bin2',    37.8,    354),
    ('QHY411-1_SDSSi_bin2',  37.8,    621),
    ('QHY411-1_SDSSg_2k',    37.8,   3977),
    ('QHY411-1_SDSSr_7k',    37.8,  12832),
    ('QHY411-1_Lum_full',   151.2,    808),
    ('QHY411-3_SDSSg_10k',  151.2,  10168),
    ('QHY411-3_SDSSr_full', 151.2,  15390),
    ('QHY411-3_Lum_full',   151.2,  18712),
    ('QHY411-3_SDSSr_19k',  151.2,  19492),
    ('QHY411-3_Lum_131k',   151.2, 134206),
]
IMAGE_ORDER = [d[0] for d in IMAGE_DEFS]
IMAGE_MP    = {d[0]: d[1] for d in IMAGE_DEFS}
IMAGE_SRC   = {d[0]: d[2] for d in IMAGE_DEFS}

def _mp_str(mp):
    """Format MP value: strip trailing zeros (4.2, 6.8, 15.3, 37.8, 151.2)."""
    return f'{mp:.1f}'.rstrip('0').rstrip('.')

def _img_label(mp, src):
    """Short LaTeX display label used in tables: 'X.X MP & N,NNN'."""
    return _mp_str(mp), fmt_src(src)

# cuml_ablation: map image filename keyword → (MP, n_sources)
# Fifteen images ordered by MP ascending, then source count ascending.
# The 151.2 MP / 428-source case (+664% penalty) is intentionally included:
# it illustrates the degenerate condition (sparse field + large frame + cuML
# init overhead) that makes the adaptive mode necessary.
# New dense-field entries (37.8 MP Eugenia/V445Pup and 151.2 MP M106/NGC2683/C2025N1)
# complete coverage from 112 to 131,397 sources.
ABLATION_FILE_MAP = [
    # 4.2 MP
    ('QSO0957',    4.2,   412),
    ('C2025A6',    4.2,   296),
    # 6.8 MP
    ('C2025R2',    6.8,   112),
    ('Atira',      6.8,  2060),   # QHY600-3_SDSSi_2k
    # 15.3 MP
    ('WASP-43-b', 15.3,   218),
    ('NGC2903',   15.3,   318),
    ('1620',      15.3,  1993),   # QHY600-4_Lum_2k
    ('hermione',  15.3,  4361),   # QHY600-4_SDSSg_4k
    ('MAXIJ1820', 15.3, 10532),   # QHY600-4_SDSSi_10k
    # 37.8 MP
    ('2012QD8',   37.8,   154),
    ('GaiaDR3',   37.8,   247),
    ('Eugenia',   37.8,  2391),   # QHY411-1_SDSSg_2k
    ('V445Pup',   37.8,  6932),   # QHY411-1_SDSSr_7k
    # 151.2 MP
    ('2025PR1',  151.2,   428),
    ('M106',     151.2, 10005),   # QHY411-3_SDSSg_10k
    ('M81',      151.2, 14241),
    ('24P_Lum',  151.2, 18888),
    ('NGC2683',  151.2, 19565),   # QHY411-3_SDSSr_19k
    ('C2025N1',  151.2, 131397),  # QHY411-3_Lum_131k
]

# cpu_baseline: map filename keyword → (MP, sources, benchmark image_label)
# (filename keyword, image_label).  Megapixels and source counts are NOT repeated
# here: they come from IMAGE_MP and IMAGE_SRC via the image label, so this table can
# never disagree with the latency tables about the same image.  They used to be
# written out alongside the keyword and went stale when the campaign changed the
# per-image source counts, leaving the same image labelled 112 sources here and 8,093
# two tables earlier.
CPU_BASELINE_FILE_MAP = [
    ('C2025A6',   'iKon936_Lum'),
    ('QSO0957',   'iKon936_SDSSg'),
    ('C2025R2',   'QHY600-3_Lum'),
    ('NGC2903',   'QHY600-4_Ha'),
    ('WASP-43-b', 'QHY600-4_SDSSg'),
    ('2012QD8',   'QHY411-1_Lum_bin2'),
    ('GaiaDR3',   'QHY411-1_SDSSi_bin2'),
    ('2025PR1',   'QHY411-1_Lum_full'),
    ('M81',       'QHY411-3_SDSSr_full'),
    ('24P',       'QHY411-3_Lum_full'),
]

# NVTX table: source counts for the single representative image per MP size
NVTX_SRC = {4.2: 296, 6.8: 112, 15.3: 318, 37.8: 154, 151.2: 428}

OOM = r' --- '  # LaTeX OOM / not-tested marker


# ── Formatting helpers ─────────────────────────────────────────────────────────
def _is_missing(v):
    return v is None or (isinstance(v, float) and np.isnan(v))

def fmt_time(v, decimals=1):
    """Latency value or OOM dash."""
    return OOM if _is_missing(v) else f'{v:.{decimals}f}'

def fmt_mb(v):
    """VRAM in MB (integer) or OOM dash."""
    return OOM if _is_missing(v) else str(int(round(v)))

def fmt_src(n):
    """Source count with LaTeX thin-space thousands separator."""
    n = int(n)
    s = f'{n:,}'.replace(',', r'\,')
    return s

_FORBIDDEN = ('QHY', 'iKon')


def save_tex(name, content):
    # The manuscript must never print a camera's commercial name.  Checked here rather
    # than remembered, because these strings live in hard-coded display lists that are
    # easy to extend without thinking about it.
    for bad in _FORBIDDEN:
        if bad in content:
            raise ValueError(f'{name} contains {bad!r}: use instrument_label()')

    path = os.path.join(OUT_DIR, name)
    with open(path, 'w') as fh:
        fh.write(content)
    print(f'  Saved  {path}')


# ── Data loading ───────────────────────────────────────────────────────────────
def _drop_warmup_per_session(grp: pd.DataFrame, n_warmup: int = 2,
                              session_gap_s: float = 1800.0) -> pd.DataFrame:
    """Drop the first n_warmup rows of each benchmark session within a group.

    A new session is detected when the gap between consecutive timestamps
    exceeds session_gap_s seconds (default 30 min).  Rows without timestamps
    are treated as a single session.
    """
    if grp['timestamp'].notna().any():
        # kind='stable' preserves CSV/measurement order for rows with equal
        # timestamps (e.g. batch-parsed 5E rows where all timestamps are the
        # same parse instant). For rows with distinct timestamps this is a no-op.
        grp = grp.sort_values('timestamp', kind='stable').reset_index(drop=True)
        ts = pd.to_datetime(grp['timestamp'], utc=True, errors='coerce')
        deltas = ts.diff().dt.total_seconds().fillna(0)
        # Mark start of each new session
        session_ids = (deltas > session_gap_s).cumsum()
        keep = []
        for _, sess in grp.groupby(session_ids, sort=False):
            keep.append(sess.iloc[n_warmup:])
        return pd.concat(keep, ignore_index=True) if keep else grp.iloc[0:0]
    else:
        return grp.iloc[n_warmup:]


def _mad_filter(grp: pd.DataFrame, col: str = 'execution_time',
                k: float = 3.0) -> pd.DataFrame:
    """Remove upper outliers using Median Absolute Deviation.

    Keeps rows where col <= median + k * MAD.  Only the upper tail is
    filtered because warmup and throttling artefacts always inflate latency;
    values below the median are never discarded.
    """
    vals = grp[col].dropna()
    if len(vals) < 4:
        return grp
    med = vals.median()
    mad = (vals - med).abs().median()
    if mad == 0:
        return grp
    upper = med + k * mad
    return grp[grp[col] <= upper]


def _detect_warmup(grp: pd.DataFrame,
                   col: str = 'execution_time',
                   tol: float = 0.10,
                   k_frac: float = 1 / 3,
                   k_min: int = 5,
                   n_min: int = 10) -> pd.DataFrame:
    """Remove leading warmup transient rows from a time-ordered group.

    Used by _drop_warmup_session_aware. Parameters match load_benchmark_adaptive.py
    exactly (tol=0.10, k_frac=1/3, k_min=5, n_min=10).

    Returns grp unchanged if N < n_min, no row exceeds threshold, or the cut
    would leave fewer than n_min survivors.
    """
    vals = grp[col].values
    n = len(vals)
    if n < n_min:
        return grp
    k = max(k_min, int(n * k_frac))
    k = min(k, n)
    stable_median = float(np.median(vals[-k:]))
    if stable_median <= 0:
        return grp
    threshold = stable_median * (1 + tol)
    cut = -1
    for i in range(n - 1, -1, -1):
        if vals[i] > threshold:
            cut = i
            break
    if cut < 0:
        return grp
    survivors = n - (cut + 1)
    if survivors < n_min:
        # The n_min guard avoids trimming when only a few repetitions would
        # be left for a reliable median, and in general that is right. But
        # it also blocks the opposite case: a LARGE, UNAMBIGUOUS WARM-UP
        # STEP at the start of a short session, where not trimming leaves a
        # median that falls BETWEEN the two regimes and represents neither.
        # It is trimmed all the same when the step is clear enough that
        # there is no ambiguity:
        #   - the head is at least 50% slower than the tail, and
        #   - at least n_floor repetitions remain for the median.
        # The case that motivated this: L40S / QHY411-3_SDSSr_19k on py3.8,
        # seven repetitions at ~63 s and five at ~34 s from the catalog
        # buffer warming up. Without the trim the cell published 59.0 s,
        # which is neither the cold regime nor the warm one. The
        # measurement protocol already states that warm-up gets discarded;
        # this enforces it when the session is short. It only applies to
        # steps AT THE START, which is the only thing this detector looks
        # at: a regime change midway through a session is NOT seen, and
        # that remains a known limitation (internal working note, not
        # included in the public repository).
        # A WARM-UP IS A MINORITY OF THE SESSION, and that is the
        # discriminant needed. Without it this exception degenerates into
        # "keep the fast tail": on H100/QHY411-3_Lum_full there are 24-rep
        # sessions that run slow for 21 and drop in the last 3, and
        # trimming there would throw away 87% of the measurements to
        # publish the final three. That is not removing a startup
        # transient, it is cherry-picking the best stretch.
        # It is therefore required that at least a third of the session
        # survive: in the cell that motivated the exception, 5 of 12 (42%)
        # survive, and in those H100 sessions, 3 of 24 (12%), which falls
        # outside it.
        n_floor = max(4, int(np.ceil(n / 3)))
        if survivors >= n_floor:
            cabeza = float(np.median(vals[:cut + 1]))
            cola = float(np.median(vals[cut + 1:]))
            if cola > 0 and cabeza / cola >= 1.5:
                return grp.iloc[cut + 1:].reset_index(drop=True)
        return grp
    return grp.iloc[cut + 1:].reset_index(drop=True)


def _drop_warmup_session_aware(grp: pd.DataFrame, n_warmup: int = 2,
                                session_gap_s: float = 1800.0) -> pd.DataFrame:
    """Like _drop_warmup_per_session but applies _detect_warmup within each session.

    The adaptive detector runs on each session's post-warmup tail independently,
    so inter-session variance never triggers false positives in the backward
    traversal. Sessions with fewer than n_min rows after the fixed drop are
    passed through untouched.
    """
    if grp['timestamp'].notna().any():
        grp = grp.sort_values('timestamp', kind='stable').reset_index(drop=True)
        ts = pd.to_datetime(grp['timestamp'], utc=True, errors='coerce')
        deltas = ts.diff().dt.total_seconds().fillna(0)
        session_ids = (deltas > session_gap_s).cumsum()
        keep = []
        for _, sess in grp.groupby(session_ids, sort=False):
            tail = sess.iloc[n_warmup:]
            tail = _detect_warmup(tail)
            if len(tail):
                keep.append(tail)
        return pd.concat(keep, ignore_index=True) if keep else grp.iloc[0:0]
    else:
        tail = grp.iloc[n_warmup:]
        return _detect_warmup(tail)


def load_benchmark_legacy(profiler_labels):
    """Original load_benchmark — fixed warmup rule, no adaptive detector, no 5C filter.

    Kept for reference and git history comparison. Not used by any table generator.
    Use load_benchmark() for all production code.
    """
    if isinstance(profiler_labels, str):
        profiler_labels = [profiler_labels]
    df = pd.read_csv(BENCHMARK_CSV)
    df['execution_time'] = pd.to_numeric(df['execution_time'], errors='coerce')
    df['gpu_name'] = df['gpu_name'].str.replace('NVIDIA ', '', regex=False)
    df['gpu_label'] = df['gpu_name'].map(GPU_LABEL_MAP).fillna(df['gpu_name'])
    df = df[df['profiler_label'].isin(profiler_labels)]
    df = df[df['image_label'].notna() & (df['image_label'] != '')]
    df = df[df['execution_time'].notna()]
    parts = []
    for _, grp in df.groupby(['machine', 'gpu_label', 'profiler_label', 'image_label'],
                              sort=False, dropna=False):
        cleaned = _drop_warmup_per_session(grp)
        cleaned = _mad_filter(cleaned)
        if len(cleaned):
            parts.append(cleaned)
    return pd.concat(parts, ignore_index=True) if parts else df


def load_benchmark(profiler_labels, csv_path=None, drop_contaminated=True):
    """Load benchmark CSV, remove warmup and outliers with session-aware adaptive detector.

    Public API for all table generators. Replaces the former fixed-warmup version.

    Pipeline per (machine, gpu_label, profiler_label, image_label) group:
      1. Sort by timestamp.
      2. Detect session boundaries (gap > 30 min); drop first 2 rows per session
         as fixed warmup, then apply adaptive warmup detector within each session
         (_drop_warmup_session_aware).
      3. Apply 3-MAD upper-outlier filter.

    drop_contaminated : bool
        Apply CONTAMINATED_SESSIONS.  True by default, because almost everything read here
        is an ABSOLUTE latency and a contaminated session falsifies it.
        SET IT TO FALSE for comparisons PAIRED WITHIN THE SAME SESSION (the two overhead
        figures): there both arms share the session, so they share the contamination and it
        cancels in the ratio, which is precisely why those figures were designed paired.
        Filtering them discards valid measurements: in one of the two figures it removed 16
        of 52 cells, all of them the ones measured on a single host during the day.

    Rows with remeasure_round in _EXCLUDED_ROUNDS (currently {'5C'}) are dropped
    before grouping. 5C data had bimodal intra-session artefacts and is superseded
    by the clean 5E re-measurement campaign.

    Parameters
    ----------
    profiler_labels : str or list of str
    csv_path : str or None
        Defaults to BENCHMARK_CSV. Pass a different path (e.g. the integrated CSV
        after 5E fusion) for validation or regeneration runs.
    """
    if isinstance(profiler_labels, str):
        profiler_labels = [profiler_labels]
    path = csv_path if csv_path is not None else latency_csv()
    df = pd.read_csv(path)
    if drop_contaminated:
        df = _drop_contaminated_sessions(df)
    df['execution_time'] = pd.to_numeric(df['execution_time'], errors='coerce')
    df['gpu_name'] = df['gpu_name'].str.replace('NVIDIA ', '', regex=False)
    df['gpu_label'] = df['gpu_name'].map(GPU_LABEL_MAP).fillna(df['gpu_name'])
    df = df[df['profiler_label'].isin(profiler_labels)]
    df = df[df['image_label'].notna() & (df['image_label'] != '')]
    df = df[df['execution_time'].notna()]
    # Drop remeasure rounds with known artefacts (bimodal 5C contamination).
    if 'remeasure_round' in df.columns:
        df = df[~df['remeasure_round'].isin(_EXCLUDED_ROUNDS)]
    # Keep only local-catalog rows.  The August-2026 campaign ran entirely against the
    # local PostgreSQL/q3c replica (GPUPHOT_USE_LOCAL_CATALOG=1), which is how the
    # system is deployed, so 'local' is the catalog these tables report.  A table only
    # ever reports one backend: mixing them would compare a network round-trip against
    # a local replica.
    if 'catalog_backend' in df.columns:
        df = df[df['catalog_backend'].fillna('vizier') == 'local']
    parts = []
    for _, grp in df.groupby(['machine', 'gpu_label', 'profiler_label', 'image_label'],
                              sort=False, dropna=False):
        cleaned = _drop_warmup_session_aware(grp)
        cleaned = _mad_filter(cleaned)
        if len(cleaned):
            parts.append(cleaned)
    return pd.concat(parts, ignore_index=True) if parts else df


def _load_benchmark_all_catalogs(profiler_labels, csv_path=None):
    """Like load_benchmark() but retains all catalog_backend values.

    Applies the same pipeline — 5C exclusion, session-aware warmup detection,
    3-MAD outlier filter — without filtering to catalog_backend='vizier'.
    Grouping includes catalog_backend so warmup/outlier removal runs
    independently per catalog within each (machine, gpu, profiler, image) cell.

    Legacy rows with no catalog_backend (NaN) are treated as 'vizier'.

    Used exclusively by gen_speedup_local_vizier() and gen_iqr_collapse(), which need a
    file holding BOTH backends.  Only the April campaign measured both on the same
    code, so it defaults to the archived April CSV rather than to BENCHMARK_CSV.
    """
    if isinstance(profiler_labels, str):
        profiler_labels = [profiler_labels]
    path = csv_path if csv_path is not None else CATALOG_BACKENDS_CSV
    df = pd.read_csv(path)
    df['execution_time'] = pd.to_numeric(df['execution_time'], errors='coerce')
    df['gpu_name'] = df['gpu_name'].str.replace('NVIDIA ', '', regex=False)
    df['gpu_label'] = df['gpu_name'].map(GPU_LABEL_MAP).fillna(df['gpu_name'])
    df = df[df['profiler_label'].isin(profiler_labels)]
    df = df[df['image_label'].notna() & (df['image_label'] != '')]
    df = df[df['execution_time'].notna()]
    if 'remeasure_round' in df.columns:
        df = df[~df['remeasure_round'].isin(_EXCLUDED_ROUNDS)]
    # Normalise catalog_backend: legacy rows (NaN) were Vizier-only
    df = df.copy()
    if 'catalog_backend' in df.columns:
        df['catalog_backend'] = df['catalog_backend'].fillna('vizier')
    else:
        df['catalog_backend'] = 'vizier'
    parts = []
    for _, grp in df.groupby(
            ['machine', 'gpu_label', 'profiler_label', 'image_label', 'catalog_backend'],
            sort=False, dropna=False):
        cleaned = _drop_warmup_session_aware(grp)
        cleaned = _mad_filter(cleaned)
        if len(cleaned):
            parts.append(cleaned)
    return pd.concat(parts, ignore_index=True) if parts else df



# ── Which session each column comes from ───────────────────────────────────────
# User decision, 2026-09-11. The criterion is NOT the date: it is the CODE.
# The paper presents a single campaign because every column comes from the
# same code; the best available measurement of each cell is taken, and when
# it was taken is not a property of the result. That is why the columns carry
# NO mark: there is nothing to mark.
#
# For that to be TRUE and not a convenience requires choosing by ARM, not by
# card, because the allocator fix (1a23734) only changes py3.12's code:
#
#   py3.12  The fix DOES change its code, so every column has to come from
#           after it, or the paper would compare different versions without
#           saying so. All six x86 cards have full post-fix coverage. The
#           two Jetsons do not, and do not need to: on aarch64 `import cuml`
#           fails (catalog.py:44 wraps it in try/except), there is no
#           allocator to hijack, and their measurement is ALREADY that of
#           the fixed code.
#
#   py3.8   cuML does not matter, so the fix does not touch its code: every
#           session is the SAME code, and the best measurement of each cell
#           is chosen. H100, A100, L40S and RTX 3060 have a clean
#           remeasurement and use it. RTX 3090 and RTX 3050 Ti keep their
#           earlier one because the new one came out contaminated
#           (production-catalog contention on one, thermal cadence step on
#           the other), and the Jetsons were not remeasured.
#
# Evidence: py38_epoch_control_per_cell.csv (plus internal working notes
# not included in the public repository).
SESSION_BY_ARM = {
    'py312_cuml_adaptive': {
        'H100 (80 GB)': 'postfix', 'A100 (80 GB)': 'postfix', 'L40S (48 GB)': 'postfix',
        'RTX 3090 (24 GB)': 'postfix', 'RTX 3060 (12 GB)': 'postfix',
        'RTX 3050 Ti (4 GB)': 'postfix',
        'Orin Super (8 GB)': 'campaign', 'Orin Nano (8 GB)': 'campaign',
    },
    'py38_baseline': {
        'H100 (80 GB)': 'postfix', 'A100 (80 GB)': 'postfix', 'L40S (48 GB)': 'postfix',
        'RTX 3060 (12 GB)': 'postfix',
        'RTX 3090 (24 GB)': 'campaign', 'RTX 3050 Ti (4 GB)': 'campaign',
        'Orin Super (8 GB)': 'campaign', 'Orin Nano (8 GB)': 'campaign',
    },
}
GPU_EPOCH = SESSION_BY_ARM['py312_cuml_adaptive']   # orden de columnas y compatibilidad


def gpu_epoch(gpu_label, profiler_label=None):
    """Which campaign a column is read from.  With the switch off, everything is the earlier one."""
    if not USE_POSTFIX_CAMPAIGN:
        return 'campaign'
    if isinstance(profiler_label, (list, tuple)):
        profiler_label = profiler_label[0] if len(profiler_label) == 1 else None
    tabla = SESSION_BY_ARM.get(profiler_label, SESSION_BY_ARM['py312_cuml_adaptive'])
    return tabla.get(gpu_label, 'campaign')


def gpu_header(gpu_label):
    """Cabecera de la columna.  Sin marca de sesión: el artículo presenta una sola campaña."""
    return GPU_TEX_HEADER[gpu_label]



def load_benchmark_by_epoch(profiler_labels, gpu_cols=None):
    """Rows of each card read from the campaign that GPU_EPOCH assigns to it.

    The single place where the per-column split is materialised, so that the tables, the
    figures and the CPU baseline cannot disagree with one another.  With the switch off,
    everything comes from the earlier campaign.
    """
    cols = list(gpu_cols) if gpu_cols is not None else list(GPU_EPOCH)
    frames = []
    for epoch in ('campaign', 'postfix'):
        sel = [g for g in cols if gpu_epoch(g, profiler_labels) == epoch]
        if not sel:
            continue
        path = LATENCY_POSTFIX_CSV if epoch == 'postfix' else BENCHMARK_CSV
        d = load_benchmark(profiler_labels, csv_path=path)
        frames.append(d[d['gpu_label'].isin(sel)])
    if not frames:
        return pd.DataFrame(columns=['gpu_label', 'image_label', 'execution_time'])
    return pd.concat(frames, ignore_index=True)


def latency_median(profiler_label, gpu_cols):
    """Return dict[image_label][gpu_label] = median latency (NaN if missing).

    Each column is read from the campaign that GPU_EPOCH assigns to it.  With the switch
    off they all fall back to the earlier campaign.
    """
    df = load_benchmark_by_epoch(profiler_label, gpu_cols)
    med = df.groupby(['gpu_label', 'image_label'])['execution_time'].median()
    result = {}
    for img in IMAGE_ORDER:
        result[img] = {}
        for gpu in gpu_cols:
            try:
                result[img][gpu] = med.loc[(gpu, img)]
            except KeyError:
                result[img][gpu] = float('nan')
    return result


def vram_median(python_ver, gpu_cols):
    """Return dict[mp][gpu_label] = median peak VRAM in MB (NaN if missing).

    Only rows with capture_complete=True are used to avoid including
    partial measurements from OOM crashes (where Nsight records the peak
    allocation before the crash rather than the true steady-state peak).
    Rows where peak_gpu_memory_MB > gpu_total_MB are also dropped: these
    are physically impossible and are Nsight artefacts (typically observed
    on cards that OOM'd mid-profiling or produced very few nsys_events).
    """
    df = pd.read_csv(MEMORY_CSV)
    df['python_ver'] = pd.to_numeric(df['python_ver'], errors='coerce')
    df['gpu_name'] = df['gpu_name'].str.replace('NVIDIA ', '', regex=False)
    df['gpu_label'] = df['gpu_name'].map(GPU_LABEL_MAP).fillna(df['gpu_name'])
    df['megapixels'] = pd.to_numeric(df['megapixels'], errors='coerce')
    df['gpu_total_MB'] = pd.to_numeric(df['gpu_total_MB'], errors='coerce')
    df['peak_gpu_memory_MB'] = pd.to_numeric(df['peak_gpu_memory_MB'], errors='coerce')
    # Keep only successful profiling captures
    if 'capture_complete' in df.columns:
        df = df[df['capture_complete'].astype(str).str.strip().str.lower() == 'true']
    # Drop physically impossible readings (artefacts from crashed/truncated profiling)
    df = df[df['peak_gpu_memory_MB'] <= df['gpu_total_MB']]
    df = df[df['python_ver'].round(2) == round(python_ver, 2)]
    mp_vals = sorted(df['megapixels'].dropna().unique())
    med = df.groupby(['gpu_label', 'megapixels'])['peak_gpu_memory_MB'].median()
    result = {}
    for mp in mp_vals:
        result[mp] = {}
        for gpu in gpu_cols:
            try:
                result[mp][gpu] = med.loc[(gpu, mp)]
            except KeyError:
                result[mp][gpu] = float('nan')
    return result, mp_vals


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def gen_speedup_local_vizier():
    """tab:speedup_local_vizier — speedup of local PostgreSQL vs Vizier catalog backend.

    For each (image, GPU, stack) cell, computes:
      speedup = median_vizier / median_local
    across the 6 × 151.2 MP images measured with both backends.

    Columns: A100 / H100 / L40S × py3.12+adaptive / py3.8 (6 combinations).
    Footer shows overall median and range across all valid cells.

    Source: data/benchmark_latency.csv (catalog_backend = vizier | local)
    """
    print('Generating body_speedup_local_vizier.tex ...')

    IMAGES_LV = [
        ('QHY411-1_Lum_full',   428,    instrument_label('QHY411-1_Lum_full') + ' Lum'),
        ('QHY411-3_SDSSg_10k',  10005,  instrument_label('QHY411-3_SDSSg_10k') + ' SDSSg'),
        ('QHY411-3_SDSSr_full', 14241,  instrument_label('QHY411-3_SDSSr_full') + r' SDSSr$_{\rm 14k}$'),
        ('QHY411-3_Lum_full',   18888,  instrument_label('QHY411-3_Lum_full') + r' Lum$_{\rm 18k}$'),
        ('QHY411-3_SDSSr_19k',  19565,  instrument_label('QHY411-3_SDSSr_19k') + r' SDSSr$_{\rm 19k}$'),
        ('QHY411-3_Lum_131k',  131397,  instrument_label('QHY411-3_Lum_131k') + r' Lum$_{\rm 131k}$'),
    ]
    PROFILERS_LV = ['py312_cuml_adaptive', 'py38_baseline']
    GPUS_LV = ['A100 (80 GB)', 'H100 (80 GB)', 'L40S (48 GB)']
    COLS_LV = [(p, g) for p in PROFILERS_LV for g in GPUS_LV]

    df = _load_benchmark_all_catalogs(PROFILERS_LV)
    med = (
        df.groupby(['profiler_label', 'gpu_label', 'image_label', 'catalog_backend'])
          ['execution_time'].median()
    )

    all_sp = []
    col_sp = {c: [] for c in COLS_LV}
    data_rows = []

    for img_key, nsrc, img_label in IMAGES_LV:
        vals = []
        for prof, gpu in COLS_LV:
            try:
                v = med.loc[(prof, gpu, img_key, 'vizier')]
                l = med.loc[(prof, gpu, img_key, 'local')]
                sp = (v / l) if (not np.isnan(v) and not np.isnan(l) and l > 0) else None
            except KeyError:
                sp = None
            vals.append(sp)
            if sp is not None:
                all_sp.append(sp)
                col_sp[(prof, gpu)].append(sp)
        cells = [f'{v:.2f}' if v is not None else '---' for v in vals]
        data_rows.append(
            f'{img_label} & {fmt_src(nsrc)} & ' + ' & '.join(cells) + r' \\'
        )

    col_meds = [
        f'{np.median(col_sp[c]):.2f}' if col_sp[c] else '---' for c in COLS_LV
    ]
    median_row = 'Median (column) & & ' + ' & '.join(col_meds) + r' \\'

    n_valid = len(all_sp)
    om   = np.median(all_sp)
    amin = min(all_sp)
    amax = max(all_sp)
    footer = (
        rf'\multicolumn{{8}}{{l}}{{\footnotesize Overall median (valid cells,'
        rf' $N={n_valid}$): {om:.2f}$\times$;\quad'
        rf' range: {amin:.2f}--{amax:.2f}$\times$}}\\'
    )

    lines = [
        r'\begin{tabular}{lrcccccc}',
        r'\toprule',
        r' & & \multicolumn{3}{c}{\textbf{py3.12\,+\,adaptive}} & \multicolumn{3}{c}{\textbf{py3.8}} \\',
        r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
        r'\textbf{Image} & \textbf{Src} & \textbf{A100} & \textbf{H100} & \textbf{L40S} & \textbf{A100} & \textbf{H100} & \textbf{L40S} \\',
        r'\midrule',
    ] + data_rows + [
        r'\midrule',
        median_row,
        footer,
        r'\bottomrule',
        r'\end{tabular}',
    ]
    save_tex('body_speedup_local_vizier.tex', '\n'.join(lines) + '\n')


def gen_iqr_collapse():
    """tab:iqr_collapse — catalogue backend effect on benchmark variability.

    Lists all (GPU, image, stack) cells where IQR_Vizier >= 5%, paired with
    IQR_local of the same cell.  IQR is expressed as a percentage of the
    median: IQR% = (Q3 − Q1) / median × 100.  Rows sorted by IQR_Vizier
    descending.  The QHY411-3 camera prefix is omitted from image labels
    (all high-IQR cells are QHY411-3 images; the caption notes this).

    Source: data/benchmark_latency.csv (catalog_backend = vizier | local)
    """
    print('Generating body_iqr_collapse.tex ...')

    GPU_SHORT_IQR = {
        'H100 (80 GB)':       'H100',
        'A100 (80 GB)':       'A100',
        'L40S (48 GB)':       'L40S',
        'RTX 3090 (24 GB)':   '3090',
        'RTX 3060 (12 GB)':   '3060',
        'RTX 3050 Ti (4 GB)': r'3050\,Ti',
    }
    IMG_IQR_DISPLAY = {
        'QHY411-3_SDSSr_full': r'SDSSr$_{\rm 14k}$',
        'QHY411-3_Lum_131k':   r'Lum$_{\rm 131k}$',
        'QHY411-3_SDSSr_19k':  r'SDSSr$_{\rm 19k}$',
        'QHY411-3_SDSSg_10k':  'SDSSg',
        'QHY411-3_Lum_full':   r'Lum$_{\rm 18k}$',
        'QHY411-1_Lum_full':   instrument_label('QHY411-1_Lum_full') + ' Lum',
    }
    STACK_SHORT = {
        'py312_cuml_adaptive': 'py312',
        'py38_baseline':       'py38',
    }

    PROFILERS_IQR = ['py312_cuml_adaptive', 'py38_baseline']

    def _iqr_pct(s):
        med = s.median()
        return float('nan') if med <= 0 else (s.quantile(0.75) - s.quantile(0.25)) / med * 100

    # Both arms come from the April CSV through the all-catalogs loader, so the two
    # columns are filtered identically and neither can pick up the August campaign,
    # which has no Vizier arm to pair against.
    df_all = _load_benchmark_all_catalogs(PROFILERS_IQR)
    df_viz = df_all[df_all['catalog_backend'] == 'vizier']
    grp_viz = df_viz.groupby(['gpu_label', 'profiler_label', 'image_label'])['execution_time']
    stats_viz = pd.DataFrame({
        'n':       grp_viz.count(),
        'iqr_pct': grp_viz.apply(_iqr_pct),
    }).reset_index()

    df_loc = df_all[df_all['catalog_backend'] == 'local']
    grp_loc = df_loc.groupby(['gpu_label', 'profiler_label', 'image_label'])['execution_time']
    stats_loc = pd.DataFrame({
        'n':       grp_loc.count(),
        'iqr_pct': grp_loc.apply(_iqr_pct),
    }).reset_index()

    merged = pd.merge(
        stats_viz.rename(columns={'n': 'n_viz', 'iqr_pct': 'iqr_pct_viz'}),
        stats_loc.rename(columns={'n': 'n_loc', 'iqr_pct': 'iqr_pct_loc'}),
        on=['gpu_label', 'profiler_label', 'image_label'],
    )
    # Filter on the displayed (rounded) value so that 4.993…% → "5.0%" is included.
    filtered = merged[merged['iqr_pct_viz'].round(1) >= 5.0].sort_values(
        'iqr_pct_viz', ascending=False
    )

    rows = []
    for _, r in filtered.iterrows():
        gpu = GPU_SHORT_IQR.get(r['gpu_label'], r['gpu_label'])
        img = IMG_IQR_DISPLAY.get(r['image_label'], r['image_label'])
        stk = STACK_SHORT.get(r['profiler_label'], r['profiler_label'])
        rows.append(
            f'{gpu} & {img} & {stk} & {int(r["n_viz"])} & {r["iqr_pct_viz"]:.1f}'
            f' & {int(r["n_loc"])} & {r["iqr_pct_loc"]:.1f}' + r' \\'
        )

    header = (
        r'\textbf{GPU} & \textbf{Image} & \textbf{Stack} & '
        r'\textbf{$N_{\rm Viz}$} & \textbf{IQR$_{\rm Viz}$\,(\%)} & '
        r'\textbf{$N_{\rm loc}$} & \textbf{IQR$_{\rm loc}$\,(\%)} \\'
    )
    # RETIRED 2026-09-11: never included in the manuscript. The calculation is kept.
    # save_tex('body_iqr_collapse.tex', _tabular(r'{llcrrrc}', header, rows))


def gen_latency_py312():
    """
    tab:latency_py312 — Median latency (s), py3.12 + adaptive cuML.
    Source: data/benchmark_latency.csv (profiler_label=py312_cuml_adaptive)
    GPUs: H100, A100, L40S, RTX 3090, RTX 3060, RTX 3050 Ti, Orin Super
    Orin Super runs py3.12 on ARM without cuML (unavailable on aarch64);
    its data is stored under profiler_label=py312_cuml_adaptive in the CSV.
    Orin Nano has no py3.12 data and is therefore omitted from this table.
    Note: RTX 3050 Ti has data for 4.2 MP; larger images will show --- until
    the full adaptive benchmark is run on that machine (images 3-7).
    """
    print('Generating body_latency_py312.tex ...')
    gpu_cols = [
        'H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)',
        'RTX 3090 (24 GB)', 'RTX 3060 (12 GB)', 'RTX 3050 Ti (4 GB)',
        'Orin Super (8 GB)',
    ]
    data = latency_median('py312_cuml_adaptive', gpu_cols)

    col_spec = r'{rr rrrrrrr}'
    header = (
        r'\textbf{MP} & \textbf{Sources}' + '\n'
        r'            & ' + ' & '.join(gpu_header(g) for g in gpu_cols) + r' \\'
    )
    rows = []
    for img in IMAGE_ORDER:
        mp  = IMAGE_MP[img]
        src = IMAGE_SRC[img]
        vals = ' & '.join(fmt_time(data[img][g]) for g in gpu_cols)
        rows.append(rf'{_mp_str(mp):5s} & {fmt_src(src):>8s} & {vals} \\')

    body = _tabular(col_spec, header, rows)
    save_tex('body_latency_py312.tex', body)


def gen_latency_py38():
    """
    tab:latency_py38 — Median latency (s), py3.8 baseline.
    Source: data/benchmark_latency.csv (profiler_label=py38_baseline)
    GPUs: H100, A100, L40S, RTX 3090, RTX 3060, RTX 3050 Ti, Orin Nano
    """
    print('Generating body_latency_py38.tex ...')
    # Orin Super gets its own column: it has 104 py3.8 repetitions
    # (2026-08-28) over two images, and omitting it would suggest the new
    # ARM module only ever existed under py3.12. The two ARM columns are
    # different modules, not two states of the same one: the NX stayed on
    # JetPack 5 (py3.8 only), and the Super was reinstalled with JetPack 6.
    gpu_cols = [
        'H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)',
        'RTX 3090 (24 GB)', 'RTX 3060 (12 GB)',
        'RTX 3050 Ti (4 GB)', 'Orin Nano (8 GB)', 'Orin Super (8 GB)',
    ]
    data = latency_median('py38_baseline', gpu_cols)

    col_spec = r'{rr rrrrrrrr}'
    header = (
        r'\textbf{MP} & \textbf{Sources}' + '\n'
        r'            & ' + ' & '.join(gpu_header(g) for g in gpu_cols) + r' \\'
    )
    rows = []
    for img in IMAGE_ORDER:
        mp  = IMAGE_MP[img]
        src = IMAGE_SRC[img]
        vals = ' & '.join(fmt_time(data[img][g]) for g in gpu_cols)
        rows.append(rf'{_mp_str(mp):5s} & {fmt_src(src):>8s} & {vals} \\')

    body = _tabular(col_spec, header, rows)
    save_tex('body_latency_py38.tex', body)


def gen_vram_merged():
    """
    tab:vram_merged — Peak GPU memory (MiB) and the py3.8 to py3.12 difference per image size.  Not a saving: the
    lower py3.12 peak is the allocator hijack returning memory to the driver, and it goes
    away with the fix.
    Uses A100-SXM4-80GB as the representative GPU (chosen because it has complete
    data for all 10 benchmark images and is the most broadly deployed 80 GB card).
    Cross-GPU variation among datacenter GPUs for the same image is <0.9% (see Figure fig:memory_comparison).
    Replaces the former tab:vram_py312, tab:vram_py38, and tab:vram_savings.
    Source: data/profiler_memory_raw.csv
    """
    print('Generating body_vram_merged.tex ...')
    rep_gpu = 'A100 (80 GB)'
    data312, mp_vals = vram_median(3.12, [rep_gpu])
    data38,  _       = vram_median(3.8,  [rep_gpu])

    col_spec = r'{r rrr}'
    header = (
        r'\textbf{MP} & \textbf{py3.8 (MiB)} & \textbf{py3.12 (MiB)} & '
        r'\textbf{Difference (\%)} \\'
    )
    rows = []
    for mp in mp_vals:
        v38  = data38[mp][rep_gpu]
        v312 = data312[mp][rep_gpu]
        if _is_missing(v38) or _is_missing(v312):
            saving_str = OOM
        else:
            # Recalculate from the same rounded-integer values displayed in the
            # table so the reader can verify the percentage from the MB columns.
            saving_str = f'{100.0 * (round(v38) - round(v312)) / round(v38):.1f}'
        rows.append(
            rf'{mp:5.1f} & {fmt_mb(v38)} & {fmt_mb(v312)} & {saving_str} \\'
        )

    body = _tabular(col_spec, header, rows)
    # RETIRED 2026-09-11: the manuscript no longer includes it (tab:vram_merged
    # switched to body_vram_ablation.tex, which shows the three allocator
    # states instead of two). The function is kept because the calculation
    # is still the source for that table and for tab:concurrency; it just
    # stops writing a .tex that nothing includes and that kept reappearing
    # as a stray file every time it was regenerated.
    # save_tex('body_vram_merged.tex', body)

    # Keep the per-GPU tables as well (still generated, but no longer \input{}'d
    # from the manuscript). Kept for reference and reproducibility.
    # Same reason: per-GPU, never included.
    # _gen_vram_all_gpus(3.12, 'body_vram_py312.tex')
    # _gen_vram_all_gpus(3.8,  'body_vram_py38.tex')


def _gen_vram_all_gpus(python_ver, filename):
    """Internal helper: per-GPU VRAM table (kept for reference, not in manuscript)."""
    gpu_cols = [
        'H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)',
        'RTX 3090 (24 GB)', 'RTX 3060 (12 GB)', 'RTX 3050 Ti (4 GB)',
    ]
    data, mp_vals = vram_median(python_ver, gpu_cols)
    col_spec = r'{r rrrrrrr}'
    header = (
        r'\textbf{MP} & '
        + ' & '.join(GPU_TEX_HEADER[g] for g in gpu_cols) + r' \\'
    )
    rows = []
    for mp in mp_vals:
        vals = ' & '.join(fmt_mb(data[mp][g]) for g in gpu_cols)
        rows.append(rf'{mp:5.1f} & {vals} \\')
    body = _tabular(col_spec, header, rows)
    save_tex(filename, body)


def gen_concurrency():
    """
    tab:concurrency — Max concurrent 4.2 MP images per GPU.
    Source: data/profiler_memory_raw.csv (4.2 MP median peak VRAM per py ver)
    Formula: floor(GPU_VRAM_total / peak_VRAM_per_image)
    GPU_VRAM_total is read from the gpu_total_MB column of the memory CSV
    (e.g. RTX 3050 Ti Laptop GPU reports 3780.8 MB, not the nominal 4096 MB).
    """
    print('Generating body_concurrency.tex ...')
    gpu_cols = [
        'H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)',
        'RTX 3090 (24 GB)', 'RTX 3060 (12 GB)', 'RTX 3050 Ti (4 GB)',
    ]
    data38,  _ = vram_median(3.8,  gpu_cols)
    data312, _ = vram_median(3.12, gpu_cols)

    # Read actual GPU total VRAM from memory CSV (more accurate than nominal spec)
    mem_df = pd.read_csv(MEMORY_CSV)
    mem_df['gpu_name'] = mem_df['gpu_name'].str.replace('NVIDIA ', '', regex=False)
    mem_df['gpu_label'] = mem_df['gpu_name'].map(GPU_LABEL_MAP).fillna(mem_df['gpu_name'])
    actual_vram_mb = mem_df.groupby('gpu_label')['gpu_total_MB'].median().to_dict()

    MP_4K = 4.2  # 4.2 MP iKon936 images are the reference workload
    gpu_short = {
        'H100 (80 GB)':       'H100',
        'A100 (80 GB)':       'A100',
        'L40S (48 GB)':       'L40S',
        'RTX 3090 (24 GB)':   'RTX 3090',
        'RTX 3060 (12 GB)':   'RTX 3060',
        'RTX 3050 Ti (4 GB)': 'RTX 3050 Ti',
    }

    col_spec = r'{lrrrr}'
    header = (
        r'\textbf{GPU} & \textbf{VRAM (GB)} & '
        r'\textbf{py3.8} & \textbf{py3.12} & \textbf{Gain (\%)} \\'
    )
    rows = []
    for gpu in gpu_cols:
        # Use the SMALLER of nominal spec and measured value.
        # Server GPUs (H100, A100) occasionally report >80 GB in NVML due to
        # unified memory reporting; capping at nominal avoids inflated concurrency.
        # Laptop GPUs (3050 Ti) report slightly less than nominal (3780.8 vs 4096 MB)
        # and the measured value is used so the +100% gain shows correctly.
        vram_total_mb = min(actual_vram_mb.get(gpu, GPU_VRAM_MB[gpu]), GPU_VRAM_MB[gpu])
        vram_gb       = GPU_VRAM_GB[gpu]
        peak38  = data38.get(MP_4K,  {}).get(gpu, float('nan'))
        peak312 = data312.get(MP_4K, {}).get(gpu, float('nan'))
        # Fallback to A100 proxy if GPU-specific VRAM data is missing.
        # Justified by the <0.3% inter-GPU variation documented in §perf:vram.
        if _is_missing(peak38):
            peak38 = data38.get(MP_4K, {}).get('A100 (80 GB)', float('nan'))
        if _is_missing(peak312):
            peak312 = data312.get(MP_4K, {}).get('A100 (80 GB)', float('nan'))
        if _is_missing(peak38) or _is_missing(peak312):
            rows.append(rf'{gpu_short[gpu]:<14s} & {vram_gb} & {OOM} & {OOM} & {OOM} \\')
            continue
        # Caption formula: floor(VRAM_total / peak_VRAM) — no safety-margin factor
        c38  = math.floor(vram_total_mb / peak38)
        c312 = math.floor(vram_total_mb / peak312)
        gain = (c312 - c38) / c38 * 100.0 if c38 > 0 else float('nan')
        gain_str = f'$+{gain:.0f}$\\,\\%' if not _is_missing(gain) else OOM
        rows.append(
            rf'{gpu_short[gpu]:<14s} & {vram_gb:2d} & {c38:2d} & {c312:2d} & {gain_str} \\'
        )

    body = _tabular(col_spec, header, rows)
    save_tex('body_concurrency.tex', body)



def vram_postfix_a100():
    """Pico post-corrección en el A100, por tamaño, en MiB.  Los cinco tamaños salen de dos sitios.

    profiler_memory_postfix.csv (bloque 5, 2026-09-10) trae 6,8, 15,3 y 37,8 MP.
    memory_allocator_fix.csv trae 4,2 y 151,2 MP, medidos antes en la misma máquina.
    UNIDADES: los dos están en MiB.  El campo peak_gpu_memory_MB de los ficheros de campaña se
    llama MB pero contiene MiB (bytes/2^20), y memory_allocator_fix.csv lleva su pico ya
    convertido en la columna pico_MiB.  Mezclarlos sin mirar el nombre daría un 4,86% de desfase
    que no existe: es 2^20/10^6.
    """
    out = {}
    d = pd.read_csv(MEMORY_POSTFIX_CSV, comment='#')
    d = d[(d['gpu_name'] == 'A100-SXM4-80GB')
          & (d['capture_complete'].astype(str).str.strip().str.lower() == 'true')]
    for mp, g in d.groupby('megapixels'):
        out[float(mp)] = g['peak_gpu_memory_MB'].median()
    f = pd.read_csv(MEMORY_ALLOC_FIX_CSV, comment='#')
    f = f[(f['gpu'] == 'A100') & (f['fix_state'] == 'post_fix')]
    for _, r in f.iterrows():
        out.setdefault(float(r['MP']), float(r['pico_MiB']))
    return out


def gen_vram_ablation():
    """tab:vram_ablation — peak device memory in the three allocator states.

    It replaces a reading that was previously split across two other tables, so that the three
    configurations sit side by side and the reader sees at once that the apparent memory
    saving of the newer stack was not a property of it but the hijacked allocator returning
    memory to the driver, and that re-seating the pool brings the peak back to the older stack.

    State columns:
      older stack        the long-standing CuPy pool, no cuML.
      newer, no pool     importing cuML installs the RAPIDS allocator unpooled: every temporary
                         pays a driver allocation and the peak falls because nothing is retained.
                         This is the configuration of the earlier latency tables.
      newer, with pool   the pool re-seated per task, which is the released version.
    Fuentes: profiler_memory_raw.csv (las dos primeras) y vram_postfix_a100() (la tercera).
    """
    print('Generating body_vram_ablation.tex ...')
    rep = 'A100 (80 GB)'
    d38, mps = vram_median(3.8, [rep])
    d312, _ = vram_median(3.12, [rep])
    post = vram_postfix_a100()

    header = (r'\textbf{MP} & \textbf{\shortstack{py3.8\\(MiB)}} & '
              r'\textbf{\shortstack{py3.12 no\\pool (MiB)}} & '
              r'\textbf{\shortstack{py3.12 with\\pool (MiB)}} & '
              r'\textbf{\shortstack{No pool vs\\py3.8 (\%)}} & '
              r'\textbf{\shortstack{With pool vs\\py3.8 (\%)}} \\')
    rows = []
    for mp in mps:
        v38, v312 = d38[mp][rep], d312[mp][rep]
        vpost = post.get(mp)
        if _is_missing(v38) or _is_missing(v312):
            continue
        # Percentages are computed from the SAME integers the table prints,
        # so the reader can redo them from what they see and not from
        # decimals that are not there.
        a, b = round(v38), round(v312)
        pre = f'${100.0 * (b - a) / a:+.1f}$'
        pos = f'${100.0 * (round(vpost) - a) / a:+.1f}$' if vpost else OOM
        rows.append(rf'{mp:5.1f} & {fmt_mb(v38)} & {fmt_mb(v312)} & '
                    rf'{fmt_mb(vpost) if vpost else OOM} & {pre} & {pos} \\')
    save_tex('body_vram_ablation.tex', _tabular(r'{r rrr rr}', header, rows))


def gen_concurrency_ablation():
    """tab:concurrency_ablation — the concurrency bound in the three allocator states.

    The plain concurrency table publishes a gain between the two stacks that its own caption then
    has to deny, because that gain was the allocator artefact. With the third column the table
    stands on its own: the bound returns to the older stack's value and the apparent doubling on
    the smallest card cancels, without the reader having to take the caption's word for it.
    Cota = parte entera de (VRAM reportada / pico a 4,2 MP).  Es aritmetica, no una concurrencia
    medida, y por eso el pie lo dice.
    """
    print('Generating body_concurrency_ablation.tex ...')
    gpu_cols = [
        'H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)',
        'RTX 3090 (24 GB)', 'RTX 3060 (12 GB)', 'RTX 3050 Ti (4 GB)',
    ]
    gpu_short = {'H100 (80 GB)': 'H100', 'A100 (80 GB)': 'A100', 'L40S (48 GB)': 'L40S',
                 'RTX 3090 (24 GB)': 'RTX 3090', 'RTX 3060 (12 GB)': 'RTX 3060',
                 'RTX 3050 Ti (4 GB)': 'RTX 3050 Ti'}
    MP_4K = 4.2
    data38, _ = vram_median(3.8, gpu_cols)
    data312, _ = vram_median(3.12, gpu_cols)
    mem_df = pd.read_csv(MEMORY_CSV)
    mem_df['gpu_name'] = mem_df['gpu_name'].str.replace('NVIDIA ', '', regex=False)
    mem_df['gpu_label'] = mem_df['gpu_name'].map(GPU_LABEL_MAP).fillna(mem_df['gpu_name'])
    actual_vram_mb = mem_df.groupby('gpu_label')['gpu_total_MB'].median().to_dict()
    # The post-fix peak was measured on the A100 and on the RTX 3050 Ti and
    # agrees between the two within 1.8%; the A100's is used for all six,
    # same as the other two columns do, whose variation across cards for
    # the same image is under 0.9%.
    peak_post = vram_postfix_a100().get(MP_4K)

    header = (r'\textbf{GPU} & \textbf{VRAM (GB)} & \textbf{py3.8} & '
              r'\textbf{\shortstack{py3.12\\no pool}} & '
              r'\textbf{\shortstack{py3.12\\with pool}} \\')
    rows = []
    for gpu in gpu_cols:
        total = min(actual_vram_mb.get(gpu, GPU_VRAM_MB[gpu]), GPU_VRAM_MB[gpu])
        p38 = data38.get(MP_4K, {}).get(gpu, float('nan'))
        p312 = data312.get(MP_4K, {}).get(gpu, float('nan'))
        if _is_missing(p38):
            p38 = data38.get(MP_4K, {}).get('A100 (80 GB)', float('nan'))
        if _is_missing(p312):
            p312 = data312.get(MP_4K, {}).get('A100 (80 GB)', float('nan'))
        cells = []
        for pk in (p38, p312, peak_post):
            cells.append(f'{math.floor(total / pk):2d}' if pk and not _is_missing(pk) else OOM)
        rows.append(rf'{gpu_short[gpu]:<14s} & {GPU_VRAM_GB[gpu]:2d} & ' + ' & '.join(cells) + r' \\')
    save_tex('body_concurrency_ablation.tex', _tabular(r'{lrrrr}', header, rows))

def gen_cuml_ablation():
    """
    tab:cuml_ablation — forcing cuML for every crossmatch against forcing cKDTree.

    Source: data/cuml_ablation_postfix1_a100_20260907.csv, one row per repetition
    (19 images x 2 arms x 5 reps, A100, current camera configs, post allocator fix).

    The version this replaces was measured in June under the April configurations, and it
    reported source counts that no longer matched the latency tables: 428 where those say
    808, 131,397 where they say 134,206.  The text quoted both as if they were the same
    number.  Measured again under the deployed configurations, the two agree, so the
    manuscript can carry one count per image.

    Two of the 190 runs produced no source count and are dropped: one returned in 3.5 s
    where its cell takes 27, and one has no time at all.  Keeping them changes neither the
    median nor the range, which is stated here because a reader is entitled to know that
    the exclusion was checked rather than assumed.
    """
    print('Generating body_cuml_ablation.tex ...')
    df = pd.read_csv(CUML_ABLATION_CSV, comment='#')
    df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')
    df['sources'] = pd.to_numeric(df['sources'], errors='coerce')
    df = df.dropna(subset=['time_s', 'sources'])

    rows = []
    keys = df.groupby('image').agg(mp=('mp', 'first'), src=('sources', 'median'))
    for img in keys.sort_values(['mp', 'src']).index:
        sub = df[df['image'] == img]
        with_cuml = sub[sub['cuml'] == 'yes']['time_s'].median()
        without = sub[sub['cuml'] == 'no']['time_s'].median()
        if pd.isna(with_cuml) or pd.isna(without):
            continue
        pct = (with_cuml - without) / without * 100
        sign = '+' if pct >= 0 else '-'
        rows.append(
            f'{_mp_str(keys.loc[img, "mp"])}   & {fmt_src(int(keys.loc[img, "src"]))} & '
            f'{with_cuml:.1f} & {without:.1f} & ' + '$' + sign + f'{abs(pct):.0f}' + '$ '
            + chr(92) * 2
        )
    header = (r'\textbf{MP} & \textbf{Sources} & \textbf{\shortstack{With\\cuML (s)}} & '
              r'\textbf{\shortstack{Without\\cuML (s)}} & \textbf{Penalty (\%)} ' + chr(92) * 2)
    body = _tabular('{rrrrr}', header, rows)
    save_tex('body_cuml_ablation.tex', body)



def gen_allocator_ablation():
    """tab:allocator_ablation — the allocator defect, gathered in one place.

    The pre-correction finding used to live split between two tables and the prose, and the three
    pieces told THE SAME measurement from different angles, which invites reading them as three
    independent results. This table joins them on the latency side.

    Source: the paired comparison built by build_allocator_fix_comparison.py, which matches the
    earlier campaign (no pool: importing cuML hijacks the CuPy allocator and every temporary pays
    a driver allocation) against the later one (pool re-seated per task). Both arms go through the
    same screening, so the two medians are cleaned identically.

    TWO PERCENTAGE COLUMNS, and it is not redundancy:
      Delta  = (with pool - without pool) / without pool. It follows from the two printed times,
               so the reader can check it, but it mixes the correction with everything else that
               changed in the days between the two campaigns.
      Net    = the correction with the epoch discounted. The older stack does not carry the
               correction, so its own ratio between the two campaigns on the same host and image
               IS the epoch and nothing else; the two effects compose multiplicatively, so the
               correction is the ratio of the two ratios. On one host the epoch reached a factor
               of three, from a weather model on the CPU, and its power limit also dropped, which
               is why its raw Delta means nothing on its own.
    The summary at the end gives the median of both per card, with the cell count behind each.
    """
    print('Generating body_allocator_ablation.tex ...')
    df = pd.read_csv(ALLOCATOR_ABLATION_CSV)
    df = df[df['profile'] == 'py312_cuml_adaptive']
    machines = [('azken', 'H100'), ('lenovo_tttserver', 'A100'), ('hp3', 'L40S')]
    by = {m: df[df['machine'] == m].set_index('image') for m, _ in machines}

    header = (r'\textbf{MP} & \textbf{Sources} & '
              + ' & '.join(
                  rf'\multicolumn{{3}}{{c}}{{\textbf{{{lbl}}}}}' for _, lbl in machines)
              + r' \\' + '\n'
              + r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}' + '\n'
              + r' & & ' + ' & '.join(
                  [r'\textbf{\shortstack{No\\pool (s)}} & \textbf{\shortstack{With\\pool (s)}} '
                   r'& \textbf{\shortstack{$\Delta$\\(\%)}}'] * 3)
              + r' \\')

    rows = []
    for img in IMAGE_ORDER:
        cells = []
        for m, _ in machines:
            t = by[m]
            if img in t.index:
                r = t.loc[img]
                cells += [fmt_time(r['v4']), fmt_time(r['fix1']),
                          f"${r['delta_pct']:+.0f}$"]
            else:
                cells += [OOM, OOM, OOM]
        rows.append(rf'{_mp_str(IMAGE_MP[img]):5s} & {fmt_src(IMAGE_SRC[img]):>8s} & '
                    + ' & '.join(cells) + r' \\')

    # Summary: raw median and median with the epoch discounted, each with its n.
    rows.append(r'\midrule')
    for col, lbl in (('delta_pct', r'\textbf{Median $\Delta$}'),
                     ('net_pct', r'\textbf{Median net}')):
        cells = []
        for m, _ in machines:
            v = pd.to_numeric(by[m][col], errors='coerce').dropna()
            cells += ['', '', rf'${v.median():+.0f}$~{{\scriptsize($n$={len(v)})}}'
                      if len(v) else OOM]
        rows.append(lbl + r' & & ' + ' & '.join(cells) + r' \\')

    save_tex('body_allocator_ablation.tex',
             _tabular(r'{rr rrr rrr rrr}', header, rows))

def gen_cpu_baseline():
    """
    tab:cpu_baseline — sep / Photutils / GPUPhot (A100 py3.12) latency comparison.
    sep and Photutils times: data/cpu_baseline_a100.csv (primary, same-machine as GPU)
                             data/cpu_baseline.csv (fallback for images not in A100 CSV)
    GPUPhot times: data/benchmark_latency.csv (A100, py312_cuml_adaptive, median)

    Using the A100-host measurements for sep/Photutils ensures the comparison is on
    identical hardware — the TTT server CPUs are 1.4–3.5× slower than the A100 host.
    """
    print('Generating body_cpu_baseline.tex ...')

    # Primary: CPU timings measured on the A100 host (same machine as GPUPhot)
    cpu_a100 = pd.read_csv(CPU_BASELINE_A100_CSV) if os.path.exists(CPU_BASELINE_A100_CSV) else pd.DataFrame()
    if not cpu_a100.empty:
        cpu_a100['sep_median_s']       = pd.to_numeric(cpu_a100['sep_median_s'],       errors='coerce')
        cpu_a100['photutils_median_s'] = pd.to_numeric(cpu_a100['photutils_median_s'], errors='coerce')

    # Fallback: TTT-server CPU timings (different hardware — kept for images missing from A100 CSV)
    cpu_ttt = pd.read_csv(CPU_BASELINE_CSV)
    cpu_ttt['sep_median_s']       = pd.to_numeric(cpu_ttt['sep_median_s'],       errors='coerce')
    cpu_ttt['photutils_median_s'] = pd.to_numeric(cpu_ttt['photutils_median_s'], errors='coerce')

    # Load GPUPhot A100 py3.12 adaptive times
    # By epoch split, not by the default file: if the A100 is post-fix,
    # this column has to be too, or it would compare pre-fix GPUPhot
    # against Sept/Photutils.
    df = load_benchmark_by_epoch('py312_cuml_adaptive', ['A100 (80 GB)'])
    a100 = df[df['gpu_label'] == 'A100 (80 GB)']
    gpuphot_med = a100.groupby('image_label')['execution_time'].median()

    col_spec = r'{rrrrr}'
    header = (
        r'\textbf{MP} & \textbf{Sources} & '
        r'\textbf{sep (s)} & \textbf{Photutils (s)} & '
        r'\textbf{\gpuphot\ (s)} \\'
    )
    rows = []
    for keyword, img_label in CPU_BASELINE_FILE_MAP:
        mp  = IMAGE_MP[img_label]
        src = IMAGE_SRC[img_label]
        # Try A100 CSV first (same-machine comparison); fall back to TTT CSV
        sep_t       = float('nan')
        photutils_t = float('nan')
        if not cpu_a100.empty:
            match_a100 = cpu_a100[cpu_a100['filename'].str.contains(keyword, na=False)]
            if not match_a100.empty:
                sep_t       = match_a100.iloc[0]['sep_median_s']
                photutils_t = match_a100.iloc[0]['photutils_median_s']
        if _is_missing(sep_t):  # fallback to TTT if not in A100 CSV
            match_ttt = cpu_ttt[cpu_ttt['filename'].str.contains(keyword, na=False)]
            if not match_ttt.empty:
                sep_t       = match_ttt.iloc[0]['sep_median_s']
                photutils_t = match_ttt.iloc[0]['photutils_median_s']

        # GPUPhot: direct lookup by image_label in the benchmark CSV
        gpu_t = gpuphot_med.get(img_label, float('nan'))

        rows.append(
            rf'{_mp_str(mp):5s} & {fmt_src(src):>8s} & '
            rf'{fmt_time(sep_t, 2)} & {fmt_time(photutils_t, 1)} & '
            rf'{fmt_time(gpu_t, 1)} \\'
        )

    body = _tabular(col_spec, header, rows)
    save_tex('body_cpu_baseline.tex', body)


def gen_nvtx_detection():
    """
    tab:nvtx_detection — detection stage of GPUPhot against the CPU tools.

    Source: data/detection_stage_vs_cpu_tools.csv (lenovo A100, 2026-09-07, post allocator
    fix).  All three tools run in the same process, on the SAME background-subtracted array
    captured from the pipeline, at min_snr=5, with no parameter tuned to equalise counts.

    Two things the previous version got wrong and this one does not.  Its "Sources" column
    published sep's count next to GPUPhot's time, which invited the reader to assume the
    comparison was at equal load; both counts are given here.  And its Photutils column did
    not state which operation it timed: this one reports the complete path
    (detect + deblend + SourceCatalog), the one comparable to sep.

    The cameras split into two populations by pca_method, and they are not the same
    comparison: with the Eigen-PSF path off, detection performs a single convolution; with
    it on, six.  The column carries the flag so the split is visible in the table itself.
    """
    print('Generating body_nvtx_detection.tex ...')
    df = pd.read_csv(DETECTION_CSV, comment='#')
    for c in ('gpuphot_cold_s', 'gpuphot_warm_s', 'sep_s', 'phot_full_s',
              'speedup_sep_cold', 'MP'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_values(['MP', 'gp_n_crop'])

    header = (r'\textbf{MP} & \textbf{PCA} & \textbf{\shortstack{N\\\gpuphot}} & '
              r'\textbf{\shortstack{N\\sep}} & \textbf{\shortstack{\gpuphot\\cold (s)}} & '
              r'\textbf{\shortstack{\gpuphot\\warm (s)}} & \textbf{sep (s)} & '
              r'\textbf{\shortstack{Photutils\\complete (s)}} & \textbf{Speedup} \\')
    rows = []
    for _, r in df.iterrows():
        rows.append(
            rf'{_mp_str(r["MP"])} & {r["pca_method"].lower()} & '
            rf'{fmt_src(int(r["gp_n_crop"]))} & {fmt_src(int(r["sep_n"]))} & '
            rf'{r["gpuphot_cold_s"]:.3f} & {r["gpuphot_warm_s"]:.3f} & '
            rf'{r["sep_s"]:.3f} & {r["phot_full_s"]:.3f} & '
            rf'{r["speedup_sep_cold"]:.1f}$\times$ \\'
        )
    body = _tabular('{rlrrrrrrr}', header, rows)
    save_tex('body_nvtx_detection.tex', body)


def gen_stage_breakdown():
    """
    tab:stage_breakdown — where the time goes, per stage, on the A100.

    Source: data/stage_breakdown.csv (lenovo A100, 2026-09-07, py3.12 + adaptive cuML,
    post allocator fix).  One stack in all three columns, one machine, one container, the
    same number of repetitions in every stage.

    Two things the May version could not say.  Its extractor summed the inclusive duration
    of every NVTX range grouped by name, so a stage nested inside another was counted twice:
    aperture photometry sits inside optimal photometry, and on the dense field those two
    alone summed to more than the whole pipeline.  Times here are self time, and the nesting
    is printed rather than hidden.  And its total was the sum of the instrumented stages
    published as if it were the pipeline's wall clock; here the two are separate rows and
    the difference gets its own line, so the reader can see how much is not instrumented.

    A dash is not a missing measurement: it is a stage this camera's configuration does not
    execute.  The iKon936 has pca_method off, so it has no Eigen-PSF, no coefficient map and
    no star projection; it has the cosmic-ray filter on, which is why an FFT convolution
    appears there outside any of the stages that nest one.

    The absolutes are not comparable with the latency tables: these run under nsys, which
    inflates the wall clock by the factor reported per image, and the latency tables are
    pre-fix.  This table is a proportional breakdown.
    """
    print('Generating body_stage_breakdown.tex ...')
    df = pd.read_csv(STAGE_BREAKDOWN_CSV, comment='#')
    df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')
    df['parent'] = df['parent'].fillna('')

    meta = df[df['stage'].str.startswith('__')]
    bdf = df[~df['stage'].str.startswith('__')]
    images = list(dict.fromkeys(df['image']))

    def m(img, key):
        v = meta[(meta['image'] == img) & (meta['stage'] == key)]['time_s']
        return float(v.iloc[0]) if len(v) else float('nan')

    def cell(img, stage, parent=''):
        v = bdf[(bdf['image'] == img) & (bdf['stage'] == stage)
                & (bdf['parent'] == parent)]['time_s']
        return float(v.iloc[0]) if len(v) else float('nan')

    def tex(name):
        return name.replace('&', r'\&')

    top = list(dict.fromkeys(bdf[bdf['parent'] == '']['stage']))
    # Only the two nestings the table means to expose.  The CSV records a stage under
    # every parent it appears in, and a stage can sit under several; printing all of them
    # would suggest a tree that is not one.
    SHOWN_NESTING = {('Optimal photometry', 'Aperture photometry'),
                     ('PSF coefficient map', 'Tile statistics')}
    kids = {}
    for _, r in bdf[bdf['parent'] != ''].iterrows():
        if (r['parent'], r['stage']) in SHOWN_NESTING:
            kids.setdefault(r['parent'], []).append(r['stage'])
    for k in kids:
        kids[k] = list(dict.fromkeys(kids[k]))
    stype = dict(zip(bdf['stage'], bdf['type']))
    BS = chr(92) * 2

    hdr = r'\textbf{Stage} & \textbf{Type}'
    COL_LABEL = {'iKon936_Lum': r'4.2\,MP',
                 'QHY411-1_Lum_full': r'151.2\,MP sparse',
                 'QHY411-3_Lum_131k': r'151.2\,MP dense'}
    for i in images:
        lab = COL_LABEL.get(i, _mp_str(df[df['image'] == i]['MP'].iloc[0]) + r'\,MP')
        hdr += r' & \textbf{\shortstack{' + lab.replace(' ', chr(92)*2) + r'}} & \textbf{\%}'
    header = hdr + ' ' + BS

    def fmt(img, stage, parent=''):
        v, tot = cell(img, stage, parent), m(img, '__wall_nsys_median__')
        return ' & --- & ---' if pd.isna(v) else f' & {v:.3f} & {v / tot * 100:.1f}'

    rows, last = [], None
    for st in top:
        t = stype.get(st, '')
        if last is not None and t != last:
            rows.append(r'\midrule')
        last = t
        rows.append(tex(st) + ' & ' + t + ''.join(fmt(i, st) for i in images) + ' ' + BS)
        for kid in kids.get(st, []):
            lbl = r'\quad of which ' + tex(kid)
            rows.append(lbl + ' & ' + t
                        + ''.join(fmt(i, kid, st) for i in images) + ' ' + BS)

    rows.append(r'\midrule')
    for label in ('Not instrumented', 'Wall clock (under nsys)'):
        cells = ''
        for img in images:
            tot = m(img, '__wall_nsys_median__')
            acc = sum(v for v in (cell(img, s) for s in top) if not pd.isna(v))
            v = tot - acc if label.startswith('Not') else tot
            cells += f' & {v:.3f} & {v / tot * 100:.1f}'
        rows.append(r'\textbf{' + label + '} & ' + cells + ' ' + BS)

    body = _tabular('{ll' + 'rr' * len(images) + '}', header, rows)
    save_tex('body_stage_breakdown.tex', body)


# ── Tabular builder ────────────────────────────────────────────────────────────
def _tabular(col_spec, header, rows):
    """Assemble a complete LaTeX tabular environment string."""
    lines = [
        f'\\begin{{tabular}}{col_spec}',
        '\\toprule',
        header,
        '\\midrule',
    ] + rows + [
        '\\bottomrule',
        '\\end{tabular}',
    ]
    return '\n'.join(lines) + '\n'


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Output directory : {OUT_DIR}')
    print(f'Benchmark CSV    : {latency_csv()}')
    print(f'Campaña          : {"POST-FIX" if USE_POSTFIX_CAMPAIGN else "agosto 2026 (pre-fix)"}')
    print()

    gen_latency_py312()
    print()
    gen_latency_py38()
    print()
    gen_speedup_local_vizier()
    print()
    gen_iqr_collapse()
    print()
    gen_vram_merged()   # replaces gen_vram_py312 + gen_vram_py38 + gen_vram_savings
    print()
    gen_concurrency()
    print()
    gen_cuml_ablation()
    print()
    gen_allocator_ablation()
    print()
    gen_vram_ablation()
    print()
    gen_concurrency_ablation()
    print()
    gen_cpu_baseline()
    print()
    gen_nvtx_detection()
    print()
    gen_stage_breakdown()
    print()
    print('Done — all table bodies generated.')
    print()
    print('Next steps:')
    print('  1. Verify tables_generated/*.tex look correct.')
    print('  2. performance.tex already uses \\input{tables_generated/body_*.tex}.')
    print('  3. Recompile: cd GPUPHOT_manuscript && pdflatex main.tex')
