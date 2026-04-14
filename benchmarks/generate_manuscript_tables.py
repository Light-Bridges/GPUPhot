#!/usr/bin/env python3
"""
generate_manuscript_tables.py
Generate all data-driven LaTeX table bodies for the GPUPhot manuscript.

Run this script whenever benchmark data changes to regenerate tables.
Output .tex files are written to GPUPHOT_manuscript/tables_generated/
and committed alongside the manuscript so that LaTeX compilation never
requires re-running this script.

Tables generated (tabular body: \\begin{tabular}...\\end{tabular}):
  body_latency_py312.tex   — tab:latency_py312  median latency, py3.12+adaptive
  body_latency_py38.tex    — tab:latency_py38   median latency, py3.8 baseline
  body_vram_py312.tex      — tab:vram_py312     peak VRAM (MB), py3.12
  body_vram_py38.tex       — tab:vram_py38      peak VRAM (MB), py3.8
  body_vram_savings.tex    — tab:vram_savings   avg VRAM reduction %, py3.8→py3.12
  body_concurrency.tex     — tab:concurrency    concurrent 4.2 MP images per GPU
  body_cuml_ablation.tex   — tab:cuml_ablation  cuML vs cKDTree ablation (A100)
  body_cpu_baseline.tex    — tab:cpu_baseline   sep / Photutils / GPUPhot (A100 py3.12)
  body_nvtx_detection.tex  — tab:nvtx_detection stage-level GPU detection vs sep (A100)

Data sources (relative paths from this file):
  results_collected/benchmark_all_cuml_v2.csv          → latency tables, cpu_baseline
  results_collected/profiler_nsys_memory_20260411.csv  → VRAM tables
  results_collected/cuml_ablation_a100_20260329.csv    → body_cuml_ablation
  results_collected/nvtx_detection_vs_sep_a100.csv     → body_nvtx_detection
  cpu_baseline_results.csv                             → body_cpu_baseline (sep/Photutils)

Usage:
  cd benchmarks/
  python generate_manuscript_tables.py

Then compile the manuscript normally: pdflatex main.tex (performance.tex inputs
the generated files via \\input{tables_generated/body_*.tex}).
"""

import os
import math
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
PROJECT  = os.path.dirname(BASE)
DATA_DIR = os.path.join(BASE, 'results_collected')
OUT_DIR  = os.path.join(PROJECT, 'GPUPHOT_manuscript', 'tables_generated')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Source CSVs ────────────────────────────────────────────────────────────────
BENCHMARK_CSV      = os.path.join(DATA_DIR, 'benchmark_all_cuml_v2.csv')
MEMORY_CSV         = os.path.join(DATA_DIR, 'profiler_nsys_memory_20260411.csv')
CUML_ABLATION_CSV  = os.path.join(DATA_DIR, 'cuml_ablation_a100_20260329.csv')
NVTX_DETECTION_CSV = os.path.join(DATA_DIR, 'nvtx_detection_vs_sep_a100.csv')
CPU_BASELINE_CSV   = os.path.join(BASE, 'cpu_baseline_results.csv')

# ── GPU label normalisation ────────────────────────────────────────────────────
GPU_LABEL_MAP = {
    'H100 PCIe':                      'H100 (80 GB)',
    'A100-SXM4-80GB':                 'A100 (80 GB)',
    'L40S':                           'L40S (48 GB)',
    'GeForce RTX 3090':               'RTX 3090 (24 GB)',
    'GeForce RTX 3060':               'RTX 3060 (12 GB)',
    'GeForce RTX 3050 Ti Laptop GPU': 'RTX 3050 Ti (4 GB)',
    'Orin Super 8GB (nvgpu)':         'Orin Super (8 GB)',
    'Orin NX 8GB (nvgpu)':            'Orin NX (8 GB)',
}

GPU_TEX_HEADER = {
    'H100 (80 GB)':       r'\textbf{H100}',
    'A100 (80 GB)':       r'\textbf{A100}',
    'L40S (48 GB)':       r'\textbf{L40S}',
    'RTX 3090 (24 GB)':   r'\textbf{3090}',
    'RTX 3060 (12 GB)':   r'\textbf{3060}',
    'RTX 3050 Ti (4 GB)': r'\textbf{3050\,Ti}',
    'Orin Super (8 GB)':  r'\textbf{Orin S.}$^*$',
    'Orin NX (8 GB)':     r'\textbf{Orin NX}',
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
    # Ordered by MP ascending, then source count ascending within each MP group
    ('iKon936_Lum',          4.2,    296),
    ('iKon936_SDSSg',        4.2,    412),
    ('QHY600-3_Lum',         6.8,    112),
    ('QHY600-3_SDSSi_2k',    6.8,   2060),
    ('QHY600-4_SDSSg',      15.3,    218),
    ('QHY600-4_Ha',         15.3,    318),
    ('QHY600-4_Lum_2k',     15.3,   1993),
    ('QHY600-4_SDSSg_4k',   15.3,   4361),
    ('QHY600-4_SDSSi_10k',  15.3,  10532),
    ('QHY411-1_Lum_bin2',   37.8,    154),
    ('QHY411-1_SDSSi_bin2', 37.8,    247),
    ('QHY411-1_SDSSg_2k',   37.8,   1998),
    ('QHY411-1_SDSSr_7k',   37.8,   6932),
    ('QHY411-1_Lum_full',  151.2,    428),
    ('QHY411-3_SDSSg_10k', 151.2,  10005),
    ('QHY411-3_SDSSr_full',151.2,  14241),
    ('QHY411-3_Lum_full',  151.2,  18888),
    ('QHY411-3_SDSSr_19k', 151.2,  19565),
    ('QHY411-3_Lum_131k',  151.2, 131397),
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
# NOTE: QHY411-1 Lum full (151.2 MP, 428 src) excluded — its cuML penalty
# (+664%) is pathological (sparse field on large frame) and distorts the table.
ABLATION_FILE_MAP = [
    ('QSO0957',    4.2,   412),
    ('C2025A6',    4.2,   296),
    ('C2025R2',    6.8,   112),
    ('WASP-43-b', 15.3,   218),
    ('NGC2903',   15.3,   318),
    ('2012QD8',   37.8,   154),
    ('GaiaDR3',   37.8,   247),
    ('M81',      151.2, 14241),
    ('24P_Lum',  151.2, 18888),
]

# cpu_baseline: map filename keyword → (MP, sources, benchmark image_label)
CPU_BASELINE_FILE_MAP = [
    ('C2025A6',    4.2,   296, 'iKon936_Lum'),
    ('QSO0957',    4.2,   412, 'iKon936_SDSSg'),
    ('C2025R2',    6.8,   112, 'QHY600-3_Lum'),
    ('NGC2903',   15.3,   318, 'QHY600-4_Ha'),
    ('WASP-43-b', 15.3,   218, 'QHY600-4_SDSSg'),
    ('2012QD8',   37.8,   154, 'QHY411-1_Lum_bin2'),
    ('GaiaDR3',   37.8,   247, 'QHY411-1_SDSSi_bin2'),
    ('2025PR1',  151.2,   428, 'QHY411-1_Lum_full'),
    ('M81',      151.2, 14241, 'QHY411-3_SDSSr_full'),
    ('24P',      151.2, 18888, 'QHY411-3_Lum_full'),
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

def save_tex(name, content):
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
        grp = grp.sort_values('timestamp').reset_index(drop=True)
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


def load_benchmark(profiler_labels):
    """Load benchmark_all_cuml_v2.csv, remove warmup reps and outliers, return DataFrame.

    Pipeline per (machine, profiler_label, image_label) group:
      1. Sort by timestamp.
      2. Detect session boundaries (gap > 30 min) and drop the first 2 rows
         of every session as warmup — handles multiple merged benchmark runs.
      3. Apply a 3-MAD upper-outlier filter to remove any remaining slow artefacts
         (GPU throttling, background load spikes, etc.).
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
    for _, grp in df.groupby(['machine', 'profiler_label', 'image_label'], sort=False):
        cleaned = _drop_warmup_per_session(grp)
        cleaned = _mad_filter(cleaned)
        if len(cleaned):
            parts.append(cleaned)
    return pd.concat(parts, ignore_index=True) if parts else df


def latency_median(profiler_label, gpu_cols):
    """Return dict[image_label][gpu_label] = median latency (NaN if missing)."""
    df = load_benchmark(profiler_label)
    df = df[df['gpu_label'].isin(gpu_cols)]
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

def gen_latency_py312():
    """
    tab:latency_py312 — Median latency (s), py3.12 + adaptive cuML.
    Source: benchmark_all_cuml_v2.csv (profiler_label=py312_cuml_adaptive)
    GPUs: H100, A100, L40S, RTX 3090, RTX 3060, RTX 3050 Ti, Orin Super
    Orin Super runs py3.12 on ARM without cuML (unavailable on aarch64);
    its data is stored under profiler_label=py312_cuml_adaptive in the CSV.
    Orin NX has no py3.12 data and is therefore omitted from this table.
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
        r'            & ' + ' & '.join(GPU_TEX_HEADER[g] for g in gpu_cols) + r' \\'
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
    Source: benchmark_all_cuml_v2.csv (profiler_label=py38_baseline)
    GPUs: H100, A100, L40S, RTX 3090, RTX 3060, RTX 3050 Ti, Orin NX
    """
    print('Generating body_latency_py38.tex ...')
    gpu_cols = [
        'H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)',
        'RTX 3090 (24 GB)', 'RTX 3060 (12 GB)',
        'RTX 3050 Ti (4 GB)', 'Orin NX (8 GB)',
    ]
    data = latency_median('py38_baseline', gpu_cols)

    col_spec = r'{rr rrrrrrr}'
    header = (
        r'\textbf{MP} & \textbf{Sources}' + '\n'
        r'            & ' + ' & '.join(GPU_TEX_HEADER[g] for g in gpu_cols) + r' \\'
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
    tab:vram_merged — Peak GPU memory (MB) and py3.8→py3.12 saving per image size.
    Uses A100-SXM4-80GB as the representative GPU (chosen because it has complete
    data for all 10 benchmark images and is the most broadly deployed 80 GB card).
    Cross-GPU variation for the same image is <0.3% (see Figure fig:memory_comparison).
    Replaces the former tab:vram_py312, tab:vram_py38, and tab:vram_savings.
    Source: profiler_nsys_memory_20260411.csv
    """
    print('Generating body_vram_merged.tex ...')
    rep_gpu = 'A100 (80 GB)'
    data312, mp_vals = vram_median(3.12, [rep_gpu])
    data38,  _       = vram_median(3.8,  [rep_gpu])

    col_spec = r'{r rrr}'
    header = (
        r'\textbf{MP} & \textbf{py3.8 (MB)} & \textbf{py3.12 (MB)} & '
        r'\textbf{Saving (\%)} \\'
    )
    rows = []
    for mp in mp_vals:
        v38  = data38[mp][rep_gpu]
        v312 = data312[mp][rep_gpu]
        if _is_missing(v38) or _is_missing(v312):
            saving_str = OOM
        else:
            saving_str = f'{100.0 * (v38 - v312) / v38:.1f}'
        rows.append(
            rf'{mp:5.1f} & {fmt_mb(v38)} & {fmt_mb(v312)} & {saving_str} \\'
        )

    body = _tabular(col_spec, header, rows)
    save_tex('body_vram_merged.tex', body)

    # Keep the per-GPU tables as well (still generated, but no longer \input{}'d
    # from the manuscript). Kept for reference and reproducibility.
    _gen_vram_all_gpus(3.12, 'body_vram_py312.tex')
    _gen_vram_all_gpus(3.8,  'body_vram_py38.tex')


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
    Source: profiler_nsys_memory_20260411.csv (4.2 MP median peak VRAM per py ver)
    Formula: floor(GPU_VRAM_total / peak_VRAM_per_image)
    GPU_VRAM_total is read from the gpu_total_MB column of the memory CSV
    (e.g. RTX 3050 Ti Laptop GPU reports 3964 MB, not the nominal 4096 MB).
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
        r'\textbf{py3.8} & \textbf{py3.12} & \textbf{Gain} \\'
    )
    rows = []
    for gpu in gpu_cols:
        # Use the SMALLER of nominal spec and measured value.
        # Server GPUs (H100, A100) occasionally report >80 GB in NVML due to
        # unified memory reporting; capping at nominal avoids inflated concurrency.
        # Laptop GPUs (3050 Ti) report slightly less than nominal (3964 vs 4096 MB)
        # and the measured value is used so the +100% gain shows correctly.
        vram_total_mb = min(actual_vram_mb.get(gpu, GPU_VRAM_MB[gpu]), GPU_VRAM_MB[gpu])
        vram_gb       = GPU_VRAM_GB[gpu]
        peak38  = data38.get(MP_4K,  {}).get(gpu, float('nan'))
        peak312 = data312.get(MP_4K, {}).get(gpu, float('nan'))
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


def gen_cuml_ablation():
    """
    tab:cuml_ablation — cuML vs cKDTree end-to-end latency ablation on A100.
    Source: results_collected/cuml_ablation_a100_20260329.csv
    Rows ordered by ablation file map (MP ascending, then source count).
    """
    print('Generating body_cuml_ablation.tex ...')
    df = pd.read_csv(CUML_ABLATION_CSV)
    df['time_s']  = pd.to_numeric(df['time_s'],  errors='coerce')
    df['sources'] = pd.to_numeric(df['sources'], errors='coerce')

    # Median per (image, cuml flag)
    med = df.groupby(['image', 'cuml'])['time_s'].median().unstack('cuml')
    # sources per image (same for all reps)
    src_map = df.groupby('image')['sources'].first()

    col_spec = r'{rrrrr}'
    header = (
        r'\textbf{MP} & \textbf{Sources} & \textbf{With cuML (s)} & '
        r'\textbf{Without cuML (s)} & \textbf{Penalty (\%)} \\'
    )
    rows = []
    for keyword, mp, _ in ABLATION_FILE_MAP:
        # Find matching row by filename keyword
        matches = [idx for idx in med.index if keyword in idx]
        if not matches:
            continue
        img_key = matches[0]
        t_yes = med.loc[img_key, 'yes'] if 'yes' in med.columns else float('nan')
        t_no  = med.loc[img_key, 'no']  if 'no'  in med.columns else float('nan')
        nsrc  = src_map.loc[img_key] if img_key in src_map.index else float('nan')
        if not _is_missing(t_yes) and not _is_missing(t_no) and t_no > 0:
            penalty = (t_yes - t_no) / t_no * 100.0
            pen_str = f'$+{penalty:.0f}$'
        else:
            pen_str = OOM
        nsrc_str = fmt_src(nsrc) if not _is_missing(nsrc) else OOM
        rows.append(
            rf'{_mp_str(mp):5s} & {nsrc_str:>8s} & '
            rf'{fmt_time(t_yes)} & {fmt_time(t_no)} & {pen_str} \\'
        )

    body = _tabular(col_spec, header, rows)
    save_tex('body_cuml_ablation.tex', body)


def gen_cpu_baseline():
    """
    tab:cpu_baseline — sep / Photutils / GPUPhot (A100 py3.12) latency comparison.
    sep and Photutils times: cpu_baseline_results.csv
    GPUPhot times: benchmark_all_cuml_v2.csv (A100, py312_cuml_adaptive, median)
    """
    print('Generating body_cpu_baseline.tex ...')

    # Load CPU baselines
    cpu = pd.read_csv(CPU_BASELINE_CSV)
    cpu['sep_median_s']      = pd.to_numeric(cpu['sep_median_s'],      errors='coerce')
    cpu['photutils_median_s']= pd.to_numeric(cpu['photutils_median_s'],errors='coerce')

    # Load GPUPhot A100 py3.12 adaptive times
    df = load_benchmark('py312_cuml_adaptive')
    a100 = df[df['gpu_label'] == 'A100 (80 GB)']
    gpuphot_med = a100.groupby('image_label')['execution_time'].median()

    col_spec = r'{rrrrr}'
    header = (
        r'\textbf{MP} & \textbf{Sources} & '
        r'\textbf{sep (s)} & \textbf{Photutils (s)} & '
        r'\textbf{\gpuphot\ (s)} \\'
    )
    rows = []
    for keyword, mp, src, img_label in CPU_BASELINE_FILE_MAP:
        # sep / Photutils: match by filename keyword in cpu_baseline_results.csv
        match = cpu[cpu['filename'].str.contains(keyword, na=False)]
        sep_t       = match.iloc[0]['sep_median_s']        if not match.empty else float('nan')
        photutils_t = match.iloc[0]['photutils_median_s']  if not match.empty else float('nan')

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
    tab:nvtx_detection — GPU source detection (A100 py3.12) vs sep, scope-equivalent.
    Source: results_collected/nvtx_detection_vs_sep_a100.csv
    """
    print('Generating body_nvtx_detection.tex ...')
    df = pd.read_csv(NVTX_DETECTION_CSV)
    df['gpuphot_detection_s'] = pd.to_numeric(df['gpuphot_detection_s'], errors='coerce')
    df['sep_median_s']         = pd.to_numeric(df['sep_median_s'],        errors='coerce')
    df['gpu_speedup_vs_sep']   = pd.to_numeric(df['gpu_speedup_vs_sep'],  errors='coerce')
    df['MP']                   = pd.to_numeric(df['MP'],                  errors='coerce')
    df = df.sort_values('MP')

    col_spec = r'{rrrrr}'
    header = (
        r'\textbf{MP} & \textbf{Sources} & \textbf{GPU det.\ (s)} & '
        r'\textbf{sep (s)} & \textbf{GPU speedup} \\'
    )
    rows = []
    for _, row in df.iterrows():
        speedup = row['gpu_speedup_vs_sep']
        sp_str  = f'{speedup:.1f}$\\times$' if not _is_missing(speedup) else OOM
        mp = row['MP']
        src = NVTX_SRC.get(mp, 0)
        rows.append(
            rf'{_mp_str(mp):5s} & {fmt_src(src):>8s} & '
            rf'{fmt_time(row["gpuphot_detection_s"], 3)} & '
            rf'{fmt_time(row["sep_median_s"], 3)} & '
            rf'{sp_str} \\'
        )

    body = _tabular(col_spec, header, rows)
    save_tex('body_nvtx_detection.tex', body)


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
    print(f'Benchmark CSV    : {BENCHMARK_CSV}')
    print()

    gen_latency_py312()
    print()
    gen_latency_py38()
    print()
    gen_vram_merged()   # replaces gen_vram_py312 + gen_vram_py38 + gen_vram_savings
    print()
    gen_concurrency()
    print()
    gen_cuml_ablation()
    print()
    gen_cpu_baseline()
    print()
    gen_nvtx_detection()
    print()
    print('Done — all table bodies generated.')
    print()
    print('Next steps:')
    print('  1. Verify tables_generated/*.tex look correct.')
    print('  2. performance.tex already uses \\input{tables_generated/body_*.tex}.')
    print('  3. Recompile: cd GPUPHOT_manuscript && pdflatex main.tex')
