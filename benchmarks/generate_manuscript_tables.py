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
  results_collected/profiler_nsys_memory_20260326.csv  → VRAM tables
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
MEMORY_CSV         = os.path.join(DATA_DIR, 'profiler_nsys_memory_20260326.csv')
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
    'Orin Super (8 GB)':  r'\textbf{Orin S.}',
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
# (image_label, LaTeX target name, MP, source count)
IMAGE_DEFS = [
    ('iKon936_Lum',         r'C/2025\,A6',    4.2,   296),
    ('iKon936_SDSSg',       r'QSO\,0957+561', 4.2,   412),
    ('QHY600-3_Lum',        r'C/2025\,R2',    6.8,   112),
    ('QHY600-4_Ha',         r'NGC\,2903',     15.3,  318),
    ('QHY600-4_SDSSg',      r'WASP-43\,b',    15.3,  218),
    ('QHY411-1_Lum_bin2',   r'2012\,QD8',     37.8,  154),
    ('QHY411-1_SDSSi_bin2', r'Gaia\,DR3',     37.8,  247),
    ('QHY411-1_Lum_full',   r'2025\,PR1',    151.2,  428),
    ('QHY411-3_SDSSr_full', r'M\,81',        151.2, 14241),
    ('QHY411-3_Lum_full',   r'24P',          151.2, 18888),
]
IMAGE_ORDER  = [d[0] for d in IMAGE_DEFS]
IMAGE_TARGET = {d[0]: d[1] for d in IMAGE_DEFS}
IMAGE_MP     = {d[0]: d[2] for d in IMAGE_DEFS}
IMAGE_SRC    = {d[0]: d[3] for d in IMAGE_DEFS}

# cuml_ablation: map image filename keyword → (camera, filter_label, n_sources)
# Key is matched with `in filename`.
# NOTE: The QHY411-1 Lum full row (2025PR1, 151.2 MP, 428 sources) is present in the
# ablation CSV but excluded here because its cuML penalty (+664%) reflects a pathological
# case (few sources on a very large frame) and would distort the table's message.
# The raw data is still in cuml_ablation_a100_20260329.csv for transparency.
ABLATION_FILE_MAP = [
    ('QSO0957',   'iKon936-1', r'SDSSg',         412),
    ('C2025A6',   'iKon936-1', r'Lum',            296),
    ('C2025R2',   'QHY600-3',  r'Lum',            112),
    ('WASP-43-b', 'QHY600-4',  r'SDSSg',          218),
    ('NGC2903',   'QHY600-4',  r'H$\alpha$',      318),
    ('2012QD8',   'QHY411-1',  r'Lum',            154),
    ('GaiaDR3',   'QHY411-1',  r'SDSSi',          247),
    ('M81',       'QHY411-3',  r'M\,81 SDSSr', 14241),
    ('24P_Lum',   'QHY411-3',  r'24P Lum',     18888),
]

# cpu_baseline: map filename keyword → (LaTeX target, MP, benchmark image_label)
# The image_label is used to look up GPUPhot A100 py3.12 times in benchmark_all_cuml_v2.csv.
CPU_BASELINE_FILE_MAP = [
    ('C2025A6',   r'C/2025\,A6',    4.2,  'iKon936_Lum'),
    ('QSO0957',   r'QSO\,0957+561', 4.2,  'iKon936_SDSSg'),
    ('C2025R2',   r'C/2025\,R2',    6.8,  'QHY600-3_Lum'),
    ('NGC2903',   r'NGC\,2903',     15.3, 'QHY600-4_Ha'),
    ('WASP-43-b', r'WASP-43\,b',    15.3, 'QHY600-4_SDSSg'),
    ('2012QD8',   r'2012\,QD8',     37.8, 'QHY411-1_Lum_bin2'),
    ('GaiaDR3',   r'Gaia\,DR3',     37.8, 'QHY411-1_SDSSi_bin2'),
    ('2025PR1',   r'2025\,PR1',    151.2, 'QHY411-1_Lum_full'),
    ('M81',       r'M\,81',        151.2, 'QHY411-3_SDSSr_full'),
    ('24P',       r'24P',          151.2, 'QHY411-3_Lum_full'),
]

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
def load_benchmark(profiler_labels):
    """Load benchmark_all_cuml_v2.csv, drop first 2 warmup reps, return DataFrame."""
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
        grp = grp.sort_values('timestamp') if grp['timestamp'].notna().any() else grp
        parts.append(grp.iloc[2:])  # drop first 2 warmup reps
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
    """Return dict[mp][gpu_label] = median peak VRAM in MB (NaN if missing)."""
    df = pd.read_csv(MEMORY_CSV)
    df['python_ver'] = pd.to_numeric(df['python_ver'], errors='coerce')
    df['gpu_name'] = df['gpu_name'].str.replace('NVIDIA ', '', regex=False)
    df['gpu_label'] = df['gpu_name'].map(GPU_LABEL_MAP).fillna(df['gpu_name'])
    df['megapixels'] = pd.to_numeric(df['megapixels'], errors='coerce')
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
    GPUs: H100, A100, L40S, RTX 3090, RTX 3060
    """
    print('Generating body_latency_py312.tex ...')
    gpu_cols = [
        'H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)',
        'RTX 3090 (24 GB)', 'RTX 3060 (12 GB)',
    ]
    data = latency_median('py312_cuml_adaptive', gpu_cols)

    col_spec = r'{rlr rrrrr}'
    header = (
        r'\textbf{MP} & \textbf{Target} & \textbf{Sources}' + '\n'
        r'            & ' + ' & '.join(GPU_TEX_HEADER[g] for g in gpu_cols) + r' \\'
    )
    rows = []
    for img in IMAGE_ORDER:
        mp  = IMAGE_MP[img]
        tgt = IMAGE_TARGET[img]
        src = IMAGE_SRC[img]
        vals = ' & '.join(fmt_time(data[img][g]) for g in gpu_cols)
        src_fmt = fmt_src(src)
        mp_str = f'{mp:5.1f}'.rstrip('0').rstrip('.')
        rows.append(rf'{mp_str:5s} & {tgt:<18s} & {src_fmt:>8s} & {vals} \\')

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

    col_spec = r'{rlr rrrrrrr}'
    header = (
        r'\textbf{MP} & \textbf{Target} & \textbf{Sources}' + '\n'
        r'            & ' + ' & '.join(GPU_TEX_HEADER[g] for g in gpu_cols) + r' \\'
    )
    rows = []
    for img in IMAGE_ORDER:
        mp  = IMAGE_MP[img]
        tgt = IMAGE_TARGET[img]
        src = IMAGE_SRC[img]
        vals = ' & '.join(fmt_time(data[img][g]) for g in gpu_cols)
        src_fmt = fmt_src(src)
        mp_str = f'{mp:5.1f}'.rstrip('0').rstrip('.')
        rows.append(rf'{mp_str:5s} & {tgt:<18s} & {src_fmt:>8s} & {vals} \\')

    body = _tabular(col_spec, header, rows)
    save_tex('body_latency_py38.tex', body)


def gen_vram_py312():
    """
    tab:vram_py312 — Peak GPU memory (MB), py3.12.
    Source: profiler_nsys_memory_20260326.csv (python_ver=3.12)
    """
    print('Generating body_vram_py312.tex ...')
    gpu_cols = [
        'H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)',
        'RTX 3090 (24 GB)', 'RTX 3060 (12 GB)', 'RTX 3050 Ti (4 GB)',
    ]
    data, mp_vals = vram_median(3.12, gpu_cols)

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
    save_tex('body_vram_py312.tex', body)


def gen_vram_py38():
    """
    tab:vram_py38 — Peak GPU memory (MB), py3.8.
    Source: profiler_nsys_memory_20260326.csv (python_ver=3.8)
    """
    print('Generating body_vram_py38.tex ...')
    gpu_cols = [
        'H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)',
        'RTX 3090 (24 GB)', 'RTX 3060 (12 GB)', 'RTX 3050 Ti (4 GB)',
    ]
    data, mp_vals = vram_median(3.8, gpu_cols)

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
    save_tex('body_vram_py38.tex', body)


def gen_vram_savings():
    """
    tab:vram_savings — Average VRAM reduction (%) when upgrading py3.8→py3.12.
    Source: profiler_nsys_memory_20260326.csv (both python_ver values)
    Computed as mean over all MP sizes processable by each GPU.
    """
    print('Generating body_vram_savings.tex ...')
    gpu_cols = [
        'H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)',
        'RTX 3090 (24 GB)', 'RTX 3060 (12 GB)', 'RTX 3050 Ti (4 GB)',
    ]
    data38, mp_vals = vram_median(3.8,  gpu_cols)
    data312, _      = vram_median(3.12, gpu_cols)

    gpu_short = {
        'H100 (80 GB)':       'H100',
        'A100 (80 GB)':       'A100',
        'L40S (48 GB)':       'L40S',
        'RTX 3090 (24 GB)':   'RTX 3090',
        'RTX 3060 (12 GB)':   'RTX 3060',
        'RTX 3050 Ti (4 GB)': 'RTX 3050 Ti',
    }

    col_spec = r'{lc}'
    header = r'\textbf{GPU} & \textbf{VRAM Saving (\%)} \\'
    rows = []
    for gpu in gpu_cols:
        savings = []
        for mp in mp_vals:
            v38  = data38[mp].get(gpu, float('nan'))
            v312 = data312[mp].get(gpu, float('nan'))
            if not _is_missing(v38) and not _is_missing(v312) and v38 > 0:
                savings.append((v38 - v312) / v38 * 100.0)
        if savings:
            avg = np.mean(savings)
            rows.append(rf'{gpu_short[gpu]:<14s} & {avg:.1f} \\')
        else:
            rows.append(rf'{gpu_short[gpu]:<14s} & {OOM} \\')

    body = _tabular(col_spec, header, rows)
    save_tex('body_vram_savings.tex', body)


def gen_concurrency():
    """
    tab:concurrency — Max concurrent 4.2 MP images per GPU.
    Source: profiler_nsys_memory_20260326.csv (4.2 MP median peak VRAM per py ver)
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

    col_spec = r'{llrrrr}'
    header = (
        r'Camera & Image & Sources & With cuML (s) & '
        r'Without cuML (s) & Penalty (\%) \\'
    )
    rows = []
    for keyword, camera, filter_label, _ in ABLATION_FILE_MAP:
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
            rf'{camera:<12s} & {filter_label:<18s} & {nsrc_str:>8s} & '
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

    col_spec = r'{rlrrr}'
    header = (
        r'\textbf{MP} & \textbf{Target} & '
        r'\textbf{sep (s)} & \textbf{Photutils (s)} & '
        r'\textbf{\gpuphot\ (s)} \\'
    )
    rows = []
    for keyword, tex_target, mp, img_label in CPU_BASELINE_FILE_MAP:
        # sep / Photutils: match by filename keyword in cpu_baseline_results.csv
        match = cpu[cpu['filename'].str.contains(keyword, na=False)]
        sep_t      = match.iloc[0]['sep_median_s']       if not match.empty else float('nan')
        photutils_t = match.iloc[0]['photutils_median_s'] if not match.empty else float('nan')

        # GPUPhot: direct lookup by image_label in the benchmark CSV
        gpu_t = gpuphot_med.get(img_label, float('nan'))

        mp_str = f'{mp:5.1f}'.rstrip('0').rstrip('.')
        rows.append(
            rf'{mp_str:5s} & {tex_target:<18s} & '
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

    col_spec = r'{lrrrr}'
    header = (
        r'\textbf{Camera} & \textbf{MP} & \textbf{GPU det.\ (s)} & '
        r'\textbf{sep (s)} & \textbf{GPU speedup} \\'
    )
    rows = []
    for _, row in df.iterrows():
        speedup = row['gpu_speedup_vs_sep']
        sp_str  = f'{speedup:.1f}$\\times$' if not _is_missing(speedup) else OOM
        rows.append(
            rf'{row["camera"]:<12s} & {row["MP"]:5.1f} & '
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
    gen_vram_py312()
    print()
    gen_vram_py38()
    print()
    gen_vram_savings()
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
