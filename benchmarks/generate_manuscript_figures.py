#!/usr/bin/env python3
"""
Generate all data figures for the GPUPhot manuscript.

Figures:
  2  - Latency vs Image Size (log-log), py3.12+adaptive recommended config
  3  - Memory py3.8 vs py3.12 (grouped bar, A100)          [data unchanged]
  4  - Heatmap GPU × image, py3.12+adaptive (10 images)
  5  - cuML Crossover synthetic benchmark                   [data unchanged]
  6  - Concurrency (grouped bar)                            [data unchanged]
  7  - py3.8 vs py3.12+adaptive: % overhead per image (real pipeline)
  8  - cuML always vs adaptive: % improvement per image

Data sources:
  benchmark_all_cuml_v2.csv              → figs 2, 4, 7, 8
  profiler_nsys_memory_summary_*.csv     → fig 3
  cuml_crossover_synthetic_*.csv         → fig 5

Outputs saved to: GPUPhotFinal/figures_profiler/  (PDF + PNG)
Then copied to:   GPUPhotFinal/GPUPHOT_manuscript/figures/
"""

import os
import math
import shutil
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
})

# Colorblind-safe palette (Wong 2011, Nature Methods)
CB_COLORS = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # bluish green
    '#CC79A7',  # reddish purple
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#F0E442',  # yellow
    '#000000',  # black
]
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BASE)
DATA_DIR = os.path.join(BASE, 'results_collected')
OUT_DIR = os.path.join(PROJECT, 'figures_profiler')
MANUSCRIPT_FIGURES_DIR = os.path.join(PROJECT, 'GPUPHOT_manuscript', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

BENCHMARK_CSV = os.path.join(DATA_DIR, 'benchmark_all_cuml_v2.csv')
MEMORY_CSV    = os.path.join(DATA_DIR, 'profiler_nsys_memory_summary_20260411.csv')
CUML_CSV      = os.path.join(DATA_DIR, 'cuml_crossover_synthetic_all_gpus_20260412.csv')

# ── GPU label normalisation ───────────────────────────────────────────────────
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
GPU_ORDER = [
    'H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)',
    'RTX 3090 (24 GB)', 'RTX 3060 (12 GB)', 'RTX 3050 Ti (4 GB)',
    'Orin Super (8 GB)', 'Orin NX (8 GB)',
]
# Short names for axes/legends
GPU_SHORT = {
    'H100 (80 GB)':       'H100',
    'A100 (80 GB)':       'A100',
    'L40S (48 GB)':       'L40S',
    'RTX 3090 (24 GB)':   'RTX 3090',
    'RTX 3060 (12 GB)':   'RTX 3060',
    'RTX 3050 Ti (4 GB)': 'RTX 3050 Ti',
    'Orin Super (8 GB)':  'Orin S.*',   # * ARM, cuML unavailable
    'Orin NX (8 GB)':     'Orin NX',
}
# GPU order for heatmap (py3.12): Orin NX excluded — no py3.12 data (runs py38_baseline only)
HEATMAP_GPU_ORDER = [
    'H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)',
    'RTX 3090 (24 GB)', 'RTX 3060 (12 GB)', 'RTX 3050 Ti (4 GB)',
    'Orin Super (8 GB)',
]

# Image label order for plots (ascending MP, then source count within MP)
IMAGE_ORDER = [
    # Ordered by MP ascending, then source count ascending within each MP group
    'iKon936_Lum',         # 4.2 MP,     296 src
    'iKon936_SDSSg',       # 4.2 MP,     412 src
    'QHY600-3_Lum',        # 6.8 MP,     112 src
    'QHY600-3_SDSSi_2k',   # 6.8 MP,   2 060 src
    'QHY600-4_SDSSg',      # 15.3 MP,    218 src
    'QHY600-4_Ha',         # 15.3 MP,    318 src
    'QHY600-4_Lum_2k',     # 15.3 MP,  1 993 src
    'QHY600-4_SDSSg_4k',   # 15.3 MP,  4 361 src
    'QHY600-4_SDSSi_10k',  # 15.3 MP, 10 532 src
    'QHY411-1_Lum_bin2',   # 37.8 MP,    154 src
    'QHY411-1_SDSSi_bin2', # 37.8 MP,    247 src
    'QHY411-1_SDSSg_2k',   # 37.8 MP,  1 998 src
    'QHY411-1_SDSSr_7k',   # 37.8 MP,  6 932 src
    'QHY411-1_Lum_full',   # 151.2 MP,   428 src
    'QHY411-3_SDSSg_10k',  # 151.2 MP, 10 005 src
    'QHY411-3_SDSSr_full', # 151.2 MP, 14 241 src
    'QHY411-3_Lum_full',   # 151.2 MP, 18 888 src
    'QHY411-3_SDSSr_19k',  # 151.2 MP, 19 565 src
    'QHY411-3_Lum_131k',   # 151.2 MP, 131 397 src
]
IMAGE_MP = {
    'iKon936_SDSSg': 4.2,      'iKon936_Lum': 4.2,
    'QHY600-3_Lum': 6.8,       'QHY600-3_SDSSi_2k': 6.8,
    'QHY600-4_Ha': 15.3,       'QHY600-4_SDSSg': 15.3,
    'QHY600-4_Lum_2k': 15.3,   'QHY600-4_SDSSg_4k': 15.3,  'QHY600-4_SDSSi_10k': 15.3,
    'QHY411-1_Lum_bin2': 37.8, 'QHY411-1_SDSSi_bin2': 37.8,
    'QHY411-1_SDSSg_2k': 37.8, 'QHY411-1_SDSSr_7k': 37.8,
    'QHY411-1_Lum_full': 151.2,
    'QHY411-3_SDSSr_full': 151.2, 'QHY411-3_Lum_full': 151.2,
    'QHY411-3_SDSSg_10k': 151.2,  'QHY411-3_SDSSr_19k': 151.2, 'QHY411-3_Lum_131k': 151.2,
}
IMAGE_SRC = {
    'iKon936_SDSSg': 412,       'iKon936_Lum': 296,
    'QHY600-3_Lum': 112,        'QHY600-3_SDSSi_2k': 2060,
    'QHY600-4_Ha': 318,         'QHY600-4_SDSSg': 218,
    'QHY600-4_Lum_2k': 1993,    'QHY600-4_SDSSg_4k': 4361,   'QHY600-4_SDSSi_10k': 10532,
    'QHY411-1_Lum_bin2': 154,   'QHY411-1_SDSSi_bin2': 247,
    'QHY411-1_SDSSg_2k': 1998,  'QHY411-1_SDSSr_7k': 6932,
    'QHY411-1_Lum_full': 428,
    'QHY411-3_SDSSr_full': 14241,  'QHY411-3_Lum_full': 18888,
    'QHY411-3_SDSSg_10k': 10005,   'QHY411-3_SDSSr_19k': 19565, 'QHY411-3_Lum_131k': 131397,
}

def _img_display(img, sep='\n'):
    """Short display label: 'X.X MP{sep}N,NNN src'."""
    mp  = IMAGE_MP[img]
    src = IMAGE_SRC[img]
    mp_s = f'{mp:.1f}'.rstrip('0').rstrip('.')
    src_s = f'{src:,}'.replace(',', '\u202f')  # narrow no-break space as thousands sep
    return f'{mp_s} MP{sep}{src_s} src'


def save(fig, name):
    """Save figure as PDF and PNG in OUT_DIR."""
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight')
        print(f'  Saved {path}')
    plt.close(fig)


def load_benchmark(profiler_labels=None):
    """Load benchmark_all_cuml_v2.csv, normalise GPU names, drop warmup rows."""
    df = pd.read_csv(BENCHMARK_CSV)
    df['execution_time'] = pd.to_numeric(df['execution_time'], errors='coerce')
    df['mp'] = pd.to_numeric(df['mp'], errors='coerce')
    df['n_sources_detected'] = pd.to_numeric(df['n_sources_detected'], errors='coerce')
    df = df[df['execution_time'].notna()]

    # Normalise GPU names (strip NVIDIA prefix, then map to label)
    df['gpu_name'] = df['gpu_name'].str.replace('NVIDIA ', '', regex=False)
    df['gpu_label'] = df['gpu_name'].map(GPU_LABEL_MAP).fillna(df['gpu_name'])

    if profiler_labels:
        df = df[df['profiler_label'].isin(profiler_labels)]

    # Drop warmup: first 2 reps per (machine, profiler_label, image_label)
    df = df[df['image_label'].notna() & (df['image_label'] != '')]
    parts = []
    for _, grp in df.groupby(['machine', 'profiler_label', 'image_label'], sort=False):
        if grp['timestamp'].notna().any():
            grp = grp.sort_values('timestamp')
        parts.append(grp.iloc[2:])
    return pd.concat(parts, ignore_index=True) if parts else df


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Latency vs Image Size (log-log), py3.12+adaptive
# ═════════════════════════════════════════════════════════════════════════════
def figure2():
    print('Figure 2: Latency vs Image Size ...')
    df = load_benchmark(['py312_cuml_adaptive'])

    # Only x86 GPUs: Jetson ARM times (~150 s for 4 MP) would collapse the log-log scale.
    # Jetson performance is discussed separately in the text.
    x86_gpus = [g for g in GPU_ORDER if 'Orin' not in g]
    df = df[df['gpu_label'].isin(x86_gpus)]

    # Median per (gpu_label, mp) — aggregates across machines and images at same size
    grp = df.groupby(['gpu_label', 'mp'])['execution_time'].median().reset_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for idx, gpu_label in enumerate(x86_gpus):
        subset = grp[grp['gpu_label'] == gpu_label].sort_values('mp')
        if subset.empty:
            continue
        color  = CB_COLORS[idx % len(CB_COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        # Need ≥2 points to draw a line; otherwise scatter only
        if len(subset) >= 2:
            ax.plot(subset['mp'], subset['execution_time'],
                    color=color, marker=marker, markersize=5,
                    label=gpu_label, zorder=3)
        else:
            ax.scatter(subset['mp'], subset['execution_time'],
                       color=color, marker=marker, s=40,
                       label=gpu_label, zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Image size (megapixels)')
    ax.set_ylabel('Median end-to-end latency (s)')
    ax.set_title('GPUPhot latency vs image size — py3.12 + adaptive cuML')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9, ncol=2)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.text(0.99, 0.02,
            'Jetson Orin (ARM) not shown — ~150 s for 4.2 MP (no RAPIDS)',
            transform=ax.transAxes, fontsize=7, color='#666666',
            ha='right', va='bottom', style='italic')
    fig.tight_layout()
    save(fig, 'fig2_latency_vs_size')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Memory py3.8 vs py3.12 (A100)  [data unchanged]
# ═════════════════════════════════════════════════════════════════════════════
def figure3():
    print('Figure 3: Memory comparison py3.8 vs py3.12 ...')
    df = pd.read_csv(MEMORY_CSV)
    a100 = df[df['gpu_name'] == 'A100-SXM4-80GB'].copy()
    mp_values = sorted(a100['megapixels'].unique())
    # Memory figure uses one value per MP size (aggregated): label = MP only
    cam_map = {mp: f'{mp:.1f}'.rstrip("0").rstrip(".") + ' MP'
               for mp in [4.2, 6.8, 15.3, 37.8, 151.2]}

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(mp_values))
    width = 0.35
    py38_vals, py312_vals = [], []
    for mp in mp_values:
        r38  = a100[(a100['megapixels'] == mp) & (a100['python_ver'].astype(str) == '3.8')]
        r312 = a100[(a100['megapixels'] == mp) & (a100['python_ver'].astype(str) == '3.12')]
        py38_vals.append(r38['median_peak_MB'].values[0]  if len(r38)  > 0 else 0)
        py312_vals.append(r312['median_peak_MB'].values[0] if len(r312) > 0 else 0)

    py38_vals  = np.array(py38_vals)
    py312_vals = np.array(py312_vals)

    ax.bar(x - width/2, py38_vals,  width, label='py3.8 (CuPy 12)',
           color='#2c3e50', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, py312_vals, width, label='py3.12 (CuPy 14)',
           color='#85c1e9', edgecolor='white', linewidth=0.5)

    for i in range(len(mp_values)):
        if py38_vals[i] > 0 and py312_vals[i] > 0:
            saving = (py38_vals[i] - py312_vals[i]) / py38_vals[i] * 100
            ax.annotate(f'{saving:.0f}%',
                        xy=(x[i] + width/2, py312_vals[i]),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color='#0072B2')

    ax.set_xlabel('Image')
    ax.set_ylabel('Peak VRAM (MB)')
    ax.set_title('Peak GPU memory: py3.8 vs py3.12 (A100-SXM4-80GB)')
    ax.set_xticks(x)
    ax.set_xticklabels([cam_map.get(mp, f'{mp} MP') for mp in mp_values], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', ls=':', alpha=0.4)
    fig.tight_layout()
    save(fig, 'fig3_memory_comparison')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Heatmap GPU × image (10 images, py3.12+adaptive)
# ═════════════════════════════════════════════════════════════════════════════
def figure4():
    print('Figure 4: Heatmap ...')
    df = load_benchmark(['py312_cuml_adaptive'])
    med = df.groupby(['gpu_label', 'image_label'])['execution_time'].median()

    # Use HEATMAP_GPU_ORDER: excludes Orin NX (no py3.12 data; all-grey columns
    # would misleadingly imply OOM rather than "not tested under this config").
    # Orin Super is included with asterisk (py3.12 ARM, cuML unavailable).
    n_imgs = len(IMAGE_ORDER)
    n_gpus = len(HEATMAP_GPU_ORDER)
    matrix = np.full((n_imgs, n_gpus), np.nan)

    for i, img in enumerate(IMAGE_ORDER):
        for j, gpu in enumerate(HEATMAP_GPU_ORDER):
            try:
                matrix[i, j] = med.loc[(gpu, img)]
            except KeyError:
                pass  # NaN = OOM or not tested

    # Row labels: MP + source count only
    row_labels = [_img_display(img) for img in IMAGE_ORDER]
    col_labels = [GPU_SHORT.get(g, g) for g in HEATMAP_GPU_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = LinearSegmentedColormap.from_list(
        'latency', ['#FFFFB2', '#FED976', '#FEB24C', '#FD8D3C',
                    '#FC4E2A', '#E31A1C', '#B10026'])
    cmap.set_bad(color='#CCCCCC')

    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    for i in range(n_imgs):
        for j in range(n_gpus):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, 'OOM', ha='center', va='center',
                        fontsize=7.5, fontweight='bold', color='#666666')
            else:
                normed = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
                tc = 'white' if normed > 0.65 else 'black'
                ax.text(j, i, f'{val:.1f}s', ha='center', va='center',
                        fontsize=7.5, fontweight='bold', color=tc)

    ax.set_xticks(range(n_gpus))
    ax.set_xticklabels(col_labels, fontsize=8.5, rotation=30, ha='right')
    ax.set_yticks(range(n_imgs))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    ax.set_title('End-to-end latency (s) — py3.12 + adaptive cuML', fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Latency (s)', fontsize=9)
    fig.tight_layout()
    save(fig, 'fig4_heatmap')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — cuML Crossover synthetic  [data unchanged]
# ═════════════════════════════════════════════════════════════════════════════
def figure5():
    print('Figure 5: cuML Crossover ...')
    df = pd.read_csv(CUML_CSV)
    # Strip "NVIDIA " prefix for robustness (older CSVs don't have it)
    df['gpu_name'] = df['gpu_name'].str.replace(r'^NVIDIA\s+', '', regex=True)
    gpu_order  = ['H100 PCIe', 'A100-SXM4-80GB', 'L40S',
                  'GeForce RTX 3090', 'GeForce RTX 3060', 'GeForce RTX 3050 Ti Laptop GPU']
    gpu_labels = {
        'H100 PCIe':                      'H100 PCIe',
        'A100-SXM4-80GB':                 'A100-SXM4',
        'L40S':                           'L40S',
        'GeForce RTX 3090':               'RTX 3090',
        'GeForce RTX 3060':               'RTX 3060',
        'GeForce RTX 3050 Ti Laptop GPU': 'RTX 3050 Ti',
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhspan(0, 1.0, alpha=0.10, color='#E31A1C', zorder=0)
    ax.axhline(y=1.0, color='#555555', linestyle='--', linewidth=1.0, zorder=2)
    ax.text(120, 1.05, 'break-even', fontsize=8, color='#555555', va='bottom')
    ax.text(80000, 0.15, 'cKDTree faster', fontsize=8, color='#B10026',
            ha='center', fontstyle='italic', alpha=0.7)
    ax.text(80000, 4.5, 'cuML faster', fontsize=8, color='#009E73',
            ha='center', fontstyle='italic', alpha=0.7)

    for idx, gpu in enumerate(gpu_order):
        subset = df[df['gpu_name'] == gpu].sort_values('N')
        if subset.empty:
            continue
        color = CB_COLORS[idx % len(CB_COLORS)]
        speedup_med = subset['speedup'].values

        # IQR band: speedup_low = cpu_q25/gpu_q75 (conservative),
        #           speedup_high = cpu_q75/gpu_q25 (optimistic).
        # Requires columns added in benchmark_cuml_crossover.py v2.
        has_iqr = all(c in subset.columns for c in ('cpu_q25', 'cpu_q75', 'gpu_q25', 'gpu_q75'))
        if has_iqr:
            sp_lo = subset['cpu_q25'] / subset['gpu_q75'].replace(0, float('nan'))
            sp_hi = subset['cpu_q75'] / subset['gpu_q25'].replace(0, float('nan'))
            ax.fill_between(subset['N'], sp_lo, sp_hi,
                            alpha=0.18, color=color, zorder=1)

        ax.plot(subset['N'], speedup_med,
                color=color,
                marker=MARKERS[idx % len(MARKERS)], markersize=4, linewidth=1.4,
                label=gpu_labels[gpu], zorder=3)

    ax.set_xscale('log')
    ax.set_xlabel('Number of sources (N)')
    ax.set_ylabel('Speedup (cKDTree time / cuML time)')
    ax.set_title('cuML vs cKDTree crossmatch: synthetic benchmark')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{int(x):,}' if x >= 1000 else f'{int(x)}'))
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    fig.tight_layout()
    save(fig, 'fig5_cuml_crossover')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Concurrency  [data unchanged]
# ═════════════════════════════════════════════════════════════════════════════
def figure6():
    print('Figure 6: Concurrency ...')
    vram_total = {
        'H100': 80*1024, 'A100': 80*1024, 'L40S': 48*1024,
        'RTX 3090': 24*1024, 'RTX 3060': 12*1024, 'RTX 3050 Ti': 4*1024,
    }
    peak_38  = 2018   # MB, py3.8  (iKon936, A100 nsys data)
    peak_312 = 1718   # MB, py3.12 (iKon936, A100 nsys data)
    gpu_order = ['H100', 'A100', 'L40S', 'RTX 3090', 'RTX 3060', 'RTX 3050 Ti']

    conc_38  = np.array([math.floor(vram_total[g]*0.95/peak_38)  for g in gpu_order])
    conc_312 = np.array([math.floor(vram_total[g]*0.95/peak_312) for g in gpu_order])

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(gpu_order))
    width = 0.35

    bars38  = ax.bar(x - width/2, conc_38,  width, label='py3.8 (CuPy 12)',
                     color='#2c3e50', edgecolor='white', linewidth=0.5)
    bars312 = ax.bar(x + width/2, conc_312, width, label='py3.12 (CuPy 14)',
                     color='#85c1e9', edgecolor='white', linewidth=0.5)

    for i, g in enumerate(gpu_order):
        if conc_38[i] > 0:
            gain = (conc_312[i] - conc_38[i]) / conc_38[i] * 100
            color = '#D55E00' if g == 'RTX 3050 Ti' else '#0072B2'
            ax.annotate(f'+{gain:.0f}%',
                        xy=(x[i] + width/2, conc_312[i]),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color=color)

    for bars in (bars38, bars312):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h/2,
                        f'{int(h)}', ha='center', va='center',
                        fontsize=7.5, color='white', fontweight='bold')

    ax.set_xlabel('GPU')
    ax.set_ylabel('Concurrent 4.2 MP images')
    ax.set_title('Theoretical concurrent images per GPU (4.2 MP, 95% VRAM)')
    ax.set_xticks(x)
    ax.set_xticklabels(gpu_order, fontsize=9, rotation=15, ha='right')
    ax.legend(fontsize=9)
    ax.grid(axis='y', ls=':', alpha=0.4)
    fig.tight_layout()
    save(fig, 'fig6_concurrency')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — py3.8 vs py3.12+adaptive: % overhead per image (real pipeline)
# ═════════════════════════════════════════════════════════════════════════════
def figure7():
    print('Figure 7: py3.8 vs py3.12+adaptive overhead ...')
    df = load_benchmark(['py38_baseline', 'py312_cuml_adaptive'])

    datacenter_gpus = ['H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)']
    df = df[df['gpu_label'].isin(datacenter_gpus)]

    med = df.groupby(['gpu_label', 'profiler_label', 'image_label'])['execution_time'].median().unstack('profiler_label')
    med = med.rename(columns={'py38_baseline': 'py38', 'py312_cuml_adaptive': 'adaptive'})
    med = med.dropna(subset=['py38', 'adaptive'])
    med['pct'] = (med['adaptive'] - med['py38']) / med['py38'] * 100
    med = med.reset_index()
    med['img_order'] = med['image_label'].map({img: i for i, img in enumerate(IMAGE_ORDER)})
    med = med.sort_values(['img_order', 'gpu_label'])

    pct_pivot = med.pivot(index='image_label', columns='gpu_label', values='pct')
    pct_pivot = pct_pivot.reindex([img for img in IMAGE_ORDER if img in pct_pivot.index])

    # QHY411-1_Lum_full is an outlier (250-300%) unrelated to cuML — pipeline overhead
    # for 151.2 MP sparse field. Cap y-axis at 100% and annotate separately.
    CAP = 100.0
    outlier_img = 'QHY411-1_Lum_full'

    x = np.arange(len(pct_pivot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 4.5))

    for i, gpu in enumerate(datacenter_gpus):
        if gpu not in pct_pivot.columns:
            continue
        raw_vals = pct_pivot[gpu].values
        capped    = np.clip(raw_vals, -CAP, CAP)
        colors    = ['#D55E00' if v > 0 else '#009E73' for v in raw_vals]
        offset    = (i - 1) * width
        ax.bar(x + offset, capped, width,
               label=GPU_SHORT[gpu], color=colors,
               edgecolor='white', linewidth=0.4, alpha=0.85)

        # Annotate bars that were clipped
        for j, (raw, cap) in enumerate(zip(raw_vals, capped)):
            if raw > CAP:
                ax.text(x[j] + offset, CAP + 1.5, f'{raw:.0f}%',
                        ha='center', va='bottom', fontsize=6.5,
                        color='#D55E00', fontweight='bold', rotation=90)

    ax.axhline(0, color='black', linewidth=0.8, zorder=5)
    ax.set_ylim(-15, CAP + 20)
    ax.set_xticks(x)
    xlabels = [_img_display(img) for img in pct_pivot.index]
    ax.set_xticklabels(xlabels, fontsize=7, rotation=0, ha='center')
    ax.set_ylabel('Overhead vs py3.8 (%)\n(positive = py3.12 slower)')
    ax.set_title('End-to-end latency: py3.12 + adaptive cuML vs py3.8 baseline\n'
                 'Datacenter GPUs — orange = py3.12 slower, green = py3.12 faster  '
                 '(values above axis capped; real value annotated)')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', ls=':', alpha=0.4)

    # MP group separators and labels (placed after ylim is set)
    mp_groups = [
        (-0.5, 1.5, '4.2 MP'), (1.5, 2.5, '6.8 MP'),
        (2.5, 4.5, '15.3 MP'), (4.5, 6.5, '37.8 MP'), (6.5, 9.5, '151.2 MP'),
    ]
    ymax = ax.get_ylim()[1]
    for xstart, xend, label in mp_groups:
        if xend < len(pct_pivot) - 0.5:
            ax.axvline(xend, color='#AAAAAA', linewidth=0.6, ls='--', zorder=1)
        mid = (xstart + xend) / 2
        ax.text(mid, ymax * 0.97, label,
                fontsize=7, color='#555555', ha='center', va='top')

    fig.tight_layout()
    save(fig, 'fig7_py38_vs_adaptive')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — cuML always vs adaptive: % improvement per image (datacenter)
# ═════════════════════════════════════════════════════════════════════════════
def figure8():
    print('Figure 8: cuML always vs adaptive improvement ...')
    df = load_benchmark(['py312_cuml_always', 'py312_cuml_adaptive'])

    datacenter_gpus = ['H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)']
    df = df[df['gpu_label'].isin(datacenter_gpus)]

    med = df.groupby(['gpu_label', 'profiler_label', 'image_label'])['execution_time'].median().unstack('profiler_label')
    med = med.rename(columns={'py312_cuml_always': 'always', 'py312_cuml_adaptive': 'adaptive'})
    med = med.dropna(subset=['always', 'adaptive'])
    # Positive = adaptive is faster (improvement)
    med['improvement'] = (med['always'] - med['adaptive']) / med['always'] * 100
    med = med.reset_index()
    med['img_order'] = med['image_label'].map({img: i for i, img in enumerate(IMAGE_ORDER)})
    med = med.sort_values(['img_order', 'gpu_label'])

    pct_pivot = med.pivot(index='image_label', columns='gpu_label', values='improvement')
    pct_pivot = pct_pivot.reindex([img for img in IMAGE_ORDER if img in pct_pivot.index])

    n_imgs = len(pct_pivot)
    x = np.arange(n_imgs)
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 4.5))

    for i, gpu in enumerate(datacenter_gpus):
        if gpu not in pct_pivot.columns:
            continue
        vals = pct_pivot[gpu].values
        colors = ['#009E73' if v > 0 else '#D55E00' for v in vals]
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width,
               label=GPU_SHORT[gpu], color=colors,
               edgecolor='white', linewidth=0.4, alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.8, zorder=5)
    ax.set_xticks(x)
    xlabels = [_img_display(img) for img in pct_pivot.index]
    ax.set_xticklabels(xlabels, fontsize=7, rotation=0, ha='center')
    ax.set_ylabel('Improvement of adaptive vs always (%)\n(positive = adaptive faster)')
    ax.set_title('Adaptive cuML vs always-on cuML\n'
                 'Datacenter GPUs — green = adaptive faster, orange = adaptive slower')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', ls=':', alpha=0.4)
    fig.tight_layout()
    save(fig, 'fig8_always_vs_adaptive')


# ═════════════════════════════════════════════════════════════════════════════
# COPY TO MANUSCRIPT
# ═════════════════════════════════════════════════════════════════════════════
def copy_to_manuscript():
    """Copy all generated fig*.{pdf,png} to GPUPHOT_manuscript/figures/."""
    os.makedirs(MANUSCRIPT_FIGURES_DIR, exist_ok=True)
    copied = 0
    for fname in sorted(os.listdir(OUT_DIR)):
        if fname.startswith('fig') and fname.endswith(('.pdf', '.png')):
            src = os.path.join(OUT_DIR, fname)
            dst = os.path.join(MANUSCRIPT_FIGURES_DIR, fname)
            shutil.copy2(src, dst)
            copied += 1
    print(f'  Copied {copied} files → {MANUSCRIPT_FIGURES_DIR}')


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Output directory : {OUT_DIR}')
    print(f'Manuscript target: {MANUSCRIPT_FIGURES_DIR}')
    print()
    figure2()
    print()
    figure3()
    print()
    figure4()
    print()
    figure5()
    print()
    figure6()
    print()
    figure7()
    print()
    figure8()
    print()
    copy_to_manuscript()
    print()
    print('Done — all figures generated and copied to manuscript.')
