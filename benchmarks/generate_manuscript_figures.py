#!/usr/bin/env python3
"""
Generate all 5 data figures for the GPUPhot manuscript.

Figures:
  2 - Latency vs Image Size (log-log)
  3 - Memory py3.8 vs py3.12 (grouped bar, A100)
  4 - Heatmap (GPU x image size, median latency)
  5 - cuML Crossover (speedup vs N sources)
  6 - Concurrency (grouped bar, concurrent images per GPU)

Outputs saved to: figures_profiler/ as PDF + PNG at 300 dpi.
"""

import os
import sys
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

# ── Global style ─────────────────────────────────────────────────────────────
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

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BASE)
DATA_DIR = os.path.join(BASE, 'results_collected')
OUT_DIR = os.path.join(PROJECT, 'figures_profiler')
os.makedirs(OUT_DIR, exist_ok=True)

UNIFIED_CSV = os.path.join(DATA_DIR, 'profiler_unified_timing_memory_20260327.csv')
MEMORY_CSV = os.path.join(DATA_DIR, 'profiler_nsys_memory_summary_20260326.csv')
CUML_CSV = os.path.join(DATA_DIR, 'cuml_crossover_synthetic_all_gpus_20260328.csv')
JETSON_CSV = os.path.join(DATA_DIR, 'jetson_orin_8gb_timing_20260329.csv')


def save(fig, name):
    """Save figure as PDF and PNG."""
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight')
        print(f'  Saved {path}')
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Latency vs Image Size (log-log)
# ═════════════════════════════════════════════════════════════════════════════
def figure2():
    print('Figure 2: Latency vs Image Size ...')
    df = pd.read_csv(UNIFIED_CSV)

    # Filter py3.12 only, and rows with timing data
    # For Orin Super, use "Orin (nvgpu)" rows which have timing from Elastic
    # For Orin NX 8GB, use separate Jetson CSV
    df312 = df[(df['pyver'].astype(str).isin(['3.12', '3.10'])) & df['median_time_s'].notna()].copy()

    # Build Orin NX data from jetson CSV
    jdf = pd.read_csv(JETSON_CSV)
    jdf_clean = jdf[jdf['type'] == 'clean']
    orin_nx = jdf_clean.groupby('megapixels')['time_s'].median().reset_index()
    orin_nx.rename(columns={'time_s': 'median_time_s'}, inplace=True)
    orin_nx['gpu_short'] = 'Orin NX 8GB'

    # Normalise GPU short names for the plot
    gpu_map = {
        'H100 PCIe': 'H100 (80 GB)',
        'A100-SXM4-80GB': 'A100 (80 GB)',
        'L40S': 'L40S (48 GB)',
        'RTX 3090': 'RTX 3090 (24 GB)',
        'RTX 3060': 'RTX 3060 (12 GB)',
        'RTX 3050 Ti Laptop GPU': 'RTX 3050 Ti (4 GB)',
        'Orin (nvgpu)': 'Orin Super (8 GB)',
        'Orin NX 8GB': 'Orin NX (8 GB)',
    }

    # Order for legend (fast to slow conceptually)
    gpu_order = [
        'H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)',
        'RTX 3090 (24 GB)', 'RTX 3060 (12 GB)', 'RTX 3050 Ti (4 GB)',
        'Orin Super (8 GB)', 'Orin NX (8 GB)',
    ]

    df312['gpu_label'] = df312['gpu_short'].map(gpu_map)
    orin_nx['gpu_label'] = orin_nx['gpu_short'].map(gpu_map)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for idx, gpu_label in enumerate(gpu_order):
        color = CB_COLORS[idx % len(CB_COLORS)]
        marker = MARKERS[idx % len(MARKERS)]

        if gpu_label == 'Orin NX (8 GB)':
            subset = orin_nx.sort_values('megapixels')
        else:
            subset = df312[df312['gpu_label'] == gpu_label].sort_values('megapixels')

        if subset.empty:
            continue

        ax.plot(
            subset['megapixels'], subset['median_time_s'],
            color=color, marker=marker, markersize=5,
            label=gpu_label, zorder=3,
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Image size (megapixels)')
    ax.set_ylabel('Median latency (s)')
    ax.set_title('End-to-end latency vs image size (py3.12)')

    # Custom tick labels
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))

    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9, ncol=2)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    fig.tight_layout()
    save(fig, 'fig2_latency_vs_size')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Memory py3.8 vs py3.12 (grouped bar, A100)
# ═════════════════════════════════════════════════════════════════════════════
def figure3():
    print('Figure 3: Memory comparison py3.8 vs py3.12 ...')
    df = pd.read_csv(MEMORY_CSV)

    # Filter A100
    a100 = df[df['gpu_name'] == 'A100-SXM4-80GB'].copy()

    # Map camera to megapixels (use the megapixels column)
    # Get unique megapixel groups
    mp_values = sorted(a100['megapixels'].unique())

    # Camera labels for x-axis
    cam_map = {4.2: 'iKon936\n4.2 MP', 6.8: 'QHY600\n6.8 MP',
               15.3: 'QHY600\n15.3 MP', 37.8: 'QHY411\n37.8 MP',
               151.2: 'QHY411\n151.2 MP'}

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(mp_values))
    width = 0.35

    py38_vals = []
    py312_vals = []
    for mp in mp_values:
        row38 = a100[(a100['megapixels'] == mp) & (a100['python_ver'].astype(str) == '3.8')]
        row312 = a100[(a100['megapixels'] == mp) & (a100['python_ver'].astype(str) == '3.12')]
        py38_vals.append(row38['median_peak_MB'].values[0] if len(row38) > 0 else 0)
        py312_vals.append(row312['median_peak_MB'].values[0] if len(row312) > 0 else 0)

    py38_vals = np.array(py38_vals)
    py312_vals = np.array(py312_vals)

    bars38 = ax.bar(x - width/2, py38_vals, width, label='py3.8 (CuPy 12)',
                    color='#2c3e50', edgecolor='white', linewidth=0.5)
    bars312 = ax.bar(x + width/2, py312_vals, width, label='py3.12 (CuPy 14)',
                     color='#85c1e9', edgecolor='white', linewidth=0.5)

    # Annotate % saving above py3.12 bars
    for i in range(len(mp_values)):
        if py38_vals[i] > 0 and py312_vals[i] > 0:
            saving = (py38_vals[i] - py312_vals[i]) / py38_vals[i] * 100
            ax.annotate(
                f'{saving:.0f}%',
                xy=(x[i] + width/2, py312_vals[i]),
                xytext=(0, 5), textcoords='offset points',
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color='#0072B2',
            )

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
# FIGURE 4: Heatmap (GPU x image size, median latency)
# ═════════════════════════════════════════════════════════════════════════════
def figure4():
    print('Figure 4: Heatmap ...')
    df = pd.read_csv(UNIFIED_CSV)

    # Use py3.12 (and py3.10 for Orin nvgpu)
    df_sel = df[df['pyver'].astype(str).isin(['3.12', '3.10'])].copy()

    # Also add Orin NX 8GB from jetson CSV
    jdf = pd.read_csv(JETSON_CSV)
    jdf_clean = jdf[jdf['type'] == 'clean']
    orin_nx_medians = jdf_clean.groupby('megapixels')['time_s'].median().to_dict()

    # GPU order for columns
    gpu_order_map = {
        'H100 PCIe': 'H100',
        'A100-SXM4-80GB': 'A100',
        'L40S': 'L40S',
        'RTX 3090': 'RTX 3090',
        'RTX 3060': 'RTX 3060',
        'RTX 3050 Ti Laptop GPU': 'RTX 3050 Ti',
        'Orin (nvgpu)': 'Orin Super',
        'Orin NX 8GB': 'Orin NX',
    }
    gpu_col_order = ['H100', 'A100', 'L40S', 'RTX 3090', 'RTX 3060',
                     'RTX 3050 Ti', 'Orin Super', 'Orin NX']

    mp_order = [4.2, 6.8, 15.3, 37.8, 151.2]
    mp_labels = ['4.2 MP', '6.8 MP', '15.3 MP', '37.8 MP', '151.2 MP']

    df_sel['gpu_label'] = df_sel['gpu_short'].map(gpu_order_map)

    # Build matrix
    matrix = np.full((len(mp_order), len(gpu_col_order)), np.nan)

    for i, mp in enumerate(mp_order):
        for j, gpu in enumerate(gpu_col_order):
            if gpu == 'Orin NX':
                val = orin_nx_medians.get(mp, np.nan)
            else:
                rows = df_sel[(df_sel['gpu_label'] == gpu) &
                              (np.isclose(df_sel['megapixels'], mp, atol=0.5))]
                if len(rows) > 0 and rows['median_time_s'].notna().any():
                    val = rows['median_time_s'].dropna().values[0]
                else:
                    val = np.nan
            matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(8, 3.5))

    # Custom colormap: yellow (fast) -> red (slow)
    cmap = LinearSegmentedColormap.from_list('latency', ['#FFFFB2', '#FED976', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#B10026'])
    cmap.set_bad(color='#CCCCCC')  # gray for NaN/OOM

    # Use log scale for color
    matrix_log = np.where(np.isnan(matrix), np.nan, matrix)
    vmin = np.nanmin(matrix_log)
    vmax = np.nanmax(matrix_log)

    im = ax.imshow(matrix_log, cmap=cmap, aspect='auto',
                   vmin=vmin, vmax=vmax)

    # Annotate cells
    for i in range(len(mp_order)):
        for j in range(len(gpu_col_order)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, 'OOM', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='#555555')
            else:
                # Choose text color based on brightness
                normed = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
                text_color = 'white' if normed > 0.65 else 'black'
                ax.text(j, i, f'{val:.1f}s', ha='center', va='center',
                        fontsize=8, fontweight='bold', color=text_color)

    ax.set_xticks(range(len(gpu_col_order)))
    ax.set_xticklabels(gpu_col_order, fontsize=8.5, rotation=30, ha='right')
    ax.set_yticks(range(len(mp_order)))
    ax.set_yticklabels(mp_labels, fontsize=9)
    ax.set_title('Median end-to-end latency by GPU and image size (py3.12)', fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Latency (s)', fontsize=9)

    fig.tight_layout()
    save(fig, 'fig4_heatmap')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5: cuML Crossover
# ═════════════════════════════════════════════════════════════════════════════
def figure5():
    print('Figure 5: cuML Crossover ...')
    df = pd.read_csv(CUML_CSV)

    # Exclude Jetsons (they don't have cuML)
    # GPU names in data: H100 PCIe, A100-SXM4-80GB, L40S, RTX 3090, RTX 3060, RTX 3050 Ti
    gpu_order = ['H100 PCIe', 'A100-SXM4-80GB', 'L40S', 'RTX 3090', 'RTX 3060', 'RTX 3050 Ti']
    gpu_labels = {
        'H100 PCIe': 'H100 PCIe',
        'A100-SXM4-80GB': 'A100-SXM4',
        'L40S': 'L40S',
        'RTX 3090': 'RTX 3090',
        'RTX 3060': 'RTX 3060',
        'RTX 3050 Ti': 'RTX 3050 Ti',
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Shade "cKDTree faster" region (below 1.0)
    ax.axhspan(0, 1.0, alpha=0.10, color='#E31A1C', zorder=0)
    ax.axhline(y=1.0, color='#555555', linestyle='--', linewidth=1.0, zorder=2)
    ax.text(120, 1.05, 'break-even', fontsize=8, color='#555555', va='bottom')

    # Label regions
    ax.text(80000, 0.15, 'cKDTree faster', fontsize=8, color='#B10026',
            ha='center', fontstyle='italic', alpha=0.7)
    ax.text(80000, 4.5, 'cuML faster', fontsize=8, color='#009E73',
            ha='center', fontstyle='italic', alpha=0.7)

    for idx, gpu in enumerate(gpu_order):
        color = CB_COLORS[idx % len(CB_COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        subset = df[df['gpu_name'] == gpu].sort_values('N')
        if subset.empty:
            continue
        ax.plot(subset['N'], subset['speedup'],
                color=color, marker=marker, markersize=5,
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
# FIGURE 6: Concurrency (grouped bar)
# ═════════════════════════════════════════════════════════════════════════════
def figure6():
    print('Figure 6: Concurrency ...')

    # VRAM totals (MB)
    vram_total = {
        'H100': 80 * 1024,
        'A100': 80 * 1024,
        'L40S': 48 * 1024,
        'RTX 3090': 24 * 1024,
        'RTX 3060': 12 * 1024,
        'RTX 3050 Ti': 4 * 1024,
    }

    # Peak VRAM for 4.2 MP iKon936 (from manuscript context / data)
    peak_38 = 2018   # MB, py3.8
    peak_312 = 1718  # MB, py3.12

    gpu_order = ['H100', 'A100', 'L40S', 'RTX 3090', 'RTX 3060', 'RTX 3050 Ti']

    conc_38 = []
    conc_312 = []
    for gpu in gpu_order:
        total = vram_total[gpu] * 0.95  # 95% usable
        conc_38.append(math.floor(total / peak_38))
        conc_312.append(math.floor(total / peak_312))

    conc_38 = np.array(conc_38)
    conc_312 = np.array(conc_312)

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(gpu_order))
    width = 0.35

    bars38 = ax.bar(x - width/2, conc_38, width, label='py3.8 (CuPy 12)',
                    color='#2c3e50', edgecolor='white', linewidth=0.5)
    bars312 = ax.bar(x + width/2, conc_312, width, label='py3.12 (CuPy 14)',
                     color='#85c1e9', edgecolor='white', linewidth=0.5)

    # Annotate gain % above py3.12 bar
    for i in range(len(gpu_order)):
        if conc_38[i] > 0:
            gain = (conc_312[i] - conc_38[i]) / conc_38[i] * 100
            text = f'+{gain:.0f}%'
        else:
            text = 'N/A'

        # Highlight RTX 3050 Ti specially
        fontw = 'bold'
        color = '#0072B2'
        if gpu_order[i] == 'RTX 3050 Ti':
            color = '#D55E00'
            fontw = 'bold'

        ax.annotate(
            text,
            xy=(x[i] + width/2, conc_312[i]),
            xytext=(0, 5), textcoords='offset points',
            ha='center', va='bottom', fontsize=8, fontweight=fontw,
            color=color,
        )

    # Add value labels on bars
    for bar_group in [bars38, bars312]:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height/2,
                        f'{int(height)}', ha='center', va='center',
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
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Output directory: {OUT_DIR}')
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
    print('All figures generated.')
