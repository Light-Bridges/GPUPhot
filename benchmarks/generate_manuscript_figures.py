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

Data sources (all in benchmarks/data/):
  benchmark_latency.csv          → figs 2, 4, 7, 8
  profiler_memory_raw.csv        → figs 3, 6 (same file, filters and pooling as the VRAM tables)
  profiler_memory_summary.csv    → (no longer read: it carries one row per camera, and taking
                                    the first one disagreed with the table at 151.2 MP)
  cuml_crossover_synthetic.csv   → fig 5

Outputs saved to: GPUPhotFinal/figures_profiler/  (PDF + PNG)
Then copied to:   GPUPhotFinal/GPUPHOT_manuscript/figures/
"""

import os
import math
import sys
import shutil
import subprocess
import numpy as np
import matplotlib.patheffects as path_effects
import pandas as pd

# Re-use the production data pipeline from the tables script so that figures
# and tables always use identical warmup/outlier/exclusion logic.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_manuscript_tables import (
    load_benchmark as _load_benchmark_tables,
    load_benchmark_by_epoch as _load_by_epoch,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
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

# Hatch and edge-color maps for per-GPU identification in bar charts (Figs 7 & 8)
HATCH_MAP = {
    'H100 (80 GB)':  '',
    'A100 (80 GB)':  '///',
    'L40S (48 GB)':  'xxx',
}
EDGE_MAP = {
    'H100 (80 GB)':  '#000000',
    'A100 (80 GB)':  '#555555',
    'L40S (48 GB)':  '#AAAAAA',
}

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BASE)
DATA_DIR = os.path.join(BASE, 'data')
DRAWIO_XML = os.path.join(PROJECT, 'GPUPhot.drawio.xml')
OUT_DIR = os.path.join(PROJECT, 'figures_profiler')
MANUSCRIPT_FIGURES_DIR = os.path.join(PROJECT, 'GPUPHOT_manuscript', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

BENCHMARK_CSV = os.path.join(DATA_DIR, 'benchmark_latency.csv')
MEMORY_RAW_CSV = os.path.join(DATA_DIR, 'profiler_memory_raw.csv')
CUML_CSV      = os.path.join(DATA_DIR, 'cuml_crossover_synthetic.csv')
CELL_STATUS_CSV = os.path.join(DATA_DIR, 'cell_status.csv')


def gpu_fig_label(gpu_label):
    """Etiqueta de la columna en una figura.

    No session mark, as in the tables: the paper presents a single campaign because
    every column comes from the same code (see SESSION_BY_ARM in generate_manuscript_tables).
    There are no two epochs to tell apart, so marking some columns and not others would
    diferencia que no existe.
    """
    return GPU_SHORT.get(gpu_label, gpu_label)



# ── GPU label normalisation ───────────────────────────────────────────────────
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
GPU_ORDER = [
    'H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)',
    'RTX 3090 (24 GB)', 'RTX 3060 (12 GB)', 'RTX 3050 Ti (4 GB)',
    'Orin Super (8 GB)', 'Orin Nano (8 GB)',
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
    'Orin Nano (8 GB)':     'Orin Nano',
}
# GPU order for heatmap (py3.12): Orin Nano excluded — no py3.12 data (runs py38_baseline only)
HEATMAP_GPU_ORDER = [
    'H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)',
    'RTX 3090 (24 GB)', 'RTX 3060 (12 GB)', 'RTX 3050 Ti (4 GB)',
    'Orin Super (8 GB)',
]

# Image label order for plots (ascending MP, then source count within MP)
IMAGE_ORDER = [
    'iKon936_Lum',
    'iKon936_SDSSg',
    'QHY600-3_SDSSi_2k',
    'QHY600-3_Lum',
    'QHY600-4_SDSSg',
    'QHY600-4_Ha',
    'QHY600-4_Lum_2k',
    'QHY600-4_SDSSg_4k',
    'QHY600-4_SDSSi_10k',
    'QHY411-1_Lum_bin2',
    'QHY411-1_SDSSi_bin2',
    'QHY411-1_SDSSg_2k',
    'QHY411-1_SDSSr_7k',
    'QHY411-1_Lum_full',
    'QHY411-3_SDSSg_10k',
    'QHY411-3_SDSSr_full',
    'QHY411-3_Lum_full',
    'QHY411-3_SDSSr_19k',
    'QHY411-3_Lum_131k',
]
IMAGE_MP = {
    'iKon936_Lum': 4.2,
    'iKon936_SDSSg': 4.2,
    'QHY600-3_SDSSi_2k': 6.8,
    'QHY600-3_Lum': 6.8,
    'QHY600-4_SDSSg': 15.3,
    'QHY600-4_Ha': 15.3,
    'QHY600-4_Lum_2k': 15.3,
    'QHY600-4_SDSSg_4k': 15.3,
    'QHY600-4_SDSSi_10k': 15.3,
    'QHY411-1_Lum_bin2': 37.8,
    'QHY411-1_SDSSi_bin2': 37.8,
    'QHY411-1_SDSSg_2k': 37.8,
    'QHY411-1_SDSSr_7k': 37.8,
    'QHY411-1_Lum_full': 151.2,
    'QHY411-3_SDSSg_10k': 151.2,
    'QHY411-3_SDSSr_full': 151.2,
    'QHY411-3_Lum_full': 151.2,
    'QHY411-3_SDSSr_19k': 151.2,
    'QHY411-3_Lum_131k': 151.2,
}
IMAGE_SRC = {
    'iKon936_Lum': 228,
    'iKon936_SDSSg': 279,
    'QHY600-3_SDSSi_2k': 3534,
    'QHY600-3_Lum': 8093,
    'QHY600-4_SDSSg': 307,
    'QHY600-4_Ha': 601,
    'QHY600-4_Lum_2k': 2938,
    'QHY600-4_SDSSg_4k': 8294,
    'QHY600-4_SDSSi_10k': 15759,
    'QHY411-1_Lum_bin2': 354,
    'QHY411-1_SDSSi_bin2': 621,
    'QHY411-1_SDSSg_2k': 3977,
    'QHY411-1_SDSSr_7k': 12832,
    'QHY411-1_Lum_full': 808,
    'QHY411-3_SDSSg_10k': 10168,
    'QHY411-3_SDSSr_full': 15390,
    'QHY411-3_Lum_full': 18712,
    'QHY411-3_SDSSr_19k': 19492,
    'QHY411-3_Lum_131k': 134206,
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
    """Load benchmark data using the same pipeline as generate_manuscript_tables.

    Delegates to _load_benchmark_tables so figures and tables always share
    identical warmup/outlier/exclusion logic (session-aware detector, 3-MAD
    filter, Vizier-only, round-5C excluded).
    """
    labels = profiler_labels if profiler_labels else []
    if not labels:
        return pd.DataFrame()
    df = _load_benchmark_tables(labels)
    # Restore mp and n_sources_detected as numeric (tables script may not add them)
    if 'mp' in df.columns:
        df['mp'] = pd.to_numeric(df['mp'], errors='coerce')
    if 'n_sources_detected' in df.columns:
        df['n_sources_detected'] = pd.to_numeric(df['n_sources_detected'], errors='coerce')
    return df


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Architecture diagram (generated from GPUPhot.drawio.xml via drawio CLI)
# ═════════════════════════════════════════════════════════════════════════════
def figure1():
    """Render fig1_architecture.{pdf,png} from GPUPhot.drawio.xml using drawio CLI.

    Requires the drawio desktop app to be installed and available in PATH.
    Install: https://github.com/jgraph/drawio-desktop/releases
      - Linux (snap):  snap install drawio
      - macOS:         brew install --cask drawio
      - Windows:       winget install JGraph.drawio
    If drawio is not available, fig1 is skipped with a warning (other figures unaffected).
    """
    print('Figure 1: Architecture diagram (drawio) ...')
    if shutil.which('drawio') is None:
        print('  WARNING: drawio not found in PATH — skipping fig1_architecture.')
        print('           Install drawio (snap install drawio) to regenerate this figure.')
        return
    if not os.path.exists(DRAWIO_XML):
        print(f'  WARNING: {DRAWIO_XML} not found — skipping fig1')
        return
    for fmt in ('pdf', 'png'):
        out = os.path.join(OUT_DIR, f'fig1_architecture.{fmt}')
        cmd = ['drawio', '-x', '-f', fmt, '--crop', '-b', '10', '-o', out, DRAWIO_XML]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f'  Saved {out}')
            else:
                print(f'  ERROR generating {fmt}: {result.stderr.strip()}')
        except Exception as exc:
            print(f'  WARNING: drawio execution failed ({exc}) — skipping fig1 {fmt}')


def _jetson_note():
    """Annotation for the ARM modules excluded from Figure 2.

    Computed rather than pinned: the figure carried a hard-coded '~150 s at 4.2 MP',
    which stopped being true once both Orin modules were measured and the two 4.2 MP
    frames diverged by a factor of six on the Orin Nano.
    """
    _j = _load_benchmark_tables(['py38_baseline', 'py312_cuml_adaptive'])
    _j = _j[_j['gpu_label'].astype(str).str.startswith('Orin')]
    if not len(_j):
        return 'Jetson Orin (ARM) not shown (no RAPIDS)'
    _m = _j.groupby(['gpu_label', 'profiler_label', 'image_label'])['execution_time'].median()
    return (f'Jetson Orin (ARM) not shown — {_m.min():.0f}\u2013{_m.max():.0f} s '
            f'at 4.2 MP (no RAPIDS)')


def _src_ticklabels(images):
    """Compact 'N src' tick labels.

    The megapixel annotation above the panel already carries the image size, so
    repeating it on every tick only costs width.  After the August-2026 campaign the
    source counts grew (131 397 -> 134 206, 14 241 -> 15 390) and the long form stopped
    fitting, overprinting its neighbours.
    """
    out = []
    for img in images:
        n = IMAGE_SRC[img]
        if n >= 10000:
            out.append(f'{n // 1000}k src')
        elif n >= 1000:
            out.append(f'{n / 1000:.1f}k src')
        else:
            out.append(f'{n} src')
    return out


def _mp_group_bounds(images):
    """First and last x index of each megapixel group, from the images actually drawn.

    These used to be written out as fixed index ranges, which mislabels the groups as
    soon as a cell drops out of the panel.
    """
    bounds = {}
    for idx, img in enumerate(images):
        mp = IMAGE_MP[img]
        bounds.setdefault(mp, [idx, idx])[1] = idx
    return bounds


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Latency vs Image Size (dodge + size ∝ log n_src)
# ═════════════════════════════════════════════════════════════════════════════
def figure2():
    """Scatter: per-image median latency, dodge by GPU, marker size ∝ log10(n_src).
    Legend positions are verified programmatically to not overlap any data point.
    """
    print('Figure 2: Latency vs Image Size (dodge + size∝log n_src) ...')
    import matplotlib.transforms as _mtf
    from matplotlib.lines import Line2D

    # Per-card epoch split (GPU_EPOCH), the same one the tables use.
    df = _load_by_epoch(['py312_cuml_adaptive'])

    # Only x86 GPUs: the ARM modules would collapse the log-log scale.  The annotation
    # below is computed from the data rather than pinned.
    x86_gpus = [g for g in GPU_ORDER if 'Orin' not in g]
    df = df[df['gpu_label'].isin(x86_gpus)]

    # Per-image median: one point per (GPU, image)
    img_med = (df.groupby(['gpu_label', 'image_label'])['execution_time']
               .median().reset_index())
    img_med['mp']    = img_med['image_label'].map(IMAGE_MP)
    img_med['n_src'] = img_med['image_label'].map(IMAGE_SRC)

    # Dodge: fixed log10 offset per GPU, centered at zero
    n_gpus = len(x86_gpus)
    DODGE_STEP = 0.030
    dodge_offsets = np.linspace(-(n_gpus - 1) / 2 * DODGE_STEP,
                                 (n_gpus - 1) / 2 * DODGE_STEP, n_gpus)
    GPU_DODGE = {g: off for g, off in zip(x86_gpus, dodge_offsets)}
    img_med['x_dodge'] = img_med.apply(
        lambda r: r['mp'] * (10 ** GPU_DODGE.get(r['gpu_label'], 0.0)), axis=1)

    # Marker size ∝ log10(n_src)
    LOG_N_MIN = np.log10(100)
    LOG_N_MAX = np.log10(131397)
    S_MIN, S_MAX = 25, 200

    def _msize(n_src):
        t = (np.log10(max(n_src, 100)) - LOG_N_MIN) / (LOG_N_MAX - LOG_N_MIN)
        return S_MIN + np.clip(t, 0, 1) * (S_MAX - S_MIN)

    img_med['msize'] = img_med['n_src'].apply(_msize)

    # Legend handle definitions (built once, reused across attempts)
    gpu_present = [g for g in x86_gpus
                   if not img_med[img_med['gpu_label'] == g].empty]
    gpu_handles = [
        Line2D([0], [0], marker='o',
               color=CB_COLORS[x86_gpus.index(g) % len(CB_COLORS)],
               markersize=7, linestyle='None',
               markeredgecolor='#333333', markeredgewidth=0.45)
        for g in gpu_present
    ]
    gpu_labels = [gpu_fig_label(g) for g in gpu_present]

    SRC_LEGEND = [(112, '~100'), (2000, '~2 k'), (19000, '~20 k'), (131397, '~130 k')]
    size_handles = [
        Line2D([0], [0], marker='o', color='#555555',
               markersize=np.sqrt(_msize(n)), linestyle='None',
               markeredgecolor='#333333', markeredgewidth=0.45, label=lbl)
        for n, lbl in SRC_LEGEND
    ]

    # ── Auto-verified legend positioning ─────────────────────────────────────
    # Start both legends in the upper area just past the 4.2 MP group.
    # Shift rightward by 0.08 axes-fraction each iteration if overlap detected.
    leg1_pos = (0.20, 0.99)  # GPU legend anchor (upper-left corner, axes fraction)
    leg2_pos = (0.42, 0.99)  # Source-count legend anchor

    final_fig = None
    overlap_log = []

    for attempt in range(4):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Image size (megapixels)')
        ax.set_ylabel('Median end-to-end latency (s)')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
        ax.grid(True, which='both', ls=':', alpha=0.4)
        ax.text(0.99, 0.02,
                _jetson_note(),
                transform=ax.transAxes, fontsize=7, color='#666666',
                ha='right', va='bottom', style='italic')

        # Scatter
        for idx, gpu_label in enumerate(x86_gpus):
            sub = img_med[img_med['gpu_label'] == gpu_label]
            if sub.empty:
                continue
            ax.scatter(sub['x_dodge'], sub['execution_time'],
                       c=CB_COLORS[idx % len(CB_COLORS)],
                       s=sub['msize'],
                       edgecolors='#333333', linewidths=0.45,
                       alpha=0.82, zorder=3)

        # Legends
        leg1 = ax.legend(handles=gpu_handles, labels=gpu_labels,
                         fontsize=8, bbox_to_anchor=leg1_pos, loc='upper left',
                         framealpha=0.95, title='GPU', title_fontsize=8)
        ax.add_artist(leg1)
        leg2 = ax.legend(handles=size_handles,
                         fontsize=8, bbox_to_anchor=leg2_pos, loc='upper left',
                         framealpha=0.95, title='Source count', title_fontsize=8)

        fig.tight_layout()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # Bounding boxes in display coordinates
        leg1_bb = leg1.get_window_extent(renderer)
        leg2_bb = leg2.get_window_extent(renderer)
        legend_bbs = [leg1_bb, leg2_bb]

        # Check each data point
        overlaps = []
        for _, row in img_med.iterrows():
            xd, yd = ax.transData.transform((row['x_dodge'], row['execution_time']))
            r_px = np.sqrt(row['msize']) * fig.dpi / 72.0 / 2.0
            pt_bb = _mtf.Bbox([[xd - r_px, yd - r_px], [xd + r_px, yd + r_px]])
            for lbb in legend_bbs:
                if lbb.overlaps(pt_bb):
                    overlaps.append((row['gpu_label'], float(row['mp']), int(row['n_src'])))
                    break

        overlap_log.append({'attempt': attempt,
                             'leg1_pos': leg1_pos, 'leg2_pos': leg2_pos,
                             'n_overlap': len(overlaps), 'overlaps': overlaps})

        if not overlaps:
            print(f'  Auto-verify attempt {attempt}: OK — no overlap. '
                  f'leg1={leg1_pos}, leg2={leg2_pos}')
            final_fig = fig
            break

        print(f'  Auto-verify attempt {attempt}: {len(overlaps)} overlapping — '
              f'{overlaps[:3]}{"..." if len(overlaps) > 3 else ""}')
        shift = 0.08
        leg1_pos = (round(leg1_pos[0] + shift, 3), leg1_pos[1])
        leg2_pos = (round(leg2_pos[0] + shift, 3), leg2_pos[1])
        plt.close(fig)
    else:
        print(f'  WARNING: overlap persists after all attempts — using last position')
        final_fig = fig

    # Report full overlap log
    for entry in overlap_log:
        status = 'OK' if entry['n_overlap'] == 0 else f"{entry['n_overlap']} overlaps"
        print(f'    attempt {entry["attempt"]}: leg1={entry["leg1_pos"]} '
              f'leg2={entry["leg2_pos"]} → {status}')

    save(final_fig, 'fig2_latency_vs_size')


# FIGURE 3 — Memory py3.8 vs py3.12 (A100)  [data unchanged]
# ═════════════════════════════════════════════════════════════════════════════
def vram_median_a100(gpu_name='A100-SXM4-80GB'):
    """Median peak VRAM per (megapixels, python_ver) for one GPU, in MiB.

    Mirrors vram_median() in generate_manuscript_tables.py exactly: same file, same two
    filters, and the same pooling of every capture of a given size.  The figure used to read
    profiler_memory_summary.csv, which carries ONE ROW PER CAMERA, and took .values[0].  At
    151.2 MP the A100 has two camera rows, so the figure published the first camera alone
    (9.4%) where the table pooled both (9.8%).  Every other size has a single camera row,
    which is why the defect showed in one cell and nowhere else.  Reading the same captures
    as the table removes the whole class, not just that cell.
    """
    df = pd.read_csv(MEMORY_RAW_CSV)
    df['gpu_name'] = df['gpu_name'].str.replace('NVIDIA ', '', regex=False)
    for col in ('python_ver', 'megapixels', 'gpu_total_MB', 'peak_gpu_memory_MB'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Keep only successful captures; drop peaks above the card, an Nsight artefact.
    if 'capture_complete' in df.columns:
        df = df[df['capture_complete'].astype(str).str.strip().str.lower() == 'true']
    df = df[df['peak_gpu_memory_MB'] <= df['gpu_total_MB']]
    df = df[df['gpu_name'] == gpu_name]
    return df.groupby(['megapixels', 'python_ver'])['peak_gpu_memory_MB'].median()


def figure3():
    print('Figure 3: Memory comparison py3.8 vs py3.12 ...')
    med = vram_median_a100()
    from generate_manuscript_tables import vram_postfix_a100
    post = vram_postfix_a100()
    mp_values = sorted({mp for mp, _ in med.index})
    # Memory figure uses one value per MP size (aggregated): label = MP only
    cam_map = {mp: f'{mp:.1f}'.rstrip("0").rstrip(".") + ' MP'
               for mp in [4.2, 6.8, 15.3, 37.8, 151.2]}

    fig, ax = plt.subplots(figsize=(7.6, 4))
    x = np.arange(len(mp_values))
    width = 0.27
    py38_vals, py312_vals, post_vals = [], [], []
    for mp in mp_values:
        py38_vals.append(med.get((mp, 3.8), 0))
        py312_vals.append(med.get((mp, 3.12), 0))
        post_vals.append(post.get(mp, 0))

    py38_vals  = np.array(py38_vals)
    py312_vals = np.array(py312_vals)
    post_vals  = np.array(post_vals)

    # Three bars, which is the point of the figure: the LOW peak of py3.12
    # was not a property of CuPy 14 but of the RAPIDS allocator returning
    # every temporary to the driver, and once the pool is re-seated the peak
    # goes back to py3.8's. With two bars, this had to be taken on faith from
    # the caption.
    ax.bar(x - width, py38_vals,  width, label='py3.8 (CuPy 12)',
           color=CB_COLORS[0], edgecolor='white', linewidth=0.5, hatch='///')
    ax.bar(x,         py312_vals, width, label='py3.12, no pool',
           color=CB_COLORS[5], edgecolor='white', linewidth=0.5)
    ax.bar(x + width, post_vals,  width, label='py3.12, pool re-seated',
           color=CB_COLORS[2], edgecolor='white', linewidth=0.5, hatch='...')

    # The label sits over the third bar and compares WITH POOL against
    # py3.8, which is the question left for the reader: does the fix cost
    # memory? The comparison without pool against py3.8 is read off the bar
    # heights and is in the table.
    for i in range(len(mp_values)):
        if py38_vals[i] > 0 and post_vals[i] > 0:
            sav = (round(post_vals[i]) - round(py38_vals[i])) / round(py38_vals[i]) * 100
            ax.annotate(f'{sav:+.1f}%', xy=(x[i] + width, post_vals[i]),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        # Gray when the difference rounds to zero: under the
                        # sign-based criterion, +0.0% came out orange and
                        # -0.0% blue, two colors for the same result.
                        color=('#666666' if abs(sav) < 0.05
                               else '#0072B2' if sav < 0 else '#D55E00'))
    ax.set_xlabel('Image')
    ax.set_ylabel('Peak VRAM (GiB)')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1024:.0f}'))
    ax.set_xticks(x)
    ax.set_xticklabels([cam_map.get(mp, f'{mp} MP') for mp in mp_values], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', ls=':', alpha=0.4)
    fig.tight_layout()
    save(fig, 'fig3_memory_comparison')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Heatmap GPU × image (10 images, py3.12+adaptive)
# ═════════════════════════════════════════════════════════════════════════════
# ── Empty-cell bookkeeping for the heatmap ───────────────────────────────────
# One mark for every empty cell.  The ledger distinguishes six causes and the figure used
# to print them, which put two vocabularies on a plot that is about latency.
#
# The mark asserts nothing, and that is deliberate.  The empty cells are not one thing:
# some were launched and ran out of memory, and the rest were never launched at all,
# because a frame of the same size class had already exhausted that card.  A label saying
# they were attempted would be false for the second group, and "n/a" would be false for
# the first.  The breakdown is printed when this runs, for the caption to carry, and the
# per-cell causes stay in build_cell_status.py with their evidence.
EMPTY_MARK = '--'


def _cell_status(profile):
    """(gpu_label, image_label) -> status, for one profile."""
    d = pd.read_csv(CELL_STATUS_CSV)
    d = d[d['profiler_label'] == profile]
    return {(g, i): s for g, i, s in
            zip(d['gpu_label'], d['image_label'], d['status'])}


def figure4():
    print('Figure 4: Heatmap ...')
    # Per-card epoch split (GPU_EPOCH), the same one the tables use.
    df = _load_by_epoch(['py312_cuml_adaptive'])
    med = df.groupby(['gpu_label', 'image_label'])['execution_time'].median()

    # Use HEATMAP_GPU_ORDER: excludes Orin Nano (no py3.12 data; all-grey columns
    # would misleadingly imply OOM rather than "not tested under this config").
    # Orin Super is included with asterisk (py3.12 ARM, cuML unavailable).
    n_imgs = len(IMAGE_ORDER)
    n_gpus = len(HEATMAP_GPU_ORDER)
    matrix = np.full((n_imgs, n_gpus), np.nan)

    # An empty cell has several possible causes and they must not be conflated:
    # the ledger built by build_cell_status.py says, per cell, whether
    # every attempt hit an out-of-memory condition, whether it failed for
    # another reason, or whether the cell was never launched on that machine.
    status = _cell_status('py312_cuml_adaptive')

    for i, img in enumerate(IMAGE_ORDER):
        for j, gpu in enumerate(HEATMAP_GPU_ORDER):
            try:
                matrix[i, j] = med.loc[(gpu, img)]
            except KeyError:
                pass  # empty: labelled below from the cell-status ledger

    # Row labels: MP + source count only
    row_labels = [_img_display(img) for img in IMAGE_ORDER]
    col_labels = [gpu_fig_label(g) for g in HEATMAP_GPU_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.cm.viridis.reversed()   # dark = high latency, bright = fast
    cmap.set_bad(color='#CCCCCC')

    vmin = max(float(np.nanmin(matrix)), 0.5)
    vmax = float(np.nanmax(matrix))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', norm=norm)

    empty_seen = set()
    for i in range(n_imgs):
        for j in range(n_gpus):
            val = matrix[i, j]
            if np.isnan(val):
                st = status.get((HEATMAP_GPU_ORDER[j], IMAGE_ORDER[i]), 'not_run')
                empty_seen.add(st)
                ax.text(j, i, EMPTY_MARK, ha='center', va='center',
                        fontsize=7.5, fontweight='bold', color='#666666')
            else:
                log_normed = (np.log10(val) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin)) if val > 0 else 0
                tc = 'white' if log_normed > 0.6 else 'black'
                ax.text(j, i, f'{val:.1f}s', ha='center', va='center',
                        fontsize=7.5, fontweight='bold', color=tc)

    ax.set_xticks(range(n_gpus))
    ax.set_xticklabels(col_labels, fontsize=8.5, rotation=30, ha='right')
    ax.set_yticks(range(n_imgs))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Latency (s)', fontsize=9)

    note = 'Dashed cells have no measurement (see caption)' if empty_seen else ''
    if empty_seen:
        tally = {s: sum(1 for g in HEATMAP_GPU_ORDER for im in IMAGE_ORDER
                        if (g, im) not in med.index
                        and status.get((g, im), 'not_run') == s)
                 for s in sorted(empty_seen)}
        print('  empty cells for the caption:',
              ', '.join(f'{k}={v}' for k, v in tally.items()),
              f'(total {sum(tally.values())})')
    if note:
        ax.set_xlabel(note, fontsize=7.5, color='#444444', labelpad=8)
    fig.tight_layout()
    save(fig, 'fig4_heatmap')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — cuML Crossover synthetic  [data unchanged]
# ═════════════════════════════════════════════════════════════════════════════
def figure5():
    print('Figure 5: cuML Crossover ...')
    # comment='#': the CSV carries its campaign provenance in the header,
    # and without this pandas would read it as data. This has happened
    # before with other files in this directory.
    df = pd.read_csv(CUML_CSV, comment='#')
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
    ax.axhspan(0, 1.0, alpha=0.08, color='#D55E00', zorder=0)
    ax.axhline(y=1.0, color='#555555', linestyle='--', linewidth=1.0, zorder=2)
    ax.text(120, 1.05, 'break-even', fontsize=8, color='#555555', va='bottom')
    ax.text(80000, 0.15, 'cKDTree faster', fontsize=8, color='#D55E00',
            ha='center', fontstyle='italic', alpha=0.85)
    ax.text(80000, 4.5, 'cuML faster', fontsize=8, color='#0072B2',
            ha='center', fontstyle='italic', alpha=0.85)

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
    # Read peak VRAM for 4.2 MP (iKon936) from the profiler CSV (A100-SXM4-80GB)
    _med = vram_median_a100()
    peak_38  = int(_med.get((4.2, 3.8),  2018))
    peak_312 = int(_med.get((4.2, 3.12), 1718))
    gpu_order = ['H100', 'A100', 'L40S', 'RTX 3090', 'RTX 3060', 'RTX 3050 Ti']

    # Read actual measured GPU total VRAM from the memory CSV, capping at the
    # nominal spec to avoid NVML inflation on server GPUs (H100/A100).
    # This matches the formula used in gen_concurrency() in generate_manuscript_tables.py.
    _short_to_label = {
        'H100': 'H100 (80 GB)', 'A100': 'A100 (80 GB)', 'L40S': 'L40S (48 GB)',
        'RTX 3090': 'RTX 3090 (24 GB)', 'RTX 3060': 'RTX 3060 (12 GB)',
        'RTX 3050 Ti': 'RTX 3050 Ti (4 GB)',
    }
    _nom_mb = {
        'H100': 80*1024, 'A100': 80*1024, 'L40S': 48*1024,
        'RTX 3090': 24*1024, 'RTX 3060': 12*1024, 'RTX 3050 Ti': 4*1024,
    }
    _mem = pd.read_csv(MEMORY_RAW_CSV)
    _mem_gl = _mem['gpu_name'].str.replace('NVIDIA ', '', regex=False).map(GPU_LABEL_MAP)
    _actual_total = _mem.assign(gl=_mem_gl).groupby('gl')['gpu_total_MB'].median().to_dict()
    vram_total = {
        g: int(min(_actual_total.get(_short_to_label[g], _nom_mb[g]), _nom_mb[g]))
        for g in gpu_order
    }

    # Formula: floor(vram_total / peak_vram) — no safety-margin factor.
    # Matches the canonical formula used in tab:concurrency (gen_concurrency()).
    conc_38  = np.array([math.floor(vram_total[g]/peak_38)  for g in gpu_order])
    conc_312 = np.array([math.floor(vram_total[g]/peak_312) for g in gpu_order])

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(gpu_order))
    width = 0.35

    bars38  = ax.bar(x - width/2, conc_38,  width, label='py3.8 (CuPy 12)',
                     color=CB_COLORS[0], edgecolor='white', linewidth=0.5, hatch='///')
    bars312 = ax.bar(x + width/2, conc_312, width, label='py3.12 (CuPy 14)',
                     color=CB_COLORS[5], edgecolor='white', linewidth=0.5)

    _top = max(conc_38.max(), conc_312.max())

    for i, g in enumerate(gpu_order):
        if conc_38[i] > 0:
            gain = (conc_312[i] - conc_38[i]) / conc_38[i] * 100
            # Colour follows the sign, as in figure 3, not the identity of the card.
            # It used to single out the 3050 Ti, which made orange mean "look here" in
            # this figure and "py3.12 uses more" in the next one.
            color = '#0072B2' if gain >= 0 else '#D55E00'
            # The sign comes from the number.  It was hard-coded as "+", which reads as
            # a property of py3.12 and would print "+-17%" the day one of these turns.
            ax.annotate(f'{gain:+.0f}%',
                        xy=(x[i] + width/2, conc_312[i]),
                        # A short bar carries its value above the bar, so the percentage
                        # has to clear it or the two overlap.
                        xytext=(0, 5 if conc_312[i] / _top >= 0.08 else 16),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color=color)

    # The py3.8 bars are hatched with white lines, so a white numeral sat on its own
    # colour and disappeared.  A dark numeral with a light halo reads on the hatched bar
    # and on the solid one alike, so both series keep the same treatment.
    _halo = [path_effects.withStroke(linewidth=2.2, foreground='white')]
    for bars in (bars38, bars312):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                # A bar of height one leaves no room for a numeral at its midpoint, and
                # the 3050 Ti is exactly the row a reader checks.  Short bars carry their
                # value above instead.
                inside = h / _top >= 0.08
                ax.text(bar.get_x() + bar.get_width()/2, h/2 if inside else h,
                        f'{int(h)}', ha='center',
                        va='center' if inside else 'bottom',
                        fontsize=7.5, color='#141413', fontweight='bold',
                        path_effects=_halo)

    ax.set_xlabel('GPU')
    ax.set_ylabel('Concurrent 4.2 MP images')
    ax.set_xticks(x)
    ax.set_xticklabels(gpu_order, fontsize=9, rotation=15, ha='right')
    ax.legend(fontsize=9)
    ax.grid(axis='y', ls=':', alpha=0.4)
    fig.tight_layout()
    save(fig, 'fig6_concurrency')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — py3.8 vs py3.12+adaptive: % overhead per image (real pipeline)
# ═════════════════════════════════════════════════════════════════════════════
# One per figure: each keeps the sessions where that figure's own two arms ran
# together, which is not the same set for py3.8-vs-adaptive as for forced-vs-adaptive.
FIX1_PAIRED_CSV = os.path.join(DATA_DIR, 'benchmark_latency_fig78_paired.csv')
FIX1_PAIRED_CSV8 = os.path.join(DATA_DIR, 'benchmark_latency_fig8_paired.csv')
DATACENTER_GPUS = ['H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)']


def _fig7_pivot(df):
    """Overhead of py3.12+adaptive over py3.8, per cell, datacenter GPUs only."""
    df = df[df['gpu_label'].isin(DATACENTER_GPUS)]
    med = (df.groupby(['gpu_label', 'profiler_label', 'image_label'])['execution_time']
             .median().unstack('profiler_label'))
    med = med.rename(columns={'py38_baseline': 'py38', 'py312_cuml_adaptive': 'adaptive'})
    med = med.dropna(subset=['py38', 'adaptive'])
    med['pct'] = (med['adaptive'] - med['py38']) / med['py38'] * 100
    piv = med.reset_index().pivot(index='image_label', columns='gpu_label', values='pct')
    return piv.reindex([img for img in IMAGE_ORDER if img in piv.index])


def _fig7_post_fix():
    """Same, measured after the allocator fix, or None if that campaign is absent.

    Only cells whose two arms ran in the same session count; the builder of that CSV
    explains why, and it is the reason this panel can be thinner than the other.
    """
    if not os.path.exists(FIX1_PAIRED_CSV):
        return None
    # drop_contaminated=False on purpose: the two arms of each cell run in
    # the SAME session, so any host contamination affects the numerator and
    # the denominator equally and cancels out in the ratio. Filtering it
    # here would drop 16 valid cells.
    d = _load_benchmark_tables(['py38_baseline', 'py312_cuml_adaptive'],
                               csv_path=FIX1_PAIRED_CSV, drop_contaminated=False)
    return _fig7_pivot(d) if len(d) else None


def _render_overhead_panel(pct_pivot, ylabel, pos_color, neg_color, name):
    """One panel of per-cell percentages, grouped by megapixels, for figures 7 and 8."""
    x = np.arange(len(pct_pivot))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.0))

    for i, gpu in enumerate(DATACENTER_GPUS):
        if gpu not in pct_pivot.columns:
            continue
        vals = pct_pivot[gpu].values
        ax.bar(x + (i - 1) * width, vals, width,
               color=[pos_color if v > 0 else neg_color for v in vals],
               edgecolor=EDGE_MAP.get(gpu, '#555555'),
               hatch=HATCH_MAP.get(gpu, ''), linewidth=0.7, alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.8, zorder=5)
    _v = pct_pivot.values[np.isfinite(pct_pivot.values)]
    lo, hi = min(0.0, float(_v.min())), float(_v.max())
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad * 1.6)      # headroom for the MP group labels
    ax.set_xticks(x)
    ax.set_xticklabels(_src_ticklabels(pct_pivot.index), fontsize=8, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', ls=':', alpha=0.4)
    ax.legend(handles=[Patch(facecolor='#BBBBBB', hatch=HATCH_MAP.get(g, ''),
                             edgecolor=EDGE_MAP.get(g, '#555555'), linewidth=0.7,
                             label=GPU_SHORT[g])
                       for g in DATACENTER_GPUS if g in pct_pivot.columns],
              fontsize=9, loc='lower left', framealpha=0.95, ncol=3)

    ymax = ax.get_ylim()[1]
    for _mp, (_fi, _li) in _mp_group_bounds(list(pct_pivot.index)).items():
        if _li + 0.5 < len(pct_pivot) - 0.5:
            ax.axvline(_li + 0.5, color='#AAAAAA', linewidth=0.6, ls='--', zorder=1)
        ax.text((_fi + _li) / 2.0, ymax * 0.97,
                f'{_mp:.1f}'.rstrip('0').rstrip('.') + ' MP',
                fontsize=7, color='#555555', ha='center', va='top')

    fig.tight_layout()
    save(fig, name)


def figure7():
    """Overhead of py3.12+adaptive over py3.8 on the datacenter GPUs, after the fix.

    Most of what this figure used to report was a defect and not a property of the
    interpreter: cuML's import replaced CuPy's allocator, and restoring it (1a23734)
    takes the median from +38.9% to +11.1%.  Reporting the campaign numbers would
    describe a system that no longer exists.  The post-fix campaign covers exactly this
    figure's population -- the three datacenter GPUs -- with both arms measured in the
    same session; the tables stay on the campaign because they need four more cards,
    one of which cannot be re-measured at all.  Absent that campaign the figure falls
    back to the campaign data rather than failing.
    """
    post = _fig7_post_fix()
    src = 'post-fix' if post is not None else 'campaign'
    print(f'Figure 7: py3.8 vs py3.12+adaptive overhead ({src}) ...')
    piv = post if post is not None else _fig7_pivot(
        load_benchmark(['py38_baseline', 'py312_cuml_adaptive']))
    _render_overhead_panel(piv, 'Overhead vs py3.8 (%)\n(positive = py3.12 slower)',
                           '#D55E00', '#009E73', 'fig7_py38_vs_adaptive')


def _fig8_pivot(df):
    """Improvement of adaptive over forced cuML, per cell, datacenter GPUs only."""
    df = df[df['gpu_label'].isin(DATACENTER_GPUS)]
    med = (df.groupby(['gpu_label', 'profiler_label', 'image_label'])['execution_time']
             .median().unstack('profiler_label'))
    med = med.rename(columns={'py312_cuml_always': 'always', 'py312_cuml_adaptive': 'adaptive'})
    if not {'always', 'adaptive'} <= set(med.columns):
        return pd.DataFrame()          # one of the two arms was never measured here
    med = med.dropna(subset=['always', 'adaptive'])
    med['pct'] = (med['always'] - med['adaptive']) / med['always'] * 100
    piv = med.reset_index().pivot(index='image_label', columns='gpu_label', values='pct')
    return piv.reindex([img for img in IMAGE_ORDER if img in piv.index])


def _fig8_post_fix():
    """Same, after the allocator fix, or None if that campaign is absent."""
    if not os.path.exists(FIX1_PAIRED_CSV8):
        return None
    # Paired within the session: see the note in _fig7_post_fix.
    d = _load_benchmark_tables(['py312_cuml_always', 'py312_cuml_adaptive'],
                               csv_path=FIX1_PAIRED_CSV8, drop_contaminated=False)
    if not len(d):
        return None
    piv = _fig8_pivot(d)
    return piv if len(piv) else None


def figure8():
    """Adaptive vs forced cuML on the datacenter GPUs, after the allocator fix.

    Same reasoning as figure 7, plus one of its own: an earlier version marked six bars
    as measured differently, because six cells whose arms had been measured ten days
    apart were re-paired while the rest kept the campaign's block design.  Here every
    cell has its two arms interleaved inside one session, so nothing needs marking.
    """
    post = _fig8_post_fix()
    src = 'post-fix' if post is not None else 'campaign'
    print(f'Figure 8: cuML always vs adaptive ({src}) ...')
    piv = post if post is not None else _fig8_pivot(
        load_benchmark(['py312_cuml_always', 'py312_cuml_adaptive']))
    _render_overhead_panel(piv, 'Improvement of adaptive vs always (%)\n(positive = adaptive faster)',
                           '#009E73', '#D55E00', 'fig8_always_vs_adaptive')


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
    figure1()
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
