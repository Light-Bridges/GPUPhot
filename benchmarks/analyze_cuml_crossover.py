#!/usr/bin/env python3
"""
Analyze the crossover point where cuML (Python 3.12) becomes faster than
CPU fallback (Python 3.8).

Uses the same Elasticsearch CSV data as analyze_times_v2.py.
Determines whether the key variable is number of sources, image size, or both.
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    """Load and deduplicate all CSV files (same logic as analyze_times_v2.py)."""
    csvs = sorted(glob.glob(os.path.join(DATA_DIR, 'es_times_*_per_hit.csv')))
    if not csvs:
        print("No CSV files found"); sys.exit(1)

    frames = []
    for f in csvs:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                frames.append(df)
        except Exception:
            pass

    df = pd.concat(frames, ignore_index=True)

    # Deduplicate
    if 'oblineid' in df.columns and 'dateproc' in df.columns:
        df = df.drop_duplicates(subset=['oblineid', 'dateproc'], keep='last')

    # Clean
    df = df[df['execution_time'] <= 1800]
    df = df[df['gpu_name'].notna() & (df['gpu_name'] != '')]

    # Python version bucket
    df['pyver'] = df['python_ver'].str.extract(r'^(\d+\.\d+)')[0]

    # Drop rows with missing dimensions
    df = df.dropna(subset=['naxis1', 'naxis2'])

    # Megapixels
    df['megapixels'] = df['naxis1'] * df['naxis2'] / 1e6

    # Image size label
    df['image_size'] = df['naxis1'].astype(int).astype(str) + 'x' + df['naxis2'].astype(int).astype(str)

    return df


def compute_paired_ratios(df):
    """
    For each (gpu_name, camera, filter, image_size) group, compute
    the median time for py3.8 and py3.12, then the ratio.
    """
    # Only x86_64 (cuML is not available on ARM)
    df = df[df['processor'] == 'x86_64'].copy()

    # Bucket Python versions
    mask_38 = df['pyver'].isin(['3.8'])
    mask_312 = df['pyver'].isin(['3.12'])
    df_38 = df[mask_38].copy()
    df_312 = df[mask_312].copy()

    group_cols = ['gpu_name', 'inmodel', 'filter', 'image_size']

    stats_38 = df_38.groupby(group_cols).agg(
        median_s_38=('execution_time', 'median'),
        n_38=('execution_time', 'count'),
        median_src_38=('n_sources_detected', 'median'),
        megapixels=('megapixels', 'first'),
    ).reset_index()

    stats_312 = df_312.groupby(group_cols).agg(
        median_s_312=('execution_time', 'median'),
        n_312=('execution_time', 'count'),
        median_src_312=('n_sources_detected', 'median'),
    ).reset_index()

    merged = stats_38.merge(stats_312, on=group_cols, how='inner')

    # Filter: require at least 10 samples per group for reliability
    merged = merged[(merged['n_38'] >= 10) & (merged['n_312'] >= 10)]

    # Ratio: >1 means 3.12 is faster, <1 means 3.8 is faster
    merged['speedup_312'] = merged['median_s_38'] / merged['median_s_312']

    # Use average of sources from both versions
    merged['median_src'] = (merged['median_src_38'] + merged['median_src_312']) / 2

    return merged


def analyze_correlations(merged):
    """Test what drives the speedup: sources, image size, or both."""
    print("=" * 70)
    print("CORRELATION ANALYSIS: What drives cuML speedup?")
    print("=" * 70)
    print(f"  Paired workloads with N>=10 per version: {len(merged)}")
    print()

    for var, label in [('median_src', 'Number of sources'),
                       ('megapixels', 'Image size (MP)')]:
        x = np.log10(merged[var].clip(lower=1))
        y = merged['speedup_312']
        rho, p = sp_stats.spearmanr(x, y)
        print(f"  Spearman({label} vs speedup_312): rho={rho:.3f}, p={p:.2e}")

    # Partial correlation: sources controlling for image size
    from numpy.linalg import lstsq
    X = np.column_stack([
        np.log10(merged['median_src'].clip(lower=1)),
        np.log10(merged['megapixels'].clip(lower=0.1)),
    ])
    y = merged['speedup_312'].values

    # Multiple regression
    X_aug = np.column_stack([np.ones(len(X)), X])
    coefs, _, _, _ = lstsq(X_aug, y, rcond=None)
    y_pred = X_aug @ coefs
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    print(f"\n  Multiple regression: speedup ~ log(sources) + log(MP)")
    print(f"    Intercept: {coefs[0]:.3f}")
    print(f"    coeff(log_sources): {coefs[1]:.3f}")
    print(f"    coeff(log_MP): {coefs[2]:.3f}")
    print(f"    R²: {r2:.3f}")

    return coefs


def find_crossover(merged):
    """Find the approximate crossover point where cuML breaks even."""
    print()
    print("=" * 70)
    print("CROSSOVER ANALYSIS")
    print("=" * 70)

    # Group by source count bins
    merged = merged.copy()
    bins = [0, 50, 100, 200, 500, 1000, 5000, 10000, 50000]
    merged['src_bin'] = pd.cut(merged['median_src'], bins=bins)

    print("\n  Speedup by source count bin:")
    print(f"  {'Source bin':>20}  {'N workloads':>12}  {'Median speedup':>15}  {'Winner':>10}")
    print("  " + "-" * 65)
    for bin_label, group in merged.groupby('src_bin', observed=True):
        if len(group) == 0:
            continue
        med_spd = group['speedup_312'].median()
        winner = "3.12+cuML" if med_spd > 1.0 else "3.8 CPU"
        print(f"  {str(bin_label):>20}  {len(group):>12}  {med_spd:>15.2f}x  {winner:>10}")

    # By megapixels
    mp_bins = [0, 5, 10, 20, 50, 100, 200]
    merged['mp_bin'] = pd.cut(merged['megapixels'], bins=mp_bins)

    print("\n  Speedup by image size bin:")
    print(f"  {'MP bin':>20}  {'N workloads':>12}  {'Median speedup':>15}  {'Winner':>10}")
    print("  " + "-" * 65)
    for bin_label, group in merged.groupby('mp_bin', observed=True):
        if len(group) == 0:
            continue
        med_spd = group['speedup_312'].median()
        winner = "3.12+cuML" if med_spd > 1.0 else "3.8 CPU"
        print(f"  {str(bin_label):>20}  {len(group):>12}  {med_spd:>15.2f}x  {winner:>10}")


def plot_crossover(merged):
    """Generate publication-quality crossover plot."""
    figdir = os.path.join(DATA_DIR, 'figures')
    os.makedirs(figdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Color by GPU
    gpu_colors = {}
    cmap = plt.cm.tab10
    for i, gpu in enumerate(sorted(merged['gpu_name'].unique())):
        gpu_colors[gpu] = cmap(i)

    # --- Panel 1: Speedup vs Sources ---
    ax = axes[0]
    for gpu, group in merged.groupby('gpu_name'):
        ax.scatter(group['median_src'], group['speedup_312'],
                   label=gpu.replace('NVIDIA ', ''), alpha=0.7, s=40,
                   color=gpu_colors[gpu], edgecolors='k', linewidth=0.3)

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Break-even')
    ax.set_xscale('log')
    ax.set_xlabel('Number of detected sources (median)')
    ax.set_ylabel('Speedup Py3.12/cuML over Py3.8/CPU')
    ax.set_title('(a) Speedup vs Source Count')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(merged['speedup_312'].max() * 1.1, 4))

    # --- Panel 2: Speedup vs Image Size ---
    ax = axes[1]
    for gpu, group in merged.groupby('gpu_name'):
        ax.scatter(group['megapixels'], group['speedup_312'],
                   label=gpu.replace('NVIDIA ', ''), alpha=0.7, s=40,
                   color=gpu_colors[gpu], edgecolors='k', linewidth=0.3)

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Break-even')
    ax.set_xscale('log')
    ax.set_xlabel('Image size (Megapixels)')
    ax.set_ylabel('Speedup Py3.12/cuML over Py3.8/CPU')
    ax.set_title('(b) Speedup vs Image Size')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(merged['speedup_312'].max() * 1.1, 4))

    plt.tight_layout()
    outpath = os.path.join(figdir, 'cuml_crossover_analysis')
    fig.savefig(outpath + '.pdf', dpi=300)
    fig.savefig(outpath + '.png', dpi=150)
    print(f"\n  Figures saved: {outpath}.pdf / .png")
    plt.close()


def print_detail_table(merged):
    """Print full detail table sorted by source count."""
    print()
    print("=" * 70)
    print("DETAIL TABLE (sorted by source count)")
    print("=" * 70)
    merged_sorted = merged.sort_values('median_src')
    print(f"  {'GPU':>25}  {'Camera':>12}  {'Filter':>6}  {'Size':>13}  {'MP':>6}  "
          f"{'Src':>7}  {'3.8(s)':>7}  {'3.12(s)':>7}  {'Speedup':>8}  {'Winner':>9}")
    print("  " + "-" * 120)
    for _, r in merged_sorted.iterrows():
        winner = "3.12" if r['speedup_312'] > 1.0 else "3.8"
        print(f"  {r['gpu_name'].replace('NVIDIA ', ''):>25}  {r['inmodel']:>12}  "
              f"{r['filter']:>6}  {r['image_size']:>13}  {r['megapixels']:>6.1f}  "
              f"{r['median_src']:>7.0f}  {r['median_s_38']:>7.1f}  {r['median_s_312']:>7.1f}  "
              f"{r['speedup_312']:>7.2f}x  {winner:>9}")


def main():
    print("Loading data...")
    df = load_data()
    print(f"  Total records: {len(df)}")

    merged = compute_paired_ratios(df)
    print(f"  Paired workloads (N>=10 each): {len(merged)}")

    analyze_correlations(merged)
    find_crossover(merged)
    print_detail_table(merged)
    plot_crossover(merged)


if __name__ == '__main__':
    main()
