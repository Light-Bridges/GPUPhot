#!/usr/bin/env python3
"""
Analyze Processing Times from GPUPhot **Profiler** Benchmark CSVs.

This script extends ``analyze_times_v2.py`` by filtering the data down to
the controlled benchmark images — those processed by the profiler containers
across multiple GPUs.  It identifies benchmark images automatically: an
OBLINEID that appears on **more than one GPU** must have been submitted via
the profiler, not production.

After filtering, it produces:

1. **Dataset summary** — images, GPUs, Python versions present.
2. **Per-image comparison** — same image on every GPU, side-by-side.
3. **Python 3.8 vs 3.12 paired comparison** — same GPU, same image.
4. **GPU ranking** — median latency per GPU across image sizes.
5. **Scalability** — latency vs megapixels per GPU.
6. **Figures** — publication-ready plots for the profiler benchmark.
7. **LaTeX tables** — ready for the manuscript.

It reuses the loading and cleaning functions from ``analyze_times_v2.py``
so that filtering, deduplication, and timeout logic stay consistent.

Usage
-----
::

    # Auto-detect profiler images from the latest benchmark CSV
    python benchmarks/analyze_times_v2_profiler.py \\
        --csv benchmarks/es_times_clean_20260325_230001_per_hit.csv

    # With all historical CSVs (profiler images are auto-detected)
    python benchmarks/analyze_times_v2_profiler.py --csv benchmarks/es_times_*_per_hit.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Reuse loading / cleaning from analyze_times_v2
from analyze_times_v2 import (
    load_data,
    prepare_dataframe,
    _bootstrap_ci,
    DEFAULT_TIMEOUT_S,
    BOOTSTRAP_CI,
    BOOTSTRAP_N,
    HAS_SCIPY,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

if HAS_SCIPY:
    from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIGURE_DIR = "figures_profiler"
TABLE_DIR = "tables_profiler"
MIN_GPUS_FOR_BENCHMARK = 2  # OBLINEID must appear on >= N GPUs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze GPUPhot profiler benchmark (controlled images only)"
    )
    p.add_argument(
        "--csv", nargs="+", default=["es_times_*_per_hit.csv"],
        help="Input CSV file(s). Supports wildcards.",
    )
    p.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S,
        help=f"Discard runs exceeding this limit (default {DEFAULT_TIMEOUT_S} s).",
    )
    p.add_argument(
        "--min-group", type=int, default=2,
        help="Minimum samples per group to include (default 2, lower than v2 because profiler has fewer reps).",
    )
    p.add_argument(
        "--min-gpus", type=int, default=MIN_GPUS_FOR_BENCHMARK,
        help=f"Minimum distinct GPUs an OBLINEID must appear on to be considered a benchmark image (default {MIN_GPUS_FOR_BENCHMARK}).",
    )
    p.add_argument(
        "--no-figures", action="store_true",
        help="Skip figure generation.",
    )
    p.add_argument(
        "--no-dedup", action="store_true",
        help="Skip deduplication (keep all reps, useful for profiler data with repeated runs).",
    )
    p.add_argument(
        "--outdir", type=str, default=".",
        help="Base output directory for figures/ and tables/ subdirs.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Profiler filtering
# ---------------------------------------------------------------------------
def filter_profiler_images(df: pd.DataFrame, min_gpus: int) -> pd.DataFrame:
    """Keep only OBLINEIDs that appear on multiple distinct GPUs."""
    if "oblineid" not in df.columns or "gpu_name" not in df.columns:
        print("  ERROR: missing oblineid or gpu_name columns.")
        return df

    gpu_per_oblineid = df.groupby("oblineid")["gpu_name"].nunique()
    benchmark_ids = gpu_per_oblineid[gpu_per_oblineid >= min_gpus].index
    filtered = df[df["oblineid"].isin(benchmark_ids)].copy()

    print(f"\n  Profiler filter: {len(benchmark_ids)} OBLINEIDs on >= {min_gpus} GPUs")
    print(f"  Profiler records: {len(filtered):,} (from {len(df):,} total)")

    return filtered


def prepare_dataframe_profiler(df: pd.DataFrame, timeout_s: float,
                               no_dedup: bool) -> pd.DataFrame:
    """Prepare dataframe, optionally skipping dedup for profiler repeated runs."""
    if no_dedup:
        # Minimal cleaning without dedup — profiler runs the same image many times
        if "obid" in df.columns:
            df = df.rename(columns={"obid": "oblineid"})
        for col in ("execution_time", "n_sources_detected", "naxis1", "naxis2",
                    "gpu_load", "gpu_mem_free", "gpu_mem_total", "gpu_mem_used", "gpu_temp"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

        n_raw = len(df)

        # Filter timeouts
        mask = df["execution_time"].notna() & (df["execution_time"] > 0) & (df["execution_time"] <= timeout_s)
        df = df[mask].copy()

        # Filter invalid GPUs
        INVALID_GPU_NAMES = {"Error retrieving GPU info", "No GPUs found"}
        if "gpu_name" in df.columns:
            df = df[~df["gpu_name"].isin(INVALID_GPU_NAMES)].copy()

        print(f"\n  Raw records:      {n_raw:>7,}")
        print(f"  After cleanup:    {len(df):>7,} (no dedup, timeout={timeout_s:.0f}s)")

        # Normalize camera names
        if "camera" in df.columns:
            atlas_cams = {"cam01", "cam02", "cam03", "cam04", "cam01, cam02, cam03, cam04"}
            df.loc[df["camera"].isin(atlas_cams), "camera"] = "Atlas 4 cams"

        # Derived columns
        if "naxis1" in df.columns and "naxis2" in df.columns:
            has_dims = df["naxis1"].notna() & df["naxis2"].notna()
            df.loc[has_dims, "image_size"] = (
                df.loc[has_dims, "naxis1"].astype(int).astype(str) + "x"
                + df.loc[has_dims, "naxis2"].astype(int).astype(str)
            )
            df["n_pixels"] = df["naxis1"] * df["naxis2"]
            df["megapixels"] = df["n_pixels"] / 1e6
        if "python_ver" in df.columns:
            df["python_ver_short"] = (
                df["python_ver"].astype(str).str.split("\n").str[0].str.split(" ").str[0]
            )
            df["pyver"] = df["python_ver"].str.extract(r'^(\d+\.\d+)')[0]
        return df
    else:
        return prepare_dataframe(df, timeout_s)


# ---------------------------------------------------------------------------
# 1. Dataset summary
# ---------------------------------------------------------------------------
def print_profiler_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("1. PROFILER BENCHMARK DATASET SUMMARY")
    print("=" * 70)

    n_images = df["oblineid"].nunique() if "oblineid" in df.columns else "?"
    n_gpus = df["gpu_name"].nunique()
    n_pyvers = df["python_ver_short"].nunique() if "python_ver_short" in df.columns else "?"

    print(f"  Benchmark images:  {n_images}")
    print(f"  GPUs:              {n_gpus}")
    print(f"  Python versions:   {n_pyvers}")
    print(f"  Total measurements: {len(df):,}")

    if "megapixels" in df.columns:
        print(f"\n  Image sizes:")
        for sz, grp in df.groupby("image_size"):
            mp = grp["megapixels"].iloc[0]
            n = len(grp)
            gpus = grp["gpu_name"].nunique()
            print(f"    {sz:>14} ({mp:6.1f} MP) — {n:4d} hits across {gpus} GPUs")

    if "gpu_name" in df.columns:
        print(f"\n  GPUs:")
        for gpu, grp in df.groupby("gpu_name"):
            gpu_short = gpu.replace("NVIDIA ", "").replace("GeForce ", "")
            pyvers = sorted(grp["pyver"].unique()) if "pyver" in grp.columns else []
            print(f"    {gpu_short:35s}  {len(grp):4d} hits  py{','.join(pyvers)}")


# ---------------------------------------------------------------------------
# 2. Per-image cross-GPU comparison
# ---------------------------------------------------------------------------
def cross_gpu_comparison(df: pd.DataFrame, min_group: int) -> pd.DataFrame:
    """For each image size, show median time per GPU and Python version."""
    print("\n" + "=" * 70)
    print("2. CROSS-GPU COMPARISON (same image, all GPUs)")
    print("=" * 70)

    group_cols = ["gpu_name", "python_ver_short", "image_size"]
    group_cols = [c for c in group_cols if c in df.columns]

    records = []
    for name, grp in df.groupby(group_cols):
        if len(grp) < min_group:
            continue
        t = grp["execution_time"].values
        ci_lo, ci_hi = _bootstrap_ci(t)
        mp = grp["megapixels"].iloc[0] if "megapixels" in grp.columns else 0

        records.append({
            **dict(zip(group_cols, name if isinstance(name, tuple) else (name,))),
            "megapixels": mp,
            "N": len(t),
            "median_s": np.median(t),
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "iqr_s": float(np.percentile(t, 75) - np.percentile(t, 25)),
            "mean_s": np.mean(t),
            "std_s": np.std(t, ddof=1) if len(t) > 1 else 0.0,
            "min_s": np.min(t),
            "max_s": np.max(t),
        })

    summary = pd.DataFrame(records)
    if summary.empty:
        print("  No groups with sufficient data.")
        return summary

    # Print grouped by image size
    summary = summary.sort_values(["megapixels", "median_s"])
    for sz in summary["image_size"].unique():
        sub = summary[summary["image_size"] == sz]
        mp = sub["megapixels"].iloc[0]
        print(f"\n  === {sz} ({mp:.1f} MP) ===")
        print(f"  {'GPU':>38}  {'Python':>8}  {'N':>4}  {'Median':>8}  {'95% CI':>15}  {'IQR':>6}  {'Min':>7}  {'Max':>7}")
        print("  " + "-" * 105)
        for _, r in sub.iterrows():
            gpu_short = r["gpu_name"].replace("NVIDIA ", "").replace("GeForce ", "")
            print(f"  {gpu_short:>38}  {r['python_ver_short']:>8}  {r['N']:>4}  "
                  f"{r['median_s']:>7.2f}s  [{r['ci95_lo']:>6.2f}, {r['ci95_hi']:>6.2f}]  "
                  f"{r['iqr_s']:>5.2f}  {r['min_s']:>6.2f}  {r['max_s']:>6.2f}")

    return summary


# ---------------------------------------------------------------------------
# 3. Python 3.8 vs 3.12 paired comparison
# ---------------------------------------------------------------------------
def python_version_comparison(df: pd.DataFrame, min_group: int) -> pd.DataFrame:
    """Compare py3.8 vs py3.12 on the same GPU and image."""
    print("\n" + "=" * 70)
    print("3. PYTHON 3.8 vs 3.12 PAIRED COMPARISON (same GPU, same image)")
    print("=" * 70)

    if "pyver" not in df.columns:
        print("  Missing pyver column.")
        return pd.DataFrame()

    # Only x86_64 for fair comparison (cuML not available on ARM)
    df_x86 = df[df["processor"] == "x86_64"].copy() if "processor" in df.columns else df.copy()

    mask_38 = df_x86["pyver"].isin(["3.8"])
    mask_312 = df_x86["pyver"].isin(["3.12"])

    group_cols = ["gpu_name", "image_size"]

    stats_38 = df_x86[mask_38].groupby(group_cols).agg(
        median_s_38=("execution_time", "median"),
        n_38=("execution_time", "count"),
        mp=("megapixels", "first"),
    ).reset_index()

    stats_312 = df_x86[mask_312].groupby(group_cols).agg(
        median_s_312=("execution_time", "median"),
        n_312=("execution_time", "count"),
    ).reset_index()

    merged = stats_38.merge(stats_312, on=group_cols, how="inner")
    merged = merged[(merged["n_38"] >= min_group) & (merged["n_312"] >= min_group)]

    if merged.empty:
        print("  No paired groups found.")
        return merged

    merged["speedup_312"] = merged["median_s_38"] / merged["median_s_312"]
    merged["diff_s"] = merged["median_s_38"] - merged["median_s_312"]
    merged = merged.sort_values(["gpu_name", "mp"])

    print(f"\n  Paired comparisons: {len(merged)}")
    print(f"  {'GPU':>35}  {'Image':>14}  {'MP':>6}  {'3.8(s)':>7}  {'3.12(s)':>8}  {'Speedup':>8}  {'Diff':>7}  {'Winner':>7}")
    print("  " + "-" * 110)
    for _, r in merged.iterrows():
        gpu_short = r["gpu_name"].replace("NVIDIA ", "").replace("GeForce ", "")
        winner = "3.12" if r["speedup_312"] > 1.0 else "3.8"
        print(f"  {gpu_short:>35}  {r['image_size']:>14}  {r['mp']:>5.1f}  "
              f"{r['median_s_38']:>7.2f}  {r['median_s_312']:>7.2f}  "
              f"{r['speedup_312']:>7.2f}x  {r['diff_s']:>+6.1f}s  {winner:>7}")

    # Statistical test per pair
    if HAS_SCIPY:
        print(f"\n  Mann-Whitney U tests (paired by GPU + image size):")
        print(f"  {'GPU':>35}  {'Image':>14}  {'U':>10}  {'p-value':>12}  {'Sig':>4}")
        print("  " + "-" * 85)
        for _, r in merged.iterrows():
            mask_g = (df_x86["gpu_name"] == r["gpu_name"]) & (df_x86["image_size"] == r["image_size"])
            a = df_x86[mask_g & mask_38]["execution_time"].values
            b = df_x86[mask_g & mask_312]["execution_time"].values
            if len(a) >= 2 and len(b) >= 2:
                stat, pval = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
                sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
                gpu_short = r["gpu_name"].replace("NVIDIA ", "").replace("GeForce ", "")
                print(f"  {gpu_short:>35}  {r['image_size']:>14}  {stat:>10.0f}  {pval:>12.4g}  {sig:>4}")

    return merged


# ---------------------------------------------------------------------------
# 4. GPU ranking
# ---------------------------------------------------------------------------
def gpu_ranking(df: pd.DataFrame) -> None:
    """Rank GPUs by overall median latency per image size."""
    print("\n" + "=" * 70)
    print("4. GPU RANKING (by median latency per image size)")
    print("=" * 70)

    if "megapixels" not in df.columns:
        return

    # Use py3.12 only for ranking (most representative)
    df_312 = df[df["pyver"] == "3.12"].copy() if "pyver" in df.columns else df.copy()

    for sz in sorted(df_312["image_size"].unique(),
                     key=lambda x: df_312[df_312["image_size"] == x]["megapixels"].iloc[0]):
        sub = df_312[df_312["image_size"] == sz]
        mp = sub["megapixels"].iloc[0]
        ranking = sub.groupby("gpu_name")["execution_time"].agg(["median", "count"]).sort_values("median")

        print(f"\n  === {sz} ({mp:.1f} MP) — py3.12 ===")
        print(f"  {'Rank':>5}  {'GPU':>38}  {'Median(s)':>10}  {'N':>4}")
        print("  " + "-" * 65)
        for rank, (gpu, row) in enumerate(ranking.iterrows(), 1):
            gpu_short = gpu.replace("NVIDIA ", "").replace("GeForce ", "")
            print(f"  {rank:>5}  {gpu_short:>38}  {row['median']:>9.2f}s  {int(row['count']):>4}")


# ---------------------------------------------------------------------------
# 5. Scalability: latency vs megapixels
# ---------------------------------------------------------------------------
def scalability_analysis(df: pd.DataFrame) -> None:
    """Analyze how latency scales with image size per GPU."""
    print("\n" + "=" * 70)
    print("5. SCALABILITY: LATENCY vs IMAGE SIZE")
    print("=" * 70)

    if not HAS_SCIPY or "megapixels" not in df.columns:
        print("  [Skipped]")
        return

    for pyv in sorted(df["pyver"].unique()) if "pyver" in df.columns else ["?"]:
        df_py = df[df["pyver"] == pyv] if pyv != "?" else df

        print(f"\n  --- Python {pyv} ---")
        for gpu, grp in df_py.groupby("gpu_name"):
            if len(grp) < 5:
                continue
            rho, pval = sp_stats.spearmanr(grp["megapixels"], grp["execution_time"])
            gpu_short = gpu.replace("NVIDIA ", "").replace("GeForce ", "")
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
            print(f"    {gpu_short:>35}  rho={rho:.3f}  p={pval:.2e}  {sig}  (N={len(grp)})")


# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
def generate_profiler_figures(df: pd.DataFrame, merged_py: pd.DataFrame,
                              outdir: str) -> None:
    """Generate publication-quality profiler benchmark figures."""
    if not HAS_PLOT:
        print("\n  [Figures skipped: matplotlib/seaborn not installed]")
        return

    figdir = Path(outdir) / FIGURE_DIR
    figdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    print(f"\n  Saving figures to {figdir}/")

    # Shorten GPU names globally
    df = df.copy()
    df["gpu_short"] = df["gpu_name"].str.replace("NVIDIA ", "").str.replace("GeForce ", "")

    # -- Fig 1: Latency vs megapixels per GPU (lines, py3.12)
    df_312 = df[df["pyver"] == "3.12"] if "pyver" in df.columns else df
    if len(df_312) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        for gpu, grp in df_312.groupby("gpu_short"):
            stats = grp.groupby("megapixels").agg(
                median_t=("execution_time", "median"),
                q25=("execution_time", lambda x: np.percentile(x, 25)),
                q75=("execution_time", lambda x: np.percentile(x, 75)),
            ).reset_index().sort_values("megapixels")
            ax.errorbar(
                stats["megapixels"], stats["median_t"],
                yerr=[stats["median_t"] - stats["q25"], stats["q75"] - stats["median_t"]],
                fmt="o-", label=gpu, capsize=3, markersize=5,
            )
        ax.set_xlabel("Image Size (Megapixels)")
        ax.set_ylabel("Median End-to-End Latency (s)")
        ax.set_title("Profiler Benchmark: Latency Scaling (Python 3.12)")
        ax.legend(title="GPU", fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.tight_layout()
        fig.savefig(figdir / "profiler_latency_vs_mp.pdf", dpi=300)
        fig.savefig(figdir / "profiler_latency_vs_mp.png", dpi=150)
        plt.close(fig)
        print(f"    profiler_latency_vs_mp.pdf")

    # -- Fig 2: Boxplot per image size, coloured by GPU (py3.12)
    if len(df_312) > 0:
        plot_df = df_312.copy()
        plot_df["mp_label"] = plot_df["megapixels"].round(1).astype(str) + " MP"
        # Sort by megapixels
        mp_order = sorted(plot_df["mp_label"].unique(),
                          key=lambda x: float(x.replace(" MP", "")))

        fig, ax = plt.subplots(figsize=(14, 7))
        sns.boxplot(
            data=plot_df, x="mp_label", y="execution_time", hue="gpu_short",
            order=mp_order, showfliers=False, ax=ax,
        )
        ax.set_ylabel("End-to-End Latency (s)")
        ax.set_xlabel("Image Size")
        ax.set_title("Profiler Benchmark: Time Distribution by Image Size (Python 3.12)")
        ax.legend(title="GPU", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        fig.savefig(figdir / "profiler_boxplot_by_size.pdf", dpi=300)
        fig.savefig(figdir / "profiler_boxplot_by_size.png", dpi=150)
        plt.close(fig)
        print(f"    profiler_boxplot_by_size.pdf")

    # -- Fig 3: Python 3.8 vs 3.12 speedup bar chart
    if not merged_py.empty:
        plot_df = merged_py.copy()
        plot_df["gpu_short"] = plot_df["gpu_name"].str.replace("NVIDIA ", "").str.replace("GeForce ", "")
        plot_df["label"] = plot_df["gpu_short"] + "\n" + plot_df["image_size"]

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#2ecc71" if s > 1 else "#e74c3c" for s in plot_df["speedup_312"]]
        bars = ax.barh(plot_df["label"], plot_df["speedup_312"], color=colors, edgecolor="k", linewidth=0.3)
        ax.axvline(x=1.0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("Speedup (Py3.12 / Py3.8)")
        ax.set_title("Python 3.12 vs 3.8 Speedup by GPU and Image Size")
        ax.invert_yaxis()

        # Annotate
        for bar, spd in zip(bars, plot_df["speedup_312"]):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{spd:.2f}x", va="center", fontsize=8)

        plt.tight_layout()
        fig.savefig(figdir / "profiler_py38_vs_312_speedup.pdf", dpi=300)
        fig.savefig(figdir / "profiler_py38_vs_312_speedup.png", dpi=150)
        plt.close(fig)
        print(f"    profiler_py38_vs_312_speedup.pdf")

    # -- Fig 4: Heatmap median latency (image size x GPU), py3.12
    if len(df_312) > 0 and "megapixels" in df_312.columns:
        pivot = df_312.pivot_table(
            values="execution_time", index="image_size", columns="gpu_short",
            aggfunc="median",
        )
        if not pivot.empty:
            # Sort rows by megapixels
            mp_map = df_312.groupby("image_size")["megapixels"].first()
            sorted_idx = mp_map.loc[mp_map.index.isin(pivot.index)].sort_values()
            pivot = pivot.reindex(sorted_idx.index)

            fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.8)))
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                        cbar_kws={"label": "Median Latency (s)"})
            ax.set_title("Profiler Benchmark: Median Latency (s) — Python 3.12")
            ax.set_ylabel("Image Size")
            plt.tight_layout()
            fig.savefig(figdir / "profiler_heatmap.pdf", dpi=300)
            fig.savefig(figdir / "profiler_heatmap.png", dpi=150)
            plt.close(fig)
            print(f"    profiler_heatmap.pdf")


# ---------------------------------------------------------------------------
# 7. LaTeX tables
# ---------------------------------------------------------------------------
def export_profiler_tables(summary: pd.DataFrame, merged_py: pd.DataFrame,
                           outdir: str) -> None:
    """Export profiler-specific LaTeX tables."""
    tbldir = Path(outdir) / TABLE_DIR
    tbldir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving LaTeX tables to {tbldir}/")

    # Table 1: Cross-GPU comparison
    if not summary.empty:
        export_cols = [c for c in ["gpu_name", "python_ver_short", "image_size",
                                    "megapixels", "N", "median_s", "ci95_lo",
                                    "ci95_hi", "iqr_s", "min_s", "max_s"]
                       if c in summary.columns]
        tex = summary[export_cols].to_latex(
            index=False, float_format="%.2f", na_rep="--",
            caption="Profiler benchmark: robust performance statistics per GPU, "
                    "Python version, and image size. Median latency with 95\\% "
                    "bootstrap CI and IQR.",
            label="tab:profiler_performance",
        )
        outpath = tbldir / "table_profiler_performance.tex"
        outpath.write_text(tex)
        print(f"    {outpath.name}")

    # Table 2: Python 3.8 vs 3.12 speedup
    if not merged_py.empty:
        export_cols = [c for c in ["gpu_name", "image_size", "mp",
                                    "n_38", "median_s_38", "n_312",
                                    "median_s_312", "speedup_312", "diff_s"]
                       if c in merged_py.columns]
        tex = merged_py[export_cols].to_latex(
            index=False, float_format="%.2f", na_rep="--",
            caption="Python 3.8 vs 3.12 comparison on controlled benchmark images. "
                    "Speedup $> 1$ indicates Python 3.12 (with cuML) is faster.",
            label="tab:profiler_py_comparison",
        )
        outpath = tbldir / "table_profiler_py_comparison.tex"
        outpath.write_text(tex)
        print(f"    {outpath.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("GPUPhot Profiler Benchmark Analysis")
    print("=" * 70)

    # Load (reuse from v2)
    df = load_data(args.csv)

    # Clean
    df = prepare_dataframe_profiler(df, args.timeout, args.no_dedup)
    print(f"  Final dataset:    {len(df):>7,} records")

    # Filter to profiler images only
    df = filter_profiler_images(df, args.min_gpus)
    if df.empty:
        print("\n  No profiler benchmark images found. Exiting.")
        return

    # Add derived columns if missing
    if "megapixels" not in df.columns and "n_pixels" in df.columns:
        df["megapixels"] = df["n_pixels"] / 1e6
    if "pyver" not in df.columns and "python_ver" in df.columns:
        df["pyver"] = df["python_ver"].str.extract(r'^(\d+\.\d+)')[0]

    # 1. Summary
    print_profiler_summary(df)

    # 2. Cross-GPU comparison
    summary = cross_gpu_comparison(df, args.min_group)

    # 3. Python 3.8 vs 3.12
    merged_py = python_version_comparison(df, args.min_group)

    # 4. GPU ranking
    gpu_ranking(df)

    # 5. Scalability
    scalability_analysis(df)

    # 6. Figures
    if not args.no_figures:
        generate_profiler_figures(df, merged_py, args.outdir)

    # 7. LaTeX tables
    export_profiler_tables(summary, merged_py, args.outdir)

    print("\n" + "=" * 70)
    print("Profiler analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
