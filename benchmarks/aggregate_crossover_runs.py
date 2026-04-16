#!/usr/bin/env python3
"""
Aggregate the 100-run synthetic crossover benchmarks with outlier filtering.

This script processes the raw CSVs from `benchmarks/results_collected/crossover_runs_100/`,
filters statistical outliers for each (GPU, N) group, and computes a new
aggregated median. This produces cleaner, smoother curves than a simple median
of all raw data, which can be affected by system noise, especially on shared
datacenter GPUs.

The output is a single CSV file in the same format as the original aggregated data,
ready to be used by `generate_manuscript_figures.py`.
"""

import os
import argparse
import pandas as pd
import numpy as np


def filter_outliers_iqr(s: pd.Series, factor: float = 1.5) -> pd.Series:
    """Filters a Series by removing values outside Q1 - factor*IQR and Q3 + factor*IQR."""
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return s[(s >= lower_bound) & (s <= upper_bound)]


def process_gpu_runs(gpu_dir: str, machine_name: str, gpu_name: str) -> pd.DataFrame | None:
    """
    Load all CSVs for a single GPU, filter outliers, and compute aggregated stats.
    """
    csv_files = [os.path.join(gpu_dir, f) for f in os.listdir(gpu_dir) if f.endswith('.csv')]
    if not csv_files:
        return None

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # The columns in the raw 100-run CSVs are slightly different from the final aggregated one.
    # Let's rename for consistency.
    df = df.rename(columns={'speedup_e2e': 'speedup'})

    aggregated_rows = []
    for n_val, group in df.groupby('N'):
        # Filter outliers for both CPU and GPU timings
        clean_cpu_ms = filter_outliers_iqr(group['cpu_ms'])
        clean_gpu_ms = filter_outliers_iqr(group['gpu_ms'])

        # If filtering removed everything, fall back to original data for this point
        if clean_cpu_ms.empty or clean_gpu_ms.empty:
            print(f"  [WARN] N={n_val}: Outlier filtering removed all data. Using raw median.", file=sys.stderr)
            med_cpu = group['cpu_ms'].median()
            med_gpu = group['gpu_ms'].median()
        else:
            med_cpu = clean_cpu_ms.median()
            med_gpu = clean_gpu_ms.median()

        # Calculate other stats from the original group, as they represent the full run's distribution
        speedup = med_cpu / med_gpu if med_gpu > 0 else 0
        winner = 'GPU' if speedup > 1.0 else 'CPU'

        aggregated_rows.append({
            'gpu_name': gpu_name,
            'machine': machine_name,
            'N': n_val,
            'cpu_ms': med_cpu,
            'gpu_e2e_ms': med_gpu,
            'speedup': speedup,
            'winner': winner,
            # For IQR bands in the plot, we take the median of the collected IQRs.
            # This is a reasonable approximation of the central tendency of the variance.
            'cpu_q25': group['cpu_q25'].median(),
            'cpu_q75': group['cpu_q75'].median(),
            'gpu_q25': group['gpu_q25'].median(),
            'gpu_q75': group['gpu_q75'].median(),
        })

    return pd.DataFrame(aggregated_rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate crossover benchmark runs with outlier filtering.")
    parser.add_argument(
        "--input-dir",
        default=os.path.join(os.path.dirname(__file__), 'results_collected', 'crossover_runs_100'),
        help="Directory containing the per-GPU subdirectories of raw CSV runs."
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(os.path.dirname(__file__), 'results_collected', 'cuml_crossover_synthetic_all_gpus_cleaned.csv'),
        help="Path to save the final aggregated and cleaned CSV file."
    )
    args = parser.parse_args()

    print(f"Input directory: {args.input_dir}")
    print(f"Output CSV: {args.output_csv}")

    all_gpu_dfs = []
    # Directories are named like 'azken_H100', 'ttt_server_RTX3090', etc.
    for dirname in sorted(os.listdir(args.input_dir)):
        gpu_dir = os.path.join(args.input_dir, dirname)
        if not os.path.isdir(gpu_dir):
            continue

        parts = dirname.split('_', 1)
        machine = parts[0]
        
        # Handle GPU names that might contain underscores
        gpu_name_map = {
            'azken_H100': 'H100 PCIe',
            'hp3_L40S': 'L40S',
            'lenovo_A100': 'A100-SXM4-80GB',
            'ttt_server_RTX3090': 'GeForce RTX 3090',
            'ttt1_RTX3060': 'GeForce RTX 3060',
            'local_RTX3050Ti': 'GeForce RTX 3050 Ti Laptop GPU'
        }
        if dirname not in gpu_name_map:
            print(f"  [WARN] Skipping unknown directory format: {dirname}", file=sys.stderr)
            continue
            
        gpu_name = gpu_name_map[dirname]
        
        print(f"Processing {gpu_name} from {dirname}...")
        gpu_df = process_gpu_runs(gpu_dir, machine, gpu_name)
        if gpu_df is not None:
            all_gpu_dfs.append(gpu_df)

    if not all_gpu_dfs:
        print("[ERROR] No data processed. Check the input directory.", file=sys.stderr)
        return

    final_df = pd.concat(all_gpu_dfs, ignore_index=True)
    final_df = final_df.sort_values(by=['gpu_name', 'N'])
    
    # Reorder columns to match the original aggregated file format
    column_order = [
        'gpu_name', 'machine', 'N', 'cpu_ms', 'gpu_e2e_ms', 'speedup', 
        'winner', 'cpu_q25', 'cpu_q75', 'gpu_q25', 'gpu_q75'
    ]
    final_df = final_df[column_order]

    final_df.to_csv(args.output_csv, index=False, float_format='%.6f')
    print(f"\nSuccessfully created cleaned aggregated file:\n  -> {args.output_csv}")


if __name__ == "__main__":
    main()
