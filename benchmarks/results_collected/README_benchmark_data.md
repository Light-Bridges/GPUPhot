# Benchmark Data — GPUPhot Profiler Analysis

## Data Collection Dates
- **Timing (Elastic)**: 2026-03-25 23:00 — 2026-03-26 03:38 UTC (main run, 7 machines)
- **Timing (Elastic)**: 2026-03-26 16:18 — 2026-03-26 19:59 UTC (hp3 L40S re-run after OOM)
- **Memory (nsys)**: 2026-03-26 (two runs: 09:54 and 15:37 UTC, 7 machines)

## Clean Data (Profiler — controlled environment)

These files contain data from **controlled benchmark runs** using the profiler
containers. Each image is processed in isolation (1 process per GPU, no other
workloads). Workers on production machines were stopped during benchmarks.

### Files

| File | Rows | Description |
|------|------|-------------|
| `profiler_timing_elastic_20260325.csv` | 3630 | End-to-end execution times (main run, 7 machines). Contains both profiler and production hits. **Filter by `group_key` on >= 2 GPUs to isolate profiler images** (10 OBLINEIDs). |
| `profiler_timing_elastic_hp3_20260326_per_hit.csv` | 2670 | hp3 L40S re-run timing (870 L40S hits). Same format, separate because hp3 was re-run after OOM. |
| `profiler_nsys_memory_20260326.csv` | 431 | Raw per-run peak GPU memory from nsys sqlite files. 7 GPUs, py3.8/3.12. Column `capture_complete=True` means nsys captured >= 5000 events (reliable). |
| `profiler_nsys_memory_summary_20260326.csv` | 61 | Aggregated (median) peak GPU memory per (machine, GPU, Python, camera). Only complete captures. |
| `profiler_unified_timing_memory_20260327.csv` | 64 | **Unified table**: median timing + median peak VRAM per (GPU, Python, image_size). Merges timing and memory sources. |
| `profiler_paired_comparison_20260327.csv` | 30 | **Paired py3.8 vs py3.12 comparison**: speedup, memory saving, per GPU and image size. Ready for manuscript tables. |

### Benchmark Images

10 FITS images selected to cover the full megapixel range:

| # | Camera | Image Size | MP | Filter | Object |
|---|--------|------------|-----|--------|--------|
| 1 | iKon936-1 | 2048x2048 | 4.2 | SDSSg | QSO0957+561 |
| 2 | iKon936-1 | 2048x2048 | 4.2 | Lum | C2025A6 |
| 3 | QHY600-3 | 3191x2129 | 6.8 | Lum | C2025R2 |
| 4 | QHY600-4 | 4787x3193 | 15.3 | SDSSg | WASP-43-b |
| 5 | QHY600-4 | 4787x3193 | 15.3 | Ha | NGC2903 |
| 6 | QHY411-1 | 7100x5325 | 37.8 | Lum | 2012QD8 |
| 7 | QHY411-1 | 7100x5325 | 37.8 | SDSSi | GaiaDR3... |
| 8 | QHY411-1 | 14200x10650 | 151.2 | Lum | 2025PR1 |
| 9 | QHY411-3 | 14200x10650 | 151.2 | Lum | 24P |
| 10 | QHY411-3 | 14200x10650 | 151.2 | SDSSr | M81 |

### Benchmark Configuration
- **Clean timing**: 2 warmup + 10 reps per image, per profiler (py3.12 and py3.8)
- **nsys profiling**: 1 warmup + 2 nsys reps per image, per profiler
- **Workers stopped** on production machines during benchmarks (`--stop-workers`)
- **Both profilers run sequentially** per machine (py3.12 first, then py3.8)

### GPUs Tested

| Machine | GPU | VRAM | Type | Notes |
|---------|-----|------|------|-------|
| lenovo_tttserver | A100-SXM4-80GB | 80 GB | Production | |
| azken | H100 PCIe | 80 GB | Production | nsys py3.8 incomplete for > 4.2 MP |
| hp3 | L40S | 48 GB | Production | OOM on first run; re-run 26 Mar |
| ttt_server | RTX 3090 | 24 GB | Dev | |
| ttt1 | RTX 3060 | 12 GB | Dev | |
| local | RTX 3050 Ti Laptop | 4 GB | Laptop | VRAM-limited for large images |
| jetson_local | Orin Super | 8 GB shared | Edge | py3.8 = Python 3.10 (JetPack 6.x) |
| jetson_orin | Orin NX 8GB | 8 GB shared | Edge | Max image: 6.8 MP (15.3 MP causes OOM). No nsys memory data (Tegra). Timing: `jetson_orin_8gb_timing_20260329.csv` |

### Known Data Issues
1. **H100 py3.8 nsys**: RESOLVED (2026-03-29). First run (26 Mar) had incomplete
   captures (< 5000 events). Re-profiled on 29 Mar with verified clean GPU (4 MiB
   baseline). All 5 image sizes now have complete captures. The consolidated CSVs
   contain the clean run data. Note: nsys measures per-process allocations, so
   other GPU processes do not contaminate peak memory readings (verified by
   comparing dirty vs clean GPU runs — identical results).
2. **hp3 L40S timing**: First run (25 Mar) had OOM kills on py3.8 for 37.8 and 151.2 MP.
   Separate re-run (26 Mar) in `profiler_timing_elastic_hp3_20260326_per_hit.csv`.
   The unified/paired tables already include both sources.
3. **RTX 3050 Ti**: Cannot process images > ~6 MP on py3.12 via nsys (OOM with nsys overhead).
   Clean timing (without nsys) works for larger images.
4. **Jetson py "3.8"**: Actually Python 3.10 (JetPack 6.x minimum). cuML unavailable on ARM,
   so both versions use CPU crossmatch — no cuML speedup expected.
5. **RTX 3090 nsys on VAST**: Container IDs (`24acfc77e2de`, `dd46e42ec08f`) could not be
   mapped to py3.8/3.12 because they are from ttt_server which shares VAST with lenovo.
   Memory values for RTX 3090 come from the iKon936 peak pattern matching.

## Dirty Data (Production — Elastic historical CSVs)

Files matching `es_times_*_per_hit.csv` in `benchmarks/` contain production
execution times collected from Elastic over various time periods. These include:

- Multiple concurrent workers processing images on the same GPU
- Variable GPU load, temperature, and memory pressure
- Mixed image types (ATLAS 4-cam mosaics, TTT/TST single-camera)
- Background processes (other Docker containers, system services)

These are useful for production performance characterization but **not for
controlled GPU comparisons**. Use `analyze_times_v2.py` for these.

## How to Reproduce

### 1. Clean timing benchmark (end-to-end latency)
```bash
# From project root, stop workers and run 2 warmup + 10 reps on all machines
cd /home/slemes/PycharmProjects/GPUPhotFinal
bash dev/run_benchmark_all_machines.sh --stop-workers --warmup 2 --reps 10 --nsys 0

# Collect results from Elastic (adjust timestamps)
python benchmarks/collect_times_from_elastic.py \
    --time-from "2026-03-25T23:00:00" --time-to "2026-03-26T04:00:00" \
    --out benchmarks/results_collected/profiler_timing_elastic_YYYYMMDD.csv
```

### 2. nsys memory profiling (peak VRAM)
```bash
# Run with nsys (1 warmup + 2 nsys reps, no clean reps)
bash dev/run_benchmark_all_machines.sh --stop-workers --warmup 1 --reps 0 --nsys 2

# sqlite files land in /app/profiling_results inside each container
# On VAST machines: /mnt/vast/samueltest/gpuphot/profiling_test_results/
# On Jetsons: /home/jetson/GPUPhot/benchmarks/benchmark_results/
# On local: benchmarks/benchmark_results/

# Extract peak memory from sqlite files (run on each machine or remotely):
# See nsys_memory_analysis.py pattern in this session's history
```

### 3. Analysis
```bash
# General production analysis (all Elastic CSVs)
python benchmarks/analyze_times_v2.py --csv benchmarks/es_times_*_per_hit.csv

# Profiler-only analysis (filters to multi-GPU OBLINEIDs)
python benchmarks/analyze_times_v2_profiler.py \
    --csv benchmarks/results_collected/profiler_timing_elastic_*.csv --no-dedup

# cuML crossover analysis
python benchmarks/analyze_cuml_crossover.py

# nsys per-stage profiling (NVTX events)
python benchmarks/analyze_profiling.py --sqlite benchmarks/benchmark_results/*.sqlite
```

## cuML vs cKDTree Crossmatch Study

### Main Conclusion

**cuML does NOT improve crossmatch performance in the GPUPhot pipeline.**

The Python 3.12 improvement over 3.8 comes from **interpreter memory optimization
and updated dependencies (CuPy 14, NumPy 2.0)**, not from cuML crossmatch.

### Why cuML is slower for 2D crossmatch

RAPIDS cuML `NearestNeighbors` uses **brute-force O(N^2)** for low-dimensional
data (2D sky coordinates). SciPy `cKDTree` is O(N log N). The real pipeline
makes **multiple crossmatch calls per image** (detection, astrometry, catalog),
each paying cuML initialization overhead. This compounds the algorithmic
disadvantage.

### Evidence

#### 1. Synthetic Benchmark (2026-03-28)

File: `cuml_crossover_synthetic_all_gpus_20260328.csv` — 60 rows, 6 GPUs

In isolation (single call), cuML wins in a narrow window on powerful GPUs:

| GPU | cuML wins from | Peak speedup | Loses again at |
|-----|----------------|--------------|----------------|
| L40S | 2K sources | 5.8x @ 20K | No (to 100K) |
| RTX 3090 | 2K sources | 3.2x @ 20K | No (to 100K) |
| H100 PCIe | 5K sources | 4.6x @ 20K | No (to 100K) |
| A100-SXM4 | 5K sources | 3.5x @ 20K | Advantage shrinks |
| RTX 3060 | 5K sources | 1.9x @ 10K | **50K sources** |
| RTX 3050 Ti | **Never** | — | CPU always wins |

#### 2. Real-Image Ablation: RTX 3050 Ti (2026-03-27)

File: `cuml_ablation_local_rtx3050ti_20260327.csv`

5 images, 3 reps, < 1K sources. cuML always slower (6-16% penalty).

#### 3. Real-Image Ablation: A100 (2026-03-29) — DECISIVE TEST

File: `cuml_ablation_a100_20260329.csv` — 10 images, 112 to 18,888 sources

**cuML is slower for ALL images**, even with 18,888 sources where the synthetic
test predicted 2x GPU speedup. The real pipeline overhead dominates:

| Image | Sources | cuML (s) | cKDTree (s) | cuML penalty |
|-------|---------|----------|-------------|--------------|
| iKon936 SDSSg | 412 | 8.2 | 4.4 | +87% |
| QHY600-4 SDSSg | 218 | 29.3 | 10.2 | +187% |
| QHY411-1 SDSSi | 247 | 24.8 | 12.5 | +98% |
| QHY411-3 SDSSr | 14,241 | 334.5 | 94.0 | +256% |
| QHY411-3 Lum | 18,888 | 273.8 | 104.4 | +162% |

### Implication for Manuscript

The narrative should NOT attribute py3.12 improvement to cuML. The correct framing:

> Python 3.12 with CuPy 14 and NumPy 2.0 reduces **peak GPU memory usage by
> 3-15%** compared to Python 3.8 with CuPy 12. This enables higher image
> concurrency per GPU. End-to-end latency is slightly higher in py3.12 due to
> cuML crossmatch overhead, but aggregate throughput improves thanks to the
> increased concurrency. Disabling cuML and using cKDTree exclusively would
> further improve py3.12 latency.

### How to Reproduce

```bash
# Synthetic crossover benchmark (inside py3.12 profiler container):
docker exec <container> python3 /app/benchmarks/benchmark_cuml_crossover.py
# Script is at: benchmarks/benchmark_cuml_crossover.py
# Outputs CSV to stdout: N,cpu_ms,gpu_e2e_ms,speedup_e2e,winner

# Real-image A/B test: create fake cuml module to block import
docker exec <container> bash -c '
    mkdir -p /tmp/nocuml/cuml/neighbors
    echo "raise ImportError(\"cuML disabled\")" > /tmp/nocuml/cuml/__init__.py
'
# Run WITH cuML (normal):
docker exec <container> python3 profile_process_image.py <image> <instrument>
# Run WITHOUT cuML:
docker exec -e PYTHONPATH=/tmp/nocuml <container> python3 profile_process_image.py <image> <instrument>
```

## Analysis Scripts

| Script | Input | Output |
|--------|-------|--------|
| `analyze_times_v2.py` | `es_times_*_per_hit.csv` | Production performance stats, figures, LaTeX tables |
| `analyze_times_v2_profiler.py` | `profiler_timing_elastic_*.csv` | Profiler-only: cross-GPU, py3.8 vs 3.12, GPU ranking |
| `analyze_cuml_crossover.py` | `es_times_*_per_hit.csv` | cuML crossover from production data (historical) |
| `analyze_profiling.py` | `*.sqlite` (nsys) | Per-stage NVTX timing breakdown (GPU/CPU/IO stages) |
| `analyze_times.py` | `es_times_*_per_hit.csv` | Legacy analysis script (predecessor of v2) |
| `benchmark_cuml_crossover.py` | (synthetic, no input) | cKDTree vs cuML crossover per GPU. Run inside container |
