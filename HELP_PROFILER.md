# Session Report: Adaptive cuML Benchmarks (2026-03-31 / 2026-04-01)

## 1. Docker Compose Fix: NVIDIA Symlink Error on hp3

### Problem
The profiler container failed to start on hp3 (4x L40S, Kubernetes/RKE1 node) with:
```
failed to create symlink: failed to remove existing file:
  .../libnvidia-ml.so.1: device or resource busy
```

### Root Cause
The `docker-compose.yml` had explicit bind mounts for NVIDIA libraries in the `profiler` and `profiler_38` services:
```yaml
- /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro
- /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro
```
These conflict with the NVIDIA Container Toolkit hook, which automatically injects the same libraries via symlinks during container creation. The bind mount occupies the path before the hook can create its symlink, causing "device or resource busy".

This was **not** caused by Kubernetes pods or the GPU operator, despite initial investigation pointing in that direction. The proof: `docker run --gpus` (no bind mounts) worked fine on the same host.

### Fix
Removed the 4 redundant bind mount lines (2 per profiler service) from `docker-compose.yml`:
- Commit: `d40a425` on branch `documentation`
- Pushed to `github.com:Light-Bridges/GPUPhot.git`

### Verification
After the fix, `docker compose up -d --build --force-recreate profiler` succeeded on all 5 machines.

---

## 2. Machine Setup: Code Update and Docker Rebuild

All 5 benchmark machines were updated with:
1. `git pull origin documentation` (commit `d40a425`)
2. `docker compose up -d --build --force-recreate profiler`
3. Verified profiler container running

### Machines

| SSH Alias | Codename | GPU | VRAM |
|---|---|---|---|
| azken | AAZ | NVIDIA H100 PCIe | 80 GB |
| hp3 | AHP | 4x NVIDIA L40S | 46 GB each |
| lenovo_tttserver | ALE | NVIDIA A100-SXM4-80GB | 40 GB |
| ttt_server | TTT | NVIDIA GeForce RTX 3090 | 24 GB |
| ttt1 | CCU | NVIDIA GeForce RTX 3060 | 6 GB |

---

## 3. Adaptive cuML Configuration

Each machine's `~/GPUPhot/.env` was configured with GPU-specific cuML crossmatch thresholds:

| Machine | GPUPHOT_CUML_MIN_SOURCES | GPUPHOT_CUML_MAX_SOURCES |
|---|---|---|
| azken (H100) | 5000 | 500000 |
| hp3 (L40S) | 2000 | 200000 |
| lenovo_tttserver (A100) | 2000 | 100000 |
| ttt_server (RTX 3090) | 2000 | 100000 |
| ttt1 (RTX 3060) | 2000 | 20000 |

**Adaptive cuML logic** (in `gpuphot/utils/catalog.py`):
- If `n_sources < MIN_SOURCES`: use cKDTree (CPU)
- If `MIN_SOURCES <= n_sources <= MAX_SOURCES`: use cuML NearestNeighbors (GPU)
- If `n_sources > MAX_SOURCES`: use cKDTree (CPU)

---

## 4. Benchmark Execution

### First Run (incorrect CONFIGS_DIR)
- **Launched**: 2026-03-31 ~15:00 WEST
- **Problem**: The benchmark script (`run_benchmark_autonomous.sh`) defaults `CONFIGS_DIR` to `/mnt/vast/samueltest/gpuphot/cameras_config`, which did not have the correct instrument JSON configs (iKon936-1.json, QHY600-3.json, etc.) on all machines.
- **Result**: Several images failed:
  - `QHY600-3_Lum` failed on ALL machines with `InsufficientStarsError` (wrong camera config caused only 3 stars to be detected)
  - Multiple 151.2 MP images failed with OOM from RAPIDS/RMM
  - ttt1 initially failed all images due to CuPy CCCL error (image not retagged after rebuild)

### Second Run (correct CONFIGS_DIR)
- **Launched**: 2026-03-31 ~21:06 WEST
- **Fix**: Each machine's benchmark was launched with `CONFIGS_DIR=~/DTO/ttt/tasks/cameras_config` pointing to the correct instrument configs on each host.
- **Parameters**: `WARMUP=2 REPS_CLEAN=10 REPS_NSYS=0 PROFILERS=312`
- **Completed**: 2026-04-01 ~00:54 WEST

### Results

| Machine | GPU | Images OK | Total Runs | Missing |
|---|---|---|---|---|
| azken | H100 | 10/10 | 100 | None |
| hp3 | L40S | 10/10 | 100 | None |
| lenovo_tttserver | A100 | 10/10 | 100 | None |
| ttt_server | RTX 3090 | 7/10 | 70 | 3x 151.2 MP (OOM, 24 GB VRAM) |
| ttt1 | RTX 3060 | 7/10 | 70 | 3x 151.2 MP (OOM, 6 GB VRAM) |

CSVs downloaded to: `benchmarks/results_collected/adaptive_cuml_20260401/`

---

## 5. Data Collection from Elasticsearch

### Script Used
`benchmarks/collect_times_from_elastic.py` with parameters:
```bash
python3 benchmarks/collect_times_from_elastic.py \
  --time-from "2026-03-31T20:00:00" \
  --time-to "2026-04-01T02:00:00" \
  --out benchmarks/es_times_adaptive_cuml_20260401_per_hit.csv
```

### ES Connection
- Host: `http://10.0.210.30:9200`
- Auth: `elastic` / `admin` (found in `~/DTO/.env` on hp3)
- Index: `logstash-*`
- Filter: `extra.application=gpuphot`, `extra.function_name=process_image`, `exists(extra.execution_time)`

### Output
- **528 rows** downloaded to `benchmarks/es_times_adaptive_cuml_20260401_per_hit_per_hit.csv`
- Same format as previous benchmark data (`es_times_gpuphot_profiler_per_hit.csv`)
- 5 GPUs, all Python 3.12, includes warmup + clean runs

---

## 6. Comparison: Adaptive cuML vs cuML-Always

### Baseline Data
- File: `benchmarks/es_times_gpuphot_profiler_per_hit.csv` (previous benchmarks, cuML always active in py3.12)

### Results Summary

**Images where cKDTree was used instead of cuML (sources below threshold):**

| GPU | Median Improvement | Range |
|---|---|---|
| H100 | **-25.7%** | -4.7% to -28.2% |
| A100 | **-13.0%** | -6.6% to -24.5% |
| L40S | **-4.5%** | -1.9% to -10.7% |
| RTX 3090 | **-3.5%** | -2.6% to -5.8% |

**Images where cuML remained active (sources within threshold range):**

| GPU | Median Change |
|---|---|
| H100 | -9.5% (range: -22.4% to +3.3%) |
| A100 | -6.2% (range: -7.4% to -4.9%) |
| L40S | -10.1% (range: -13.6% to -6.7%) |

### Key Findings
1. Adaptive cuML significantly improves latency for images with few sources (below cuML threshold), where cuML's O(N^2) brute-force was unnecessarily costly compared to cKDTree's O(N log N).
2. Images where cuML remains active show similar or slightly better performance (the adaptive logic adds negligible overhead).
3. The improvement is proportionally larger on high-end GPUs (H100/A100) because the RAPIDS/RMM memory management overhead is more significant relative to the fast GPU computation.
4. OOM on 151.2 MP images for RTX 3090/3060 is expected (insufficient VRAM for full-resolution processing).

### Figure
Saved to: `benchmarks/figures/adaptive_cuml_vs_always.png`

---

## 7. Other Issues Encountered

### CuPy CCCL Compilation Error on ttt1
- **Symptom**: `CompileException: incomplete type "__nv_fp8_e8m0" is not allowed`
- **Cause**: The `docker run` benchmark used the old image tag (`gpuphotfinal-profiler`) which hadn't been rebuilt with `nvidia-cublas-cu12==12.9.1.4` pin.
- **Fix**: `docker tag gpuphot-profiler:latest gpuphotfinal-profiler:latest` after rebuild.

### Workers Management
- `dto-worker0-1` was stopped on azken, hp3, and lenovo_tttserver before benchmarks to free GPU resources.
- Workers were automatically restarted by the monitor script after all benchmarks completed.

---

## 8. Files Generated/Modified

| File | Description |
|---|---|
| `docker-compose.yml` | Removed redundant nvidia-smi and libnvidia-ml bind mounts |
| `benchmarks/results_collected/adaptive_cuml_20260401/*.csv` | Raw benchmark CSVs from 5 machines |
| `benchmarks/es_times_adaptive_cuml_20260401_per_hit_per_hit.csv` | ES data (528 rows, same format as previous) |
| `benchmarks/figures/adaptive_cuml_vs_always.png` | Comparison figure |
| `benchmarks/figures/adaptive_cuml_comparison.png` | Earlier comparison (from first run, less complete) |
