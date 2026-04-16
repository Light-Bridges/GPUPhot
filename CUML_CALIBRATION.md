# cuML Crossmatch Calibration

GPUPhot can optionally use [RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/) for
GPU-accelerated catalog cross-matching.  Because cuML uses a brute-force O(N²) algorithm
while the CPU fallback (cKDTree) is O(N log N), **GPU is only faster within a specific
source-count window** that depends on your hardware.  This guide explains how to find that
window for your GPU and configure GPUPhot accordingly.

---

## When to run this calibration

Run the calibration once per GPU model, after installing cuML (see [INSTALL.md](INSTALL.md)).
You do not need to re-run it unless you change hardware.

**Skip this entirely if:**

- You are on ARM / Jetson (cuML is not available on ARM).
- Your GPU has ≤ 4 GB VRAM (GPU crossmatch is rarely beneficial).
- You prefer to keep the default CPU-only crossmatch (`GPUPHOT_USE_CUML_CROSSMATCH=0`).

---

## Prerequisites

| Requirement | Notes |
|---|---|
| x86_64 host with NVIDIA GPU | cuML is not available on ARM |
| CUDA 11.x or 12.x drivers | Match the cuML build flavour |
| Docker with NVIDIA runtime | `docker run --gpus all` must work |
| `gpuphotfinal-profiler-1` container running | Or the equivalent profiler image for your setup |

Start the profiler container if it is not already running:

```bash
docker compose up -d profiler
```

---

## Quick start

Run the calibration script inside the profiler container and read the recommendation
printed at the end:

```bash
docker exec gpuphotfinal-profiler-1 \
    python3 /app/benchmarks/benchmark_cuml_crossover.py \
    --logspace 25 100 200000 --auto-refine
```

The script prints a CSV to stdout (suitable for saving) and the recommendation to stderr.
To capture both:

```bash
docker exec gpuphotfinal-profiler-1 \
    python3 /app/benchmarks/benchmark_cuml_crossover.py \
    --logspace 25 100 200000 --auto-refine \
    > cuml_crossover_$(hostname)_$(date +%Y%m%d).csv
```

The recommendation is printed to stderr even when stdout is redirected, so you will see it
in the terminal regardless.

---

## Understanding the output

Progress lines are written to stderr during the run:

```
=== Phase 1: initial sweep ===
  N=   3258: CPU=10.2ms  GPU=8.3ms  speedup=1.23x  [GPU]  n=9
  N=   4472: CPU=14.9ms  GPU=12.8ms  speedup=1.17x  [GPU]  n=11
  ...
=== Phase 2: refining around crossover zones ===
  ...

# ── Recommendation for NVIDIA GeForce RTX 3050 Ti Laptop GPU ──
#   GPU win zone (all):    N ≈ 2781–14284  (peak 1.76x at N=11565)
#   GPU win zone (robust): N ≈ 3258–13549  (speedup ≥ 1.15x throughout)
#
#   → GPUPHOT_CUML_MIN_SOURCES=3258    # robust boundary (speedup ≥ 1.15x)
#   → GPUPHOT_CUML_MAX_SOURCES=13549   # robust boundary (speedup ≥ 1.15x)
```

### Two win zones

| Zone | Meaning | Use for |
|---|---|---|
| **all** | Every N where GPU was faster in the sweep, including the uncertain boundary edges (speedup ≈ 1.0) | Reference / understanding the full range |
| **robust** | Only N where speedup ≥ `--min-speedup` (default 1.15×) | Setting the thresholds in your `.env` |

**Use the robust zone thresholds** — the boundary edges of the full zone sit where CPU and
GPU are nearly equal (speedup ≈ 1.0), making them sensitive to measurement noise and
system load.  The robust zone excludes those fragile edges.

### The `n=` indicator

Each row shows how many iterations the adaptive loop needed before the measurement
stabilised (`n=`).  Low values (5–8) mean the timing was clean.  `(!max_samples)` means
the measurement hit the hard cap before converging — this is normal for very small N
(<500 sources) where absolute times are tiny and OS jitter dominates.

---

## Applying the thresholds

Edit your `.env` file (project root) and set the three cuML variables:

```bash
# .env
GPUPHOT_USE_CUML_CROSSMATCH=0       # 0 = adaptive (use cuML only inside the window)
GPUPHOT_CUML_MIN_SOURCES=3258       # from the "robust" recommendation
GPUPHOT_CUML_MAX_SOURCES=13549      # from the "robust" recommendation
```

> **`GPUPHOT_USE_CUML_CROSSMATCH=0` is the adaptive mode.**  With `MIN` and `MAX` set,
> GPUPhot automatically chooses cuML when the source count falls inside the window and
> falls back to cKDTree otherwise.  Setting it to `1` forces cuML for *all* source counts,
> which is slower — do not do this.

Recreate the worker container to pick up the new environment:

```bash
docker compose up -d --force-recreate worker
```

### Verify in Python

```python
import gpuphot
print(gpuphot.crossmatch.backend_for(n_sources=5000))  # should print 'cuml'
print(gpuphot.crossmatch.backend_for(n_sources=500))   # should print 'ckdtree'
```

---

## Reference table: known GPUs (2026-04)

These values were measured with the default calibration settings
(`--logspace 25 100 200000 --auto-refine`, adaptive CV 8 %, robust threshold 1.15×).
They are a starting point — always run the calibration on your own machine.

| GPU | VRAM | MIN_SOURCES | MAX_SOURCES | Peak speedup |
|---|---|---|---|---|
| H100 PCIe 80 GB | 80 GB | ~5 000 | > 500 000 | 4.6× @ 20 K |
| A100-SXM4-80GB | 80 GB | ~5 000 | > 100 000 | 3.5× @ 20 K |
| L40S | 48 GB | ~2 000 | > 200 000 | 5.8× @ 20 K |
| RTX 3090 | 24 GB | ~2 000 | > 100 000 | 3.2× @ 20 K |
| RTX 3060 | 12 GB | ~5 000 | ~40 000 | 1.9× @ 10 K |
| RTX 3050 Ti (laptop) | 4 GB | ~3 258 | ~13 549 | 1.8× @ 11 K |

> **Warning:** These are synthetic benchmarks (single crossmatch call).  In the real
> pipeline, multiple crossmatches per image plus cuML initialisation overhead reduce the
> advantage.  Always verify with a real-image test before enabling cuML in production.

---

## Advanced options

| Option | Default | Description |
|---|---|---|
| `--logspace N MIN MAX` | `25 100 200000` | N log-spaced source counts to test |
| `--auto-refine` | off | Add extra points around crossover zones for finer boundaries |
| `--refine-points K` | `5` | Points inserted per crossover zone |
| `--target-cv CV` | `0.08` | Convergence threshold (CV of last window); lower = more stable, slower |
| `--max-samples N` | `20` | Hard cap on iterations per data point |
| `--min-speedup X` | `1.15` | Minimum speedup to count as a reliable GPU win |
| `--validate` | off | Re-measure Phase 1 boundary points after pre-warm to confirm consistency |
| `--warmup-n N` | max of grid | Source count used for the global GPU pre-warm before Phase 2 |

### Tightening the convergence

On a noisy or heavily loaded system, raise `--target-cv` to 0.12 (faster, less strict) or
lower it to 0.05 (slower, more stable).  If many rows show `(!max_samples)`, raise
`--max-samples` to 30:

```bash
docker exec gpuphotfinal-profiler-1 \
    python3 /app/benchmarks/benchmark_cuml_crossover.py \
    --logspace 25 100 200000 --auto-refine \
    --target-cv 0.05 --max-samples 30
```

### Changing the robust threshold

If your GPU shows very consistent measurements, you can lower `--min-speedup` to 1.05 to
extend the window slightly.  On a variable system, raise it to 1.25 for more conservative
thresholds:

```bash
--min-speedup 1.25   # conservative
--min-speedup 1.05   # aggressive
```

---

## Platform limitations

| Platform | cuML available | Notes |
|---|---|---|
| x86_64 + CUDA 11/12 | Yes | Full support |
| ARM / Jetson (JetPack 5) | No | Falls back to cKDTree automatically |
| ARM / Jetson (JetPack 6 + Python 3.12) | Planned | Not yet available as of 2026-04 |
| macOS / CPU-only | No | No CUDA |

On platforms without cuML, GPUPhot ignores `GPUPHOT_USE_CUML_CROSSMATCH` and always uses
cKDTree.  No calibration is needed.

---

## See also

- [INSTALL.md](INSTALL.md) — installing cuML and its RAPIDS dependencies
- [DOCKER.md](DOCKER.md) — managing the profiler container
- [benchmarks/README.md](benchmarks/README.md) — full benchmark suite documentation
