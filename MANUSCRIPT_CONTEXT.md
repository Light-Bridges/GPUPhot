# GPUPhot Manuscript Context — Ground Truth for Writing

> This document contains verified findings, benchmark results, and narrative
> guidance for writing the GPUPhot manuscript. It is the authoritative source
> of truth — do NOT contradict these findings based on assumptions or outdated
> information. All data referenced here is in `benchmarks/results_collected/`.

## 1. What GPUPhot Is

GPUPhot is a **GPU-accelerated Python pipeline for real-time photometry and
astrometry** of astronomical CCD/CMOS images. It processes images from the
**Telescopios Robóticos del Teide (TTT/TST)** — a robotic multi-telescope
facility at the Teide Observatory (IAC, Tenerife) that generates thousands
of images per night from multiple instruments (iKon936, QHY600, QHY411).

The pipeline runs as a **distributed system**: Celery workers consume FITS
images from a task queue, process them on GPU, and store photometric catalogs
and transient alerts. It is designed to operate **autonomously** on
heterogeneous GPU hardware, from NVIDIA H100 datacenter cards to Jetson Orin
edge devices.

### Key Differentiators (what makes it publishable)

1. **Full pipeline on GPU**: not just one operation — photometry, PSF modeling,
   background estimation, source detection, and aperture photometry all run on
   GPU via CuPy/CUDA. Only astrometry solving and catalog queries use CPU.

2. **Adaptive memory management**: the pipeline adapts to available GPU VRAM —
   batching FFT planes across CUDA streams, tiling large images, and monitoring
   memory pressure in real-time to avoid OOM on small GPUs.

3. **Distributed real-time processing**: Celery workers with GPU affinity,
   Docker containerization, and centralized logging (Elasticsearch) enable
   autonomous operation across a heterogeneous GPU fleet.

4. **Reproducibility**: Docker containers pin all dependencies (CuPy, NumPy,
   astrometry solver). The same container image runs on datacenter and edge.

5. **Tested across 8 GPUs**: from H100 (80 GB) to Jetson Orin NX (8 GB shared),
   covering 4 orders of magnitude in image size (4.2 to 151.2 megapixels).

## 2. The Python 3.8 vs 3.12 Story — CRITICAL NARRATIVE

### What we initially assumed (WRONG)

That Python 3.12 with RAPIDS cuML would be faster than Python 3.8 because
cuML accelerates the catalog crossmatch on GPU.

### What the data actually shows (CORRECT)

**Python 3.12 is SLOWER in end-to-end latency but uses LESS GPU memory.**

The improvement comes from **updated dependencies** (CuPy 14, NumPy 2.0),
not from cuML. cuML actually HURTS performance.

### Evidence

#### End-to-end timing (Elastic profiler data, 25-26 Mar 2026)

py3.8 is faster in 28 of 30 GPU/image pairs. Only two exceptions:
- L40S at 15.3 MP: py3.12 is 1.10x faster
- H100 at 37.8 MP: py3.12 is 1.15x faster

Data: `profiler_paired_comparison_20260327.csv`

#### GPU memory (nsys profiling, 26-29 Mar 2026)

py3.12 uses **6-8% less peak VRAM** consistently across all GPUs:

| GPU | Avg VRAM saving (py3.12 vs py3.8) |
|-----|-----------------------------------|
| H100 PCIe | 8.1% |
| A100-SXM4-80GB | 8.1% |
| L40S | 8.1% |
| RTX 3090 | 6.0% |
| RTX 3060 | 7.7% |
| RTX 3050 Ti | 7.8% |

Data: `profiler_nsys_memory_summary_20260326.csv`

Specific examples (A100):
- iKon936 (4.2 MP): 2018 MB (py3.8) vs 1718 MB (py3.12) = 300 MB saved (15%)
- QHY411-3 (151 MP): 31006 MB vs 27998 MB = 3007 MB saved (10%)

#### cuML crossmatch is ALWAYS slower in the real pipeline

Ablation test on A100 (29 Mar 2026): py3.12 with cuML enabled vs disabled.
10 images, 112 to 18,888 sources. cuML is 87-256% SLOWER for ALL images.

| Image | Sources | With cuML | Without cuML | cuML penalty |
|-------|---------|-----------|--------------|--------------|
| iKon936 SDSSg | 412 | 8.2s | 4.4s | +87% |
| QHY600-4 SDSSg | 218 | 29.3s | 10.2s | +187% |
| QHY411-3 SDSSr | 14,241 | 334.5s | 94.0s | +256% |

Data: `cuml_ablation_a100_20260329.csv`

**Why**: cuML uses brute-force O(N^2) for 2D NearestNeighbors; scipy cKDTree
is O(N log N). The pipeline makes multiple crossmatch calls per image, each
paying cuML initialization overhead.

Synthetic crossover benchmark confirms: even in isolation, cuML only wins in
a narrow window (2K-50K sources on powerful GPUs), and NEVER on small GPUs.
Data: `cuml_crossover_synthetic_all_gpus_20260328.csv`

### Correct manuscript narrative

> Python 3.12 with CuPy 14 and NumPy 2.0 reduces peak GPU memory usage by
> 6-8% compared to Python 3.8 with CuPy 12. This reduction enables processing
> more images concurrently on the same GPU — for example, on the RTX 3050 Ti
> (4 GB), the iKon936 image footprint drops from 2018 MB to 1718 MB, enabling
> two concurrent images instead of one. End-to-end latency per image is
> slightly higher in py3.12, but aggregate throughput improves through
> increased concurrency.
>
> The RAPIDS cuML library, while included in the py3.12 environment, does not
> improve crossmatch performance for 2D source matching. cuML's brute-force
> nearest-neighbor implementation is algorithmically inferior to scipy's
> cKDTree (O(N^2) vs O(N log N)) for the low-dimensional case. We recommend
> using cKDTree as the default crossmatch backend.

### What NOT to write

- Do NOT claim cuML accelerates the pipeline
- Do NOT present py3.12 as "faster" without qualifying it as memory-efficient
- Do NOT attribute the memory improvement to "Python 3.12 interpreter" — it's
  the dependency versions (CuPy 14, NumPy 2.0) that matter

## 3. Hardware Tested

| Machine | GPU | VRAM | Type | CPU | Role |
|---------|-----|------|------|-----|------|
| AAZ | H100 PCIe | 80 GB | Datacenter | AMD EPYC 9124 | Production (TTT/TST) |
| ALE | A100-SXM4-80GB | 80 GB | Datacenter | Intel Xeon Gold 6354 | Production (TTT/TST) |
| AHP | L40S | 48 GB | Datacenter | Intel Xeon Platinum 8468 | Production (TTT/TST) |
| TTT | RTX 3090 | 24 GB | Consumer | Intel Core i9-10940X | Dev server |
| CCU | RTX 3060 | 12 GB | Consumer | Intel Core i5-10400F | Dev server |
| LLB | RTX 3050 Ti Laptop | 4 GB | Laptop | Intel Core i7-12700H | Development |
| JET | Orin Super | 8 GB shared | Edge (Tegra) | ARM Cortex-A78AE | Edge deployment |
| JON | Orin NX 8GB | 8 GB shared | Edge (Tegra) | ARM Cortex-A78AE | Edge deployment |

### Jetson specifics
- cuML is NOT available on ARM (no RAPIDS wheels for aarch64)
- py3.8 on Jetson is actually Python 3.10 (JetPack 6.x minimum)
- Memory is shared between CPU and GPU (unified memory)
- nsys does not capture GPU memory allocation events on Tegra
- Orin NX 8GB can only process images up to 6.8 MP (15.3 MP causes OOM crash)

## 4. Benchmark Images

10 FITS images covering 4.2 to 151.2 megapixels. For each resolution
(except 6.8 MP), two images with different source densities were selected
to assess the impact of source count on processing time. This is a
deliberate design choice: the sparse/dense pairing reveals that source
density can affect end-to-end latency by up to 1.7× at 151.2 MP, an
effect that would be masked if only one image per size were tested.

| Camera | Size | MP | Object | Sources | Field | Instrument |
|--------|------|----|--------|---------|-------|------------|
| iKon936-1 | 2048x2048 | 4.2 | C/2025 A6 | 296 | Sparse | TTT3 |
| iKon936-1 | 2048x2048 | 4.2 | QSO 0957+561 | 412 | Dense | TTT3 |
| QHY600-3 | 3191x2129 | 6.8 | C/2025 R2 | 112 | Sparse | TTT3 |
| QHY600-4 | 4787x3193 | 15.3 | WASP-43-b | 218 | Sparse | TTT2 |
| QHY600-4 | 4787x3193 | 15.3 | NGC 2903 | 318 | Dense | TTT2 |
| QHY411-1 | 7100x5325 | 37.8 | 2012 QD8 | 154 | Sparse | TTT1 |
| QHY411-1 | 7100x5325 | 37.8 | GaiaDR3... | 247 | Dense | TTT1 |
| QHY411-3 | 14200x10650 | 151.2 | 2025 PR1 | 428 | Sparse | TST |
| QHY411-3 | 14200x10650 | 151.2 | M81 | 14241 | Dense | TST |
| QHY411-3 | 14200x10650 | 151.2 | 24P | 18888 | Very dense | TST |

Production ATLAS images (9576x6376, 61.1 MP) have 6K-160K sources — these
are the workloads where crossmatch performance matters most.

## 5. Key Performance Numbers for the Manuscript

### Latency (median, py3.12, single image, no concurrency)

| Image | H100 | A100 | L40S | RTX 3090 | RTX 3060 | RTX 3050 Ti | Orin Super |
|-------|------|------|------|----------|----------|-------------|------------|
| 4.2 MP | 8.5s | 5.7s | 6.1s | 10.0s | 7.4s | 47.5s | 23.7s |
| 6.8 MP | 7.7s | 5.2s | 6.0s | 9.6s | 7.4s | — | — |
| 15.3 MP | 10.8s | 10.8s | 11.4s | 17.5s | 20.1s | — | — |
| 37.8 MP | 8.7s | 14.9s | 16.4s | 22.8s | 27.4s | — | — |
| 151.2 MP | 51.5s | 84.5s | 86.5s | — | — | — | — |

Note: H100 is fastest overall for large images; A100 is fastest for small images.

### Memory (median peak VRAM, py3.12)

| Image | H100 | A100 | L40S | RTX 3090 | RTX 3060 | RTX 3050 Ti |
|-------|------|------|------|----------|----------|-------------|
| 4.2 MP | 1718 MB | 1718 MB | 1718 MB | 1718 MB | 1718 MB | 1718 MB |
| 6.8 MP | 5613 MB | 5613 MB | 5613 MB | 5613 MB | 5613 MB | — |
| 15.3 MP | 8300 MB | 8241 MB | 8241 MB | 8237 MB | 8241 MB | — |
| 37.8 MP | 6849 MB | 6801 MB | 6801 MB | 6801 MB | 6796 MB | 3779 MB |
| 151.2 MP | 28052 MB | 27998 MB | 27998 MB | 23698 MB | — | — |

Note: Peak VRAM is nearly identical across GPUs for the same image — the
pipeline's memory usage is determined by image size, not GPU architecture.
The RTX 3050 Ti shows lower values for large images because it runs out of
VRAM and the pipeline adapts (fewer concurrent FFT planes).

### Concurrency implications

On the A100 (80 GB), py3.12 enables:
- 4.2 MP images: 49 concurrent (vs 42 with py3.8) = +17%
- 151.2 MP images: 3 concurrent (vs 2 with py3.8) = +50%

On the RTX 3050 Ti (4 GB), py3.12 enables:
- 4.2 MP images: 2 concurrent (vs 1 with py3.8) = **2x throughput**

## 5b. Validation Strategy — Two-Paper Split

This manuscript (A&C, software paper) covers ONLY:
- Pipeline correctness (deterministic detection, WCS solution, catalog output)
- Performance benchmarks (timing, memory, scalability)
- Software architecture and reproducibility

Scientific validation of photometric/astrometric accuracy is deferred to:
**Alarcon et al. (in prep.)** — the companion paper authored by Miguel.

That paper will cover:
- Photometric accuracy (magnitude residuals vs reference catalogs)
- Astrometric precision (RMS vs Gaia DR3)
- PSF modeling fidelity (residual flux after subtraction)
- Completeness, contamination, depth analysis
- Comparison with classical pipelines (SExtractor, Photutils) on science metrics

**Do NOT add scientific validation data to this manuscript.** If reviewers
ask for it, point to the companion paper. The validation section (Section 6
in the manuscript) confirms functional correctness only — the pipeline runs,
produces catalogs, and the outputs are structurally valid. The science
quality assessment belongs in Alarcon et al.

## 6. Related Work and Comparison Points

The manuscript MUST compare with:

- **SExtractor / Source Extractor++**: the standard CPU photometry tool.
  GPUPhot replaces the full SExtractor workflow on GPU.
- **sep (Python wrapper for SExtractor)**: CPU-based, single-threaded.
  Useful as direct CPU baseline.
- **Photutils (astropy)**: Python photometry library, CPU-only.
  GPUPhot implements equivalent operations on GPU.
- **IRAF/PyRAF**: legacy pipeline, being deprecated.
- **AstrOmatic suite**: SExtractor, SCAMP, SWarp — CPU pipeline.

NOT compared with (and should explain why):
- **LSST/Rubin pipeline**: different scale (survey-level), not real-time.
- **DRAGONS (Gemini)**: different scope (instrument-specific reduction).

## 7. Reference Manuscript: GpuFitsCrypt

The lead author has a previous manuscript on **GpuFitsCrypt** — a GPU-accelerated
FITS encryption library. Structural patterns to reuse:

- Same author, similar GPU+astronomy domain
- GpuFitsCrypt manuscript structure: architecture diagram, GPU kernel description,
  benchmark tables with multiple GPUs, Docker reproducibility section
- Cross-reference: GPUPhot uses GpuFitsCrypt for encrypted FITS I/O
- The encryption layer is NOT the focus of the GPUPhot manuscript — mention it
  as a dependency, not as a contribution

Location: `GPUPHOT_manuscript/` and `GpuFitsCrypt_manuscript/` in the repo.

## 8. Data Files Reference

All in `benchmarks/results_collected/`:

### Timing (end-to-end latency from Elasticsearch)
- `profiler_timing_elastic_20260325.csv` — main benchmark run (7 machines)
- `profiler_timing_elastic_hp3_20260326_per_hit.csv` — L40S re-run

### Memory (peak VRAM from nsys)
- `profiler_nsys_memory_20260326.csv` — raw per-run data (429 rows)
- `profiler_nsys_memory_summary_20260326.csv` — medians (62 rows)

### Combined tables (ready for manuscript)
- `profiler_paired_comparison_20260327.csv` — py3.8 vs 3.12, timing + memory
- `profiler_unified_timing_memory_20260327.csv` — all data merged

### cuML investigation
- `cuml_ablation_a100_20260329.csv` — real images, cuML always slower
- `cuml_ablation_local_rtx3050ti_20260327.csv` — small GPU, cuML always slower
- `cuml_crossover_synthetic_all_gpus_20260328.csv` — synthetic crossover, 6 GPUs

### CPU baseline comparison
- `cpu_baseline_results.csv` — sep + Photutils vs 10 benchmark images (in `benchmarks/`)
- `benchmark_cpu_baselines.py` — script to reproduce (in `benchmarks/`)

### Edge devices
- `jetson_orin_8gb_timing_20260329.csv` — Orin NX 8GB (max 6.8 MP)

### Scripts
- `benchmark_cuml_crossover.py` — synthetic crossover benchmark (user-runnable)
- `analyze_times_v2_profiler.py` — profiler data analysis
- `analyze_times_v2.py` — production data analysis
- `analyze_profiling.py` — nsys NVTX stage breakdown

### Full documentation
- `README_benchmark_data.md` — complete methodology, known issues, reproduction steps

## 9. Figures Needed for the Manuscript

1. **Architecture diagram** (Figure 1): FITS ingestion -> Celery queue ->
   GPU worker -> photometry/astrometry -> catalog output. Show Docker
   container boundary and GPU/CPU split.

2. **Latency vs image size** (Figure 2): log-log plot, one line per GPU,
   error bars from IQR. Data: `profiler_unified_timing_memory_20260327.csv`

3. **Memory comparison py3.8 vs py3.12** (Figure 3): grouped bar chart,
   5 image sizes, showing peak VRAM for both versions.
   Data: `profiler_paired_comparison_20260327.csv`

4. **GPU heatmap** (Figure 4): rows = image sizes, columns = GPUs,
   cells = median latency. Color scale: yellow (fast) to red (slow).
   Already generated: `figures_profiler/profiler_heatmap.pdf`

5. **cuML crossover** (Figure 5): speedup vs N for 6 GPUs, with break-even
   line at 1.0. Data: `cuml_crossover_synthetic_all_gpus_20260328.csv`
   Include note that real pipeline does not benefit.

6. **Concurrency scaling** (Figure 6): theoretical concurrent images per GPU
   based on VRAM, py3.8 vs py3.12. Derived from memory data.

## 10. CPU Baseline Comparison — Scope-Mismatched (CRITICAL)

**Data file**: `benchmarks/cpu_baseline_results.csv`
**Date**: 24 March 2026
**Script**: `benchmarks/benchmark_cpu_baselines.py`
**Tools compared**: sep (SExtractor Python) and Photutils vs 10 benchmark images

### Raw results

| MP | Image | sep (s) | Photutils (s) | GPUPhot A100 py3.12 (s) | GPUPhot/sep | GPUPhot/Phot |
|----|-------|---------|---------------|-------------------------|-------------|--------------|
| 4.2 | QSO0957+561 | 0.19 | 1.9 | 5.7 | 30x slower | 3.1x slower |
| 6.8 | C2025R2 | 0.29 | 3.1 | 5.2 | 18x slower | 1.7x slower |
| 15.3 | WASP-43-b | 0.98 | 7.9 | 10.8 | 11x slower | 1.4x slower |
| 37.8 | 2012QD8 | 1.49 | 13.5 | 14.8 | 10x slower | 1.1x slower |
| 37.8 | GaiaDR3... | 3.33 | 20.6 | 14.8 | 4x slower | **0.7x faster** |
| 151.2 | 2025PR1 | 8.18 | 125.8 | 84.5 | 10x slower | **0.7x faster** |
| 151.2 | M81 | 8.83 | 73.3 | 84.5 | 10x slower | 1.2x slower |

### Why this is NOT a fair comparison

| Tool | Operations performed |
|------|---------------------|
| sep | Detection + aperture photometry only (~2 stages) |
| Photutils | Detection + aperture photometry only (~2 stages) |
| **GPUPhot** | Detection + aperture photometry + PSF modeling + FFT background estimation + multi-catalog crossmatch + astrometric calibration (CPU) + structured catalog output (~7 stages) |

sep does ~2 operations. GPUPhot does ~7. The raw timing difference reflects
**scope**, not inefficiency. GPUPhot's 5.7s for a 4.2 MP image includes
astrometry solving (CPU), Vizier catalog queries (network I/O), and
crossmatching — none of which sep or Photutils perform.

### Key observations

1. **GPUPhot vs Photutils at 37.8-151 MP**: GPUPhot becomes faster than
   Photutils for large images (0.7x = GPUPhot wins). This is where GPU
   parallelism starts to compensate for the additional pipeline stages.

2. **sep is always fastest** for raw detection — it is a C extension with
   decades of optimization. But sep produces no astrometric solution, no
   PSF model, no crossmatched catalog, and no transient alerts.

3. **The honest comparison** would isolate the detection+photometry stages
   of GPUPhot (using nsys NVTX per-stage timing from `analyze_profiling.py`)
   and compare only those against sep. This is available in the nsys data
   but not yet extracted as a separate table.

### Correct manuscript narrative

> Direct wall-clock comparison with sep and Photutils is presented for
> transparency, but is not scope-equivalent. sep performs source detection
> and circular aperture photometry (two operations); GPUPhot executes a
> complete autonomous pipeline including PSF fitting, FFT-based background
> estimation, multi-catalog crossmatch, astrometric calibration via
> Astrometry.net, and structured catalog output (seven stages). For images
> exceeding 37 megapixels, GPUPhot's end-to-end time becomes competitive
> with Photutils despite performing substantially more operations, due to
> GPU parallelism in the photometric stages. A per-stage comparison using
> nsys NVTX profiling is provided in Section 8 to isolate the detection
> stage timing.

### What NOT to write

- Do NOT claim GPUPhot is faster than sep — it is not, and never will be
  for detection-only workloads
- Do NOT present the 7-30x difference as a failure — explain it is a scope
  difference, then highlight where GPU pays off (large images, full pipeline)
- Do NOT skip this comparison — reviewers WILL ask for it, and having it
  ready (with honest framing) is far better than omitting it

## 11. Known Limitations to Address Honestly

1. **cuML does not help**: must be stated as a finding, not hidden
2. **py3.12 is slower per-image**: the trade-off must be explicit
3. **Jetson Orin 8GB very limited**: max 6.8 MP, unstable for larger images
4. **GPUPhot slower than sep in raw timing**: scope-mismatched comparison,
   must be framed correctly (see Section 10 above)
5. **Astrometry is CPU-only**: the astrometry solver (Astrometry.net) runs
   on CPU; only the source detection and photometry are GPU-accelerated
6. **Single-GPU only**: no multi-GPU support within a single image; scaling
   is through Celery workers (one GPU per worker)
7. **Per-stage fair comparison extracted (A100, 4 of 5 images)**: NVTX data
   shows GPU detection is 2.3-10.5x faster than sep on identical images.
   Data: `nvtx_detection_vs_sep_a100.csv`. 151.2 MP sqlite was corrupted.
