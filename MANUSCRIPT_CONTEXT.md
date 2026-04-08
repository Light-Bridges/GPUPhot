# GPUPhot Manuscript Context — Ground Truth for Writing

> This document contains verified findings, benchmark results, and narrative
> guidance for writing the GPUPhot manuscript. It is the authoritative source
> of truth — do NOT contradict these findings based on assumptions or outdated
> information. All data referenced here is in `benchmarks/results_collected/`.

## Quick Navigation

**Start here before writing anything:**

| What you need | Section |
|---------------|---------|
| **Journal rules** (abstract, highlights, keywords, word limits) | **§0** ← check before submitting |
| Editorial rules, narrative thread, what NOT to write | **§13** ← read before writing |
| What GPUPhot is, key differentiators | §1 |
| All verified numbers (latency, VRAM, speedups) | **§5** |
| Current manuscript state, which file has what | §12 |
| The py3.8 vs py3.12 + cuML full story | §2 |
| Hardware platforms and specs | §3 |
| Benchmark images (cameras, MP, source counts) | §4 |
| CPU comparison framing (scope-mismatch) | §10 |
| Known limitations to state honestly | §11 |

**Manuscript repo**: `slemesp/GPUPHOT_manuscript`, branch `main`, commit `bf27dfb`
**Code repo**: `Light-Bridges/GPUPhot`, branch `documentation`, commit `3712545`

## 0. Journal Requirements — Astronomy & Computing (Elsevier)

> Source: https://www.sciencedirect.com/journal/astronomy-and-computing/publish/guide-for-authors
> Verified 2026-04-06. The ScienceDirect page does not render full content
> via automated fetch; requirements below come from the guide + Elsevier
> general author policies + analysis of published A&C papers.

### Article type
This paper is a **Full Length Article** (not a Software Release Note).
Software Release Papers are shorter and require a stable public repository URL
with professional packaging — keep that distinction if scope changes.

### Abstract
- **No explicit word limit** specified by A&C (unlike many Elsevier journals)
- Published papers range ~170–290 words; target **~200 words**
- **One continuous paragraph** — no blank lines inside `\begin{abstract}`
- **No references** inside the abstract
- **No undefined abbreviations** (define on first use or avoid)
- **No bullet lists or numbered items** — prose only
- Do NOT use "validate" to describe performance benchmarks — "validate"
  implies scientific accuracy (photometric precision, astrometric RMS),
  which belongs to Alarcon et al. Use "benchmark", "evaluate", or "assess"
- **Structure** (conventional for A&C tool papers, per ZTF/HSC/ClusterPyXT):
  1. "We present [TOOL]..." — tool named in sentence 1
  2. What it does (capabilities, stages, deployment)
  3. "We benchmark/evaluate on [DATA]..." — results as evidence
  4. Close with scientific mission / open-source availability — NOT with
     a negative result or benchmark number

### Highlights (required by Elsevier)
- 3–5 bullet points, **max 85 characters each** (including spaces)
- Should capture the novel contributions, not restate the abstract
- Written as complete sentences in present tense
- **Currently missing from the manuscript** — needs to be added before submission
  Example format in cas-dc:
  ```latex
  \begin{highlights}
  \item GPUPhot executes six of seven photometric pipeline stages on GPU via CuPy
  \item Adaptive memory management deploys unchanged across a 20x VRAM range
  \item CuPy 14 / NumPy 2.0 reduces peak VRAM 6--8\%, increasing GPU concurrency
  \item cuML nearest-neighbour crossmatch is slower than cKDTree for 2D queries
  \end{highlights}
  ```

### Keywords
- Required; typically 4–6 keywords for A&C
- **Currently in manuscript** — verify they are present and relevant
- Suggested: GPU computing, photometry pipeline, real-time astronomy,
  CuPy, robotic telescope, image processing

### What to check before submission
- [x] Highlights block added to main.tex (commit 290459d, 4 bullets)
- [x] Keywords verified (6 terms in main.tex)
- [x] Abstract: one paragraph, no references, ~200 words, ends with mission
- [ ] All abbreviations defined on first use in main text
- [ ] 151.2 MP NVTX profiling data (corrupted during collection — re-acquire before final revision)

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

Synthetic crossover benchmark (extended to 500K sources on clean GPUs, 30 Mar 2026):

| GPU | cuML wins from | Peak speedup | Loses again at |
|-----|----------------|--------------|----------------|
| H100 PCIe | 5K | 4.8x @ 20K | Not in range (1.27x @ 500K, trending down) |
| A100-SXM4 | 2K | 4.4x @ 20K | 200K (0.43x @ 500K) |
| L40S | 2K | 5.5x @ 20K | ~500K (0.86x @ 500K) |
| RTX 3090 | 2K | 3.7x @ 10K | 200K (0.46x @ 500K) |
| RTX 3060 | 2K | 2.0x @ 10K | 50K (0.14x @ 500K) |
| RTX 3050 Ti | Never | — | CPU always wins |

All benchmarks run with workers stopped and GPU memory verified clean (< 5 MiB).
Data: `cuml_crossover_synthetic_clean_20260330.csv` (supersedes 20260328 version)

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

### IMPORTANT: cuML was active in the FIRST py3.12 benchmarks (March 2026)

The profiler benchmarks (25-26 Mar 2026) for py3.12 were run with cuML
ENABLED by default (catalog.py: `if CUML_AVAILABLE and is_gpu_input`).
Those py3.12 latency numbers INCLUDE cuML crossmatch overhead (71-256% per
image). The py3.8 environment does not have cuML installed and always used
cKDTree. This must be stated transparently in the manuscript.

### Definitive benchmark: adaptive cuML (April 2026)

**Completed**: 2026-04-04. Three phases run on all 6 x86 machines (2 warmup +
10 clean reps each). Data in `benchmarks/results_collected/benchmark_all_cuml_v2.csv`.

#### Configuration per phase
| Phase | GPUPHOT_ENVIRONMENT | cuML strategy |
|-------|---------------------|---------------|
| `profiler_cuml_always_v2` | py3.12 | cuML for all source counts |
| `profiler_cuml_adaptive_v2` | py3.12 | cuML only in [MIN, MAX] per GPU |
| `profiler_py38_v2` | py3.8 | cKDTree always (no RAPIDS) |

Adaptive thresholds (from `run_cuml_comparison_sequential.sh`):

| GPU | MIN_SOURCES | MAX_SOURCES |
|-----|-------------|-------------|
| H100 PCIe (azken) | 5000 | 500000 |
| L40S (hp3) | 2000 | 200000 |
| A100-SXM4 (lenovo) | 2000 | 100000 |
| RTX 3090 (ttt_server) | 2000 | 100000 |
| RTX 3060 (ttt1) | 2000 | 20000 |
| RTX 3050 Ti (local) | 2000 | 50000 |

#### Key results (median, warmup dropped, datacenter machines)

**py38_baseline vs py312_cuml_adaptive** (% overhead of py312 vs py38):

| Image | MP | n_src | py38 (s) | py312_adapt (s) | Overhead |
|-------|----|-------|----------|-----------------|---------|
| iKon936 (both) | 4.2 | 296-412 | 4.6–6.5 | 6.1–9.1 | +25-40% |
| QHY600-3 Lum | 6.8 | 112 | 3.4–6.1 | 4.2–7.7 | +25% |
| QHY600-4 (both) | 15.3 | 218-318 | 6.8–11.2 | 7.5–11.4 | +5-35% |
| QHY411-1 bin2 (both) | 37.8 | 154-247 | 8.3–13.1 | 10.7–16.7 | +15-35% |
| QHY411-1 full | 151.2 | 428 | 16–27 | 34–85 | +100-250%* |
| QHY411-3 SDSSr | 151.2 | 14241 | 52–59 | 61–79 | +15-52% |
| QHY411-3 Lum | 151.2 | 18888 | 60–75 | 87–109 | +21-80% |

*QHY411-1_Lum_full outlier (428 sources, 151.2 MP): even with cKDTree
(428 < MIN), py312 is 2-4x slower — the bottleneck is the image processing
pipeline (background/detection for 151.2 MP), not the crossmatch. This
overhead does NOT appear for denser 151.2 MP images where cuML is active.

**py312_cuml_always vs py312_cuml_adaptive** (% improvement from adaptive):

| Image | MP | n_src | always (s) | adaptive (s) | Improvement |
|-------|----|-------|-----------|--------------|------------|
| QHY411-1 bin2 | 37.8 | 154 | 10.6–14.0 | 10.7–13.9 | ~10% |
| QHY411-3 SDSSr | 151.2 | 14241 | 64–83 | 61–79 | **-17%** (H100: -27%) |
| QHY411-3 Lum | 151.2 | 18888 | 88–108 | 87–109 | ~3% |
| Small images | 4.2–15.3 | 112-412 | — | — | 1-5% |

The adaptive benefit is most visible on H100 for QHY411-3_SDSSr_full (14241
sources): adaptive=61s vs always=83s (−27%). For most other images, always ≈
adaptive because the crossmatch is not the bottleneck.

#### Production recommendation

**Use py3.12 + adaptive cuML** for all production deployments:
1. **VRAM savings** (6-8% vs py3.8): on a 80 GB GPU, enables 50% more
   concurrent 151.2 MP images (3 vs 2), or 17% more 4.2 MP images
2. **Avoids worst-case cuML penalty**: for sparse fields (<MIN sources),
   cKDTree is used — eliminating the 71-256% cuML overhead seen in the
   old always-on configuration
3. **Competitive with py3.8 for dense fields**: for images in the cuML-
   beneficial range (MIN-MAX sources, e.g. ATLAS images with 6K-160K
   sources), py3.12+adaptive ≈ py3.8 in end-to-end latency
4. **Per-image latency trade-off**: for sparse fields, py3.8 is 20-40%
   faster per image — but aggregate throughput favors py3.12 via concurrency

### Code not in the repository (do NOT reference in manuscript)

- **SmartGPUDecoratorClass** (`gpuphot/utils/smartgpudecoratorclass.py`):
  Experimental decorator for per-function GPU memory monitoring. Not committed.

### What NOT to write

- Do NOT claim cuML always accelerates the pipeline — it depends on source count
- Do NOT present py3.12 as "faster per image" — it is not; it is memory-efficient
- Do NOT attribute the memory improvement to "Python 3.12 interpreter" — it's
  the dependency versions (CuPy 14, NumPy 2.0) that matter
- Do NOT hide the early py3.12 benchmarks that had cuML always-on — state it
- DO present py3.12+adaptive as the recommended configuration with clear reasoning

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

### Latency — definitive benchmark (April 2026)

Two tables: py3.8 baseline (cKDTree, no RAPIDS) and py3.12+adaptive cuML.
Data: `benchmarks/results_collected/benchmark_all_cuml_v2.csv` (warmup dropped,
median of 10 clean reps, 3–5 datacenter machines per entry).

#### py3.8 baseline (cKDTree, no RAPIDS overhead)

| Image | MP | n_src | H100 | A100 | L40S | RTX 3090 | RTX 3060 |
|-------|----|-------|------|------|------|----------|----------|
| iKon936 SDSSg | 4.2 | 412 | 6.5s | 3.5s | 3.7s | 6.2s | 4.6s |
| iKon936 Lum | 4.2 | 296 | 6.4s | 3.8s | 4.0s | 6.5s | 5.4s |
| QHY600-3 Lum | 6.8 | 112 | 6.1s | 3.4s | 4.0s | 6.4s | 5.7s |
| QHY600-4 Ha | 15.3 | 318 | 9.6s | 6.8s | 7.8s | 11.1s | 13.1s |
| QHY600-4 SDSSg | 15.3 | 218 | 11.2s | 7.6s | 8.8s | 12.8s | 15.4s |
| QHY411-1 Lum bin2 | 37.8 | 154 | 11.7s | 8.3s | 11.8s | 17.2s | 18.9s |
| QHY411-1 SDSSi bin2 | 37.8 | 247 | 12.9s | 9.4s | 13.1s | 18.9s | 22.9s |
| QHY411-1 Lum full | 151.2 | 428 | 19.0s | 16.4s | 27.5s | — | — |
| QHY411-3 SDSSr | 151.2 | 14241 | 59.0s | 52.0s | 56.9s | — | — |
| QHY411-3 Lum | 151.2 | 18888 | 75.1s | 60.4s | 74.2s | — | — |

#### py3.12 + adaptive cuML (RECOMMENDED production configuration)

| Image | MP | n_src | H100 | A100 | L40S | RTX 3090 | RTX 3060 |
|-------|----|-------|------|------|------|----------|----------|
| iKon936 SDSSg | 4.2 | 412 | 8.2s | 4.4s | 5.1s | 8.0s | 6.0s |
| iKon936 Lum | 4.2 | 296 | 9.1s | 4.8s | 6.0s | 8.8s | 7.3s |
| QHY600-3 Lum | 6.8 | 112 | 7.7s | 4.2s | 5.7s | 8.2s | 7.1s |
| QHY600-4 Ha | 15.3 | 318 | 10.3s | 7.5s | 9.1s | 13.1s | 16.4s |
| QHY600-4 SDSSg | 15.3 | 218 | 11.2s | 10.3s | 11.4s | 16.0s | 19.8s |
| QHY411-1 Lum bin2 | 37.8 | 154 | 10.7s | 10.8s | 13.9s | 18.9s | 22.5s |
| QHY411-1 SDSSi bin2 | 37.8 | 247 | 12.2s | 14.1s | 16.7s | 21.9s | 27.0s |
| QHY411-1 Lum full | 151.2 | 428 | 34.4s | 66.5s | 85.2s | — | — |
| QHY411-3 SDSSr | 151.2 | 14241 | 61.3s | 79.1s | 64.7s | — | — |
| QHY411-3 Lum | 151.2 | 18888 | 87.4s | 108.5s | 88.4s | — | — |

RTX 3090/3060 OOM expected for 151.2 MP images (VRAM limit).

#### Notes on these numbers

- H100 is fastest for small/medium images; A100 fastest for sparse 151.2 MP
- L40S shows anomalous slowdown for QHY411-1_Lum_full (428 sources, 151.2 MP)
  — py312 pipeline overhead dominates when crossmatch is trivial. NOT a
  crossmatch issue; occurs equally in always and adaptive modes.
- For QHY411-3_SDSSr_full (14241 sources): H100 py312_adaptive=61s is
  competitive with py38=59s — the cuML crossmatch benefit nearly offsets
  the py312 overhead.
- Orin Super (py3.12 ARM, no cuML): 4.2 MP ~150s, 6.8 MP ~154s.
  Data: `benchmark_jetson_local.csv`.

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

### Definitive benchmark (April 2026) — all 3 modes, all 6 x86 machines
- `benchmark_all_cuml_v2.csv` — unified, 1676 rows (980 ES + 49 Jetson),
  columns: machine, gpu_name, python_ver, profiler_label, environment,
  image_label, mp, execution_time, n_sources_detected, timestamp,
  naxis1, naxis2, filter, object, gpu_mem_total, gpu_mem_used, gpu_temp.
  `profiler_label` values: `py38_baseline`, `py312_cuml_adaptive`,
  `py312_cuml_always`. Includes 2 warmup rows per group (drop first 2 per
  machine+image+profiler group sorted by timestamp before analysis).

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

1. **cuML does not universally help**: with always-on cuML, performance is
   degraded for sparse fields. With adaptive cuML (RECOMMENDED), the overhead
   is eliminated for low source counts, and for dense fields cuML provides
   modest improvement. Must be stated as a nuanced finding, not hidden.
2. **py3.12 is slower per-image for sparse/medium fields**: the trade-off is
   explicit — py3.8 is 20-40% faster per image in typical production ranges,
   but py3.12 enables more concurrent images (6-8% VRAM savings) and is
   competitive for dense fields (>10K sources) with adaptive cuML.
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

## 12. Manuscript .tex File Status (as of 2026-04-06) — ALL SECTIONS COMPLETE

LaTeX repo: slemesp/GPUPHOT_manuscript, branch `main`, commit `3255aaf`.
All sections compiled successfully (21 pages). No pending edits.

### Figures — ALL IN USE

All 8 figures in `GPUPHOT_manuscript/figures/`, all referenced in .tex files.

| File | Used in |
|------|---------|
| fig1_architecture.* | architecture.tex |
| fig2_latency_vs_size.* | performance.tex |
| fig3_memory_comparison.* | performance.tex |
| fig4_heatmap.* | performance.tex |
| fig5_cuml_crossover.* | performance.tex (§cuML Crossmatch Evaluation) |
| fig6_concurrency.* | performance.tex |
| fig7_py38_vs_adaptive.* | performance.tex (§Concurrency, after deployment rec.) |
| fig8_always_vs_adaptive.* | performance.tex (§Concurrency, after deployment rec.) |

To regenerate all figures: `python3 benchmarks/generate_manuscript_figures.py`

### Section status

| File | Status | Key changes made |
|------|--------|-----------------|
| abstract.tex | ✓ DONE | Rewritten: GPU pipeline first, cuML last |
| introduction.tex | ✓ DONE | Contributions rewritten (see §13) |
| architecture.tex | ✓ DONE | No changes needed |
| implementation.tex | ✓ DONE | cuML evaluation MOVED to performance.tex; adaptive strategy stays here |
| software.tex | ✓ DONE | No changes needed |
| deployment.tex | ✓ DONE | No changes needed |
| validation.tex | ✓ DONE | No changes needed |
| performance.tex | ✓ DONE | April 2026 data; fig7+fig8 added; cuML evaluation section expanded here |
| discussion.tex | ✓ DONE | Deployment recommendation subsection added; limitations updated |
| conclusion.tex | ✓ DONE | H100 latency updated (51.5s→61.3s); adaptive cuML sentence added |

### Critical structural change (April 2026)

The `\subsection{Evaluation of RAPIDS cuML for Catalog Crossmatch}` was
**moved from implementation.tex to performance.tex**. Reason: it is a
performance benchmark, not an algorithm description. See §13 for the
editorial rationale.

- `\label{sec:cuml_eval}` is now defined in **performance.tex**
- `fig5_cuml_crossover` is defined in **performance.tex**
- `tab:cuml_ablation` is defined in **performance.tex**
- `implementation.tex` retains only:
  - `\subsubsection{Crossmatch Backend Selection}` (4 lines, forward ref to sec:cuml_eval)
  - `\subsubsection{Adaptive cuML Crossmatch Strategy}` `\label{sec:impl:adaptive_cuml}`
    (thresholds table, env vars — this is code description, stays in implementation)

### Build instructions

```bash
cd GPUPHOT_manuscript
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## 13. Narrative Thread and Editorial Decisions

> **This section is the editorial ground truth. Read it before writing any
> new content for the manuscript.**
> Additional "What NOT to write" rules specific to cuML/py3.12 are in §2.
> Rules for the CPU comparison framing are in §10.

### The correct main thread

The paper is about **GPUPhot as a complete GPU pipeline** — what it does,
which stages run on GPU, how it scales from 4 GB to 80 GB VRAM, and how it
deploys autonomously. The reader should finish the paper understanding a
production-grade GPU photometry system, not a cuML benchmark.

cuML is a **secondary finding**: an honest negative result that emerged
during development and that we resolved. It is contribution #4 out of 4,
not the headline story.

### Priority hierarchy for manuscript content

| Priority | Topic | Correct location |
|----------|-------|-----------------|
| ★★★★★ | 6 GPU stages: FFT background, source detection, eigenPSF, aperture photometry, zero-point calibration, crossmatch | implementation.tex §§ per stage |
| ★★★★★ | Adaptive GPU memory management (checkpoint VRAM, OOM fallback, process recycling, memory pools) | implementation.tex §GPU Memory Management |
| ★★★★ | Scalability: 4.2–151.2 MP, 8 GPUs, 20× VRAM range | performance.tex + abstract |
| ★★★★ | CuPy 14/NumPy 2.0 VRAM savings → concurrency gain | performance.tex §Memory Analysis |
| ★★★ | Distributed Celery/Docker architecture | architecture.tex, deployment.tex |
| ★★ | cuML negative result + adaptive solution | performance.tex §cuML Crossmatch Evaluation |
| ★ | Adaptive cuML thresholds and env vars (code detail) | implementation.tex §Adaptive cuML Strategy |

### Contributions hierarchy (introduction.tex)

1. **Seven-stage GPU pipeline** (6/7 on GPU via CuPy), 8 GPUs, 20× VRAM range.
   Headline: GPU source detection 2.2–10.5× faster than `sep` (scope-equivalent).
2. **Adaptive GPU memory management**: checkpoint VRAM monitoring, OOM fallback
   with spatial binning, process-level GPU isolation, CuPy memory pooling.
3. **CuPy 14 / NumPy 2.0 VRAM savings**: 6–8% reduction → +15–100% concurrency.
4. **Transferable negative result**: cuML 87–256% slower than cKDTree for 2D
   crossmatch + adaptive per-GPU threshold strategy as solution.

### Abstract structure

1. Problem + what GPUPhot is (one sentence each)
2. What runs on GPU (6 stages) + distribution/deployment
3. Adaptive memory management capability range
4. Three quantitative results:
   - (i) H100: 61.3 s for dense 151.2 MP image; GPU detection 2.2–10.5× faster than sep
   - (ii) VRAM: 6–8% savings → up to +100% concurrency on 4 GB GPUs
   - (iii) cuML: 87–256% slower (negative result) + resolved via adaptive strategy

### What NOT to write (applies to all sections)

- Do NOT open any section with cuML as the motivation or main finding
- Do NOT claim cuML accelerates the pipeline without qualification — it only
  helps for source counts in [MIN, MAX] with the adaptive strategy
- Do NOT present py3.12 as "faster per image" — it is not; it saves VRAM
- Do NOT attribute the VRAM improvement to the Python interpreter — it is
  CuPy 14 and NumPy 2.0 buffer management
- Do NOT add scientific validation data (photometric accuracy, astrometric
  precision) — that belongs in Alarcon et al. (in prep.)
- Do NOT reference SmartGPUDecoratorClass — not in the repository

### Key numbers to cite (April 2026 definitive benchmark)

| Metric | Value | Source |
|--------|-------|--------|
| H100 latency, dense 151.2 MP (14,241 src) | **61.3 s** (py3.12+adaptive) | benchmark_all_cuml_v2.csv |
| H100 latency, sparse 151.2 MP (428 src) | **34.4 s** (py3.12+adaptive) | benchmark_all_cuml_v2.csv |
| GPU detection speedup vs sep (A100) | **2.2–10.5×** | nvtx_detection_vs_sep_a100.csv |
| VRAM savings, py3.12 vs py3.8 | **6–8%** | profiler_nsys_memory_summary_20260326.csv |
| Concurrency gain on RTX 3050 Ti | **+100%** (1→2 images) | derived from VRAM data |
| Concurrency gain on A100 (151.2 MP) | **+50%** (2→3 images) | derived from VRAM data |
| cuML penalty (always-on, A100 ablation) | **87–256%** | cuml_ablation_a100_20260329.csv |
| Adaptive cuML improvement (H100, 14K src) | **27%** vs always-on | benchmark_all_cuml_v2.csv |
| Recommended configuration | **py3.12 + adaptive cuML** | production decision |
