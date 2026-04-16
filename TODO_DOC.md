# Documentation TODO

Gaps identified during documentation review (2026-04-16).
Ordered by priority: Critical → Useful → Nice-to-have.

---

## CRITICAL

### C1 — Error handling (USAGE.md)
GPUPhot defines 8+ custom exceptions in `gpuphot/exceptions.py`
(`GPUPhotError`, `InsufficientStarsError`, `MoffatFitError`,
`UnableToAstrometrizeError`, etc.) but USAGE.md has no error-handling section.
Users calling `process_image()` do not know what to catch or what each
exception means.
**Action:** Add "Error Handling" section to USAGE.md with exception hierarchy
and example try/except patterns.
**Status:** done — USAGE.md section 7

### C2 — `forced_values` in instrument configs (USAGE.md)
The config parser supports a `forced_values` block to override incorrect or
missing FITS header values (gain, read noise, etc.).  This is critical for
real-world data with broken headers, but there is no example in USAGE.md
section 1.
**Action:** Add subsection with JSON example showing `forced_values` usage
to USAGE.md section 1 (Instrument Configuration).
**Status:** done — USAGE.md section 1.2

### C3 — GPU memory requirements & OOM guidance (INSTALL.md)
INSTALL.md only says "16 GB VRAM recommended" without quantitative detail.
We have measured peak VRAM data from nsys profiling across all 6 GPUs:

Reference workload: 4.2 MP image (iKon936), full pipeline
  - py3.8  (CuPy 12): ~2018 MB peak VRAM
  - py3.12 (CuPy 14): ~1718 MB peak VRAM  (-15 %, memory optimisation)

Theoretical concurrent images per GPU (floor(VRAM × 0.95 / peak)):
  | GPU              | VRAM  | py3.8 | py3.12 |
  |------------------|-------|-------|--------|
  | H100 PCIe        | 80 GB |  38   |  44    |
  | A100-SXM4        | 80 GB |  38   |  44    |
  | L40S             | 48 GB |  22   |  26    |
  | RTX 3090         | 24 GB |  11   |  13    |
  | RTX 3060         | 12 GB |   5   |   6    |
  | RTX 3050 Ti      |  4 GB |   1   |   2    |

Image sizes tested (from benchmark data):
  - 4.2 MP  (296–412 sources)  — all GPUs OK
  - 6.8 MP  (112–2060 sources) — all GPUs OK
  - 15.3 MP (218–4361 sources) — RTX 3050 Ti OOMs, others OK

`reset_cupy_allocators()` in `gpuphot.utils.gpu` can be called between
batches to release the CuPy memory pool if VRAM is tight.
**Action:** Add "GPU Memory Requirements" subsection to INSTALL.md with
the table above, OOM notes, and reset_cupy_allocators() guidance.
**Status:** done — INSTALL.md Troubleshooting section

---

## USEFUL

### U1 — Database schema & result querying (DOCKER.md or DATABASE.md)
Docker deployment writes photometry results to PostgreSQL table `imaphot`
(via `gpuphot_worker/database_insert_utils.py`).  `database_search_utils.py`
exists for querying but is undocumented.  Users don't know where results go
or how to retrieve them.
**Action:** Add "Result Persistence" section to DOCKER.md or create DATABASE.md
with table schema and example queries.
**Status:** pending

### U2 — Celery task results: TTL, serialisation, worker crash (USAGE.md)
USAGE.md mentions `task.get()` but not Redis TTL, DataFrame serialisation,
or what happens if a worker crashes before persisting to the DB.
**Action:** Extend USAGE.md section 4.4 with a "Managing Results" note.
**Status:** pending

### U3 — Supported filters & filter mapping (USAGE.md or CUSTOM_CATALOG.md)
The pipeline maps filter names to canonical codes (g, r, i, …) for
cross-matching.  Supported filters per catalog are listed in CUSTOM_CATALOG.md
but the mapping logic and how to add custom filters is not documented.
**Action:** Add "Supported Filters" subsection to USAGE.md section 1 or
CUSTOM_CATALOG.md.
**Status:** pending

### U4 — Astrometry solver timeouts & recovery (USAGE.md)
`get_solver()` is documented briefly but timeout behaviour,
`AstrometrizationTimeoutError`, and recovery strategies are not explained.
**Action:** Add "Troubleshooting Astrometry" subsection to USAGE.md section 2.
**Status:** pending

---

## NICE-TO-HAVE

### N1 — Logging API: decorators, handlers, Logstash (USAGE.md)
`hierarchical_debug` decorator, `NotifyingHandler`, and `LOGSTASH_LOGGING`
env var are useful for production monitoring but are not mentioned in user docs.
**Action:** Brief "Advanced Logging" note in USAGE.md or logger/README.md.
**Status:** pending
