# gpuphot.stats

Utilities for statistical operations used in the gpuphot pipeline.

Modules:
- `reduction.py` — Image stack reduction helpers (subpixel registration, sigma-clipping, frame registration).
- `subpixel.py` — High-precision subpixel registration (Guizar-Sicairos method, CuPy port).
- `subpixel_masked.py` — Masked normalized cross-correlation implementation (Padfield).
- `s_util.py` — Small supporting utilities (dtype handling).
- `common.py` — Module version and default constants.

Notes:
- Many functions in this package have GPU-accelerated (CuPy) implementations. Ensure CuPy is available for GPU execution.
- NVTX annotations are used for profiling; if NVTX is not installed, the code falls back to no-op decorators.
- No functional changes were made in this documentation pass—only docstrings, module headers and English translations were added.
