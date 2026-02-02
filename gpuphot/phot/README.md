# gpuphot.phot

The `gpuphot.phot` package contains the core photometry algorithms used by
GPUPhot. The code is performance-focused and uses CuPy (GPU) for heavy
numeric computation, with CPU fallbacks where necessary.

Modules overview
----------------
- `background.py` — Local background estimation (FFT-based) and tile-based
  percentile recomposition.
- `conv.py` — FFT convolution helpers, kernel generators and utilities for
  aperture and Gaussian kernels.
- `cosmetics.py` — Small image filters (salt-and-pepper removal, cosmic-ray
  filtering) used as pre-processing steps.
- `photo_gpu.py` — The main processing pipeline: star detection, PSF fitting,
  aperture corrections, optimized photometry and overall image calibration.
- `psf.py` — PSF detection, creation of star datasets, grouping stars into
  clusters, eigen-PSF extraction and related helpers.
- `utils.py` — Small utilities for tile decomposition, percentile calculations
  and recomposition helpers.

Design notes
------------
- The implementation heavily optimizes for GPUs using CuPy and offloads CPU
  paths only when necessary. The code contains many NVTX ranges for
  profiling and careful calls to free GPU memory.
- `photo_gpu.py` is intentionally large; it orchestrates the full
  photometric pipeline. Consider reviewing `FIXERS.md` for suggested refactors.

Environment variables and runtime
---------------------------------
The photometry code reads certain environment and configuration values from
`instrument_configs` and other parser modules. During runtime it expects
CuPy-enabled CUDA environment for GPU acceleration; fallback to CPU may be
slower.

Non-documentation fixes
-----------------------
Any functional changes discovered while documenting files were recorded in
`FIXERS.md` within this directory. Please review and address those in a
separate code-fix pass.

Testing recommendations
-----------------------
- Add unit tests for each utility (conv, utils) using small synthetic arrays.
- For `photo_gpu.py` and `psf.py`, add integration tests that exercise both
  GPU and CPU paths (mock GPUtil or run in CI with and without CUDA).

Examples
--------
- Quick background estimation:

```python
from gpuphot.phot.background import get_local_background_fft
import cupy as cp
img = cp.random.rand(1024, 1024)
bg, std = get_local_background_fft(img, pxscale=0.5)
```

- Get a singleton logger for the phot module:

```python
from gpuphot.logger.hierarchical_logging import setup_logger
logger = setup_logger('gpuphot.phot')
logger.info('Starting photometry')
```

Contact
-------
If you want, I can proceed to implement fixes in `FIXERS.md` in a dedicated
branch and open a PR for review.
