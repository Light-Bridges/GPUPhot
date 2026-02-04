# Call tree — start from `process_image` (iteration 1)

**Root file:** `gpuphot/phot/photo_gpu.py`  
**Note:** Partial document. Expand nodes marked with \[EXPAND\] in subsequent iterations.

- `process_image` — `gpuphot/phot/photo_gpu.py` \[ROOT; \[EXPAND\]\]
  - Observed direct calls (first pass):
    - `calibrate_image` — `gpuphot/phot/photo_gpu.py` \[EXPAND\]
    - `perform_opt_photometry` — `gpuphot/phot/photo_gpu.py` \[EXPAND\]
    - `batch_aperture_photometry` — `gpuphot/phot/photo_gpu.py` \[EXPAND\]
    - referenced utility functions (possible external modules):
      - `create_processor` (instance `ImageProcessor`) \[search in project\]
      - `crossmatch_sources` \[search in project\]
      - `create_aperture_corrections_map` \[search in project\]
      - `find_aperture_corrections` \[search in project\]
      - `get_aper_kernel` \[search in project\]
      - `fill_image`, `get_local_background_fft`, `get_mean_std`, `gen_apm_filter`, `convolve_fft`, `decompose_into_tiles` \[EXPAND\]

- `calibrate_image` — `gpuphot/phot/photo_gpu.py` \[EXPAND\]
  - GPU usage: convert input to CuPy `cp.asarray(imdata)`
  - Important internal steps to expand:
    - NaN/Inf cleanup
    - star detection (min_snr)
    - astrometry/references (zp, color_range)
    - PSF/PCA function calls (pca_method)
    - return: `(calib_dict, astrometry_dict, photometry_df)`

- `perform_opt_photometry` — `gpuphot/phot/photo_gpu.py` \[EXPAND\]
  - Main blocks (by number):
    - Selection of central stars
    - SNR processing and validation (`crossmatch_sources`)
    - Opening radius configuration
    - `create_aperture_corrections_map` and `find_aperture_corrections`
    - `batch_aperture_photometry`
    - Signal/noise calculation, setting `np.polyfit`
    - Calculation of optimal flow and generation of `extra_info`
  - Conditionals to mark:
    - Fallbacks between CPU/GPU (`.get()` / `.item()` transfers)
    - Validation of enough stars (`InsufficientStarsError`)
    - Using `n_images` for noise scaling

- `batch_aperture_photometry` — `gpuphot/phot/photo_gpu.py` \[EXPAND\]
  - Variants in the file: multiple commented implementations and one active (annotated)
  - Critical steps:
    - calculation of opening areas (`get_aper_kernel`)
    - FFT and radius convolution loop
    - 2D vs 3D handling (n_images)
    - memory transfers and management (`mempool`, OOM handling)
    - GPU↔CPU transfer points documented in `process_image` / `perform_opt_photometry`
  - Conditional branches:
    - input as `np.ndarray` vs `cp.ndarray`
    - behavior in OOM (returns `None` or exception)

## Pending tasks / next steps (per iterations)

1. Open the complete definition of `process_image` in `gpuphot/phot/photo_gpu.py` and list direct / conditional calls with line of code (start-end). \[PRIORITY\]
2. Expand each listed call by opening its file and repeat recursively to base utilities (`get_aper_kernel`, `fill_image`, etc.). Mark optional dependencies (`cuml`, Python versions).
3. Identify and write down all GPU↔CPU transfer points and their estimated cost (e.g.: `.get()`, `.item()`, large array copies).
4. Generate sections of `callgraph.md` with:
   - Node: `func` — `path:line_start-line_end`
     - Type: CPU / GPU / Mixed
     - Direct calls: [...]
     - Conditionals / flags: [...]
     - Notes: transfers, exceptions, estimated cost
5. Deliver partial iterations per file to avoid timeouts and facilitate review.

## Proposed format for node expansion

- Node: `func_name` — `path/to/file.py:line_start-line_end`  
  - Type: CPU \| GPU \| Mixed  
  - Direct calls:
    - `callee()` — `module/path.py:lineno`
  - Conditionals / flags:
    - `if use_cuml: ... else: fallback`
  - Notes: GPU↔CPU transfers, possible exceptions, estimated cost.