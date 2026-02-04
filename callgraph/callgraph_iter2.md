# Call tree — start from `process_image` (iteration 2)

**Root file:** `gpuphot/phot/photo_gpu.py`  
**Goal of this iteration:** Expand the `process_image` node with direct calls, classify type (CPU/GPU/Mixed) and list the subnodes to open in the next pass.

## Root node

- `process_image` — `gpuphot/phot/photo_gpu.py` \[ROOT; \[EXPAND MORE\]\]
  - Type: Mixed (CPU input -> GPU processing and various transfers)
  - Signature/purpose: Main orchestrator that receives image data, options and returns calibration/photometry results.
  - General structure (logical blocks):
    1. Input preparation (ensure format, convert to `cp.asarray` if available)
    2. Calibration (call `calibrate_image`)
    3. Optimized photometry (call `perform_opt_photometry`)
    4. Aperture photometry batch / alternatives (call `batch_aperture_photometry`)
    5. Collection of results, `.get()`/`.item()` transfers and memory cleaning
  - Direct calls observed:
    - `calibrate_image()` — `gpuphot/phot/photo_gpu.py`
      - Type: Mixed (mostly GPU for computations; CPU for adjustments/returns)
      - Notes: convert input to `cp.asarray`; clear NaN/Inf; detects fonts; configure calibration parameters.
    - `perform_opt_photometry()` — `gpuphot/phot/photo_gpu.py`
      - Type: Mixed (GPU for convolutions and additions; CPU for `np.polyfit` adjustments and validations)
      - Notes: select stars, build correction maps, call `batch_aperture_photometry`.
    - `batch_aperture_photometry()` — `gpuphot/phot/photo_gpu.py`
      - Type: GPU intensive (FFT, convolutions, mempool)
      - Notes: multiple variants; intensive use of `cupy.fft` and handling of OOM.
    - `create_processor()` — (factory; locate)
      - Type: CPU (instantiation); produces `ImageProcessor` used by `process_image`.
    - Referenced utilities (locate and expand):
      - `crossmatch_sources` — locate (possible CPU/external structure usage)
      - `create_aperture_corrections_map` — locate
      - `find_aperture_corrections` — locate
      - `get_aper_kernel` — locate (generates opening kernels)
      - `fill_image` — locate
      - `get_local_background_fft` — locate
      - `get_mean_std` — locate
      - `gen_apm_filter` — locate
      - `convolve_fft` — locate
      - `decompose_into_tiles` — locate

  - Critical points / GPU↔CPU transfers:
    - Use of `.get()` in intermediate results to use `numpy`/`scipy`/`np.polyfit`.
    - Scale of final data returned to the user (possible conversion to `np.ndarray`).
    - Atomic operations or reductions that return scalars and use `.item()`.

  - Exceptions/validations detected:
    - `InsufficientStarsError` in `perform_opt_photometry`
    - Shape and type validations (np.ndarray vs cp.ndarray)
    - OOM handling in `batch_aperture_photometry` (mempool release, fallback or exception)

## Immediate subnodes to analyze (priority and note)

1. `calibrate_image` — open and document internal calls, conditions and lines (PRIORITY A).
2. `perform_opt_photometry` — open and expand each block: star selection, map creation, calls to `batch_aperture_photometry` (PRIORITY A).
3. `batch_aperture_photometry` — identify active variant, locate commented implementations, and handle OOM/mempool notes (PRIORITY A).
4. Locate and open listed utilities (`get_aper_kernel`, `convolve_fft`, `decompose_into_tiles`, etc.) and classify type and conditions (PRIORITY B).
5. Detect optional dependencies in the bodies of these functions (`cuml`, `cupyx.scipy`, Python versions) and mark fallback branches (PRIORITY B).

## Actions done/pending for next iteration

- Generated this expanded node of `process_image` with type classification and list of utilities to open.
- Pending: open each listed function and add:
  - Line range (start-end)
  - Direct internal and conditional calls- Marks on GPU↔CPU transfers and estimated cost
  - Notes on bibliographic references found in docstrings/code
- Expected result after the next iteration: file `docs/callgraph_iter3.md` with the subnodes `calibrate_image`, `perform_opt_photometry` and `batch_aperture_photometry` broken down with lines and recursive calls.

## Target format per node (example for next iterations)

- Node: `func_name` — `path/to/file.py:line_start-line_end`  
  - Type: CPU \| GPU \| Mixed  
  - Direct calls:
    - `callee()` — `module/path.py:line`  
  - Conditionals / flags:
    - `if cuml_available: ... else: fallback`  
  - Notes: GPU↔CPU transfers, exceptions, references to papers (if any).