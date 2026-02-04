# Call tree — start from `process_image` (iteration 3)

**Root file:** `gpuphot/phot/photo_gpu.py`  
**Goal of this iteration:** Expand the `calibrate_image`, `perform_opt_photometry` and `batch_aperture_photometry` nodes with direct calls, type (CPU/GPU/Mixed), transfer points and next steps.

## Node: `process_image` — `gpuphot/phot/photo_gpu.py` \[ROOT\]
- Type: Mixed
- Summary: Orchestrator. Calls `calibrate_image`, `perform_opt_photometry`, `batch_aperture_photometry` and handles `.get()`/`.item()` transfers.
- Subnodes analyzed in this iteration: `calibrate_image`, `perform_opt_photometry`, `batch_aperture_photometry`.

---

## Node: `calibrate_image` — `gpuphot/phot/photo_gpu.py` \[EXPAND: lines, confirm\]
- Type: Mixed (input → GPU, settings → CPU)
- Purpose: normalize image, clean NaN/Inf, detect initial sources, estimate background/noise and prepare calibration parameters.
- Direct calls / frequent utilities to locate:
  - `get_local_background_fft()` — locate (possible GPU/FFT)
  - `get_mean_std()` — locate (reduction, possible transfer to CPU)
  - `gen_apm_filter()` — locate (generates aperture mask/filter)
  - `decompose_into_tiles()` — locate (if there is tile processing)
- Transfer points:
  - Input conversion: `cp.asarray(imdata)` if CuPy available.
  - Reductions or adjustments using `numpy` → `.get()` or `np.array`.
- Conditionals/flags to look for:
  - Use of `cuml`/other accelerators for detection/clustering.
  - Parameters `min_snr`, `pca_method`, `zp`, `color_range`.
- Earrings:
  - Add range of lines (start-end).
  - Open and include internal calls with lines and cost notes (memory/CPU-GPU).

---

## Node: `perform_opt_photometry` — `gpuphot/phot/photo_gpu.py` \[EXPAND: lines, confirm\]
- Type: Mixed
- Purpose: optimize flow/noise by openings; select calibration stars; adjust aperture corrections.
- Direct calls / utilities:
  - `crossmatch_sources()` — locate (possible CPU)
  - `create_aperture_corrections_map()` — locate
  - `find_aperture_corrections()` — locate
  - `batch_aperture_photometry()` — `gpuphot/phot/photo_gpu.py` (central call)
  - `np.polyfit`, `np.log10` (CPU operations after `.get()`)
- Critical points:
  - Validation of number of stars (`InsufficientStarsError`).
  - Intermediate transfers for polynomial adjustment and index selection (`.get()`, `.item()`).
- Conditionals:
  - Branches when enough SNR or stars are missing.
  - Fallbacks between GPU/CPU implementation of crossmatch or optimizations.
- Earrings:
  - Add range of lines and extract calls with their conditions/branches.

---

## Node: `batch_aperture_photometry` — `gpuphot/phot/photo_gpu.py` \[EXPAND: lines, confirm\]
- Type: GPU intensive
- Purpose: calculate flows by openings for a list of radios; Using FFT for convolution with aperture kernels.
- Direct calls / utilities:
  - `get_aper_kernel()` — locate (generate kernel per radio)
  - `fill_image()` — locate (if used for normalization)
  - `convolve_fft()` — locate (if there is a wrapper)
  - Direct use of `cupy.fft.rfft2`, `cupy.fft.irfft2`, `cp.roll`
- Memory management:
  - Use of `cp.get_default_memory_pool()` and release `mempool.free_all_blocks()`
  - OOM strategies: capture, free mempool, try to fallback or abort.
- Transfer points:
  - Final photometry result converted to `np.ndarray` if returned to CPU.
  - Aggregations/statistics that perform `.get()` for use in Numpy/Scipy.
- Earrings:
  - Identify active variant in the file (commented vs active).
  - Add lines and recursive calls with cost notes.

---

## Next tasks (next iteration)
1. Open `gpuphot/phot/photo_gpu.py` and extract line ranges (start-end) for `calibrate_image`, `perform_opt_photometry`, `batch_aperture_photometry`. Add them to each node.
2. Open the listed utility files (`get_local_background_fft`, `get_mean_std`, `gen_apm_filter`, `convolve_fft`, `decompose_into_tiles`, `get_aper_kernel`, `fill_image`, `crossmatch_sources`, `create_aperture_corrections_map`, `find_aperture_corrections`) and document internal calls and conditionals.
3. Mark related branches with optional dependencies (`cuml`, `cupyx.scipy`) and note fallbacks.
4. Add references to lines of code and examples of GPU↔CPU transfer when detected.

## Notes
- Maintain this document as an iterative guide; each open file will generate a section with `path:line_start-line_end`, type and direct calls.