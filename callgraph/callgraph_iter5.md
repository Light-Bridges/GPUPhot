# Call tree — Iteration 5

Objective: Document key features affecting `process_image` detected in iteration 4. I cover: background, conv, catalog (crossmatch) and psf (detection/PSF/grouping). Each node includes: range, type (CPU/GPU/Mixed), direct calls, transfer points, conditionals for optional deps and estimated effort.

Brief plan (what I did here):
- I opened the target files and extracted the main functions with their line ranges.
- Documented internal calls, explicit transfers (.get(), cp.asarray, .item()) and optional dependency detections (cuML).
- I proposed next steps and priorities.

Checklist for this iteration
- [x] `gpuphot/phot/background.py::get_local_background_fft` (lines 25-89)
- [x] `gpuphot/phot/conv.py` (functions: convolve_fft 28-80, get_mean_std 84-109, get_aper_kernel 164-187, fill_image 191-203, gen_apm_filter 207-244, batch_aper_kernel 248-289, fill_nan_fft 293-319)
- [x] `gpuphot/utils/catalog.py` (crossmatch wrapper and impls: _crossmatch_sources_gpu_impl 48-92, _crossmatch_sources_cpu_impl 97-134, crossmatch_sources 137-215; CUML detection 35-43)
- [x] `gpuphot/phot/psf.py` (detect_isolated_stars 123-205, create_star_dataset 209-321, _group_star_dataset_cpu_impl 324-389, _group_star_dataset_gpu_impl 392-466, group_star_dataset 471-541, get_eigen_psfs 544-573, detect_sources_psf 678-716, create_coeff_map 599-644, fit_moffat 791-879)
- [x] `docs/callgraph_iter5.md` file created with one summary per node.


---

Node: get_local_background_fft — `gpuphot/phot/background.py:25-89`
- Type: GPU (uses CuPy) — Mixed if input was NumPy (converted to cp.array internally).
- Purpose: estimate local and optionally std background, using gaussian_filter, partitioned into tiles and filled with NaNs by FFT.
- Direct calls/usage:
  - Convert the input to `cp.array` if it is not `cp.ndarray` (line 52) => CPU->GPU if applicable.
  - gaussian_filter(image, sigma=2) (line 54) — GPU operation.
  - get_mean_std(...) (line 55) — import from `gpuphot/phot/conv.py` (can use FFTs on GPU and return CuPy arrays).
  - free_gpu_mem() (lines 57, 64, 69, 77) — explicit memory management.
  - decompose_into_tiles / calculate_tile_percentiles / recompose_from_percentiles (lines 59-61) — tiling operations (these utilities are in `gpuphot/phot/utils` or `gpuphot/phot/utils.py`).
  - fill_nan_fft(...) (line 82) loop until NaNs are filled (use conv.py fill_nan_fft).
- GPU↔CPU transfer points:
  - If the input `image` is NumPy, the function does `cp.array(image)` (CPU->GPU) at line 52.
  - Use `.get()` not explicit here, but internal calls (for example on `get_mean_std`) can make transfers.
- Optional conditionals/deps: none explicit per library; use CuPy directly.
- Effort: Low/Medium — compact but important function (uses tiling and iterative filling).
- Next: Open tiling utilities (`decompose_into_tiles`, `calculate_tile_percentiles`, `recompose_from_percentiles`) if transfer details are needed.

---

Node: convolve_fft — `gpuphot/phot/conv.py:28-80`
- Type: GPU (CuPy) / Mixed if NumPy arrays are passed but module convention usually requires CuPy.
- Purpose: FFT convolution with optional padding and roll/crop.
- Direct calls:
  - cp.fft.rfft2 / cp.fft.irfft2 (lines 56, 69)
  - cp.multiply for frequency multiplication (line 63)
  - use of cp.pad (line 50)
- Transfer points: none explicit (works on GPU if receiving CuPy arrays). If you receive NumPy the conversion would be before the call.
- Performance notes: padding can increase memory significantly (discussed on line 51). There are multiple nvtx ranges for perf tracing.
- Effort: Medium (short and critical function for performance).

Node: get_mean_std — `gpuphot/phot/conv.py:84-109`
- Type: GPU
- Purpose: calculate mean and std using `gen_apm_filter` + `convolve_fft` (uses CuPy).
- Direct calls:
  - gen_apm_filter(lk) (line 99)
  - convolve_fft(im_g, k_app) (line 100)
  - convolve_fft(im_g * im_g, k_app) for std (line 102)- Transfers: not explicit; works at CuPy. If im_g comes from NumPy, the conversion to cp.asarray occurs at line 98.
- Effort: Low.

Node: get_aper_kernel — `gpuphot/phot/conv.py:164-187`
- Type: GPU (generates kernel in CuPy and returns kernel and area as CuPy)
- Purpose: generate circular kernel (float radius) and area.
- Notes: raise ValueError if size even (line 176). Used extensively by `batch_aperture_photometry` and `aperture_photometry`.
- Effort: Low.

Node: fill_image — `gpuphot/phot/conv.py:191-203`
- Type: CPU/GPU neutral (only shape calculation using numpy/cupynumeric)
- Purpose: rounds dimensions to powers of 2 for FFT (returns shape).
- Effort: Low.

Node: gen_apm_filter — `gpuphot/phot/conv.py:207-244`
- Type: GPU
- Purpose: generates aperture filter (normalized circular mask) and optional normalization.
- Direct calls: use cp.indices, cp.sum, boolean operations.
- Effort: Low.

Node: batch_aper_kernel — `gpuphot/phot/conv.py:248-289`
- Type: GPU
- Purpose: generate batch kernels (vectorized) for multiple radios.
- Direct calls: conversion of lists/np arrays to cp.array (line 260), mask broadcasting (line 282).
- Effort: Low/Medium (useful for optimizations if many kernels are generated in a row).

Node: fill_nan_fft — `gpuphot/phot/conv.py:293-319`
- Type: GPU
- Purpose: fill NaNs using convolutions and neighbor masks.
- Direct calls:
  - gen_apm_filter(lk, li) (line 309)
  - convolve_fft(not_nan_mask, k_app) and convolve_fft(image_zeroed, k_app) (lines 312, 315)
- Transfers: not explicit; works at CuPy.
- Effort: Medium (it is important for `get_local_background_fft` and `create_aperture_corrections_map_gpu`).

---

Node: crossmatch (cuML/CPU) — `gpuphot/utils/catalog.py` (summary range)
- CUML detection (lines 35-43): `CUML_AVAILABLE` boolean set if `cuml` imported successfully.
- GPU impl: `_crossmatch_sources_gpu_impl` — `gpuphot/utils/catalog.py:48-92` (uses cuML NearestNeighbors)
  - Requires inputs CuPy arrays.
  - Returns indexes as CuPy arrays.
  - In case of error it throws RuntimeError.
- CPU impl: `_crossmatch_sources_cpu_impl` — `gpuphot/utils/catalog.py:97-134` (uses SciPy KDTree)
  - Requires inputs NumPy arrays.
- Wrapper: `crossmatch_sources` — `gpuphot/utils/catalog.py:137-215`
  - Detects input type (CuPy vs NumPy) and if `CUML_AVAILABLE` tries GPU.
  - If GPU fails or is not available, transfer (CPU fallback): `source_np = source_coords.get()` and `ref_np = ref_coords.get()` (lines 186-189) (GPU->CPU transfer)
  - Call CPU impl and then, if the original input was GPU, transfer the results CPU->GPU with `cp.asarray` (lines 205-207).
- Key points / transfers:
  - GPU->CPU transfer in fallback (lines 186-189).
  - Transfer CPU->GPU results (lines 205-207).
- Effort: Medium. Critical for `perform_opt_photometry` and `find_aperture_corrections_gpu` decisions.

---

Node: detect_isolated_stars — `gpuphot/phot/psf.py:123-205`
- Type: GPU (uses CuPy and GPU convs), but mixes with small operations on CPU (e.g. get_centroids_distance_kdtree which uses KDTree on CPU on `.get()` of coords).
- Flow / direct calls:
  - Build kernel with `gaussian_kernel` (conv.py)
  - Call `adaptive_memory_management` (utils.gpu)
  - Calls `detect_sources_psf` (line 170) which returns coordinates and `conv_sigma` (this function uses `convolve_fft`)
  - Calculate distances with `get_centroids_distance_kdtree(coor_f.get())` (line 175) => TRANSFER GPU->CPU (`.get()`)
  - Use `.get()` in distances and sorting for small metrics (line 195 `snr[m].get() + dist[m.get()]`)
  - Returns `coor_f` as a CuPy array.
- Transfer points:
  - `coor_f.get()` for KDTree-based distances (line 175): GPU->CPU (moderate arrays)
  - use of `.get()` in sorting (line 195)
- Effort: High (important feature with CPU/GPU mix and KDTree dependencies)

Node: create_star_dataset — `gpuphot/phot/psf.py:209-321`
- Type: GPU (operates on CuPy; convert coords/img to cp.asarray if necessary)
- Flow:
  - Iterates coordinates and generates cutouts (subimages) for each candidate, applies filters and saves in `star_dataset_full` (CuPy arrays).- At the end it filters and returns `selected_star_dataset`, `selected_coords`, `selected_scaling_dataset` as CuPy arrays.
- Transfer points: none explicit (convert inputs to cp.asarray if they are not).
- Effort: Medium/High (coordinate loop; potentially memory intensive)

Node: Grouping (group_star_dataset) — `gpuphot/phot/psf.py`:
- GPU impl: `_group_star_dataset_gpu_impl` — `gpuphot/phot/psf.py:392-466`
  - Use cuML AgglomerativeClustering and cuML pairwise_distances if `CUML_CLUSTERING_AVAILABLE`.
  - If GPU fails, wrapper transfers to CPU and uses `_group_star_dataset_cpu_impl`.
- CPU impl: `_group_star_dataset_cpu_impl` — `gpuphot/phot/psf.py:324-389` (sklearn Agglomerative clustering + reassignment of small groups using distances)
- Wrapper: `group_star_dataset` — `gpuphot/phot/psf.py:471-541`
  - If input is CuPy and cuML clustering is available, try GPU.
  - If GPU fails or unavailable, TRANSFER GPU->CPU (coords.get() line 513) and use CPU impl. If original input was GPU, transfer result back with `cp.asarray` (line 534).
- Transfer points:
  - coords.get() (GPU->CPU) on fallback (line 513)
  - cp.asarray(result_np) (CPU->GPU) on fallback return (line 534)
- Effort: Medium (important due to CPU/GPU branches and possible transfer cost)

Node: get_eigen_psfs — `gpuphot/phot/psf.py:544-573`
- Type: Mixed (uses PCA; converts `normed_star_dataset.reshape(...).get()` to line 568 -> TRANSFER GPU->CPU for PCA.fit)
- Flow: flattens stars and calls `PCA.fit` (to CPU or cuML PCA depending on availability). In the current implementation it calls `.get()` and uses PCA (possible CPU-bound). Returns eigen_psfs (NumPy/CPU arrays or converted according to impl).
- Transfer points: `.get()` on line 568 before `pca.fit`.
- Effort: Medium.

Node: detect_sources_psf — `gpuphot/phot/psf.py:678-716`
- Type: GPU
- Flow:
  - Use `convolve_fft` with `flipped_psf` (line 704)
  - If there are eigen_psfs and coeff_map, add convolutions and multiply by coeff_map (line 706-708)
  - Calculate A = calculate_kernel_area(...) (line 709) and conv_ima_sigma = conv_ima_pca / rms / sqrt(A)
  - Call `find_local_max` and `find_local_centroid` (lines 711-712)
- Transfer points: none explicit (operates in CuPy). Some operations may involve reductions that return scalars in CPU later, but you don't see `.get()` here.
- Effort: Medium/High (key for detection and affects `detect_isolated_stars`).

Node: create_coeff_map — `gpuphot/phot/psf.py:599-644`
- Type: GPU
- Flow: Place coefficients on a map, use `decompose_into_tiles` and `calculate_tile_nanmean_sigclip`, fill NaNs with `fill_nan_fft` (line 634) and recompose_from_percentiles.
- Transfer points: none explicit; everything in CuPy unless you use utilities that do `.get()`.
- Effort: Medium.

Node: fit_moffat — `gpuphot/phot/psf.py:791-879`
- Type: Mixed (operates with xp = cp.get_array_module -> uses CuPy when GPU is present, but lmfit requires NumPy -> does `.asnumpy()` / `.asnumpy()` on lines 853-856)
- Flow:
  - Prepare r, Z arrays in the appropriate module (np or cp)
  - If xp == cp, convert `r` and `Z` to NumPy with `cp.asnumpy` before calling `lmfit` (lines 853-856)
  - Returns `r, Z, result, fwhm, fwhm_err` (NumPy arrays and scalars)
- Transfer points:
  - cp.asnumpy(r) and cp.asnumpy(Z) (GPU->CPU) to use `lmfit` on CPU
- Effort: Medium (important for FWHM calculation; inevitable transfers due to lmfit)

---

Next steps (Proposed Iteration 6)
1. Generate `docs/callgraph_iter5.json` with node structure (id, path_range, type, calls, transfers, optional_deps). This will allow you to convert to DOT or display.
2. Open tiling utilities and referenced helpers (`gpuphot/phot/utils.py` or `gpuphot/phot/utils/*`) to check if they use `.get()` or internal transfers.
3. Add automatic validation scripts:
   - Verify that `path:start-end` points to a real `def` and that the body contains the detected calls.
   - Search for `.get()`, `.item()`, `cp.asarray`, `try: import cuml` patterns in the ranges.4. Optional: create a small DOT viewer with the current nodes (export JSON -> DOT).

If you want me to do any of these steps now (for example: generate the JSON, open the tiling utilities, or create and run the automatic validator), tell me which one and I will do it in this same iteration.