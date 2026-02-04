# Call tree — start from `process_image` (iteration 4)

Root file: `gpuphot/phot/photo_gpu.py`

Goal of this iteration: extract line ranges (start-end) and document the immediate nodes around `process_image` with direct calls, type (CPU/GPU/Mixed), transfer points and optional dependencies detected in the code.

Quick summary of what was done:
- Read `gpuphot/phot/photo_gpu.py` and located the key functions.
- Documented the nodes: `process_image`, `perform_opt_photometry`, `batch_aperture_photometry`, `create_aperture_corrections_map_gpu`, `find_aperture_corrections_gpu`, `calibrate_image`, `batch_aperture_photometry` (commented and active variants), and invoked utilities.

Format per node (template used):
- Node: `id` — `path:start-end`
  - Type: CPU | GPU | Mixed
  - Summary: short purpose
  - Direct calls within the range (callee — path:line_call)
  - GPU↔CPU transfer points (lines/mechanism)
  - Optional conditionals/deps (how they are detected)
  - Notes / estimated cost / next steps

---

Node: process_image — gpuphot/phot/photo_gpu.py:625-697
- Type: Mixed (mostly CPU orchestrating calls that execute work on GPU)
- Summary: High entry point. Extract parameters from the header, construct `params`, call `calibrate_image`, and update the header with astrometry/photometry.
- Direct calls observed:
  - calibrate_image(...) — gpuphot/phot/photo_gpu.py:669 (call site)
  - update_header_with_astrometry(...) — gpuphot/phot/photo_gpu.py:690 (import from utils.headers)
  - update_header_with_photometry(...) — gpuphot/phot/photo_gpu.py:692
- Observed transfer points: none direct in `process_image` (transfers occur in callee).
- Optional conditionals/deps: use `DefaultConfig.DEFAULT_PROCESSING_PARAMS` (configurable); can propagate flags that condition branches on streets (e.g. `pca_method`, `min_snr` if activated from params).
- Notes: exact range detected. Next step: open `calibrate_image` (already included below) to expand GPU/CPU flow.

---

Node: calibrate_image — gpuphot/phot/photo_gpu.py:2837-3145
- Type: Mixed (input -> GPU with `cp.asarray`, calculations and detection on GPU; then transfers to CPU for scalars and astrometry)
- Summary: Normalize image, clean NaN/Inf, estimate background, detect stars, build star dataset, obtain PSF, calculate FWHM, prepare data for `perform_opt_photometry`.
- Direct calls observed within range:
  - cp.asarray(imdata) (line ~2849) — CPU->GPU conversion
  - get_local_background_fft(img_cp, scale, get_std=False, **kwargs) — gpuphot/phot/photo_gpu.py:2903 (call site)
  - detect_isolated_stars(...) — imported from .psf
  - create_star_dataset(...) — psf helper
  - stack_sigmaclip(...) — utils.stats
  - fit_moffat(psf) — psf helper
  - get_eigen_psfs(...), project_all_stars_onto_eigenpsfs(...), create_coeff_map(...) — PSF PCA path
  - detect_sources_psf(...) — source detection
  - perform_opt_photometry(...) — gpuphot/phot/photo_gpu.py:3080 (call site)
  - astrometrice2(...) — utils.astro (astrometry, may fail)
  - catalog_results(...), get_zeropoint(...), get_maglim(...), crossmatch_sources(...) — CPU utilities
- GPU↔CPU transfer points:
  - Conversion: `img_cp = cp.asarray(imdata)` (CPU->GPU)
  - Scalar extraction: `fluxsky = np.round(m.get(), 6)` (GPU->CPU) — line near 2931
  - In the end, `dfm` is constructed with `optimal_coords[:, 1]` etc. `perform_opt_photometry` returns NumPy arrays (via `.get()` on its return), so the final transfer happens in the callee — here `calibrate_image` receives `optimal_flux, optimal_noise, optimal_coords, extra_info` and then uses `optimal_coords` to build the DataFrame (already on CPU).
- Optional conditionals/deps detected:
  - `pca_method` activates the PCA branch of PSF (`get_eigen_psfs`, `project_all_stars_onto_eigenpsfs`)
  - If `len(sources) < 5` throws `InsufficientStarsError`.
  - Use of `mem_pool.free_all_blocks()` and `reset_cupy_allocators()` to manage GPU memory.
- Notes / effort: Medium (200+ lines, GPU/CPU mix and several utility calls). Next: open `get_local_background_fft`, `detect_isolated_stars`, `perform_opt_photometry`.---

Node: get_local_background_fft — called in calibrate_image (call site: gpuphot/phot/photo_gpu.py:2903)
- Source file: imported from `..phot.background` (look for file `gpuphot/phot/background.py` or similar in the project).
- Recommended actions: open `gpuphot/phot/background.py` and locate `get_local_background_fft` to document its type (probable use of FFT on GPU) and transfer points.

---

Node: perform_opt_photometry — gpuphot/phot/photo_gpu.py:700-1009
- Type: Mixed (GPU intensive with one-time transfers to CPU for polyfit and returns)
- Summary: Star selection, SNR calculation, creation of aperture correction maps (call create_aperture_corrections_map_gpu), batch photometry, calculation of photometric parameters and radius optimization.
- Direct calls observed within range:
  - adaptive_memory_management(...) — utils.gpu
  - crossmatch_sources(...) — utils.catalog (called in GPU mode, may raise ImportError if GPU dependency is missing)
  - create_aperture_corrections_map_gpu(...) — gpuphot/phot/photo_gpu.py:832 (call site)
  - find_aperture_corrections_gpu(...) — gpuphot/phot/photo_gpu.py:840 (call site)
  - batch_aperture_photometry(...) — gpuphot/phot/photo_gpu.py:850 (call site)
  - xp = cp.get_array_module(...) -> dynamic use numpy/cupy (decide polyfit path)
  - xp.polyfit / xp.log10 -> if xp==cp try polyfit on GPU; in many environments `cupy.polyfit` may not exist or behave differently → the active version uses `.get()` or exceptions and finally `np.polyfit` on CPU in commented branches
  - returns: opt_signal.get(), opt_total_noise.get(), opt_coords.get(), extra_info — final transfer GPU->CPU
- GPU↔CPU transfer points:
  - Various one-time transfers:
    - center_conv_snr_np = center_conv_snr.get() and opt_radii_np = opt_radii_gpu.get() for polyfit (lines near block8)
    - ret: opt_signal.get(), opt_total_noise.get(), opt_coords.get() (line 1009)
    - use of `.item()` in multiple places for scalars (1 element transfers)
  - Internally `create_aperture_corrections_map_gpu` and `find_aperture_corrections_gpu` handle transfers based on GPU/CPU grouping success.
- Optional conditionals/deps:
  - GPU crossmatch failure handling: except ImportError -> log and raise
  - Module runtime selection (xp = cp.get_array_module) that decides CPU/GPU routes for polyfit
  - InsufficientStarsError in validations
- Notes/effort: High (extensive feature with many branches, transfers and loops). Next: Document `create_aperture_corrections_map_gpu`, `find_aperture_corrections_gpu`, `batch_aperture_photometry` in detail.

---

Node: create_aperture_corrections_map_gpu — gpuphot/phot/photo_gpu.py:388-565
- Type: Mixed (GPU attempt for clustering and calculation; CPU fallback with transfers)
- Summary: Calculates aperture curves with `batch_aperture_photometry` on `unit_star_dataset`, groups stars into clusters and calculates corrections per cluster. If `group_star_dataset` returns `np.ndarray` it falls to CPU path with transfers.
- Direct calls observed:
  - batch_aperture_photometry(unit_star_dataset, None, positions_psf, radii) — gpuphot/phot/photo_gpu.py:418
  - group_star_dataset(...) — imported from .psf (wrapper that can return cp.ndarray or np.ndarray)
  - calculate_aperture_corrections_gpu(...) — gpuphot/phot/photo_gpu.py:?? (call inside loop)
  - calculate_aperture_corrections(...) — numpy version (in CPU branch)
  - cp.asarray(...) and .get() on CPU branches
- GPU↔CPU transfer points:
  - Transfer of `aperture_curves_cp.get()` curves to the CPU fallback (line ~433)
  - Transfer of `coords.get()` coords if necessary
  - Final transfer: CPU -> GPU results with `cp.asarray` (line ~523)
- Optional conditionals/deps:
  - `if isinstance(labels, cp.ndarray)` determines GPU path vs CPU fallback
- Notes/effort: Medium (not very long function but with loops and transfers). Important because it triggers costly transfers.

---

Node: find_aperture_corrections_gpu — gpuphot/phot/photo_gpu.py:301-353- Type: GPU
- Summary: Perform crossmatch between `sources` and `cluster_centers` trying to keep everything on GPU; indexes fixes by tile_idx.
- Direct calls:
  - crossmatch_sources(sources, cluster_centers, thres_px=int(cp.max(sources))) — utils.catalog (may return cp.ndarray or np.ndarray depending on implementation)
- Transfer points:
  - Does not explicitly transfer in its normal flow; assumes inputs on GPU. Raises TypeError if it receives indexes on CPU.
- Notes: Low effort; critical node to ensure that crossmatch implements fallback.

---

Node: batch_aperture_photometry (active deployment) — gpuphot/phot/photo_gpu.py:1789-2423
- Type: GPU (accepts NumPy and CuPy; transfer CPU->GPU if necessary)
- Summary: v4 implementation: handles NumPy or CuPy inputs, transfers to GPU if necessary, calculates areas, prepares FFTs and processes planes with streams for concurrency; handles OOM and returns None on failure.
- Direct calls observed:
  - cp.asarray() — initial transfers if inputs are NumPy
  - get_aper_kernel(...) — kernel generation (multiple calls in spoke loop)
  - fill_image(...) — to calculate fft_shape
  - cp.fft.rfft2 / cp.fft.irfft2, cp.roll — FFT/roll operations
  - process_plane inner function (use get_aper_kernel and convolve via FFT)
- GPU↔CPU transfer points:
  - At startup: if `img_ori_input` is np.ndarray -> `img_gpu = cp.asarray(img_ori_input)` (CPU->GPU)
  - Similar for background
  - Returns: final_flux, final_back_flux, areas are CuPy if success else None
  - Position/radii management: inputs `positions` and `radii` must be cp.ndarray in call
- Conditionals:
  - if positions.size == 0 returns empty arrays
  - if there is OOM in process_plane it returns None and marks `success=False`
  - Concurrency: adjust `max_concurrent` streams
- Notes/effort: High (long, concurrency, OOM handling). Important for memory and performance optimizations.

---

Other utilities detected (continue in future iterations):
- get_aper_kernel — gpuphot/phot/conv.py (use in batch_aperture_photometry and aperture_photometry). Open `gpuphot/phot/conv.py`.
- convolve_fft, fill_image, gen_apm_filter — imported from `.conv` (open same file).
- crossmatch_sources — `gpuphot/utils/catalog.py` (open to see CPU/GPU/fallbacks implementation)
- group_star_dataset, calculate_aperture_corrections_gpu, calculate_aperture_corrections — already in this file
- get_local_background_fft — `gpuphot/phot/background.py` (open)

---

Recommendations/next steps (Suggested Iteration 5):
1. Open and document `gpuphot/phot/background.py` — search for `get_local_background_fft` and extract its range/branches (GPU vs CPU, cuFFT vs fallback).
2. Open `gpuphot/phot/conv.py` — document `get_aper_kernel`, `convolve_fft`, `fill_image`, `gen_apm_filter` (direct impact on `batch_aperture_photometry`).
3. Open `gpuphot/utils/catalog.py` — identify `crossmatch_sources` and its strategy (cuml/CPU fallback) and how it returns cp.ndarray vs np.ndarray.
4. Open `gpuphot/psf` helpers if you want to break down internals (`group_star_dataset`, `create_star_dataset`, PCA helpers).
5. Generate structured output (JSON) with nodes ready to generate a DOT/visualization graph. Include fields `path_range`, `tipo`, `calls`, `transfers`, `optional_deps`.

Suggested size per iteration:
- Iteration 5: 6–10 functions (background, conv, catalog crossmatch, get_aper_kernel, group_star_dataset, create_aperture_corrections_map_gpu internals)

Proposed quick validation:
- For each node included above: verify that `path:start-end` matches `def` and `return` found (already verified for these nodes). In the next iteration I automate a script that validates ranges and searches for patterns `.get()`, `.item()`, `try/except ImportError`.

---

Final notes:
- I have prioritized nodes that cause costly transfers and runtime decisions (crossmatch, group_star_dataset, batch photometry). This allows optimizations and documentation to be focused on critical points.- If you want, in the next iteration I can: (A) open `gpuphot/phot/background.py` and `gpuphot/phot/conv.py` and document them with the same format, or (B) generate a JSON with the current nodes ready for visualization.