# process_image kwargs and decision points

This document explains how `process_image` receives and forwards processing parameters (kwargs), which keys are available by default, which keys are passed but not present in defaults, and the main decision/fallback points (GPU vs CPU, custom Vizier function, exceptions).

> Source of truth lines referenced are from the repository files. Key call sites:
> - `ImageProcessor.process_image` — `gpuphot/image_processor.py` (line 74): `params = {**self.processing_params, **kwargs}` — kwargs override defaults.
> - `process_image` — `gpuphot/phot/photo_gpu.py` (lines ~624–696): high-level entry point; merges defaults similarly and calls `calibrate_image(..., **params)`.
> - `calibrate_image` — `gpuphot/phot/photo_gpu.py` (lines ~2836–2842): signature shows explicit named args + `**kwargs`; forwards `**kwargs` to deeper functions.
> - `__getVizier` — `gpuphot/utils/catalog.py` (lines ~254–333): handles `custom_vizier_search_func`, `vizier_timeout`, `vizier_row_limit`, `vizier_cache` and does validation/fallback.
> - `crossmatch_sources` (wrapper) — `gpuphot/utils/catalog.py` (lines ~136–215): chooses GPU (cuML) vs CPU (KDTree) and implements fallback and transfers.

## Short summary
- The `ImageProcessor` builds `params = {**processing_params, **kwargs}` and calls the core `process_image` with `**params`. Thus any key you pass to `ImageProcessor.process_image(..., key=value)` will override the corresponding default in `processing_params`.
- `calibrate_image` and many other functions accept `**kwargs` and forward them. Keys may therefore be consumed at different depths of the pipeline.
- Some keys come from `DefaultConfig.DEFAULT_PROCESSING_PARAMS` (documented defaults). Others are supported by functions but are not in the default config — they should be documented explicitly.

---

## Table of keys (defaults vs not)

| Key name | Default? | Where used (file:lines) | Effect / notes |
|---|---:|---|---|
| border | yes | `gpuphot/instrument_config_parser.py` DEFAULT (lines 114–129) ; used in `calibrate_image` (photo_gpu.py ~2934–2938) | Pixel border ignored for star detection |
| center_factor | yes | DEFAULT (114–129); used in `calibrate_image` (photo_gpu.py ~2946–2951) | Fraction to pick center region for reference stars |
| color_range | yes | DEFAULT; used in `calibrate_image` -> `get_zeropoint` (photo_gpu.py ~3094) | Color tolerance for photometric calibration |
| CR_filt | yes | DEFAULT; used in `calibrate_image` (photo_gpu.py ~2923–2929) | Apply cosmic-ray filter if True |
| SP_filt | yes | DEFAULT; used in `calibrate_image` (photo_gpu.py ~2925–2929) | Apply salt-and-pepper filter if True |
| pca_method | yes | DEFAULT; used in `calibrate_image` (photo_gpu.py ~2988–3003) | Enable PCA-based PSF modeling |
| tile_section | yes | DEFAULT; used while building coeff maps (photo_gpu.py ~2998–3000) | Tile size for coefficient map |
| tile_section_psf | yes | DEFAULT; passed to `perform_opt_photometry` (photo_gpu.py ~3039–3053) | PSF tile size for aperture corrections |
| max_stars_ref | yes | DEFAULT; used to select reference PSF stars (photo_gpu.py ~2962–2964) | Maximum number of stars for PSF reference |
| min_conv_snr | yes | DEFAULT; used in `perform_opt_photometry` (photo_gpu.py ~736–741) | Minimum convolution SNR for isolated-star selection |
| zp_maxmag | yes | DEFAULT; used in `calibrate_image` when querying catalog (photo_gpu.py ~3099–3115) | Maximum magnitude to include for zeropoint calculation |

Non-default (found in code; not present in DEFAULT_PROCESSING_PARAMS):

| Key name | Where used (file:lines) | Effect / notes |
|---|---|---|
| custom_vizier_search_func | `gpuphot/utils/catalog.py` `__getVizier` (lines ~254–333) | If provided (callable), `__getVizier` will call it. On timeout/error/invalid result fallback to Vizier. Useful to inject private catalog access or caching.
| vizier_timeout | `__getVizier` (lines ~256–328) | Timeout used for queries (passed to `TimeoutExecutor` and Vizier constructor).
| vizier_row_limit | `__getVizier` (lines ~256–333) | Row limit to pass to Vizier.
| vizier_cache | `__getVizier` (lines ~256–333) | Whether to use Vizier cache when querying.
| expected_columns | `__getVizier` and `catalog_results` (catalog.py ~254–333; ~380–400) | Columns expected from catalog query; used for result validation.

Notes:
- `lum_gmag_coeff` and `lum_rmag_coeff` appear in defaults and are used inside `catalog_results` (photo_gpu/photo calibs) (catalog.py lines ~476–481).
- There may exist other keys consumed by lower-level functions (e.g., `get_local_background_fft`, `detect_sources_psf`, `create_aperture_corrections_map_gpu`): they accept `**kwargs`. To be exhaustive, run an automatic search for `kwargs.get('...')` and `['...']` usage across the repo. See next steps.

---

## Decision nodes & fallbacks (to include in the flow graph)

1. kwargs merge (priority)
   - `ImageProcessor.process_image`: `params = {**self.processing_params, **kwargs}` (file: `gpuphot/image_processor.py`, line 74). This is a global rule: any key in kwargs overrides the default.

2. cuML / GPU crossmatch decision
   - `gpuphot/utils/catalog.py`: module-level detection of cuML availability sets `CUML_AVAILABLE` (lines ~34–43).
   - `crossmatch_sources` wrapper checks `is_gpu_input` (is input a `cupy.ndarray`) and `CUML_AVAILABLE` (lines ~169–176) and attempts `_crossmatch_sources_gpu_impl`.
   - On exception in GPU KNN, wrapper logs and falls back to CPU: transfers arrays to CPU, runs `_crossmatch_sources_cpu_impl`, then (if original input was GPU) transfers results back to GPU (lines ~178–211).
   - Graph node: (is_input_gpu? and CUML_AVAILABLE?) -> try GPU crossmatch -> (success) GPU indices returned; (GPU exception) -> transfer->CPU KDTree -> (return; possibly transfer back)

3. custom Vizier function decision
   - `__getVizier` checks `custom_vizier_search_func is not None` (catalog.py lines ~296–316). If present, it runs the function inside a `TimeoutExecutor` and validates the DataFrame with `_is_valid_result`. If invalid/timeout/error -> log -> fallback to Vizier built-in query (lines ~317–333).
   - Graph node: (custom_vizier_search_func?) -> call -> (valid result?) yes -> use; no/exception/timeout -> fallback to Vizier.

4. fit_moffat / PSF failure
   - `calibrate_image` calls `fit_moffat(psf)` inside try/except (photo_gpu.py lines ~2974–2980). On failure it raises `MoffatFitError()` and aborts the normal flow.

5. insufficient stars error
   - Several places check star counts and raise `InsufficientStarsError` when thresholds are not met (photo_gpu.py ~2940–2942, ~2965–2967, `perform_opt_photometry` ~809–813). In the graph these are error/early-exit nodes.

6. capture CUDA exceptions / allocator reset
   - `process_image` is decorated with `@capture_cuda_exception` (photo_gpu.py lines ~622–625). There's also a `finally: reset_cupy_allocators()` in `process_image` (photo_gpu.py ~696–698). Represent OOM/CUDA exceptions as a generic catch/fallback box.

## Example: passing a custom Vizier search function

```python
# Example usage with ImageProcessor
from gpuphot.image_processor import create_processor

def my_private_vizier(coocenter, catalog, radius, mag_limit, ref_filter, row_limit, expected_columns, timeout):
    # Implement custom query to your private DB or API
    # Must return a pandas.DataFrame or None
    return my_dataframe_or_none

proc = create_processor('my_instrument')
phot_df, header = proc.process_image(
    imdata, imheader,
    custom_vizier_search_func=my_private_vizier,
    vizier_timeout=30,
    vizier_row_limit=500
)
```

Notes for the example:
- If `my_private_vizier` raises an exception or returns an invalid/empty result, `__getVizier` will log and fall back to the Vizier query.
- `custom_vizier_search_func` is not part of `DEFAULT_PROCESSING_PARAMS` by default, so it must be passed via `kwargs` when calling `process_image` (or set in instrument config JSON if you extend the parser).

---

## Recommended next steps (small, safe improvements)
1. Add `callgraph/process_image_kwargs.md` (this file has been created). Translate to English (done). If you want Spanish, I can add a localized version.
2. Add a small script `tools/extract_kwargs.py` to discover `kwargs.get(...)` and `params['...']` uses across the repo and generate a JSON summary; then use `tools/json_to_md.py` to generate/refresh this MD. I can implement that next.
3. Add a short paragraph in `callgraph/README.md` or `README.md` clarifying that `kwargs` override `processing_params` and enumerating important non-default kwargs (e.g., `custom_vizier_search_func`, `vizier_timeout`, ...).
4. Optionally extend the DOT generator to put decision nodes for: `CUML_AVAILABLE`, `is_gpu_input`, `custom_vizier_search_func` present, `fit_moffat` success/failure. I can update `callgraph/process_image_graph_iter6.dot` if you want.

---

## Automated scan summary (repo-wide)

I ran an automated scan (`tools/extract_kwargs.py`) across the repository to find `kwargs` usage, `params[...]` indexing and functions accepting `**kwargs`.

- Repo root scanned: /home/slemes/PycharmProjects/GPUPhotFinal
- Files scanned: 25797
- Unique keys detected: 2243

The full JSON report has been written to `tools/kwargs_index.json` (note: it can be large). Below are the top keys by number of occurrences (truncated):

- dtype (141 occurrences)
- klass (86 occurrences)
- name (85 occurrences)
- axis (78 occurrences)
- out (71 occurrences)
- dim (53 occurrences)
- value (45 occurrences)
- width, label (43 occurrences each)
- color (42 occurrences)

If you want an exhaustive, filtered export (only gpuphot/ keys, or only astrophot-specific keys such as `custom_vizier_search_func`, `min_conv_snr`, `lum_gmag_coeff`), I can run the extractor with narrower filters and append the small JSON subset to this file.

## Keys explicitly found inside `gpuphot/` (filtered)

During an AST-based scan over `gpuphot/`, these keys were found explicitly referenced via `kwargs.get(...)` or via `params[...]` in our code (not in site-packages):

- `lum_gmag_coeff` (found in `gpuphot/utils/catalog.py`, line ~476)
- `lum_rmag_coeff` (found in `gpuphot/utils/catalog.py`, line ~477)
- `min_conv_snr` (mentioned in comments/old code; present in defaults)
- `center_factor` (mentioned in comments/old code; present in defaults)
- `do_pad` (present in `DEFAULT_PROCESSING_PARAMS`, see `gpuphot/instrument_config_parser.py`)

Note: some keys appear only as defaults (in `DEFAULT_PROCESSING_PARAMS`) and are not explicitly referenced as `kwargs.get(...)` in the codebase; they still propagate because `processing_params` is merged and forwarded via `**params`.

### Example: tracing the `do_pad` key
- Origin: `DEFAULT_PROCESSING_PARAMS` includes `'do_pad': True` (file: `gpuphot/instrument_config_parser.py`).
- Merge: In `ImageProcessor.process_image` we do `params = {**self.processing_params, **kwargs}` (file: `gpuphot/image_processor.py`, line 74). Any `do_pad` you pass in kwargs will override the default.
- Forward: `ImageProcessor` calls core `process_image(..., **params)` (file: `gpuphot/image_processor.py`, line 77). The core `process_image` (file: `gpuphot/phot/photo_gpu.py`) also merges defaults and passes `**params` to `calibrate_image` and other deep functions.
- Consumption: `do_pad` is not referenced directly with `kwargs.get('do_pad')` in `gpuphot/phot/photo_gpu.py` but it may be used indirectly by lower-level functions (e.g., `convolve_fft`, `fill_image`, `get_local_background_fft`) because `calibrate_image` and `aperture_photometry` forward `**kwargs` to them. Therefore the effective usage path is:

  `ImageProcessor.process_image(kwargs) -> process_image(params) -> calibrate_image(..., **kwargs) -> <deeper functions>(**kwargs)`

If you want an exact line where `do_pad` is consumed, we must search for `kwargs.get('do_pad'` or `['do_pad']` across `gpuphot/`. A quick AST scan found no explicit `kwargs.get('do_pad')` usage; so its use is implicit via forwarded `**params`.

### Example: tracing `custom_vizier_search_func`
- `process_image` / `calibrate_image` forward `**kwargs` down to `catalog_results` (photo_gpu.py lines ~3084–3086 and ~3113–3114).
- `catalog_results` calls `__getVizier(..., **kwargs)` (catalog.py lines ~419, ~437, etc.) and `__getVizier` explicitly checks `custom_vizier_search_func` (catalog.py lines ~296–316). Therefore `custom_vizier_search_func` is directly consumable when supplied to `process_image`.

---

If you want, I can now:
- Append to this MD an exhaustive list of all `gpuphot/` keys found by the AST scan (with file:line for each occurrence), or
- Produce a compact CSV `tools/kwargs_gpuphot.csv` listing (key, file, line, context) for easy review.

Which do you prefer?

## Generated mapping artifacts

I generated two machine-readable artifacts you can inspect to validate which default keys are consumed where (all filtered to `gpuphot/`):

- `tools/default_key_consumers.json` — full JSON mapping of each default key to functions that either accept it as a named parameter, access it via `kwargs.get(...)`/`kwargs['...']`, or accept `**kwargs` (potential consumer).
- `tools/default_key_consumers.csv` — compact CSV (columns: key, file, function, lineno, reason) for quick review in a spreadsheet.

Use those files to audit every default key and verify no undocumented keys are missing.

### Correction / clarification about `do_pad`
- Earlier this doc noted `do_pad` might be only forwarded implicitly. Clarification: `do_pad` is indeed included in `DEFAULT_PROCESSING_PARAMS` (see `gpuphot/instrument_config_parser.py`, line 119) and it is *consumed directly* by `convolve_fft` which declares `do_pad: bool` in its signature (file: `gpuphot/phot/conv.py`, line 28). `convolve_fft` is called with `**kwargs` from several places (for example `get_mean_std`, `aperture_photometry`, and elsewhere), therefore passing `do_pad` in `process_image(..., do_pad=...)` will reach and control padding behavior in the core convolution routine.

- In short: `do_pad` is an explicit, supported parameter (default True) and is consumed by `gpuphot.phot.conv.convolve_fft`.

---

If this looks good I can:
- produce a filtered CSV with only keys that are used (param or kwargs.get) rather than all 'accepts_kwargs' potentials, or
- generate a per-key markdown table (automatically) showing the exact functions that consume each default key (one page per key) and append it to `callgraph/`.

Which do you prefer?
