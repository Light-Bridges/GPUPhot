# Instrument configuration defaults

This directory contains the default instrument configuration used by the GPUPhot
image processing pipeline. The canonical file is `default.json` and it provides
project-wide defaults and mappings that the per-instrument processors will use
when extracting metadata from FITS headers and performing image reduction.

Purpose
-------
- Provide a central, human-editable set of header keyword mappings (`header_keywords`) so
  the pipeline can translate heterogeneous FITS headers into canonical internal
  names.
- Offer fallback camera specifications (`camera_specs`) used when headers are
  missing or incomplete.
- Contain processing tuning parameters (`processing_params`) and reduction
  options (`image_reduction`) that can be adjusted for performance or memory
  constraints.
- Allow strict overrides via `forced_values` (Priority 1) for cases where the
  instrument header is known to be wrong or unreliable.
- Map a wide range of filter names to internal standard codes using
  `filter_map`, which drives catalog selection and photometric calibration.

File: `default.json`
---------------------
Key sections and description:

- `header_keywords` (Priority 2):
  - A mapping from internal canonical keys (used by the pipeline) to the FITS
    header keyword names expected in incoming images. When the pipeline reads a
    header, it first tries to translate keys using this mapping.

- `camera_specs` (Priority 3):
  - Lowest priority defaults. If a value is not present in `forced_values` or
    the FITS header, the pipeline will use the value from `camera_specs`.
  - Use these only when a sensible default for an instrument is known.

- `processing_params`:
  - Internal tuning parameters controlling pipeline execution and algorithmic stages:
    - `tile_section` (int): Size in pixels of image tiles for background and detection (default: 1000).
    - `tile_section_psf` (int): Size in pixels of image tiles for PSF calculation (default: 2500).
    - `center_factor` (float): Central region fraction used for PSF and zeropoint estimation (default: 0.7).
    - `border` (int): Border margin in pixels excluded from source extraction (default: 50).
    - `pca_method` (bool): Enable Principal Component Analysis (PCA) for spatial PSF variation (default: false in default.json).
    - `SP_filt` (bool): Enable salt-and-pepper defect filtering (default: false in default.json).
    - `CR_filt` (bool): Enable cosmic-ray filtering pre-processing (default: false).
    - `do_pad` (bool): Enable image padding during tile operations (default: false).
    - `astrom` (bool): Enable astrometric solving stage (default: false).
    - `max_stars_ref` (int): Maximum reference stars per tile used for PSF / zeropoint fit (default: 15).
    - `min_conv_snr` (float): Minimum SNR for candidate stars used in convolution (default: 300).
    - `color_range` (float): Color difference threshold for reference catalog cross-matching (default: 0.6).
    - `zp_maxmag` (float): Upper magnitude cutoff for catalog matching and zeropoint (default: 21).
    - `lum_gmag_coeff` / `lum_rmag_coeff` (float): Luminosity blend coefficients for synthetic filters (default: 0.5 each).

- `image_reduction`:
  - Controls automatic binning and cropping behavior applied by the worker
    when configured or when memory limitations occur. The `apply_reduction`
    option accepts strings: `"always"`, `"never"`, or `"on_failure"`.

- `forced_values` (Priority 1):
  - Strict overrides. Values declared here will win over both FITS header
    values and `camera_specs`. Use sparingly and document why a forced value
    is needed.

- `filter_map`:
  - Translates common (and sometimes ambiguous) filter names found in headers
    to a small set of internal standard codes (e.g., `SDSSr`, `Lum`). This
    mapping is used to select the correct photometric catalog and calibration
    strategy.

Conventions & priorities
-----------------------
- Resolution order for a parameter:
  1. `forced_values` (highest priority)
  2. value extracted from the FITS header using `header_keywords`
  3. `camera_specs` (fallback defaults)

- The file intentionally includes `//` comments to explain each field. It is
  **not valid standard JSON** and cannot be parsed with Python's `json` module.
  The pipeline uses the `json-with-comments` library (see `requirements.txt`)
  to handle these files.

Editing and per-instrument overrides
-----------------------------------
- Best practice: do not edit `default.json` to suit a single instrument. Instead,
  create a new JSON config file for that instrument containing only the keys
  that you want to override (the instrument-specific config directory is
  configured via `INSTRUMENT_CONFIG_BASE_PATH`).

- If you must change `default.json`, add an explanatory comment with the
  rationale and date. Keep changes minimal and test processors against a
  representative set of FITS headers.

Example: overriding the gain for one instrument (`my_instrument.json`):

```json
{
  "forced_values": {
    "gain": 0.33
  }
}
```

This file (and per-instrument overrides) will be read by the project's
instrument configuration parser at runtime.
