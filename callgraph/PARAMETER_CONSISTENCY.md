# Analysis of Parameter Consistency in GPUPhot

**Version:** 1.0
**Date:** 2024-08-02

## 1. Objective

This document audits the consistency between the default values defined in `DEFAULT_PROCESSING_PARAMS` and the default values in the function signatures that ultimately consume these parameters.

The goal is to identify any discrepancies that could lead to different behaviors depending on whether a function is called from the high-level `process_image` pipeline (which injects the configuration) or called directly (which would use the function's default).

## 2. Methodology

The analysis follows these steps:
1.  List all parameters from `DefaultConfig.DEFAULT_PROCESSING_PARAMS`.
2.  Trace each parameter through the call stack, starting from `calibrate_image`.
3.  Identify the final function that uses the parameter.
4.  Compare the default value from the configuration with the default value in the function's signature.
5.  Document any inconsistencies and their potential impact.

## 3. Parameter Consistency Analysis

The following table summarizes the findings for each parameter in `DEFAULT_PROCESSING_PARAMS`.

| Parameter | Config Default | Consuming Function | Function Default | Consistent? | Analysis & Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `border` | `20` | `calibrate_image` | `10` | **No** | **Discrepancy Found.** If `calibrate_image` is called directly without specifying `border`, it will use a 10-pixel border, whereas the standard pipeline uses 20. This could lead to more edge-effect-contaminated sources being included in a direct call. |
| `center_factor` | `0.7` | `calibrate_image` | `0.7` | Yes | Consistent. |
| `color_range` | `0.6` | `get_zeropoint` | `0.3` | **No** | **Discrepancy Found.** The `get_zeropoint` function is called from `calibrate_image`, which passes the parameter. However, if called directly, it uses a much stricter solar color range (0.3 vs 0.6), which would select fewer reference stars for ZP calculation. |
| `CR_filt` | `False` | `calibrate_image` | `False` | Yes | Consistent. |
| `do_pad` | `True` | `convolve_fft` | `True` | Yes | Consistent. The parameter is passed via `**kwargs` and the function signature matches. |
| `lum_gmag_coeff`| `0.5` | `catalog_results` | N/A | Yes | This parameter is consumed from `kwargs` and does not have a direct default in the function signature. The logic relies on the value passed from the configuration. |
| `lum_rmag_coeff`| `0.5` | `catalog_results` | N/A | Yes | Same as `lum_gmag_coeff`. |
| `max_stars_ref`| `15` | `calibrate_image` | `15` | Yes | Consistent. |
| `min_conv_snr` | `300` | `perform_opt_photometry` | `300.0` | Yes | Consistent (type difference float/int is handled by Python). |
| `min_snr` | `5` | `calibrate_image` | `5` | Yes | Consistent. This value is then passed down to `detect_isolated_stars` and `detect_sources_psf`. |
| `pca_method` | `True` | `calibrate_image` | `True` | Yes | Consistent. |
| `SP_filt` | `True` | `calibrate_image` | `True` | Yes | Consistent. |
| `tile_section` | `1000`| `calibrate_image` | `1000`| Yes | Consistent. |
| `tile_section_psf`| `3000`| `calibrate_image` | `2500`| **No** | **Discrepancy Found.** If `calibrate_image` is called directly, it will use a tile size of 2500 for grouping stars in the aperture correction map, potentially creating more, smaller groups than the standard pipeline's 3000. |
| `zp_maxmag` | `21` | `calibrate_image` | `21` | Yes | Consistent. |

## 4. Summary of Inconsistencies

Three significant discrepancies were found:

1.  **`border`**:
    -   Config: `20`
    -   `calibrate_image` default: `10`
    -   **Impact**: Direct calls to `calibrate_image` will be less restrictive with sources near the edge.

2.  **`color_range`**:
    -   Config: `0.6`
    -   `get_zeropoint` default: `0.3`
    -   **Impact**: Direct calls to `get_zeropoint` will use a much smaller color range, potentially finding fewer calibration stars and affecting the zero-point stability.

3.  **`tile_section_psf`**:
    -   Config: `3000`
    -   `calibrate_image` default: `2500`
    -   **Impact**: Direct calls to `calibrate_image` will use smaller tiles for PSF grouping, which could alter the aperture correction map, especially in dense fields.

## 5. Recommendations

The existence of these discrepancies poses a risk to the predictability and maintainability of the code. A developer making a direct call to a lower-level function might get unexpected results that differ from the standard pipeline.

**Primary Recommendation: Unify Default Values**

The most robust solution is to make the default values in the function signatures consistent with those in `DEFAULT_PROCESSING_PARAMS`. This ensures that the behavior is the same regardless of how the function is called.

-   In `gpuphot/phot/photo_gpu.py`, the signature of `calibrate_image` should be modified:
    -   `border: int = 20`
    -   `tile_section_psf: int = 3000`
-   In `gpuphot/utils/astro.py` (or wherever `get_zeropoint` is located), its signature should be modified:
    -   `solar_filter=0.6`

This approach establishes `DEFAULT_PROCESSING_PARAMS` as the single source of truth for default behavior, which is a sound engineering practice.

**Alternative (Less Recommended): Documentation Only**

If modifying the function signatures is undesirable (e.g., due to legacy reasons or API stability concerns), the discrepancies should be explicitly and clearly documented in the docstring of each affected function. For example, the docstring for `calibrate_image` should state: *"Note: The default `border` value of 10 in this function differs from the standard pipeline configuration default of 20."*

This is less ideal as it puts the burden on the developer to notice the warning, but it is better than having no information at all.

Given the goal of creating a high-quality, maintainable system, the **primary recommendation (unification) is strongly advised.**
