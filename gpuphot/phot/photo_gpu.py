# SPDX-License-Identifier: MIT
"""
High-level photometry pipeline for gpuphot.

This module contains the primary high-level orchestration of the photometry
pipeline: image calibration, background estimation, source detection, PSF
handling, aperture photometry (batch), and final catalog/zero-point
calculation. The implementation is optimized for GPUs using CuPy and includes
fallbacks and carefully documented transfer points between CPU and GPU.

Only documentation-level edits were applied in this pass. No functional code
was modified. Any required runtime fixes are recorded in `FIXERS.md` in the
same directory.
"""

from __future__ import annotations

import gc
import os
import time
import traceback
import warnings

import cupy as cp
import numpy as _numpy
from ..phot.cosmetics import CR_filter, SP_filter
try:
    import cupynumeric as np
except ImportError:
    import numpy as np
except Exception:
    import numpy as np

import nvtx
import pandas as pd
from astropy.wcs import WCS
# import tensorflow as tf
from cupyx.scipy.ndimage import convolve, label, sum as nd_sum, mean as nd_mean, maximum_filter, \
    median_filter, laplace, binary_dilation

from .conv import fill_image, fill_nan_fft, get_aper_kernel, convolve_fft, gen_apm_filter
from .psf import create_coeff_map, create_star_dataset, detect_isolated_stars, detect_sources_psf, fit_moffat, \
    get_eigen_psfs, group_star_dataset, project_all_stars_onto_eigenpsfs, recreate_normed_star
from ..exceptions import InsufficientStarsError, MoffatFitError, capture_cuda_exception, UnableToAstrometrizeError, \
    DataValidationError
from ..instrument_config_parser import HeaderKey, DefaultConfig
from ..logger.hierarchical_logging import setup_logger, hierarchical_debug
from ..phot.background import get_local_background_fft
from ..stats.reduction import stack_sigmaclip
from ..utils.astro import astrometrice2, get_astrometry_params, get_maglim, get_target_snr, get_zeropoint, \
    plate_scale_px
from ..utils.catalog import catalog_results, crossmatch_sources
from ..utils.gpu import free_gpu_mem, reset_cupy_allocators, adaptive_memory_management
from ..utils.headers import update_header_with_astrometry, update_header_with_photometry

logger = setup_logger(__name__)



### # @hierarchical_debug(logger)
@nvtx.annotate('gen_moff_filter', category='phot.photo_gpu')
def gen_moff_filter(alpha, beta, **kwargs):
    """
    Generate a Moffat filter.

    :param alpha: Alpha parameter for Moffat filter.
    :type alpha: float
    :param beta: Beta parameter for Moffat filter.
    :type beta: float
    :return: Moffat filter kernel and kernel size.
    :rtype: tuple(cupy.ndarray, int)
    """
    fw = alpha * (2 * np.sqrt(2 ** (1 / beta) - 1))
    sigma_r = fw / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    lk = np.ceil(sigma_r).astype(np.int16) * 4
    k_dim = 2 * lk + 1, 2 * lk + 1
    indi = cp.indices(k_dim)
    r2 = (lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2
    ker = (1 + r2 / alpha ** 2) ** -beta
    ksum = cp.sum(ker)
    ksum2 = cp.sum(ker * ker)
    n = k_dim[0] ** 2
    k_app = (ker - ksum / n) / (ksum2 - ksum * ksum / n)

    return k_app, lk


### # @hierarchical_debug(logger)
@nvtx.annotate('get_sky', category='phot.photo_gpu')
def get_sky(im_g, fw, qt=90, mem=cp.get_default_memory_pool(), **kwargs):
    """
    Estimate the sky background and RMS noise.

    :param im_g: Input image.
    :type im_g: cupy.ndarray
    :param fw: Full width at half maximum.
    :type fw: float
    :param qt: Quantile for sky estimation, by default 90.
    :type qt: float, optional
    :param mem: Memory pool, by default cp.get_default_memory_pool().
    :type mem: cupy.cuda.memory.MemoryPool, optional
    :return: Estimated sky background, RMS noise, and memory usage.
    :rtype: tuple(cupy.ndarray, cupy.ndarray, int)
    """
    sigma_r = fw / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    lk = np.ceil(sigma_r).astype(np.int16) * 4
    k_app = gen_apm_filter(2 * lk)
    fot_m = convolve(im_g, k_app, origin=(0, 0))
    fot_m2 = convolve(im_g * im_g, k_app, origin=(0, 0))
    del k_app
    fot_m2 = cp.sqrt(fot_m2 - fot_m * fot_m)
    cut1 = cp.percentile((fot_m * fot_m2).flatten(), qt)
    mask = fot_m * fot_m2 > cut1
    mm = mem.used_bytes()
    fot_m[mask] = None
    fot_m2[mask] = None
    del (cut1, mask)
    fot_m = cov_nan(fot_m, 20)
    fot_m2 = cov_nan(fot_m2, 20)

    return fot_m, fot_m2, mm



### # @hierarchical_debug(logger)
@nvtx.annotate('gen_moff_filter2', category='phot.photo_gpu')
def gen_moff_filter2(alpha, beta, **kwargs):
    """
     Generate a Moffat filter with adjusted alpha.

     .. deprecated::
         This function is unused and will be removed in a future version.
         Use :func:`gen_moff_filter` instead.

     :param alpha: Alpha parameter for Moffat filter.
     :type alpha: float
     :param beta: Beta parameter for Moffat filter.
     :type beta: float
     :return: Moffat filter kernel and kernel size.
     :rtype: tuple(cupy.ndarray, int)
     """
    warnings.warn(
        "gen_moff_filter2 is deprecated and will be removed in a future version. "
        "Use gen_moff_filter instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    alpha = alpha / 2
    fw = alpha * (2 * np.sqrt(2 ** (1 / beta) - 1))
    sigma_r = fw / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    lk = np.ceil(sigma_r).astype(np.int16) * 4
    k_dim = 2 * lk + 1, 2 * lk + 1
    indi = cp.indices(k_dim)
    r2 = (lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2
    ker = (1 + r2 / alpha ** 2) ** -beta
    ksum = cp.sum(ker)
    k_app = ker / ksum

    return k_app, lk


@nvtx.annotate('calculate_aperture_corrections_gpu', category='phot.photo_gpu')
def calculate_aperture_corrections_gpu(corr: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Calculate aperture corrections using CuPy (GPU).

    Each star's growth curve is normalized by its maximum value, which under the
    pipeline's default radii (up to ~7·FWHM, well beyond the PSF support) corresponds
    to the total stamp flux. Outliers are rejected using a 2.5·MAD criterion, robust
    to non-Gaussian deviations from PSFs that depart significantly from the local mean.

    :param corr: Array of aperture curves (n_stars, n_radii), potentially with NaNs.
                 Expected shape (n_stars_in_cluster, n_radii).
    :type corr: cupy.ndarray
    :return: Aperture correction factors and errors for the cluster.
    :rtype: tuple(cupy.ndarray, cupy.ndarray) Both shape (n_radii,)
    """
    if not cp.issubdtype(corr.dtype, cp.floating):
        corr = corr.astype(cp.float64)

    if corr.size == 0 or corr.shape[0] == 0:
        if corr.ndim == 2 and corr.shape[1] > 0:
            n_radii = corr.shape[1]
            dtype_out = corr.dtype if cp.issubdtype(corr.dtype, cp.floating) else cp.float64
            return cp.full(n_radii, cp.nan, dtype=dtype_out), cp.full(n_radii, cp.nan, dtype=dtype_out)
        else:
            return cp.array([]), cp.array([])

    # Normalize each curve by its own max (= flux at largest aperture = stamp total)
    max_vals = cp.nanmax(corr, axis=1, keepdims=True)
    mask_broadcasted = cp.broadcast_to(max_vals > 0, corr.shape)
    corr_normalized = cp.where(mask_broadcasted, corr / max_vals, cp.nan)

    # Robust outlier rejection: 2.5·MAD
    corr_med = cp.nanmedian(corr_normalized, axis=0)
    abs_dev = cp.abs(corr_normalized - corr_med[cp.newaxis, :])
    mad = 1.4826 * cp.nanmedian(abs_dev, axis=0)
    epsilon = 1e-9
    mad = cp.where(mad == 0, epsilon, mad)
    cmask = abs_dev > 2.5 * mad[cp.newaxis, :]
    corr_cleaned = cp.where(cmask, cp.nan, corr_normalized)

    # Final correction factor and uncertainty
    corr_fact = cp.nanmean(corr_cleaned, axis=0)
    corr_err = cp.nanstd(corr_cleaned, axis=0) / cp.sqrt(cp.sum(~cp.isnan(corr_cleaned), axis=0))

    # Handle radii where all stars became NaN
    corr_fact = cp.where(cp.isnan(corr_fact), 0.0, corr_fact)
    corr_err = cp.where(cp.isnan(corr_err), 0.0, corr_err)

    return corr_fact, corr_err


### # @hierarchical_debug(logger)
@nvtx.annotate('calculate_aperture_corrections', category='phot.photo_gpu')
def calculate_aperture_corrections(corr: np.ndarray) -> cp.ndarray:
    """
    Calculate aperture corrections for all stars in an optimized way.

    .. deprecated::
        This CPU-only version is deprecated and will be removed in a future version.
        Use :func:`calculate_aperture_corrections_gpu` instead.

    :param corr: Array of aperture corrections.
    :type corr: numpy.ndarray
    :return: Aperture correction factors and errors.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    warnings.warn(
        "calculate_aperture_corrections is deprecated and will be removed in a future version. "
        "Use calculate_aperture_corrections_gpu instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if corr.size == 0 or corr.shape[0] == 0:
        if corr.ndim == 2 and corr.shape[1] > 0:
            n_radii = corr.shape[1]
            return np.full(n_radii, np.nan), np.full(n_radii, np.nan)
        else:
            return np.array([]), np.array([])

    # Normalize (handle potential division by zero/nan)
    max_vals = np.nanmax(corr, axis=1, keepdims=True)
    # Use np.divide with where clause
    corr_normalized = np.divide(corr, max_vals, where=(max_vals != 0), out=np.full_like(corr, np.nan))

    # Iterative outlier rejection
    corr_fa = np.nanmedian(corr_normalized, axis=0)
    corr_e = np.nanstd(corr_normalized, axis=0)
    corr_fa_bc = corr_fa[np.newaxis, :]
    corr_e_bc = corr_e[np.newaxis, :]

    # Handle case where corr_e might be zero
    corr_e_bc = np.where(corr_e_bc == 0, 1e-9, corr_e_bc)  # Add small epsilon if std is zero

    with np.errstate(invalid='ignore'):  # Ignore comparison with NaN warnings
        cmask = np.abs(corr_normalized - corr_fa_bc) > corr_e_bc
    corr_cleaned = np.where(cmask, np.nan, corr_normalized)

    # Final factor and error
    corr_fact = np.nanmean(corr_cleaned, axis=0)
    corr_err = np.nanstd(corr_cleaned, axis=0) / np.sqrt(np.sum(~np.isnan(corr_cleaned), axis=0))  # Standard error

    # Replace NaNs in final result (e.g., if all values for a radius were outliers)
    corr_fact = np.nan_to_num(corr_fact, nan=0.0)  # Replace NaN with 0.0
    corr_err = np.nan_to_num(corr_err, nan=0.0)  # Replace NaN with 0.0

    return corr_fact, corr_err


@nvtx.annotate('find_aperture_corrections_gpu', category='phot.photo_gpu')
def find_aperture_corrections_gpu(sources: cp.ndarray, corrections: cp.ndarray, correction_errors: cp.ndarray,
                                  cluster_centers: cp.ndarray,
                                  opt_rad_idx: cp.ndarray = None, **kwargs) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Find the aperture correction for each source using GPU indexing and GPU crossmatch.

    :param sources: Array of source coordinates (GPU).
    :type sources: cupy.ndarray
    :param corrections: Array of aperture corrections (GPU) (n_clusters, n_radii).
    :type corrections: cupy.ndarray
    :param correction_errors: Array of aperture correction errors (GPU) (n_clusters, n_radii).
    :type correction_errors: cupy.ndarray
    :param cluster_centers: Array of cluster centers (GPU).
    :type cluster_centers: cupy.ndarray
    :param opt_rad_idx: Array of optimal radii indices (GPU, optional) (n_sources,).
    :type opt_rad_idx: cupy.ndarray or None
    :return: Array of aperture corrections and errors for each source (GPU).
    :rtype: tuple(cupy.ndarray, cupy.ndarray)
    """
    nvtx_range = nvtx.start_range('find_aperture_corrections_gpu', category='phot.photo_gpu', color='teal')

    # Call the main crossmatch function. It will handle GPU/CPU automatically.
    # Since inputs (sources, cluster_centers) are CuPy, it will attempt GPU first.
    cm_range = nvtx.start_range('call_crossmatch_wrapper', category='utils.catalog')
    _, tile_idx = crossmatch_sources(sources, cluster_centers, thres_px=int(cp.max(sources)))
    nvtx.end_range(cm_range)
    # tile_idx will be CuPy array if successful GPU or if fallback occurred from GPU input

    if len(tile_idx) != len(sources):
        nvtx.end_range(nvtx_range)
        raise ValueError(f"Crossmatch returned unexpected indices...")
    if not isinstance(tile_idx, cp.ndarray):
        # This indicates CPU path was used AND original input was CPU - shouldn't happen here
        nvtx.end_range(nvtx_range)
        raise TypeError("Crossmatch did not return CuPy array as expected.")

    # Indexing on GPU
    gpu_index_range = nvtx.start_range('gpu_indexing_corrections', category='phot.photo_gpu')
    if opt_rad_idx is None:
        aperture_corrections = corrections[tile_idx, :]
        aperture_correction_errors = correction_errors[tile_idx, :]
    else:
        if not isinstance(opt_rad_idx, cp.ndarray):
            nvtx.end_range(gpu_index_range)
            nvtx.end_range(nvtx_range)
            raise TypeError("opt_rad_idx must be CuPy")
        aperture_corrections = corrections[tile_idx, opt_rad_idx]
        aperture_correction_errors = correction_errors[tile_idx, opt_rad_idx]
    nvtx.end_range(gpu_index_range)

    nvtx.end_range(nvtx_range)
    return aperture_corrections, aperture_correction_errors



@nvtx.annotate('create_aperture_corrections_map_gpu', category='phot.photo_gpu')
def create_aperture_corrections_map_gpu(image_shape: tuple, block_size: int, unit_star_dataset: cp.ndarray,
                                        coords: cp.ndarray, radii: cp.ndarray, **kwargs) -> tuple[
    cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Calculates aperture corrections map, attempting GPU acceleration for grouping
    and correction calculation, with fallback to CPU where necessary.

    Returns results on GPU.

    :param image_shape: Tuple (height, width) of the image.
    :type image_shape: tuple
    :param block_size: Size of the tiles.
    :type block_size: int
    :param unit_star_dataset: Dataset of unit stars (GPU).
    :type unit_star_dataset: cupy.ndarray
    :param coords: Coordinates of stars (GPU).
    :type coords: cupy.ndarray
    :param radii: Radii for aperture photometry (GPU).
    :type radii: cupy.ndarray
    :return: Aperture corrections, errors, and cluster centers (GPU).
    :rtype: tuple(cupy.ndarray, cupy.ndarray, cupy.ndarray)
    """
    nvtx_range = nvtx.start_range('create_aperture_corrections_map_gpu_v3', category='phot.photo_gpu', color='purple')
    mempool = cp.get_default_memory_pool()

    # 1. Calculate aperture photometry curves (GPU) - No changes
    phot_range = nvtx.start_range('photometry_curves', category='phot.photo_gpu')
    n_stars_psf, psf_h, psf_w = unit_star_dataset.shape
    positions_psf = cp.array([[psf_h // 2, psf_w // 2]], dtype=cp.int32)
    aperture_curves_cp, _, _ = batch_aperture_photometry(unit_star_dataset, None, positions_psf, radii)
    aperture_curves_cp = aperture_curves_cp.squeeze(axis=-1)  # Shape (n_stars_psf, n_radii)
    n_radii = aperture_curves_cp.shape[1]
    nvtx.end_range(phot_range)

    # --- KEY STEP: CALL THE GROUPING WRAPPER ---
    # Pass the GPU coordinates directly. The wrapper will decide whether to use GPU or CPU.
    grouping_range = nvtx.start_range('group_star_dataset_dispatch', category='phot.psf')
    if block_size < image_shape[0] or block_size < image_shape[1]:
        avg_group_size = int(
            max(5, coords.shape[0] / (np.prod(image_shape) / min(max(image_shape), block_size) ** 2)))
        min_group_size_val = max(3, avg_group_size // 2)

        # This function NOW can return cp.ndarray or np.ndarray
        labels = group_star_dataset(coords, avg_group_size=avg_group_size,
                                    min_group_size=min_group_size_val)
    else:
        # If the block size is larger than the image, treat all stars as one group
        labels = cp.zeros(coords.shape[0], dtype=cp.int32)  # Single group label for all stars
    nvtx.end_range(grouping_range)

    # --- KEY STEP: CHECK THE TYPE OF LABELS TO DECIDE THE PATH ---

    if isinstance(labels, cp.ndarray):
        # ----- SUCCESSFUL GPU PATH -----
        # Labels is on GPU, aperture_curves_cp is on GPU, coords is on GPU.
        # Goal: Calculate corrections and centers (if possible) on GPU.
        nvtx_gpu_path = nvtx.start_range('correction_path_gpu', category='phot.photo_gpu', color='lime')
        # logger.debug("Grouping successful on GPU. Proceeding with GPU correction calculation.")

        # We don't need to transfer coords or aperture_curves to CPU.

        unique_labels_gpu = cp.unique(labels)
        n_clusters = len(unique_labels_gpu)

        # Pre-allocate result arrays on GPU
        aperture_corrections_final_gpu = cp.full((n_clusters, n_radii), cp.nan, dtype=aperture_curves_cp.dtype)
        aperture_correction_errors_final_gpu = cp.full((n_clusters, n_radii), cp.nan, dtype=aperture_curves_cp.dtype)

        # Calculate centers - Option A: GPU (preferred to avoid transfers)
        cluster_centers_gpu = cp.full((n_clusters, coords.shape[1]), cp.nan, dtype=coords.dtype)

        calc_corr_range = nvtx.start_range('calculate_corrections_gpu_loop', category='phot.photo_gpu')
        for idx, label_val_gpu in enumerate(unique_labels_gpu):  # Iterate using GPU labels
            label_val_np = label_val_gpu.item()  # Value for the map

            # --- GPU Operations inside the loop ---
            gpu_inner_range = nvtx.start_range(f'gpu_calc_cluster_{label_val_np}', category='phot.photo_gpu')
            mask_gpu = (labels == label_val_gpu)
            n_in_cluster = cp.sum(mask_gpu)

            if n_in_cluster > 0:
                cluster_curves_gpu = aperture_curves_cp[mask_gpu]

                aper_corr_gpu, aperr_corr_err_gpu = calculate_aperture_corrections_gpu(cluster_curves_gpu)

                aperture_corrections_final_gpu[idx, :] = aper_corr_gpu
                aperture_correction_errors_final_gpu[idx, :] = aperr_corr_err_gpu

                # Calculate center on GPU
                cluster_centers_gpu[idx, :] = cp.mean(coords[mask_gpu], axis=0)
            # else: Empty cluster, already pre-filled with NaN
            nvtx.end_range(gpu_inner_range)

        nvtx.end_range(calc_corr_range)

        # No final CPU->GPU transfer needed for corrections/errors/centers

        nvtx.end_range(nvtx_gpu_path)

    else:  # isinstance(labels, np.ndarray)
        # ----- CPU PATH (Fallback or original CPU input if it existed) -----
        # Labels is NumPy (labels_np = labels).
        # aperture_curves_cp is on GPU. coords is on GPU.
        # We need to transfer curves to CPU and use coords_np (which the wrapper already got if it fell back).
        nvtx_cpu_path = nvtx.start_range('correction_path_cpu', category='cpu_ops', color='orange')
        # logger.debug("Grouping fell back to CPU or input was CPU. Proceeding with CPU correction calculation.")

        labels_np = labels  # Rename for clarity

        # --- Required Transfer: Curves GPU -> CPU ---
        transfer_curves_range = nvtx.start_range('transfer_curves_for_cpu_calc', category='transfer', color='red')
        aperture_curves_np = aperture_curves_cp.get()
        nvtx.end_range(transfer_curves_range)
        # Free GPU memory for curves if large
        del aperture_curves_cp
        mempool.free_all_blocks()
        gc.collect()

        # --- Required Transfer: Coords GPU -> CPU (if not already done in wrapper fallback) ---
        # To be safe, get coords_np here if the wrapper didn't (although it should)
        # Or better, assume that if labels is np.ndarray, coords_np exists internally in the wrapper
        # For center calculation, we need coords_np
        transfer_coords_range = nvtx.start_range('get_coords_for_cpu_center_calc', category='transfer', color='red')
        coords_np = coords.get()  # Transfer to calculate centers
        del coords
        nvtx.end_range(transfer_coords_range)

        # --- CPU Code (very similar to the original) ---
        calc_corr_range = nvtx.start_range('calculate_corrections_cpu_loop', category='cpu_ops')
        unique_labels_np = np.unique(labels_np)
        n_clusters = len(unique_labels_np)

        # Use lists to aggregate CPU results
        cluster_centers_list = []
        aperture_corrections_list = []
        aperture_correction_errors_list = []

        for label in unique_labels_np:
            mask_np = (labels_np == label)
            n_in_cluster = np.sum(mask_np)

            if n_in_cluster > 0:
                cluster_coords_np = coords_np[mask_np]
                cluster_curves_np = aperture_curves_np[mask_np]

                # *** CALL THE NUMPY VERSION OF CORRECTIONS ***
                aper_corr_np, aperr_corr_err_np = calculate_aperture_corrections(cluster_curves_np)

                aperture_corrections_list.append(aper_corr_np)
                aperture_correction_errors_list.append(aperr_corr_err_np)
                cluster_centers_list.append(np.mean(cluster_coords_np, axis=0))
            else:
                # Append NaNs if the cluster is empty
                aperture_corrections_list.append(np.full(n_radii, np.nan))
                aperture_correction_errors_list.append(np.full(n_radii, np.nan))
                cluster_centers_list.append(np.full(coords_np.shape[1], np.nan))

        # Aggregate results (NumPy)
        aperture_corrections_np = np.array(aperture_corrections_list)
        aperture_correction_errors_np = np.array(aperture_correction_errors_list)
        cluster_centers_np = np.array(cluster_centers_list)
        nvtx.end_range(calc_corr_range)

        # --- Required Final Transfer: Results CPU -> GPU ---
        transfer_results_range = nvtx.start_range('transfer_cpu_results_to_gpu', category='transfer', color='red')
        aperture_corrections_final_gpu = cp.asarray(aperture_corrections_np)
        aperture_correction_errors_final_gpu = cp.asarray(aperture_correction_errors_np)
        cluster_centers_gpu = cp.asarray(cluster_centers_np)
        nvtx.end_range(transfer_results_range)

        nvtx.end_range(nvtx_cpu_path)

    # --- End of Conditional Block ---

    # Free memory that is no longer needed (coords could be freed earlier if centers are calculated on GPU)
    # Deliberately not deleting coords here in case it's needed outside
    mempool.free_all_blocks()
    gc.collect()

    nvtx.end_range(nvtx_range)  # End overall function range
    return aperture_corrections_final_gpu, aperture_correction_errors_final_gpu, cluster_centers_gpu



@capture_cuda_exception
@hierarchical_debug(logger)
@nvtx.annotate('process_image', category='phot.photo_gpu')
def process_image(imdata, imheader, header_descriptions=None, **kwargs):
    """
    Process an image using the specified parameters and translate headers.

    :param imdata: The image data to be processed.
    :type imdata: numpy.ndarray
    :param imheader: The original FITS header associated with the image.
    :type imheader: astropy.io.fits.header.Header
    :param header_descriptions: Descriptions of the header keywords (optional).
    :type header_descriptions: dict or None
    :param kwargs: Additional parameters for processing that override default settings.
    :return: A tuple containing the data frame of processed results and the updated header.
    :rtype: tuple(pandas.DataFrame, astropy.io.fits.header.Header)
    :raises UnableToAstrometrizeError: If the image cannot be astrometrized.
    """
    try:
        # parameters from header
        scale = plate_scale_px(imheader[HeaderKey.PXSIZE.value], imheader[HeaderKey.FOCALEN.value])
        n_images = imheader[HeaderKey.TOTIMA.value]
        gain = imheader[HeaderKey.GAIN.value]
        try:
            rdnoise = imheader[HeaderKey.GAIN.value] * imheader[HeaderKey.BIASSTD.value]
        except:
            rdnoise = imheader[HeaderKey.RDNOISE.value]
        exptime = imheader[HeaderKey.EXPT1.value]
        satlevel = imheader[HeaderKey.SATLEVEL.value]
        target_ra = imheader[HeaderKey.POINTRA.value] * 15
        target_dec = imheader[HeaderKey.POINTDEC.value]
        # try:
        site_elevation = imheader[HeaderKey.SITEELEV.value]
        # except:
        #     site_elevation = imheader[HeaderKey.SITEALT.value]
        site_latitude = imheader[HeaderKey.SITELAT.value]
        site_longitude = imheader[HeaderKey.SITELONG.value]
        date_obs = imheader[HeaderKey.DATE_OBS.value]
        filter = imheader[HeaderKey.FILTER.value]
        binning = imheader[HeaderKey.XBINNING.value]

        # default parameters
        default_params = DefaultConfig.DEFAULT_PROCESSING_PARAMS

        # Update default parameters with any provided in kwargs
        params = {**default_params, **kwargs}

        # Call calibrate_image with updated parameters
        dfm, h_wcs, dic_calib = calibrate_image(
            imdata, filter, binning,
            scale, gain, rdnoise, exptime, satlevel,
            target_ra, target_dec, n_images=n_images,
            # SP_filt=params['SP_filt'],
            # CR_filt=params['CR_filt'],
            # border=params['border'],
            # center_factor=params['center_factor'],
            # pca_method=params['pca_method'],
            # tile_section=params['tile_section'],
            # max_stars_ref=params['max_stars_ref'],
            # min_snr=params['min_snr'],
            # color_range=params['color_range'],
            # tile_section_psf=params['tile_section_psf'],
            **params
        )

        if dfm is None:
            raise UnableToAstrometrizeError()

        # Update header
        imheader = update_header_with_astrometry(imheader, h_wcs, site_latitude, site_longitude, site_elevation,
                                                 date_obs,
                                                 header_descriptions)
        imheader = update_header_with_photometry(imheader, dic_calib, header_descriptions)

        return dfm, imheader
    finally:
        reset_cupy_allocators()


@nvtx.annotate('perform_opt_photometry', category='phot.photo_gpu')
def perform_opt_photometry(img_ori: cp.ndarray, back: cp.ndarray, conv_ima_sigma: cp.ndarray,
                           source_coord: cp.ndarray, isolated_coord: cp.ndarray,
                           tile_section_psf: int, star_dataset: cp.ndarray, fwhm: float,
                           gain: float, rdnoise: float, n_images: int = 1,
                           center_factor: float = 1.0,
                           min_conv_snr: float = 300.0, **kwargs) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Optimize the aperture photometry process to find the best radii for signal-to-noise ratio (SNR) for each star.
    (GPU-focused, Option B with cuML crossmatch).

    :param img_ori: Input image. Without back subtraction.
    :type img_ori: cupy.ndarray
    :param back: Background image.
    :type back: cupy.ndarray
    :param conv_ima_sigma: Convolution sigma map.
    :type conv_ima_sigma: cupy.ndarray
    :param source_coord: Coordinates of sources.
    :type source_coord: cupy.ndarray
    :param isolated_coord: Coordinates of isolated stars.
    :type isolated_coord: cupy.ndarray
    :param tile_section_psf: Size of PSF tiles.
    :type tile_section_psf: int
    :param star_dataset: Dataset of stars.
    :type star_dataset: cupy.ndarray
    :param fwhm: Full-width at half-maximum of the PSF.
    :type fwhm: float
    :param gain: Gain value for flux conversion.
    :type gain: float
    :param rdnoise: Read noise value of the detector.
    :type rdnoise: float
    :param n_images: Number of images (default is 1).
    :type n_images: int
    :param center_factor: Fraction defining the central region for selecting isolated stars (default is 1.0).
    :type center_factor: float
    :param min_conv_snr: Minimum convolutional SNR for selecting isolated stars (default is 300.0).
    :type min_conv_snr: float
    :return: Tuple containing the optimized fluxes, noise, coordinates, and extra information.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, dict)
    :raises InsufficientStarsError: If there are not enough isolated stars for processing.
    :raises DataValidationError: If input data is invalid or insufficient.
    """
    overall_range = nvtx.start_range('perform_opt_photometry_optimized_gpu_crossmatch', category='phot.photo_gpu',
                                     color='cyan')

    mempool = cp.get_default_memory_pool()
    logger.info(f"Starting adaptive photometry. Initial GPU Memory: Used={mempool.used_bytes() / 1e9:.2f} GB")

    adaptive_memory_management(mempool)

    # BLOQUE 1: Select central stars
    block1_range = nvtx.start_range('center_stars_selection', category='phot.photo_gpu', color='yellow')
    center_factor = min(center_factor, 1.0)
    h, w = img_ori.shape[-2:]
    xmin = int(w * 0.5 * (1 - center_factor))
    xmax = int(w * 0.5 * (1 + center_factor))
    ymin = int(h * 0.5 * (1 - center_factor))
    ymax = int(h * 0.5 * (1 + center_factor))
    center_mask = (isolated_coord[:, 0] > ymin) & (isolated_coord[:, 0] < ymax) & \
                  (isolated_coord[:, 1] > xmin) & (isolated_coord[:, 1] < xmax)
    nvtx.end_range(block1_range)

    # BLOQUE 2: Obtain convolutional SNR
    block2_range = nvtx.start_range('snr_source_validation_gpu', category='phot.photo_gpu', color='orange')
    conv_snr = conv_ima_sigma[
        cp.round(source_coord[:, 0]).astype(cp.int32), cp.round(source_coord[:, 1]).astype(cp.int32)]
    pos_conv_snr_mask = conv_snr > 0
    source_coord = source_coord[pos_conv_snr_mask]
    conv_snr = conv_snr[pos_conv_snr_mask]

    del pos_conv_snr_mask

    # --- NO TRANSFER NEEDED ---
    # Call GPU crossmatch directly with CuPy arrays
    cm_range = nvtx.start_range('crossmatch_sources_gpu_call', category='utils.catalog_gpu')
    try:
        # Match isolated_coord_center TO source_coord_filt
        _, source_coords_matched_idx = crossmatch_sources(
            isolated_coord[center_mask], source_coord, thres_px=3
        )
        del _
        # source_coords_matched_idx is now a CuPy array of indices into source_coord_filt
    except ImportError as e:
        logger.error(f"ImportError using GPU crossmatch: {e}. Cannot proceed.")
        raise
    except Exception as e:
        logger.error(f"Error during GPU crossmatch in Block 2: {e}")
        # Decide how to handle: fallback to CPU? Raise error?
        raise RuntimeError("GPU crossmatch failed in Block 2") from e
    nvtx.end_range(cm_range)
    # --- End GPU Crossmatch ---

    conv_snr_isol = conv_ima_sigma[
        cp.round(isolated_coord[:, 0]).astype(cp.int32), cp.round(isolated_coord[:, 1]).astype(cp.int32)]
    conv_snr_mask = conv_snr_isol > min_conv_snr

    del conv_ima_sigma

    # Fallback for weak fields: if there are still too few, take the brightest ones
    # above a minimum floor, so that the growth curve remains usable.
    num_good_isolated = cp.sum(conv_snr_mask)
    if num_good_isolated < 10:
        usable = conv_snr_isol > 30.0
        if cp.sum(usable) >= 3:
            order = cp.argsort(conv_snr_isol)[::-1][:10]
            conv_snr_mask = cp.zeros_like(conv_snr_isol, dtype=bool)
            conv_snr_mask[order[conv_snr_isol[order] > 30.0]] = True
            logger.info(f"Weak field: aperture corrections from {int(cp.sum(conv_snr_mask))} faint stars, "
                        f"median SNR {float(cp.median(conv_snr_isol[conv_snr_mask])):.1f}")

    del conv_snr_isol

    num_good_isolated = cp.sum(conv_snr_mask)
    if num_good_isolated < 3:
        nvtx.end_range(block2_range)
        nvtx.end_range(overall_range)
        raise InsufficientStarsError(num_stars=num_good_isolated.item())

    nvtx.end_range(block2_range)

    # BLOQUE 3: Aperture radii setup
    block3_range = nvtx.start_range('aperture_radii_setup', category='phot.photo_gpu', color='yellow')
    stamp_half = star_dataset.shape[1] // 2  # asume star_dataset shape (N, h, w)
    max_radii_phot = float(min(stamp_half - 2,
                            np.ceil(5 * fwhm)))  # rango para photometría
    max_radii_corr = float(min(stamp_half - 1,     # rango para construir C(r)
                                np.ceil(6 * fwhm) + 1))
    min_radii = float(np.ceil(0.75 * fwhm))
    radii_phot = cp.arange(int(min_radii), int(max_radii_phot) + 1, 1, dtype=cp.float64)
    radii_corr = cp.arange(int(min_radii), int(max_radii_corr) + 1, 1, dtype=cp.float64)
    if radii_corr.size == 0 or radii_phot.size == 0:
        nvtx.end_range(block3_range)
        nvtx.end_range(overall_range)
        raise DataValidationError("Input data is invalid or insufficient")
    nvtx.end_range(block3_range)

    # BLOQUE 4: Find aperture corrections
    block4_range = nvtx.start_range('aperture_corrections_mapping_gpu', category='phot.photo_gpu', color='purple')
    adaptive_memory_management(mempool)
    # create_aperture_corrections_map_gpu still has internal CPU steps for grouping
    corrections, correction_errors, cluster_centers = create_aperture_corrections_map_gpu(
        (h, w), tile_section_psf, star_dataset[conv_snr_mask],
        isolated_coord[conv_snr_mask], radii_corr
    )
    n_phot = len(radii_phot)
    corrections = corrections[:, :n_phot]
    correction_errors = correction_errors[:, :n_phot]
    del conv_snr_mask, star_dataset, isolated_coord

    # find_aperture_corrections_gpu now uses GPU crossmatch, no external transfers needed
    aperture_corrections, aperture_correction_errors = find_aperture_corrections_gpu(
        source_coord, corrections, correction_errors, cluster_centers
    )

    del corrections, correction_errors, cluster_centers
    nvtx.end_range(block4_range)

    # BLOQUE 5: Get batch photometry
    block5_range = nvtx.start_range('batch_aperture_photometry', category='phot.photo_gpu', color='green')
    adaptive_memory_management(mempool, force_free=True)
    source_flux, back_flux, area = batch_aperture_photometry(img_ori, back, cp.round(source_coord).astype(cp.int32),
                                                             radii_phot)
    if source_flux is None:
        # batch_aperture_photometry signals GPU OOM by returning None fluxes
        # (see its docstring); indexing them below would die with an opaque
        # "'NoneType' object is not subscriptable" three layers away from the
        # actual cause.
        raise MemoryError(
            "batch_aperture_photometry returned no flux: GPU ran out of memory "
            "during aperture photometry (see preceding OOM log entries)")
    del img_ori, back
    # if img_ori.ndim == 3:  # Handle averaging if needed
    #     source_flux = cp.mean(source_flux, axis=0)
    #     if back_flux is not None: back_flux = cp.mean(back_flux, axis=0)
    nvtx.end_range(block5_range)

    # BLOQUE 7: Calculate photometric parameteres
    block7_range = nvtx.start_range('photometric_parameters_calc_gpu', category='phot.photo_gpu', color='lime')

    center_isolated_flux = source_flux[:, source_coords_matched_idx]
    center_isolated_aper_corr = aperture_corrections[source_coords_matched_idx].T
    center_isolated_aper_corr_err = aperture_correction_errors[source_coords_matched_idx].T

    center_isolated_signal = center_isolated_flux / center_isolated_aper_corr

    # Stochastic noise terms (reduced by 1/N from frame averaging)
    center_isolated_back_noise_sq   = cp.abs(back_flux[:, source_coords_matched_idx]) * gain
    center_isolated_read_noise_sq   = area.reshape(-1, 1) * rdnoise ** 2
    center_isolated_source_noise_sq = center_isolated_flux * gain / center_isolated_aper_corr ** 2
    center_isolated_corr_noise_sq = (
        center_isolated_flux * gain * center_isolated_aper_corr_err 
        / center_isolated_aper_corr ** 2
    ) ** 2

    center_isolated_total_noise = cp.sqrt(
        (center_isolated_source_noise_sq + 
        center_isolated_back_noise_sq + 
        center_isolated_read_noise_sq) / n_images
        + center_isolated_corr_noise_sq          # not divided by N
    ) / gain

    del center_isolated_source_noise_sq, center_isolated_back_noise_sq
    del center_isolated_read_noise_sq, center_isolated_corr_noise_sq
    del center_isolated_flux, center_isolated_aper_corr_err, center_isolated_aper_corr

    center_isolated_snr = center_isolated_signal / center_isolated_total_noise
    center_conv_snr = conv_snr[source_coords_matched_idx]

    del source_coords_matched_idx, center_isolated_signal, center_isolated_total_noise

    nvtx.end_range(block7_range)

    # BLOQUE 8: Find optimal aperture radius
    block8_range = nvtx.start_range('optimal_radii_calculation', category='phot.photo_gpu', color='orange')

    xp = cp.get_array_module(center_isolated_snr)
    opt_radii_idx = xp.nanargmax(center_isolated_snr, axis=0)
    opt_radii = radii_phot[opt_radii_idx]
    del opt_radii_idx, center_isolated_snr
    if len(center_conv_snr) == 0 or len(opt_radii) == 0:
        raise DataValidationError("center_conv_snr or opt_radii is empty.")

    polyfit_range = nvtx.start_range('polyfit', category='gpu_ops' if xp == cp else 'cpu_ops')
    try:
        # Filter saturated points (those pinned at r_min or r_max_phot edges)
        not_saturated = (opt_radii > min_radii) & (opt_radii < max_radii_phot)

        if int(xp.sum(not_saturated)) < 10:
            pov = xp.polyfit(xp.log10(center_conv_snr), xp.log10(opt_radii), 1, cov=False)
        else:
            # Inverted fit: bin in integer r_opt, fit log(SNR) vs log(r), then invert.
            # This avoids the bias from r_opt's pixel discretisation that affects
            # a direct log(r) vs log(SNR) OLS fit.
            r_clean = opt_radii[not_saturated]
            log_snr_clean = xp.log10(center_conv_snr[not_saturated])
            unique_r = xp.unique(r_clean)

            log_snr_medians = []
            log_snr_mads = []
            r_used = []
            for r_val in unique_r:
                m = r_clean == r_val
                if int(xp.sum(m)) < 5:
                    continue
                vals = log_snr_clean[m]
                med = xp.median(vals)
                mad = 1.4826 * xp.median(xp.abs(vals - med))
                log_snr_medians.append(float(med))
                log_snr_mads.append(float(mad))
                r_used.append(float(r_val))

            if len(r_used) < 3:
                pov = xp.polyfit(xp.log10(center_conv_snr), xp.log10(opt_radii), 1, cov=False)
            else:
                log_r_used = xp.log10(xp.asarray(r_used, dtype=opt_radii.dtype))
                log_snr_med_arr = xp.asarray(log_snr_medians, dtype=opt_radii.dtype)
                # log_snr_mad_arr = xp.asarray(log_snr_mads, dtype=opt_radii.dtype)
                # weights = 1.0 / xp.maximum(log_snr_mad_arr, 1e-3) ** 2

                pov_inv = xp.polyfit(log_r_used, log_snr_med_arr, 1)#, w=weights)
                slope_inv, intercept_inv = pov_inv[0], pov_inv[1]
                # Invert to recover log(r) = a*log(SNR) + b form
                slope = 1.0 / slope_inv
                intercept = -intercept_inv / slope_inv
                pov = xp.stack([slope, intercept])

        del opt_radii, center_conv_snr
    except Exception as e:
        nvtx.end_range(polyfit_range)
        nvtx.end_range(block8_range)
        raise DataValidationError(f"Polyfit failed: {e}")

    nvtx.end_range(polyfit_range)
    nvtx.end_range(block8_range)

    # BLOQUE 9: Calculate optimal flux and noise
    block9_range = nvtx.start_range('optimal_flux_calculation', category='phot.metadata', color='lime')

    source_opt_rad = cp.round(
        cp.fmax(cp.fmin((10 ** (pov[0] * cp.log10(conv_snr) + pov[1])), max_radii_phot), min_radii)
    ).astype(int)
    source_opt_rad_idx = cp.searchsorted(radii_phot, source_opt_rad, side='right') - 1

    del conv_snr

    opt_aperture_corrections = aperture_corrections[cp.arange(source_coord.shape[0]), source_opt_rad_idx]
    opt_aperture_correction_errors = aperture_correction_errors[cp.arange(source_coord.shape[0]), source_opt_rad_idx]
    opt_flux = source_flux[source_opt_rad_idx, cp.arange(source_coord.shape[0])]

    del source_flux, aperture_correction_errors, aperture_corrections

    mask = opt_flux > 0
    opt_flux = opt_flux[mask]
    opt_aper_corr = opt_aperture_corrections[mask]
    opt_corr_err = opt_aperture_correction_errors[mask]

    del opt_aperture_correction_errors, opt_aperture_corrections

    opt_signal = opt_flux / opt_aper_corr

    opt_back_noise_sq = cp.abs(back_flux[source_opt_rad_idx, cp.arange(source_coord.shape[0])])[mask] * gain
    opt_read_noise_sq = area[source_opt_rad_idx][mask] * rdnoise ** 2

    del source_opt_rad_idx, back_flux

    opt_source_noise_sq = opt_flux * gain / opt_aper_corr ** 2
    opt_corr_noise_sq = (opt_flux * gain * opt_corr_err / opt_aper_corr ** 2) ** 2

    opt_total_noise = cp.sqrt(
        opt_source_noise_sq / n_images +
        opt_back_noise_sq / n_images +
        opt_read_noise_sq / n_images +
        opt_corr_noise_sq
    ) / gain
    opt_coords = source_coord[mask]

    del source_coord, opt_back_noise_sq, opt_read_noise_sq, opt_corr_noise_sq, opt_source_noise_sq, opt_flux, opt_corr_err

    nvtx.end_range(block9_range)

    # BLOQUE 10: Obtain aditional information for header purposes
    block10_range = nvtx.start_range('extra_info_generation', category='phot.metadata', color='yellow')
    extra_info = {}
    ref_snr_values = [10, 100, 250, 1000]
    opt_final_snr = cp.divide(opt_signal, opt_total_noise)
    for snr_ref in ref_snr_values:
        diff_snr = cp.abs(opt_final_snr - snr_ref)
        ref_idx_cp = cp.argmin(diff_snr)
        tx4_range = nvtx.start_range(f'transfer_extra_info_snr{snr_ref}', category='transfer_scalar', color='pink')
        ref_idx_np = ref_idx_cp.item()
        try:
            rad_val = source_opt_rad[mask][ref_idx_np].item()
            corr_val = opt_aper_corr[ref_idx_np].item()
            extra_info[f'RAD{snr_ref}'] = int(rad_val)
            extra_info[f'CORR{snr_ref}'] = round(corr_val, 3)
        except IndexError:
            continue

        nvtx.end_range(tx4_range)

    del source_opt_rad, mask, opt_aper_corr, opt_final_snr
    nvtx.end_range(block10_range)

    # Transferencia final GPU -> CPU para return
    # final_transfer_range = nvtx.start_range('final_gpu_to_cpu_transfer', category='transfer', color='red')
    # opt_signal_np = opt_signal.get()
    # opt_total_noise_np = opt_total_noise.get()
    # opt_coords_np = opt_coords.get()
    # nvtx.end_range(final_transfer_range)

    # Cleanup opcional
    # ... del ...
    # mempool.free_all_blocks()
    # gc.collect()

    nvtx.end_range(overall_range)
    return opt_signal.get(), opt_total_noise.get(), opt_coords.get(), extra_info



@nvtx.annotate('batch_aperture_photometry', category='phot.photo_gpu', color='blue')
def batch_aperture_photometry(
        # Accept both types for image and background
        img_ori_input: cp.ndarray | np.ndarray,
        back_input: cp.ndarray | np.ndarray | None,
        # Positions and radii expected on GPU from calling function
        positions: cp.ndarray,
        radii: cp.ndarray,
        **kwargs
) -> tuple[cp.ndarray | None, cp.ndarray | None, cp.ndarray]:
    """
    Perform aperture photometry with adaptive input handling and internal background subtraction (v4).
    Uses CUDA streams for potentially concurrent plane processing.

    Accepts image and background as NumPy or CuPy arrays, transfers to GPU if needed,
    performs background subtraction internally, and uses delayed output allocation.

    :param img_ori_input: Original image data (GPU or CPU). Background will be subtracted internally.
    :type img_ori_input: cupy.ndarray or numpy.ndarray
    :param back_input: Background image (GPU or CPU, optional). Must match img dimensions if provided.
    :type back_input: cupy.ndarray or numpy.ndarray or None
    :param positions: Positions of sources (GPU) (n_sources, 2) -> [[row, col], ...]. Must be integer type.
    :type positions: cupy.ndarray (int32/64)
    :param radii: Aperture radii (GPU, ideally float32).
    :type radii: cupy.ndarray
    :param kwargs: Potential memory thresholds: 'mem_safe_ratio', 'mem_warning_ratio', 'mem_critical_ratio'.
    :return: Tuple containing flux (background subtracted), background flux (if back provided),
             and aperture area per radius (all CuPy arrays).
             Flux/BackFlux can be None if an OOM error occurred.
    :rtype: tuple(cupy.ndarray | None, cupy.ndarray | None, cupy.ndarray)
    :raises ValueError: If radii is empty or input types are incorrect.
    :raises TypeError: If input types are inconsistent or not array-like.
    """
    nvtx_range = nvtx.start_range('batch_aperture_photometry', category='phot.photo_gpu', color='blue')
    mempool = cp.get_default_memory_pool()

    # Preferred processing dtype (float32 saves memory, float64 more precise)
    # Let's stick to float64 as in the original for now, can be changed later if needed
    processing_dtype = cp.float64

    # --- Input Validation and GPU Transfer ---
    transfer_nvtx = nvtx.start_range('input_transfer_check', category='phot.setup', color='orange')
    img_gpu = None
    back_gpu = None

    # Validate and transfer Image
    if isinstance(img_ori_input, _numpy.ndarray):
        logger.info(f"batch_photometry received NumPy image, transferring to GPU.")
        transfer_start = time.time()
        try:
            # Transfer directly with the target processing dtype
            img_gpu = cp.asarray(img_ori_input, dtype=processing_dtype)
            # No need to free here, let Python GC handle img_ori_input when it goes out of scope
        except Exception as e:
            logger.error(f"Failed to transfer input image to GPU: {e}", exc_info=True)
            nvtx.end_range(transfer_nvtx)
            nvtx.end_range(nvtx_range)
            raise MemoryError("Failed to allocate image on GPU") from e
        transfer_end = time.time()
        logger.info(
            f"Image transfer took {transfer_end - transfer_start:.2f}s. Mem: {mempool.used_bytes() / 1e9:.2f} GB")
    elif isinstance(img_ori_input, cp.ndarray):
        # Ensure correct dtype, avoid copy if already correct
        if img_ori_input.dtype != processing_dtype:
            logger.info(f"Converting input GPU image to {processing_dtype}")
            img_gpu = img_ori_input.astype(processing_dtype, copy=True)  # Explicit copy if dtype changes
        else:
            img_gpu = img_ori_input  # No copy needed
    else:
        raise TypeError(f"img_ori_input must be NumPy or CuPy array, got {type(img_ori_input)}")

    del img_ori_input

    # Validate and transfer Background (if provided)
    if back_input is not None:
        if isinstance(back_input, _numpy.ndarray):
            if back_input.shape != img_gpu.shape[-back_input.ndim:]:  # Check shape against GPU image dims
                raise ValueError(f"Background shape {back_input.shape} mismatch with image shape {img_gpu.shape}.")
            logger.info(f"batch_photometry received NumPy background, transferring to GPU.")
            transfer_start = time.time()
            try:
                back_gpu = cp.asarray(back_input, dtype=processing_dtype)
            except Exception as e:
                logger.error(f"Failed to transfer input background to GPU: {e}", exc_info=True)
                del img_gpu  # Clean up already transferred image
                mempool.free_all_blocks()
                gc.collect()
                nvtx.end_range(transfer_nvtx)
                nvtx.end_range(nvtx_range)
                raise MemoryError("Failed to allocate background on GPU") from e
            transfer_end = time.time()
            logger.info(
                f"Background transfer took {transfer_end - transfer_start:.2f}s. Mem: {mempool.used_bytes() / 1e9:.2f} GB")
        elif isinstance(back_input, cp.ndarray):
            if back_input.shape != img_gpu.shape[-back_input.ndim:]:
                raise ValueError(f"Background shape {back_input.shape} mismatch with image shape {img_gpu.shape}.")
            # Ensure correct dtype, avoid copy if already correct
            if back_input.dtype != processing_dtype:
                logger.info(f"Converting input GPU background to {processing_dtype}")
                back_gpu = back_input.astype(processing_dtype, copy=True)  # Explicit copy
            else:
                back_gpu = back_input  # No copy needed
        else:
            raise TypeError(f"back_input must be NumPy/CuPy array or None, got {type(back_input)}")

        del back_input
    else:
        back_gpu = None  # Explicitly None if not provided

    # Validate Radii and Positions (expected on GPU)
    if not isinstance(positions, cp.ndarray):
        # Maybe transfer if numpy? For now, enforce CuPy input.
        raise TypeError("positions must be a CuPy array.")
    if not isinstance(radii, cp.ndarray):
        raise TypeError("radii must be a CuPy array.")
    # Ensure radii has the correct dtype for processing if needed, but keep original precision for area
    radii_proc = radii.astype(processing_dtype, copy=False)

    nvtx.end_range(transfer_nvtx)
    logger.debug(f"Inputs transferred/validated. GPU Memory: Used={mempool.used_bytes() / 1e9:.2f} GB")

    # --- Validations and Area Calculation ---
    if radii.size == 0:
        raise ValueError("Radii array cannot be empty.")
    area_calc_range = nvtx.start_range('area_kernel_params', category='phot.setup')
    areas = cp.zeros(len(radii), dtype=cp.float64)  # Keep area precision high
    try:
        # Use original radii for area calculation precision
        max_radius_area = cp.max(radii).item()
        kernel_size_area = int(2 * np.ceil(max_radius_area) + 1)
        if kernel_size_area % 2 == 0: kernel_size_area += 1
        for i, r_orig in enumerate(radii):  # Iterate over original radii
            # get_aper_kernel likely returns float64 kernel, area calculation is sensitive
            temp_kernel, areas[i] = get_aper_kernel(r_orig.item(), size=kernel_size_area)
            del temp_kernel
    except Exception as e:
        logger.error(f"Error calculating areas: {e}", exc_info=True)
        areas.fill(cp.nan)  # Mark areas as invalid
    nvtx.end_range(area_calc_range)

    # Handle empty positions *after* area calculation
    if positions.size == 0:
        logger.warning("Positions array is empty. Returning empty flux results.")
        n_radii_out = len(radii)
        n_images_out = img_gpu.shape[0] if img_gpu.ndim == 3 else 1
        # Ensure output shape matches image dimensions
        if img_gpu.ndim == 3:
            out_shape = (n_images_out, n_radii_out, 0)
        elif img_gpu.ndim == 2:
            out_shape = (n_radii_out, 0)
        else:  # Handle 1D or other unexpected dims if necessary
            logger.error(f"Unexpected image dimension: {img_gpu.ndim}")
            nvtx.end_range(nvtx_range)
            return None, None, areas  # Or raise error

        nvtx.end_range(nvtx_range)
        # Return results matching processing_dtype
        return cp.zeros(out_shape, dtype=processing_dtype), \
            cp.zeros(out_shape, dtype=processing_dtype) if back_gpu is not None else None, \
            areas

    # --- FFT Setup ---
    image_shape_orig = img_gpu.shape
    is_3d = img_gpu.ndim == 3
    n_images = image_shape_orig[0] if is_3d else 1
    img_h = image_shape_orig[1] if is_3d else image_shape_orig[0]
    img_w = image_shape_orig[2] if is_3d else image_shape_orig[1]
    n_radii = len(radii)
    n_positions = len(positions)

    # Determine kernel size based on max radius using the processing dtype version
    try:
        max_radius = cp.nanmax(radii_proc).item()  # Use radii_proc here
        kernel_size = int(2 * np.ceil(max_radius) + 1)
        if kernel_size % 2 == 0: kernel_size += 1
    except ValueError:  # Handles case where radii_proc contains only NaNs
        logger.error("Could not determine valid kernel size from radii (all NaNs?). Aborting.")
        nvtx.end_range(nvtx_range)
        # Clean up GPU memory before returning
        del img_gpu, back_gpu, radii_proc, positions
        mempool.free_all_blocks()
        gc.collect()
        return None, None, areas  # Return None for fluxes, but calculated areas

    padding = (kernel_size - 1) // 2
    try:
        # Calculate FFT shape based on image dimensions + padding
        fft_shape = fill_image((img_h + 2 * padding, img_w + 2 * padding))
    except Exception as e:
        logger.error(f"Error calculating FFT shape: {e}", exc_info=True)
        nvtx.end_range(nvtx_range)
        del img_gpu, back_gpu, radii_proc, positions
        mempool.free_all_blocks()
        gc.collect()
        return None, None, areas

    logger.debug(
        f"Using FFT shape: {fft_shape} for image ({img_h}, {img_w}), max radius {max_radius}, kernel size {kernel_size}")

    # --- Function to process a single 2D image plane ---
    # (process_plane remains largely the same as provided in the prompt,
    #  it correctly uses the passed 'stream' argument)
    def process_plane(plane_idx, img_plane_subtracted_gpu, back_plane_gpu, stream):
        # ... (Existing process_plane implementation from the prompt) ...
        # Important: Ensure it returns (plane_flux, plane_back_flux)
        # Ensure internal variables are cleaned up (del) before returning
        # Ensure it handles potential OOM within the plane processing gracefully
        # and returns (None, None) in case of failure.

        # <<< PASTE THE ORIGINAL process_plane FUNCTION CODE HERE >>>
        # Make sure it accepts 'stream' as the last argument and uses it internally.
        # It should look something like this:

        # with stream:
        #    plane_nvtx = None
        #    # ... rest of the function ...
        #    try:
        #       # ... calculations ...
        #       return plane_flux, plane_back_flux # On success
        #    except Exception as e:
        #       logger.error(...)
        #       # ... cleanup internal variables ...
        #       return None, None # On failure
        #    finally:
        #       if plane_nvtx: nvtx.end_range(plane_nvtx)

        # --- Placeholder for the actual process_plane function ---
        # Replace this placeholder with your actual function implementation
        # Ensure it correctly uses the 'stream' argument passed to it.
        with stream:
            plane_nvtx = None
            fft_img_range = None
            radius_nvtx = None
            kernel_range = None
            conv_range = None
            conv_back_range = None
            plane_flux = None
            img_c, back_c = None, None
            kernel_c = None
            positions_local = positions  # Use positions directly from outer scope
            radii_local = radii_proc  # Use processing radii from outer scope

            try:
                plane_nvtx = nvtx.start_range(f'process_plane_{plane_idx}', category='phot.plane')
                logger.debug(
                    f"Processing plane {plane_idx} on stream {stream.ptr}. Mem: {mempool.used_bytes() / 1e9:.2f} GB")

                # Perform FFTs (on subtracted image and original background if available)
                fft_img_range = nvtx.start_range(f'fft_plane_{plane_idx}', category='phot.fft')
                try:
                    # FFT requires float input for rfft2 (real->complex).
                    # Input arrays are already processing_dtype.
                    img_c = cp.fft.rfft2(img_plane_subtracted_gpu, s=fft_shape)
                    if back_plane_gpu is not None:
                        back_c = cp.fft.rfft2(back_plane_gpu, s=fft_shape)
                    else:
                        back_c = None
                except Exception as e:
                    logger.error(f"FFT failed for plane {plane_idx}: {e}", exc_info=True)
                    raise  # Propagate error to be caught by outer try-except
                finally:
                    if fft_img_range: nvtx.end_range(fft_img_range)

                # Allocate output for *this plane* (matching processing dtype)
                plane_flux = cp.zeros((n_radii, n_positions), dtype=processing_dtype)
                plane_back_flux = cp.zeros((n_radii, n_positions),
                                           dtype=processing_dtype) if back_plane_gpu is not None else None

                # --- Loop over radii ---
                for i, r in enumerate(radii_local):  # Use processing radii
                    kernel = None
                    prod_img = None
                    convolved_img = None
                    convolved_img_rolled = None
                    convolved_img_cropped = None
                    prod_back = None
                    convolved_back = None
                    convolved_back_rolled = None
                    convolved_back_cropped = None
                    radius_nvtx = None
                    kernel_range = None
                    conv_range = None
                    conv_back_range = None
                    kernel_c = None  # Needs to be defined before try

                    try:
                        r_item = r.item()  # Get scalar value for NVTX/logging
                        radius_nvtx = nvtx.start_range(f'radius_{r_item:.2f}', category='phot.radius_iter')

                        # --- Kernel FFT ---
                        kernel_range = nvtx.start_range(f'kernel_fft_r={r_item:.2f}', category='phot.conv')
                        try:
                            # Use original radius for get_aper_kernel if precision matters there,
                            # but the kernel used for convolution should match processing_dtype.
                            # Assuming get_aper_kernel uses the scalar radius value.
                            # We use the corresponding original radius `radii[i]` if needed,
                            # otherwise `r_item` from `radii_proc` is fine.
                            # Let's assume `r_item` is sufficient here for kernel shape.
                            kernel_orig_precision, _ = get_aper_kernel(r_item, size=kernel_size)  # Recalculate kernel
                            kernel = kernel_orig_precision.astype(processing_dtype,
                                                                  copy=False)  # Cast to processing dtype
                            # We need the conjugate for convolution theorem multiplication
                            kernel_c = cp.conj(cp.fft.rfft2(kernel, s=fft_shape))
                            del kernel, kernel_orig_precision  # Free kernel memory
                            kernel = None
                        except Exception as e:
                            logger.error(f"Kernel FFT failed for radius {r_item} in plane {plane_idx}: {e}",
                                         exc_info=True)
                            raise  # Propagate
                        finally:
                            if kernel_range: nvtx.end_range(kernel_range)

                        # --- Convolution (Subtracted Image for Flux) ---
                        conv_range = nvtx.start_range(f'ifft_roll_index_img_r={r_item:.2f}', category='phot.conv')
                        try:
                            prod_img = img_c * kernel_c
                            # Output of irfft2 matches input FFT precision (float if img_c was complex)
                            convolved_img = cp.fft.irfft2(prod_img, s=fft_shape)
                            del prod_img
                            prod_img = None  # Free memory
                            # Rolling introduces temporary copy/view
                            convolved_img_rolled = cp.roll(convolved_img, shift=[padding, padding], axis=[0, 1])
                            del convolved_img
                            convolved_img = None  # Free memory
                            # Cropping introduces view or copy depending on contiguity
                            convolved_img_cropped = convolved_img_rolled[:img_h, :img_w]
                            del convolved_img_rolled
                            convolved_img_rolled = None  # Free memory
                            # Perform indexing (advanced indexing creates a copy)
                            plane_flux[i, :] = convolved_img_cropped[positions_local[:, 0], positions_local[:, 1]]
                            del convolved_img_cropped
                            convolved_img_cropped = None  # Free memory
                        except Exception as e:
                            logger.error(
                                f"Image convolution/indexing failed for radius {r_item} in plane {plane_idx}: {e}",
                                exc_info=True)
                            raise  # Propagate
                        finally:
                            if conv_range: nvtx.end_range(conv_range)

                        # --- Convolution (Original Background for Back Flux) ---
                        if back_c is not None and plane_back_flux is not None:
                            conv_back_range = nvtx.start_range(f'ifft_roll_index_back_r={r_item:.2f}',
                                                               category='phot.conv')
                            try:
                                prod_back = back_c * kernel_c
                                convolved_back = cp.fft.irfft2(prod_back, s=fft_shape)
                                del prod_back
                                prod_back = None
                                convolved_back_rolled = cp.roll(convolved_back, shift=[padding, padding], axis=[0, 1])
                                del convolved_back
                                convolved_back = None
                                convolved_back_cropped = convolved_back_rolled[:img_h, :img_w]
                                del convolved_back_rolled
                                convolved_back_rolled = None
                                plane_back_flux[i, :] = convolved_back_cropped[
                                    positions_local[:, 0], positions_local[:, 1]]
                                del convolved_back_cropped
                                convolved_back_cropped = None
                            except Exception as e:
                                logger.error(
                                    f"Background convolution/indexing failed for radius {r_item} in plane {plane_idx}: {e}",
                                    exc_info=True)
                                raise  # Propagate
                            finally:
                                if conv_back_range: nvtx.end_range(conv_back_range)

                        # --- Radius Cleanup ---
                        del kernel_c
                        kernel_c = None  # Clean up kernel FFT explicitly

                    except (cp.cuda.runtime.CUDARuntimeError, MemoryError) as e_radius:
                        logger.error(f"OOM or CUDA Error during radius r={r_item} in plane {plane_idx}: {e_radius}",
                                     exc_info=False)  # Less verbose exc_info
                        # Clean up potential intermediate arrays from this radius iteration.
                        # Rebinding to None drops the reference just like `del`, but cannot
                        # raise: a bare `del` dies with UnboundLocalError when the OOM struck
                        # before the name was assigned, and that error would replace e_radius
                        # and hide the real OOM from the log and the outer handler.
                        kernel_c = prod_img = convolved_img = convolved_img_rolled = convolved_img_cropped = None
                        prod_back = convolved_back = convolved_back_rolled = convolved_back_cropped = None
                        mempool.free_all_blocks()  # Try to free memory
                        gc.collect()
                        raise e_radius  # Propagate error to outer handler for the plane
                    except Exception as e_gen_radius:
                        logger.error(f"Unexpected error during radius r={r_item} in plane {plane_idx}: {e_gen_radius}",
                                     exc_info=True)
                        raise e_gen_radius
                    finally:
                        if radius_nvtx: nvtx.end_range(radius_nvtx)
                # --- End Radii Loop ---

                # --- Successful Plane Cleanup ---
                logger.debug(f"Finished radii for plane {plane_idx}. Cleaning up FFT data.")
                del img_c, back_c  # Delete FFT transforms for the plane
                img_c, back_c = None, None  # Ensure they are None
                # No need to call free_all_blocks here, let the outer loop manage batch cleanup
                # gc.collect() # Avoid frequent GC inside the hot loop if possible
                return plane_flux, plane_back_flux

            except (cp.cuda.runtime.CUDARuntimeError, MemoryError) as e_plane:
                logger.error(f"OOM or CUDA Error processing plane {plane_idx}: {e_plane}",
                             exc_info=False)  # Less verbose
                # Ensure cleanup of major allocations if error occurred mid-plane.
                # None-rebind instead of `del`: the radius handler above has already
                # unbound kernel_c on its way here, so `del kernel_c` would raise
                # UnboundLocalError inside this handler and mask the real OOM.
                img_c = back_c = plane_flux = kernel_c = None
                # Try to free memory before returning None
                mempool.free_all_blocks()
                gc.collect()
                return None, None  # Signal failure for this plane
            except Exception as e_gen_plane:
                logger.error(f"Unexpected error processing plane {plane_idx}: {e_gen_plane}", exc_info=True)
                img_c = back_c = plane_flux = kernel_c = None
                mempool.free_all_blocks()
                gc.collect()
                return None, None  # Signal failure
            finally:
                # Make sure NVTX range is ended even if exceptions occur
                if plane_nvtx: nvtx.end_range(plane_nvtx)

    # --- End process_plane function ---

    # --- Main Processing Loop ---
    fft_loop_range = nvtx.start_range('fft_convolution_loop', category='phot.photo_gpu')
    final_flux = None
    final_back_flux = None
    success = True  # Overall success flag

    # Create a fixed number of streams for controlled concurrency
    # Limit concurrency to avoid excessive intermediate memory usage
    max_concurrent = min(100, n_images) if n_images > 1 else 1  # Ensure at least 1 stream
    logger.info(f"Using {max_concurrent} concurrent streams for processing {n_images} planes.")
    streams = [cp.cuda.Stream() for _ in range(max_concurrent)]

    # --- 3D Image Cube Processing ---
    if is_3d:
        all_plane_flux_list = [None] * n_images  # Preallocate list for results
        all_plane_back_flux_list = [None] * n_images if back_gpu is not None else None

        # Store futures: (plane_idx, stream_obj, future_result_tuple)
        # We don't need the stream_obj later if we sync all, but good for debug
        futures_in_flight = []

        for batch_start in range(0, n_images, max_concurrent):
            batch_nvtx = nvtx.start_range(f'batch_{batch_start}-{min(batch_start + max_concurrent, n_images) - 1}',
                                          category='phot.batch')
            batch_end = min(batch_start + max_concurrent, n_images)
            batch_size = batch_end - batch_start
            streams_this_batch = streams[:batch_size]  # Get streams for this batch

            # --- 1. Launch work for the current batch asynchronously ---
            launch_nvtx = nvtx.start_range(f'launch_batch', category='phot.launch')
            logger.debug(f"Launching batch: planes {batch_start} to {batch_end - 1}")
            current_batch_futures = []
            for i in range(batch_size):
                plane_idx = batch_start + i
                stream = streams_this_batch[i]

                # Prepare data for the plane (do this outside 'with stream' if it involves complex setup,
                # but simple slicing and subtraction are fine inside)
                # The actual process_plane call must be inside 'with stream'
                img_plane = img_gpu[plane_idx]
                back_plane = back_gpu[plane_idx] if back_gpu is not None else None

                # Perform subtraction - this intermediate needs memory
                # Do it inside the stream context so it's scheduled correctly.
                with stream:
                    img_subtracted = img_plane - back_plane if back_plane is not None else img_plane.copy()  # Use copy if no subtraction to avoid modifying input

                    # Call process_plane - this queues operations on the stream
                    # It returns GPU arrays immediately, computation happens later.
                    future_result_tuple = process_plane(plane_idx, img_subtracted, back_plane, stream)

                    # Store the future result along with its index and stream
                    current_batch_futures.append((plane_idx, stream, future_result_tuple))

                    # Explicitly delete the temporary subtracted image *within the stream's context*
                    # This might not free memory immediately but signals intent earlier.
                    del img_subtracted

            nvtx.end_range(launch_nvtx)  # End launch phase NVTX

            # --- 2. Synchronize streams for this batch ---
            sync_nvtx = nvtx.start_range(f'sync_batch', category='phot.sync')
            logger.debug(f"Synchronizing {len(streams_this_batch)} streams for batch starting at {batch_start}...")
            sync_start_time = time.time()
            for i in range(batch_size):
                # Sync each stream used in this batch
                streams_this_batch[i].synchronize()
            sync_end_time = time.time()
            logger.debug(f"Batch synchronization took {sync_end_time - sync_start_time:.3f} s")
            nvtx.end_range(sync_nvtx)  # End sync phase NVTX

            # --- 3. Collect results for this batch ---
            collect_nvtx = nvtx.start_range(f'collect_batch', category='phot.collect')
            logger.debug(f"Collecting results for batch starting at {batch_start}...")
            for plane_idx, stream, future_result in current_batch_futures:
                # Now that the stream is synchronized, the data is ready on GPU
                plane_flux, plane_back_flux = future_result

                if plane_flux is not None:
                    all_plane_flux_list[plane_idx] = plane_flux
                    if all_plane_back_flux_list is not None and plane_back_flux is not None:
                        all_plane_back_flux_list[plane_idx] = plane_back_flux
                    elif all_plane_back_flux_list is not None and plane_back_flux is None and back_gpu is not None:
                        # Handle case where background existed but failed for this plane
                        logger.warning(
                            f"Background processing failed for plane {plane_idx}, but flux calculation succeeded.")
                        # Keep None in the list for this plane's background flux
                else:
                    # process_plane returned None, indicating failure for this plane
                    logger.error(f"Processing failed for plane {plane_idx}. Flux result will be missing.")
                    success = False  # Mark overall process as potentially incomplete/failed
                    # Keep None in the list for this plane's flux (and back_flux if applicable)

            del plane_idx, stream, future_result

            nvtx.end_range(collect_nvtx)  # End collect phase NVTX

            # --- Cleanup after batch ---
            # Clear the futures list for the completed batch
            del current_batch_futures
            # Force garbage collection and free unused GPU memory blocks *after* each batch
            # This is crucial when processing many planes to avoid OOM
            cleanup_nvtx = nvtx.start_range('batch_cleanup', category='phot.cleanup')
            logger.debug(f"Cleaning memory after batch. Used: {mempool.used_bytes() / 1e9:.2f} GB")
            mempool.free_all_blocks()
            gc.collect()
            logger.debug(f"Memory after cleanup. Used: {mempool.used_bytes() / 1e9:.2f} GB")
            nvtx.end_range(cleanup_nvtx)

            nvtx.end_range(batch_nvtx)  # End overall batch NVTX

        # --- Stack results *after* processing all batches ---
        if success and any(f is not None for f in all_plane_flux_list):  # Check if any plane succeeded
            stack_range = nvtx.start_range('stack_results', category='phot.postproc')
            logger.info(
                f"Stacking final results from {sum(1 for f in all_plane_flux_list if f is not None)} successful planes...")
            try:
                # Filter out None values before stacking if any planes failed
                valid_fluxes = [f for f in all_plane_flux_list if f is not None]
                if valid_fluxes:
                    final_flux = cp.stack(valid_fluxes, axis=0)
                    del valid_fluxes
                else:  # All planes failed
                    logger.error("All planes failed processing. Returning None for flux.")
                    final_flux = None
                    success = False

                if all_plane_back_flux_list is not None:
                    valid_back_fluxes = [bf for bf in all_plane_back_flux_list if bf is not None]
                    if valid_back_fluxes:
                        final_back_flux = cp.stack(valid_back_fluxes, axis=0)
                    # If valid_back_fluxes is empty but final_flux exists, it means background failed everywhere or wasn't calculated
                    # Keep final_back_flux as None in this case or if final_flux is None

            except Exception as e_stack:
                logger.error(f"Error during final stacking: {e_stack}", exc_info=True)
                success = False
                final_flux, final_back_flux = None, None  # Ensure reset on stacking failure
            finally:
                del all_plane_flux_list, all_plane_back_flux_list  # Clear intermediate list memory
                mempool.free_all_blocks()
                gc.collect()
                if stack_range: nvtx.end_range(stack_range)
        elif not success:
            logger.error("Overall processing failed or was incomplete. Returning None for fluxes.")
            final_flux, final_back_flux = None, None
        else:  # Case: success=True but all_plane_flux_list was empty or all None (shouldn't happen if positions weren't empty)
            logger.warning("Processing finished, but no valid plane results were collected.")
            final_flux, final_back_flux = None, None


    # --- 2D Image Processing ---
    else:  # Case 2D (n_images = 1)
        logger.debug(f"Processing 2D image.")
        stream = streams[0]  # Use the first stream
        img_subtracted = None  # Define before try block
        try:
            with stream:
                # Perform subtraction within the stream context
                img_subtracted = img_gpu - back_gpu if back_gpu is not None else img_gpu.copy()  # Use copy if no subtraction

                # Call process_plane for the single plane (index 0)
                final_flux, final_back_flux = process_plane(0, img_subtracted, back_gpu, stream)

                del img_subtracted  # Delete intermediate

            # Synchronize the single stream used
            stream.synchronize()

            if final_flux is None:
                logger.critical(f"Processing failed for the 2D image.")
                success = False  # Mark failure

        except Exception as e_2d:
            logger.error(f"Error processing 2D image: {e_2d}", exc_info=True)
            success = False
            final_flux, final_back_flux = None, None
            # Cleanup potential intermediate
            del img_subtracted
            mempool.free_all_blocks()
            gc.collect()

    nvtx.end_range(fft_loop_range)
    # --- End FFT Loop ---

    # Final cleanup of main inputs (img_gpu, back_gpu might have been deleted earlier on error)
    final_cleanup_range = nvtx.start_range('final_cleanup', category='phot.cleanup')
    logger.debug("Performing final cleanup of input arrays.")
    try:
        del img_gpu
    except NameError:
        pass
    try:
        del back_gpu
    except NameError:
        pass
    try:
        del radii_proc
    except NameError:
        pass
    # positions is input, maybe don't delete here unless sure it's a copy
    # del positions
    mempool.free_all_blocks()
    gc.collect()
    logger.debug(f"Final memory usage: {mempool.used_bytes() / 1e9:.2f} GB")
    nvtx.end_range(final_cleanup_range)

    nvtx.end_range(nvtx_range)  # End overall function range

    if success:
        logger.info(f"Batch aperture photometry completed.")
    else:
        logger.error(f"Batch aperture photometry finished with errors or failures.")

    # Ensure return types match expectation even on failure (None or CuPy array)
    if not success:
        return None, None, areas
    else:
        return final_flux, final_back_flux, areas



### # @hierarchical_debug(logger)
@nvtx.annotate('calibrate_image', category='phot.photo_gpu')
def calibrate_image(imdata: np.ndarray, filter: str, binning:int, scale: float, gain: float, rdnoise: float,
                    exptime: float, satlevel: float, target_ra: float, target_dec: float = None, n_images: int = 1,
                    SP_filt: bool = True, CR_filt: bool = False, border: int = 20, center_factor: float = 0.7,
                    pca_method: bool = True, max_stars_ref: int = 15, min_snr: int = 5,
                    color_range: float = 0.6, tile_section_psf: int = 3000, zp_maxmag: float = 21,
                    sip_order: int = 3, peak_to_total_max: float = 0.3, **kwargs):
    """
    Calibrate an image.

    :param imdata: Image data.
    :type imdata: numpy.ndarray
    :param filter: Filter used for the image.
    :type filter: str
    :param binning: Binning factor for the image.
    :type binning: int
    :param scale: Image scale, in arcsec/pixel.
    :type scale: float
    :param gain: Gain value, in e-/ADU.
    :type gain: float
    :param rdnoise: Read noise value, in e-.
    :type rdnoise: float
    :param exptime: Exposure time, in seconds.
    :type exptime: float
    :param satlevel: Saturation level, in ADU.
    :type satlevel: float
    :param target_ra: Target right ascension, in degrees.
    :type target_ra: float
    :param target_dec: Target declination, in degrees.
    :type target_dec: float or None
    :param n_images: Number of stacked images.
    :type n_images: int
    :param SP_filt: Whether to apply a Salt-and-Pepper filter.
    :type SP_filt: bool
    :param CR_filt: Whether to apply a cosmic ray filter.
    :type CR_filt: bool
    :param border: Distance from the border where sources are ignored.
    :type border: int
    :param center_factor: Factor to select the center of the image for reference calculations.
    :type center_factor: float
    :param pca_method: Whether to use PCA for PSF fitting.
    :type pca_method: bool
    :param tile_section: Tile section size for background estimation.
    :type tile_section: int
    :param max_stars_ref: Maximum number of stars to use for reference PSF.
    :type max_stars_ref: int
    :param min_snr: Minimum SNR for source detection.
    :type min_snr: int
    :param color_range: Color range around B-V = 0.65 for zeropoing calculations.
    :type color_range: float
    :param tile_section_psf: Tile section size for aperture corrections variations.
    :type tile_section_psf: int
    :param peak_to_total_max: Maximum peak-to-total flux ratio for a stamp to enter the
                              PSF model; above it the stamp is treated as a hot pixel.
                              The ratio of a real star depends on how well the PSF is
                              sampled (~0.03 at 5 px FWHM, but much higher when the FWHM
                              approaches one pixel), so severely undersampled cameras may
                              need a value above the 0.3 default. Set per instrument in
                              the camera JSON.
    :type peak_to_total_max: float
    :return: Calibration dictionary, astrometry dictionary, photometry dataframe.
    :rtype: tuple(dict, dict, pandas.DataFrame)
    :raises InsufficientStarsError: If less than 5 isolated stars are detected.
    :raises MoffatFitError: If there's an error fitting Moffat to reference PSF.
    """

    mempool = cp.get_default_memory_pool()

    # with gpu_array_manager(imdata, mempool) as img_cp:
    img_cp = cp.asarray(imdata)
    imadata_shape = imdata.shape
    del imdata

    # Protect bad prereduction
    img_cp[cp.isinf(img_cp) | cp.isnan(img_cp)] = 0

    # Get background
    back, _ = get_local_background_fft(img_cp, scale, get_std=False, **kwargs)

    del _

    rms = cp.sqrt(back * gain / binning**2 + rdnoise**2) / gain / cp.sqrt(n_images)
    xmin = int(imadata_shape[1] * 0.5 * (1 - 0.3))
    xmax = int(imadata_shape[1] * 0.5 * (1 + 0.3))
    ymin = int(imadata_shape[0] * 0.5 * (1 - 0.3))
    ymax = int(imadata_shape[0] * 0.5 * (1 + 0.3))
    m = cp.median(back[ymin:ymax, xmin:xmax])
    s = cp.std(back[ymin:ymax, xmin:xmax])
    mask = cp.abs(back[ymin:ymax, xmin:xmax] - m) < 3 * s
    m = cp.median(back[ymin:ymax, xmin:xmax][mask])

    del mask
    fluxsky = np.round(m.get(), 6)
    del m
    if fluxsky < 0: logger.warning('Median background flux is negative')
    dic_calib = {'FLUXSKY': fluxsky}

    # Detect isolated stars
    if CR_filt:
        img = CR_filter(img_cp - back)
    elif SP_filt:
        img = SP_filter(img_cp - back)
    else:
        img = img_cp - back

    # del img_cp
    adaptive_memory_management(mempool)

    sources = detect_isolated_stars(img[border:-border, border:-border],
                                    rms[border:-border, border:-border],
                                    scale, sat_lim=satlevel * 0.8, **kwargs)

    sources = sources + border

    if len(sources) < 5:
        logger.error('Less than 5 isolated stars detected')
        raise InsufficientStarsError(num_stars=len(sources))

    star_dataset, coord, scaling = create_star_dataset(img, sources, scale)
    del sources
    center_factor = np.min((center_factor, 1))
    xmin = int(imadata_shape[1] * 0.5 * (1 - center_factor))
    xmax = int(imadata_shape[1] * 0.5 * (1 + center_factor))
    ymin = int(imadata_shape[0] * 0.5 * (1 - center_factor))
    ymax = int(imadata_shape[0] * 0.5 * (1 + center_factor))

    # Find reference psf
    unit_star_dataset = star_dataset.astype(cp.double) / scaling[:, 3][:, None, None]
    center_mask = (coord[:, 1] > xmin) & (coord[:, 1] < xmax) & (coord[:, 0] > ymin) & (coord[:, 0] < ymax)
    # peak_to_total_ratio = cp.std(unit_star_dataset, axis=(1, 2))
    peak_to_total_ratio = scaling[:, 0] / scaling[:, 3]
    mask_star_dataset = peak_to_total_ratio < peak_to_total_max
    unit_star_dataset = unit_star_dataset[mask_star_dataset]
    coord = coord[mask_star_dataset]
    star_dataset = star_dataset[mask_star_dataset]

    star_dataset_ref = unit_star_dataset[center_mask[mask_star_dataset]]
    star_dataset_ref = star_dataset_ref[cp.argsort(scaling[center_mask & mask_star_dataset, 3])[::-1]]
    star_dataset_ref = star_dataset_ref[:max_stars_ref, :, :]
    del center_mask, peak_to_total_ratio, scaling
    if star_dataset_ref.shape[0] < 5:
        logger.error('Less than 5 isolated stars detected. Image may be too crowded or too noisy')
        raise InsufficientStarsError(num_stars=star_dataset_ref.shape[0])

    psf, _ = stack_sigmaclip(star_dataset_ref, n=2)

    del star_dataset_ref, _

    psf = psf / cp.sum(psf)
    try:
        _, _, _, fwhm, _ = fit_moffat(psf)
        del _
    except:
        logger.error('Error fitting Moffat to reference PSF')
        raise MoffatFitError()
    if fwhm < 1:
        logger.error('Error fitting Moffat to reference PSF')
        raise MoffatFitError()

    dic_calib['FWHM'] = fwhm
    eigen_psfs, coeff_map = None, None
    fwhm = max(fwhm, 2)


    if pca_method:
        # Create PSF deviations dataset (each star minus the reference PSF).
        # Both have unit flux by construction, so sum(d_i) = 0 per stamp.
        unit_star_dataset_dev = unit_star_dataset - psf

        # Weighted-PCA approximation: normalise each deviation by its own
        # standard deviation. This down-weights stars with low SNR (whose
        # deviations are dominated by noise) and lets stars with cleaner
        # signal contribute more to the principal components.
        norm_factor = cp.std(unit_star_dataset_dev, axis=(1, 2))[:, None, None]
        unit_star_dataset_dev_norm = unit_star_dataset_dev / norm_factor

        # PCA on the normalised deviations. sklearn centres per-pixel internally.
        eigen_psfs = get_eigen_psfs(unit_star_dataset_dev_norm, n_components=5)
        eigen_psfs = cp.asarray(eigen_psfs)

        # Projection on the same (normalised) space, then rescale coefficients
        # to recover their physical magnitude in the original deviation space.
        coefficients_norm = project_all_stars_onto_eigenpsfs(unit_star_dataset_dev_norm, eigen_psfs)
        coefficients = coefficients_norm * norm_factor.squeeze()[:, None]
        coeff_map = create_coeff_map(imadata_shape, coord, coefficients.T, scale, tile_section=tile_section_psf)

        del coefficients
        mempool.free_all_blocks()
        gc.collect()

        x1 = int(imadata_shape[1] * 0.25)
        x2 = int(imadata_shape[1] * 0.75)
        y1 = int(imadata_shape[0] * 0.25)
        y2 = int(imadata_shape[0] * 0.75)
        x = np.array([x1, x2, x1, x2])
        y = np.array([y1, y2, y2, y1])
        fwhm_lab = ['FWHMLL', 'FWHMLR', 'FWHMUL', 'FWHMUR']
        # fwhms = np.zeros(len(fwhm_lab))
        for point in range(len(fwhm_lab)):
            psf_l = psf + recreate_normed_star(coeff_map, eigen_psfs, (x[point], y[point]))
            try:
                _, _, _, fwhm_l, _ = fit_moffat(psf_l)
                fwhm_l = np.round(fwhm_l, 2)
            except:
                fwhm_l = 0
            dic_calib[fwhm_lab[point]] = fwhm_l

    del unit_star_dataset
    # Detect sources
    sources, conv_ima_sigma = detect_sources_psf(img, rms, fwhm, psf, eigen_psfs, coeff_map, min_snr=min_snr, **kwargs)
    sources = sources[(sources[:, 0] > border) & (sources[:, 0] < img.shape[0] - border) & (sources[:, 1] > border) & (
            sources[:, 1] < img.shape[1] - border)]

    del coeff_map, img, rms, psf
    mempool.free_all_blocks()

    # Perform optimized photometry

    # if 'img_cp' not in locals():
    #     img_cp = cp.asarray(imdata)
    # img_cp = cp.asarray(imdata)
    # with gpu_array_manager(imdata, mempool) as img_cp:

    # Note: center_factor and min_conv_snr use function defaults, not DEFAULT_PROCESSING_PARAMS

    optimal_flux, optimal_noise, optimal_coords, extra_info = perform_opt_photometry(
        img_ori=img_cp,
        back=back,
        conv_ima_sigma=conv_ima_sigma,
        source_coord=sources,
        isolated_coord=coord,
        tile_section_psf=tile_section_psf,
        star_dataset=star_dataset,
        fwhm=fwhm,
        gain=gain,
        rdnoise=rdnoise,
        n_images=n_images,
        # center_factor=kwargs.get('center_factor', 1.0),
        # min_conv_snr=kwargs.get('min_conv_snr', 300.0)
    )

    del img_cp, back, conv_ima_sigma, mask_star_dataset, star_dataset, coord, sources

    # del conv_ima_sigma, sources, back, mask_star_dataset, star_dataset, coord
    # mempool.free_all_blocks()
    # At this point, the memory has been freed
    reset_cupy_allocators()

    dic_calib.update(extra_info)

    dfm = pd.DataFrame({'xcentroid': optimal_coords[:, 1],
                        'ycentroid': optimal_coords[:, 0],
                        'flux': optimal_flux,
                        'noise': optimal_noise,
                        'snr': optimal_flux / optimal_noise})

    del optimal_coords, optimal_flux, optimal_noise

    dfm_ast = dfm.loc[dfm.snr > min_snr]
    dfm_ast = dfm_ast.sort_values('snr', ascending=False).dropna().reset_index(drop=True)

    # Astrometrize
    h_wcs = astrometrice2(dfm_ast, scale, target_ra, target_dec, imadata_shape, sip_order=sip_order, **kwargs)
    del dfm_ast
    if h_wcs == {}:
        logger.error('Astrometry failed')
        return dfm, h_wcs, dic_calib

    else:
        # Photometrize
        coocenter, FOV, scale = get_astrometry_params(h_wcs, imadata_shape)
        result, catalog, ref_filter = catalog_results(coocenter, FOV / 2,
                                                        filter, maglimit=zp_maxmag, **kwargs)
        dic_calib['CATALOG'] = catalog
        dic_calib['CATBAND'] = ref_filter

        w = WCS(h_wcs)
        dfm.loc[:, 'RA'], dfm.loc[:, 'DEC'] = w.all_pix2world(dfm.xcentroid.values, dfm.ycentroid.values, 1)
        # dfm.loc[:, 'RA'] = ra
        # dfm.loc[:, 'DEC'] = dec
        photo_dict = get_zeropoint(result, dfm, exptime, center_lims=(xmin, xmax, ymin, ymax),
                                    solar_filter=color_range, dist_thres_px=1.5 * fwhm * scale / 3600)
        dic_calib.update(photo_dict)

        # Calculate limiting magnitude
        try:
            mag = dic_calib['ZP'] - 2.5 * np.log10(dfm.flux.values / exptime)
            snr = dfm.snr.values
            maglim3 = get_maglim(mag, snr, 3)
            zp_maxmag = np.min([maglim3, 23])
            del mag, snr
        except Exception as e:
            maglim3 = 0
            zp_maxmag = 23
            logger.warning('Error calculating limiting magnitude: {}'.format(e))
        dic_calib['MAGLIM'] = maglim3

        #Get all the sources
        logger.info('Getting all sources from catalog until magnitude limit {}'.format(zp_maxmag))
        result, catalog, ref_filter = catalog_results(coocenter, FOV / 2,
                                                        filter, maglimit=zp_maxmag, **kwargs)
        w = WCS(h_wcs)
        dfm.loc[:, 'RA'], dfm.loc[:, 'DEC'] = w.all_pix2world(dfm.xcentroid.values, dfm.ycentroid.values, 1)

        # Check catalog coincidence
        dfm_idx, catalog_idx = crossmatch_sources(dfm[['RA', 'DEC']].values,
                                                    result[['RA', 'DEC']].values,
                                                    thres_px=1.5 * fwhm * scale / 3600)

        # Add astrometric errors to dfm
        dfm.loc[dfm_idx, 'RAERR'] = dfm.loc[dfm_idx, 'RA'].values - result.loc[catalog_idx, 'RA'].values
        dfm.loc[dfm_idx, 'DECERR'] = dfm.loc[dfm_idx, 'DEC'].values - result.loc[catalog_idx, 'DEC'].values

        del dfm_idx, catalog_idx, result

        dic_calib['RAPREC'] = np.round(np.nanmedian(dfm.RAERR) * 3600, 3)
        dic_calib['DECPREC'] = np.round(np.nanmedian(dfm.DECERR) * 3600, 3)
        dic_calib['RADISP'] = np.round(np.nanstd(dfm.RAERR) * 3600, 3)
        dic_calib['DECDISP'] = np.round(np.nanstd(dfm.DECERR) * 3600, 3)

        # Calculate target SNR
        try:
            if target_ra and target_dec:
                target_snr = get_target_snr(dfm, target_ra, target_dec, dist_thres_px=1.5 * fwhm)
            else:
                target_snr = 0
        except Exception as e:
            target_snr = 0
            logger.warning('Error calculating target SNR: {}'.format(e))
        dic_calib['OBJECSNR'] = np.round(target_snr, 2)

    return dfm, h_wcs, dic_calib


### # @hierarchical_debug(logger)
@nvtx.annotate('aperture_photometry', category='phot.photo_gpu')
def aperture_photometry(img, positions, aper_rad, **kwargs):
    """
    Perform aperture photometry.

    .. deprecated::
        This function is deprecated and will be removed in a future version.
        Use :func:`batch_aperture_photometry` instead, which supports streams
        and OOM handling.

    :param img: Image data.
    :type img: cupy.ndarray
    :param positions: Positions of sources.
    :type positions: cupy.ndarray
    :param aper_rad: Aperture radius.
    :type aper_rad: int
    :return: Flux and area of the aperture.
    :rtype: tuple(cupy.ndarray, float)
    """
    warnings.warn(
        "aperture_photometry is deprecated and will be removed in a future version. "
        "Use batch_aperture_photometry instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    kernel, area = get_aper_kernel(aper_rad, size=2 * int(aper_rad) + 1)
    conv_ima = convolve_fft(img, kernel, **kwargs)
    positions = cp.array(cp.round(positions)).astype(cp.int32)
    flux = conv_ima[positions[:, 0], positions[:, 1]]
    del conv_ima, kernel, positions

    return flux, area


### # @hierarchical_debug(logger)
@nvtx.annotate('get_fwhm_mof', category='phot.photo_gpu')
def get_fwhm_mof(model, img, step=50, ns=25, mins=3, **kwargs):
    """
    Get the full width at half maximum using Moffat model.

    :param model: Model to use for FWHM estimation.
    :type model: object
    :param img: Image data.
    :type img: numpy.ndarray
    :param step: Step size for sampling.
    :type step: int
    :param ns: Number of samples.
    :type ns: int
    :param mins: Minimum number of stars.
    :type mins: int
    :return: Mean FWHM, standard deviation of FWHM, mean alpha, and mean beta.
    :rtype: tuple(float, float, float, float)
    """
    ims, cs = sample_im(img, step, ns)
    pred2 = model.predict(ims)
    alpha, beta, nstar, fwhm = pred_mof(pred2)
    fws = fwhm[nstar >= np.mean(nstar)]

    return np.mean(fws), np.std(fws), np.mean(alpha), np.mean(beta)


### # @hierarchical_debug(logger)
@nvtx.annotate('cov_nan', category='phot.photo_gpu')
def cov_nan(img, nc=10, **kwargs):
    """
    Fill NaN values in an image using convolution.

    :param img: Input image.
    :type img: cupy.ndarray
    :param nc: Number of chunks.
    :type nc: int
    :return: Image with NaN values filled.
    :rtype: cupy.ndarray
    """
    delta = np.round(img.shape[0] / nc).astype(int)
    for i in range(nc):
        for j in range(nc):
            ii = img[delta * i:delta * (i + 1), delta * j:delta * (j + 1)]
            ii[cp.isnan(ii)] = cp.nanmean(ii)
            img[delta * i:delta * (i + 1), delta * j:delta * (j + 1)] = ii
    img[cp.isnan(img)] = cp.nanmean(img)
    img[cp.isinf(img)] = cp.nanmean(img)

    return img




### # @hierarchical_debug(logger)
@nvtx.annotate('sample_im', category='phot.photo_gpu')
def sample_im(img, nc=50, ns=100, **kwargs):
    """
    Sample an image.

    :param img: Input image.
    :type img: numpy.ndarray
    :param nc: Number of chunks.
    :type nc: int
    :param ns: Number of samples.
    :type ns: int
    :return: Sampled image chunks and their maximum values.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    delta = np.round((img.shape[0] - 512) / nc).astype(int)
    pi = 512 * 512
    lim = []
    cmax = []
    np.random.seed(42)
    for k in range(ns):
        i = np.random.randint(nc)
        j = np.random.randint(nc)
        ii = img[delta * i:delta * i + 512, delta * j:delta * j + 512]
        ima = np.median(ii)
        ii = np.log(abs(ii) / ima) / 100
        li = ii.shape[0] * ii.shape[1]
        ii = np.hstack([ii.flatten(), [0] * (pi - li)])
        ii[np.isnan(ii)] = 0
        ii[np.isinf(ii)] = 0
        lim = lim + [ii]
        cmax = cmax + [ima]
    iac2 = np.asarray(lim)
    icmax2 = np.asarray(cmax)
    iac2 = iac2.reshape(-1, 512, 512, 1)

    return iac2, icmax2


### # @hierarchical_debug(logger)
@nvtx.annotate('pred_mof', category='phot.photo_gpu')
def pred_mof(pred, **kwargs):
    """
    Predict Moffat parameters.

    :param pred: Predicted values.
    :type pred: numpy.ndarray
    :return: Alpha, beta, number of stars, and FWHM.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    alpha = pred[:, 1]
    beta = pred[:, 0] * 0.4 + 4.565
    nstar = pred[:, 2] * 200
    fwhm = 2 * alpha * np.sqrt(2 ** (1 / beta) - 1)

    return alpha, beta, nstar, fwhm




### # @hierarchical_debug(logger)
@nvtx.annotate('sigma_clip', category='phot.photo_gpu')
def sigma_clip(img, sclip, **kwargs):
    """
    Perform sigma clipping on an image.

    :param img: Input image.
    :type img: cupy.ndarray
    :param sclip: Sigma clipping factor.
    :type sclip: float
    :return: Mean and standard deviation of the clipped image.
    :rtype: tuple(float, float)
    """
    img0 = img.copy()
    for i in range(5):
        imed = cp.nanmean(img0)
        rms = cp.nanstd(img0)
        img0[img0 >= imed + sclip * rms] = cp.nan
        img0[img0 <= imed - sclip * rms] = cp.nan
    del img0

    return imed, rms


### # @hierarchical_debug(logger)
@nvtx.annotate('gen_gauss_filter', category='phot.photo_gpu')
def gen_gauss_filter(fw, **kwargs):
    """
    Generate a Gaussian filter.

    :param fw: Full width at half maximum.
    :type fw: float
    :return: Gaussian filter kernel and kernel size.
    :rtype: tuple(cupy.ndarray, int)
    """
    sigma_r = fw / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_r2 = sigma_r * sigma_r
    lk = np.ceil(sigma_r).astype(np.int16) * 4
    k_dim = 2 * lk + 1, 2 * lk + 1
    indi = cp.indices(k_dim)
    r2 = (lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2
    ker = cp.exp(-r2 / 2 / sigma_r2)
    ksum = cp.sum(ker)
    ksum2 = cp.sum(ker * ker)
    n = k_dim[0] ** 2
    k_app = (ker - ksum / n) / (ksum2 - ksum * ksum / n)

    return k_app, lk


# ### # @hierarchical_debug(logger)
@nvtx.annotate('detect_gpu', category='phot.photo_gpu')
def detect_gpu(img, sky, rms, sdet, mode='g', fw=1, alpha=0, beta=0, minpix
=4, mincut=10, mem=cp.get_default_pinned_memory_pool(), **kwargs):
    """
    Detect sources in an image using GPU.

    :param img: Input image.
    :type img: cupy.ndarray
    :param sky: Sky background.
    :type sky: cupy.ndarray
    :param rms: RMS noise.
    :type rms: cupy.ndarray
    :param sdet: Detection threshold in sigma units.
    :type sdet: float
    :param mode: Detection mode ('g' for Gaussian).
    :type mode: str
    :param fw: Full width at half maximum.
    :type fw: float
    :param alpha: Moffat alpha parameter (not used for Gaussian mode).
    :type alpha: float
    :param beta: Moffat beta parameter (not used for Gaussian mode).
    :type beta: float
    :param minpix: Minimum number of pixels for a valid detection.
    :type minpix: int
    :param mincut: Minimum flux cut for a valid detection.
    :type mincut: float
    :param mem: GPU memory pool.
    :type mem: cupy.cuda.MemoryPool
    :return: DataFrame of detected sources, mask of small detections, and memory usage.
    :rtype: tuple(pandas.DataFrame, cupy.ndarray, int)
    """
    gf, lk = gen_gauss_filter(fw)
    g = convolve(img - sky, gf, origin=(0, 0))
    g1 = (g / rms > sdet).astype(cp.int32)
    del rms
    mem.free_all_blocks()
    label_im, nb_labels = label(g1)
    ids0 = cp.asarray([range(nb_labels + 1)])
    npix = nd_sum(g1, label_im, ids0)
    ids = ids0[(npix > mincut) & (npix > 0)]
    idm = ids0[(npix <= minpix) & (npix > 0)]
    npix = npix[(npix > mincut) & (npix > 0)]
    if minpix == 0:
        mask = False
    else:
        mask = cp.isin(label_im, cp.asarray(idm))
    del g1
    mem.free_all_blocks()
    idx = cp.indices(img.shape, dtype=cp.int16)
    im1 = g * idx
    x = nd_mean(im1[0, :, :], label_im, ids) / nd_mean(g, label_im, ids)
    y = nd_mean(im1[1, :, :], label_im, ids) / nd_mean(g, label_im, ids)
    el = nd_mean(im1[0, :, :] * im1[1, :, :], label_im, ids) / nd_mean(g,
                                                                       label_im, ids)
    el = el - x * y
    coor = x.astype(cp.int16), y.astype(cp.int16)
    mm = cp.get_default_memory_pool().used_bytes()
    flux = g[coor]
    del g
    mem.free_all_blocks()
    res = np.asarray([y.get(), x.get(), flux.get(), npix.get(), el.get()]
                     ).transpose().reshape((-1, 5))

    del x, y, flux, npix, el

    df = pd.DataFrame(res, columns=['xcentroid', 'ycentroid', 'flux',
                                    'npix', 'elip'])

    return df, mask, mm


### # @hierarchical_debug(logger)
@nvtx.annotate('get_peak_image', category='phot.photo_gpu')
def get_peak_image(img, positions, aper_rad, **kwargs):
    """
    Get peak values in an image at specified positions.

    :param img: Input image.
    :type img: cupy.ndarray
    :param positions: Positions to check for peaks.
    :type positions: cupy.ndarray
    :param aper_rad: Aperture radius for peak detection.
    :type aper_rad: int
    :return: Peak values at specified positions.
    :rtype: cupy.ndarray
    """
    lk = 2 * aper_rad
    img_m = maximum_filter(img, size=lk)
    positions = cp.array(cp.round(positions)).astype(cp.int32)
    P = img_m[positions[:, 0], positions[:, 1]]
    del img_m, positions

    return P
 


if __name__ == '__main__':
    from astropy.io import fits

    directory_path = os.path.join(os.path.dirname(__file__), '..', '..',
                                  'tests', 'data')
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.fits'):
                if 'TTT1' in file:
                    image_path = os.path.join(root, file)
                    try:
                        print(f'Processing {image_path}')

                        image_cp = cp.asarray(fits.getdata(image_path))
                        resul = SP_filter(image_cp)
                        end_time = time.time()

                    except Exception as e:
                        print(f'Error processing {image_path}: {e}')
                        print('Traceback:')
                        traceback.print_exc()
