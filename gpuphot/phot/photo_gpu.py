from __future__ import annotations

import gc
import time
import traceback

import cupy as cp
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
from ..utils.gpu import free_gpu_mem, reset_cupy_allocators, maybe_free_arrays
from ..utils.headers import update_header_with_astrometry, update_header_with_photometry

logger = setup_logger(__name__)


# ### # @hierarchical_debug(logger)
# def get_solver():
#     """
#     Get the astrometry solver with index files.
#
#     Returns
#     -------
#     astrometry.Solver
#         Configured astrometry solver instance.
#     """
#     if os.path.exists('/data'):
#         cache = '/data/astrometry_cache'
#     else:
#         cache = '/mnt/data/astrometry_cache'
#     solver = astrometry.Solver(astrometry.series_5200.index_files(
#         cache_directory=cache, scales={0, 1, 2, 3, 4, 5, 6}) + astrometry.
#                                series_4100.index_files(cache_directory=cache, scales={7, 8, 9, 10,
#                                                                                       11}))
#
#     return solver


# def init_gpu():
#     """ """
#     logger.debug('Tensorflow version ' + tf.__version__)
#     gpus = tf.config.list_physical_devices('GPU')
#     logger.debug('GPUs:', gpus)
#     tf.config.set_logical_device_configuration(gpus[0], [tf.config.
#                                                LogicalDeviceConfiguration(memory_limit=1024)])


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
@nvtx.annotate('SP_filter', category='phot.photo_gpu')
def SP_filter(img, filter_size=3, high_threshold_factor=10,
              low_threshold_factor=5, scaling_factor=1.4826, **kwargs):
    """
    Apply a median filter to remove salt-and-pepper noise.

    :param img: Input image.
    :type img: cupy.ndarray
    :param filter_size: Size of the median filter (default is 3).
    :type filter_size: int, optional
    :param high_threshold_factor: Factor to determine the high threshold for noise detection (default is 10).
    :type high_threshold_factor: float, optional
    :param low_threshold_factor: Factor to determine the low threshold for noise detection (default is 5).
    :type low_threshold_factor: float, optional
    :param scaling_factor: Scaling factor for estimating the standard deviation from MAD (default is 1.4826).
    :type scaling_factor: float, optional
    :return: Filtered image.
    :rtype: cupy.ndarray
    """
    med_filter = median_filter(img, size=filter_size)
    dif = img - med_filter
    med = cp.nanmedian(dif)
    ms = scaling_factor * cp.nanmedian(cp.abs(dif - med))
    mask = (dif > med + high_threshold_factor * ms) | (dif < med -
                                                       low_threshold_factor * ms)
    img[mask] = med_filter[mask]
    del med_filter, dif, med, ms, mask

    return img


### # @hierarchical_debug(logger)
@nvtx.annotate('CR_filter', category='phot.photo_gpu')
def CR_filter(img, thres=3, **kwargs):
    """
    Apply a cosmic ray filter to an image.

    :param img: Input image.
    :type img: cupy.ndarray
    :param thres: Threshold for cosmic ray detection (default is 3).
    :type thres: float, optional
    :return: Filtered image.
    :rtype: cupy.ndarray
    """

    img = cp.asarray(img, dtype=cp.float32)
    mask = laplace(img)  # > 100
    mask = cp.abs(mask - cp.mean(mask)) > thres * cp.std(mask)
    mask = binary_dilation(mask, cp.ones((3, 3)))
    img_filled = img.copy()
    img_filled[mask] = cp.nan
    n = cp.sum(cp.isnan(img_filled))
    while n > 0:
        img_filled = fill_nan_fft(img_filled, 3, 0, min_neighbors=5)
        if n == cp.sum(cp.isnan(img_filled)):
            break
        else:
            n = cp.sum(cp.isnan(img_filled))
    ref = img_filled < cp.percentile(img_filled, 95)

    mask = mask & ref
    img[mask] = cp.nan
    del img_filled, mask, ref
    n = cp.sum(cp.isnan(img))
    while n > 0:
        img = fill_nan_fft(img, 3, 0, min_neighbors=5)
        if n == cp.sum(cp.isnan(img)):
            break
        else:
            n = cp.sum(cp.isnan(img))

    free_gpu_mem()
    return img


### # @hierarchical_debug(logger)
@nvtx.annotate('gen_moff_filter2', category='phot.photo_gpu')
def gen_moff_filter2(alpha, beta, **kwargs):
    """
     Generate a Moffat filter with adjusted alpha.

     :param alpha: Alpha parameter for Moffat filter.
     :type alpha: float
     :param beta: Beta parameter for Moffat filter.
     :type beta: float
     :return: Moffat filter kernel and kernel size.
     :rtype: tuple(cupy.ndarray, int)
     """
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

    :param corr: Array of aperture curves (n_stars, n_radii), potentially with NaNs.
                 Expected shape (n_stars_in_cluster, n_radii).
    :type corr: cupy.ndarray
    :return: Aperture correction factors and errors for the cluster.
    :rtype: tuple(cupy.ndarray, cupy.ndarray) Both shape (n_radii,)
    """
    # Ensure input is float for calculations involving NaN/Inf
    if not cp.issubdtype(corr.dtype, cp.floating):
        corr = corr.astype(cp.float64)  # Use float64 for precision

    if corr.size == 0 or corr.shape[0] == 0:
        # Handle empty input case gracefully
        if corr.ndim == 2 and corr.shape[1] > 0:
            n_radii = corr.shape[1]
            # Ensure output dtype matches potential input dtype if float
            dtype_out = corr.dtype if cp.issubdtype(corr.dtype, cp.floating) else cp.float64
            return cp.full(n_radii, cp.nan, dtype=dtype_out), cp.full(n_radii, cp.nan, dtype=dtype_out)
        else:
            # Cannot determine n_radii, return empty
            return cp.array([]), cp.array([])

    # Normalize each curve by its own max
    max_vals = cp.nanmax(corr, axis=1, keepdims=True)  # Shape (n_stars, 1)

    # --- FIXED NORMALIZATION APPROACH ---
    # First broadcast the mask to match output shape
    mask_broadcasted = cp.broadcast_to(max_vals != 0, corr.shape)
    # Then use cp.where() function instead of 'where' parameter
    corr_normalized = cp.where(mask_broadcasted,
                               corr / max_vals,  # Division with implicit broadcasting
                               cp.nan)
    # --- END FIXED NORMALIZATION ---

    # Iterative outlier rejection (using median and std dev)
    corr_fa = cp.nanmedian(corr_normalized, axis=0)  # Shape (n_radii,)
    corr_e = cp.nanstd(corr_normalized, axis=0)  # Shape (n_radii,)

    # Add epsilon to std dev if it's zero
    epsilon = 1e-9
    # Ensure corr_e is float before comparison if input wasn't
    if not cp.issubdtype(corr_e.dtype, cp.floating):
        corr_e = corr_e.astype(cp.float64)
    corr_e = cp.where(corr_e == 0, epsilon, corr_e)

    # Expand dims for broadcasting comparison against (n_stars, n_radii) array
    corr_fa_bc = corr_fa[cp.newaxis, :]  # Shape (1, n_radii)
    corr_e_bc = corr_e[cp.newaxis, :]  # Shape (1, n_radii)

    # Identify outliers (comparison broadcasts correctly)
    cmask = cp.abs(corr_normalized - corr_fa_bc) > corr_e_bc  # Shape (n_stars, n_radii)
    # Replace outliers with NaN using cp.where (function, not argument)
    corr_cleaned = cp.where(cmask, cp.nan, corr_normalized)

    # Final correction factor and error
    corr_fact = cp.nanmean(corr_cleaned, axis=0)  # Shape (n_radii,)
    corr_err = cp.nanstd(corr_cleaned, axis=0)  # Shape (n_radii,)

    # Handle cases where all values for a radius became NaN
    # Ensure outputs are float before isnan check
    if not cp.issubdtype(corr_fact.dtype, cp.floating):
        corr_fact = corr_fact.astype(cp.float64)
    if not cp.issubdtype(corr_err.dtype, cp.floating):
        corr_err = corr_err.astype(cp.float64)

    corr_fact = cp.where(cp.isnan(corr_fact), 0.0, corr_fact)
    corr_err = cp.where(cp.isnan(corr_err), 0.0, corr_err)

    return corr_fact, corr_err


### # @hierarchical_debug(logger)
@nvtx.annotate('calculate_aperture_corrections', category='phot.photo_gpu')
def calculate_aperture_corrections(corr: np.ndarray) -> cp.ndarray:
    """
    Calculate aperture corrections for all stars in an optimized way.

    :param corr: Array of aperture corrections.
    :type corr: numpy.ndarray
    :return: Aperture correction factors and errors.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """

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
    corr_err = np.nanstd(corr_cleaned, axis=0)

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
    _, tile_idx = crossmatch_sources(sources, cluster_centers, thres_px=float(cp.iinfo(cp.int32).max))
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


#
# ### # @hierarchical_debug(logger)
# @nvtx.annotate('find_aperture_corrections', category='phot.photo_gpu')
# def find_aperture_corrections(sources: cp.ndarray, corrections: np.ndarray, correction_errors: np.ndarray,
#                               cluster_centers: np.ndarray,
#                               opt_rad_idx: np.array = None, **kwargs) -> cp.ndarray:
#     """
#     Find the aperture correction for each source.
#
#     :param sources: Array of source coordinates.
#     :type sources: cupy.ndarray
#     :param corrections: Array of aperture corrections.
#     :type corrections: numpy.ndarray
#     :param correction_errors: Array of aperture correction errors.
#     :type correction_errors: numpy.ndarray
#     :param cluster_centers: Array of cluster centers.
#     :type cluster_centers: numpy.ndarray
#     :param opt_rad_idx: Array of optimal radii indices (optional).
#     :type opt_rad_idx: numpy.ndarray or None
#     :return: Array of aperture corrections and errors for each source.
#     :rtype: tuple(cupy.ndarray, cupy.ndarray)
#     """
#     _, tile_idx = crossmatch_sources(sources.get(), cluster_centers, thres_px=int(cp.max(sources)))
#     if opt_rad_idx is None:
#         aperture_corrections = corrections[tile_idx, :]
#         aperture_correction_errors = correction_errors[tile_idx, :]
#     else:
#         aperture_corrections = corrections[tile_idx, opt_rad_idx]
#         aperture_correction_errors = correction_errors[tile_idx, opt_rad_idx]
#     return aperture_corrections, aperture_correction_errors


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
    avg_group_size = int(
        max(coords.shape[0], coords.shape[0] / (np.prod(image_shape) / min(max(image_shape), block_size) ** 2)))
    min_group_size_val = max(1, min(avg_group_size // 2, 5))

    # This function NOW can return cp.ndarray or np.ndarray
    labels = group_star_dataset(coords, avg_group_size=avg_group_size,
                                min_group_size=min_group_size_val)
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

        # Map numpy labels to output array index
        # Doing the mapping on CPU is simpler
        unique_labels_np_for_map = unique_labels_gpu.get()
        label_to_idx_map = {label_val: idx for idx, label_val in enumerate(unique_labels_np_for_map)}

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

                # *** CALL THE GPU VERSION OF CORRECTIONS ***
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
    # Always return GPU arrays
    return aperture_corrections_final_gpu, aperture_correction_errors_final_gpu, cluster_centers_gpu


### # @hierarchical_debug(logger)
@nvtx.annotate('create_aperture_corrections_map', category='phot.photo_gpu')
def create_aperture_corrections_map(image_shape: tuple, block_size: int, unit_star_dataset: cp.ndarray,
                                    coords: cp.ndarray, radii: np.ndarray, **kwargs):
    """
    Calculate aperture corrections for all stars in an optimized way.

    :param image_shape: Tuple (height, width) of the image.
    :type image_shape: tuple
    :param block_size: Size of the tiles.
    :type block_size: int
    :param unit_star_dataset: Dataset of unit stars.
    :type unit_star_dataset: cupy.ndarray
    :param coords: Coordinates of stars.
    :type coords: cupy.ndarray
    :param radii: Radii for aperture photometry.
    :type radii: numpy.ndarray
    :return: Aperture corrections, errors, and cluster centers.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    # calculate aperture photometry curves
    positions = cp.array([[unit_star_dataset.shape[1] // 2, unit_star_dataset.shape[2] // 2]])
    aperture_curves, _, _ = batch_aperture_photometry(unit_star_dataset, None, positions, radii)
    aperture_curves = aperture_curves[:, :, 0].get()

    # cluster stars
    avg_group_size = int(
        max(len(coords), len(coords) / (np.prod(image_shape) / min(max(image_shape), block_size) ** 2)))
    labels = group_star_dataset(coords, avg_group_size=avg_group_size, min_group_size=min(avg_group_size, 5))

    unique_labels = np.unique(labels)
    cluster_centers = []
    aperture_corrections = []
    aperture_correction_errors = []
    for label in unique_labels:
        cluster_points = coords[labels == label]
        cluster_aperture_curves = aperture_curves[labels == label]

        aper_corr, aperr_corr_err = calculate_aperture_corrections(cluster_aperture_curves)
        aperture_corrections.append(aper_corr)
        aperture_correction_errors.append(aperr_corr_err)
        cluster_centers.append(np.mean(cluster_points, axis=0))

    # calculate aperture corrections
    aperture_corrections = np.array(aperture_corrections)
    aperture_correction_errors = np.array(aperture_correction_errors)
    cluster_centers = np.array(cluster_centers)

    return aperture_corrections, aperture_correction_errors, cluster_centers


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

        # default parameters
        default_params = DefaultConfig.DEFAULT_PROCESSING_PARAMS

        # Update default parameters with any provided in kwargs
        params = {**default_params, **kwargs}

        # Call calibrate_image with updated parameters
        dfm, h_wcs, dic_calib = calibrate_image(
            imdata, filter,
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


@nvtx.annotate('perform_opt_photometry_optimized_gpu_crossmatch', category='phot.photo_gpu')
def perform_opt_photometry_optimized_gpu_crossmatch(img: cp.ndarray, back: cp.ndarray, conv_ima_sigma: cp.ndarray,
                                                    source_coord: cp.ndarray, isolated_coord: cp.ndarray,
                                                    tile_section_psf: int, star_dataset: cp.ndarray, fwhm: float,
                                                    gain: float, rdnoise: float, n_images: int = 1,
                                                    center_factor: float = 1.0,
                                                    min_conv_snr: float = 300.0) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Optimized aperture photometry process (GPU-focused, Option B with cuML crossmatch).
    """
    overall_range = nvtx.start_range('perform_opt_photometry_optimized_gpu_crossmatch', category='phot.photo_gpu',
                                     color='cyan')
    mempool = cp.get_default_memory_pool()

    # BLOQUE 1: Selección de estrellas centrales (GPU mask, CPU limits)
    block1_range = nvtx.start_range('center_stars_selection', category='phot.photo_gpu', color='yellow')
    # ... (Sin cambios respecto a la versión anterior) ...
    center_factor = min(center_factor, 1.0)
    h, w = img.shape[-2:]
    xmin = int(w * 0.5 * (1 - center_factor))
    xmax = int(w * 0.5 * (1 + center_factor))
    ymin = int(h * 0.5 * (1 - center_factor))
    ymax = int(h * 0.5 * (1 + center_factor))
    center_mask = (isolated_coord[:, 0] > ymin) & (isolated_coord[:, 0] < ymax) & \
                  (isolated_coord[:, 1] > xmin) & (isolated_coord[:, 1] < xmax)
    nvtx.end_range(block1_range)

    # BLOQUE 2: Procesamiento SNR y validación de fuentes (GPU crossmatch)
    block2_range = nvtx.start_range('snr_source_validation_gpu', category='phot.photo_gpu', color='orange')
    # ... (Cálculo de conv_snr y filtrado inicial igual que antes) ...
    source_row_idx = cp.clip(cp.rint(source_coord[:, 0]), 0, h - 1).astype(cp.int32)
    source_col_idx = cp.clip(cp.rint(source_coord[:, 1]), 0, w - 1).astype(cp.int32)
    conv_snr = conv_ima_sigma[source_row_idx, source_col_idx]
    pos_conv_snr_mask = conv_snr > 0
    source_coord_filt = source_coord[pos_conv_snr_mask]
    conv_snr_filt = conv_snr[pos_conv_snr_mask]
    isolated_coord_center = isolated_coord[center_mask]

    # --- NO TRANSFER NEEDED ---
    # Call GPU crossmatch directly with CuPy arrays
    cm_range = nvtx.start_range('crossmatch_sources_gpu_call', category='utils.catalog_gpu')
    try:
        # Match isolated_coord_center TO source_coord_filt
        _, source_coords_matched_idx = crossmatch_sources(
            isolated_coord_center, source_coord_filt, thres_px=3
        )
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

    # ... (Resto del Bloque 2: validación de isolated stars, etc. sin cambios) ...
    isolated_row_idx = cp.clip(cp.rint(isolated_coord[:, 0]), 0, h - 1).astype(cp.int32)
    isolated_col_idx = cp.clip(cp.rint(isolated_coord[:, 1]), 0, w - 1).astype(cp.int32)
    conv_snr_isol = conv_ima_sigma[isolated_row_idx, isolated_col_idx]
    conv_snr_mask_isol = conv_snr_isol > min_conv_snr
    final_isolated_mask = center_mask & conv_snr_mask_isol

    num_good_isolated = cp.sum(final_isolated_mask)
    if num_good_isolated < 10:
        final_isolated_mask = conv_snr_isol > min_conv_snr * 0.5

    num_good_isolated = cp.sum(final_isolated_mask)
    if num_good_isolated < 3:
        nvtx.end_range(block2_range)
        nvtx.end_range(overall_range)
        raise InsufficientStarsError(num_stars=num_good_isolated.item())
    isolated_coord_for_map = isolated_coord[final_isolated_mask]
    star_dataset_for_map = star_dataset[final_isolated_mask]
    nvtx.end_range(block2_range)

    # BLOQUE 3: Configuración radios (Sin cambios)
    block3_range = nvtx.start_range('aperture_radii_setup', category='phot.photo_gpu', color='yellow')
    # ... (igual que antes) ...
    max_radii = float(np.ceil(7 * fwhm))
    min_radii = float(np.ceil(0.75 * fwhm))
    radii = cp.arange(int(min_radii), int(max_radii) + 1, 1, dtype=cp.float64)
    if radii.size == 0:
        nvtx.end_range(block3_range)
        nvtx.end_range(overall_range)
        raise DataValidationError("Input data is invalid or insufficient")
    nvtx.end_range(block3_range)

    # BLOQUE 4: Mapa de correcciones (Llama a helpers que usan GPU crossmatch internamente)
    block4_range = nvtx.start_range('aperture_corrections_mapping_gpu', category='phot.photo_gpu', color='purple')
    # create_aperture_corrections_map_gpu still has internal CPU steps for grouping
    corrections, correction_errors, cluster_centers = create_aperture_corrections_map_gpu(
        (h, w), tile_section_psf, star_dataset_for_map,
        isolated_coord_for_map, radii
    )
    # find_aperture_corrections_gpu now uses GPU crossmatch, no external transfers needed
    aperture_corrections, aperture_correction_errors = find_aperture_corrections_gpu(
        source_coord_filt, corrections, correction_errors, cluster_centers
    )
    nvtx.end_range(block4_range)

    # BLOQUE 5: Fotometría por lotes (Sin cambios)
    block5_range = nvtx.start_range('batch_aperture_photometry', category='phot.photo_gpu', color='green')
    # ... (igual que antes, llama a batch_aperture_photometry adaptada) ...
    source_coord_int = cp.rint(source_coord_filt).astype(cp.int32)
    source_coord_int[:, 0] = cp.clip(source_coord_int[:, 0], 0, h - 1)
    source_coord_int[:, 1] = cp.clip(source_coord_int[:, 1], 0, w - 1)
    source_flux, back_flux, area = batch_aperture_photometry(img, back, source_coord_int, radii)
    if img.ndim == 3:  # Handle averaging if needed
        source_flux = cp.mean(source_flux, axis=0)
        if back_flux is not None: back_flux = cp.mean(back_flux, axis=0)
    nvtx.end_range(block5_range)

    # BLOQUE 7: Cálculo señal/ruido (Sin cambios, ya era full GPU)
    block7_range = nvtx.start_range('photometric_parameters_calc_gpu', category='phot.photo_gpu', color='lime')
    # Uses source_coords_matched_idx (now CuPy from GPU crossmatch)
    # ... (lógica igual que antes) ...
    aperture_corrections_matched = aperture_corrections[source_coords_matched_idx]
    aperture_errors_matched = aperture_correction_errors[source_coords_matched_idx]
    source_flux_matched = source_flux[:, source_coords_matched_idx]
    back_flux_matched = back_flux[:, source_coords_matched_idx] if back_flux is not None else None
    aperture_corrections_matched_t = aperture_corrections_matched.T
    aperture_errors_matched_t = aperture_errors_matched.T
    epsilon = 1e-9
    center_isolated_signal = cp.divide(source_flux_matched, aperture_corrections_matched_t + epsilon)
    area_col = area.reshape(-1, 1)
    center_isolated_back_noise_sq = cp.zeros_like(source_flux_matched)
    if back_flux_matched is not None: center_isolated_back_noise_sq = cp.abs(back_flux_matched) * gain
    center_isolated_read_noise_sq = area_col * rdnoise ** 2
    corr_sq = aperture_corrections_matched_t ** 2
    center_isolated_source_noise_sq = cp.divide(source_flux_matched * gain, corr_sq + epsilon)
    center_isolated_corr_noise_sq = cp.power(
        cp.divide(source_flux_matched * gain * aperture_errors_matched_t, corr_sq + epsilon), 2)
    total_noise_sq_sum = (
            center_isolated_source_noise_sq / n_images + center_isolated_back_noise_sq / n_images + center_isolated_read_noise_sq / n_images + center_isolated_corr_noise_sq)
    total_noise_sq_sum = cp.maximum(total_noise_sq_sum, 0)
    center_isolated_total_noise = cp.sqrt(total_noise_sq_sum) / gain
    center_isolated_snr = cp.divide(center_isolated_signal, center_isolated_total_noise + epsilon)
    center_conv_snr = conv_snr_filt[source_coords_matched_idx]  # Already CuPy
    nvtx.end_range(block7_range)

    # BLOQUE 8: Optimización radio (Transferencia CPU solo para polyfit)
    block8_range = nvtx.start_range('optimal_radii_calculation', category='phot.photo_gpu', color='orange')
    # ... (igual que antes, usa cp.nanargmax, .get() para polyfit) ...
    opt_radii_idx = cp.nanargmax(center_isolated_snr, axis=0)
    opt_radii_gpu = radii[opt_radii_idx]
    # --- Transfer Point (Required by np.polyfit) ---
    tx3_range = nvtx.start_range('transfer_for_polyfit', category='transfer', color='red')
    center_conv_snr_np = center_conv_snr.get()  # center_conv_snr is CuPy
    opt_radii_np = opt_radii_gpu.get()
    nvtx.end_range(tx3_range)
    # --- End Transfer ---
    valid_fit_mask_np = (center_conv_snr_np > 0) & (opt_radii_np > 0) & np.isfinite(center_conv_snr_np) & np.isfinite(
        opt_radii_np)
    if np.sum(valid_fit_mask_np) < 2:
        nvtx.end_range(block8_range)
        nvtx.end_range(overall_range)
        raise DataValidationError("Not enough valid points for polyfit")
    center_conv_snr_fit = center_conv_snr_np[valid_fit_mask_np]
    opt_radii_fit = opt_radii_np[valid_fit_mask_np]
    polyfit_range = nvtx.start_range('polyfit_cpu', category='cpu_ops', color='blue')
    try:
        log10_conv_snr_fit = np.log10(center_conv_snr_fit)
        log10_opt_radii_fit = np.log10(opt_radii_fit)
        pov = np.polyfit(log10_conv_snr_fit, log10_opt_radii_fit, 1, cov=False)
    except Exception as e:
        nvtx.end_range(polyfit_range)
        nvtx.end_range(block8_range)
        nvtx.end_range(overall_range)
        raise DataValidationError(f"Polyfit failed: {e}")
    nvtx.end_range(polyfit_range)
    nvtx.end_range(block8_range)

    # BLOQUE 9: Cálculo flujo/ruido óptimos (Sin cambios, ya era full GPU)
    block9_range = nvtx.start_range('optimal_flux_calculation', category='phot.metadata', color='lime')
    # ... (lógica igual que antes, opera en GPU) ...
    valid_conv_snr_mask = (conv_snr_filt > 0) & cp.isfinite(conv_snr_filt)
    source_opt_rad = cp.full_like(conv_snr_filt, cp.nan)
    source_opt_rad_idx = cp.full_like(source_opt_rad, -1, dtype=cp.int32)
    if cp.any(valid_conv_snr_mask):
        log10_conv_snr_valid = cp.log10(conv_snr_filt[valid_conv_snr_mask])
        log10_opt_rad_valid = pov[0] * log10_conv_snr_valid + pov[1]
        opt_rad_intermediate_valid = cp.power(10, log10_opt_rad_valid)
        max_radii_val = radii[-1].item()
        min_radii_val = radii[0].item()
        source_opt_rad_valid = cp.rint(cp.clip(opt_rad_intermediate_valid, min_radii_val, max_radii_val)).astype(
            radii.dtype)
        source_opt_rad[valid_conv_snr_mask] = source_opt_rad_valid
    valid_opt_rad_mask = ~cp.isnan(source_opt_rad)
    if cp.any(valid_opt_rad_mask):
        source_opt_rad_for_search = source_opt_rad[valid_opt_rad_mask]
        indices_valid = cp.searchsorted(radii, source_opt_rad_for_search, side='left')
        indices_valid = cp.clip(indices_valid, 0, len(radii) - 1)
        source_opt_rad_idx[valid_opt_rad_mask] = indices_valid
    final_valid_mask = (source_opt_rad_idx != -1)
    opt_aperture_corrections = cp.full_like(source_opt_rad, cp.nan)
    opt_aperture_correction_errors = cp.full_like(source_opt_rad, cp.nan)
    opt_flux = cp.full_like(source_opt_rad, cp.nan)
    if cp.any(final_valid_mask):
        indices_for_final = source_opt_rad_idx[final_valid_mask]
        source_indices_final = cp.where(final_valid_mask)[0]
        opt_aperture_corrections[final_valid_mask] = aperture_corrections[source_indices_final, indices_for_final]
        opt_aperture_correction_errors[final_valid_mask] = aperture_correction_errors[
            source_indices_final, indices_for_final]
        opt_flux[final_valid_mask] = source_flux[indices_for_final, source_indices_final]
    positive_flux_mask = final_valid_mask & (opt_flux > 0)
    if cp.sum(positive_flux_mask) == 0:
        nvtx.end_range(block9_range)
        nvtx.end_range(overall_range)
        raise DataValidationError("Input data is invalid or insufficient")
    opt_flux_final = opt_flux[positive_flux_mask]
    opt_aper_corr_final = opt_aperture_corrections[positive_flux_mask]
    opt_corr_err_final = opt_aperture_correction_errors[positive_flux_mask]
    source_opt_rad_final = source_opt_rad[positive_flux_mask]
    source_opt_rad_idx_final = source_opt_rad_idx[positive_flux_mask]
    final_source_indices = cp.where(positive_flux_mask)[0]
    opt_signal_final = cp.divide(opt_flux_final, opt_aper_corr_final + epsilon)
    opt_back_flux_final = cp.zeros_like(opt_flux_final)
    if back_flux is not None: opt_back_flux_final = back_flux[source_opt_rad_idx_final, final_source_indices]
    opt_area_final = area[source_opt_rad_idx_final]
    opt_back_noise_sq = cp.abs(opt_back_flux_final) * gain
    opt_read_noise_sq = opt_area_final * rdnoise ** 2
    opt_corr_sq = opt_aper_corr_final ** 2
    opt_source_noise_sq = cp.divide(opt_flux_final * gain, opt_corr_sq + epsilon)
    opt_corr_noise_sq = cp.power(cp.divide(opt_flux_final * gain * opt_corr_err_final, opt_corr_sq + epsilon), 2)
    opt_total_noise_sq_sum = (
            opt_source_noise_sq / n_images + opt_back_noise_sq / n_images + opt_read_noise_sq / n_images + opt_corr_noise_sq)
    opt_total_noise_sq_sum = cp.maximum(opt_total_noise_sq_sum, 0)
    opt_total_noise_final = cp.sqrt(opt_total_noise_sq_sum) / gain
    opt_coords_final = source_coord_filt[final_source_indices]
    nvtx.end_range(block9_range)

    # BLOQUE 10: Info extra (Transferencia escalares)
    block10_range = nvtx.start_range('extra_info_generation', category='phot.metadata', color='yellow')
    # ... (igual que antes, usa cp.argmin, .item()) ...
    extra_info = {}
    ref_snr_values = [10, 100, 250, 1000]
    opt_final_snr = cp.divide(opt_signal_final, opt_total_noise_final + epsilon)
    if opt_final_snr.size > 0:
        for snr_ref in ref_snr_values:
            diff_snr = cp.abs(opt_final_snr - snr_ref)
            ref_idx_cp = cp.argmin(diff_snr)
            tx4_range = nvtx.start_range(f'transfer_extra_info_snr{snr_ref}', category='transfer_scalar', color='pink')
            ref_idx_np = ref_idx_cp.item()
            try:
                rad_val = source_opt_rad_final[ref_idx_np].item()
                corr_val = opt_aper_corr_final[ref_idx_np].item()
                extra_info[f'RAD{snr_ref}'] = int(rad_val)
                extra_info[f'CORR{snr_ref}'] = round(corr_val, 3)
            except IndexError:
                extra_info[f'RAD{snr_ref}'] = -1
                extra_info[f'CORR{snr_ref}'] = -1.0
            nvtx.end_range(tx4_range)
    else:
        for snr_ref in ref_snr_values:
            extra_info[f'RAD{snr_ref}'] = -1
            extra_info[f'CORR{snr_ref}'] = -1.0
    nvtx.end_range(block10_range)

    # Transferencia final GPU -> CPU para return
    final_transfer_range = nvtx.start_range('final_gpu_to_cpu_transfer', category='transfer', color='red')
    opt_signal_np = opt_signal_final.get()
    opt_total_noise_np = opt_total_noise_final.get()
    opt_coords_np = opt_coords_final.get()
    nvtx.end_range(final_transfer_range)

    # Cleanup opcional
    # ... del ...
    # mempool.free_all_blocks()
    # gc.collect()

    nvtx.end_range(overall_range)
    return opt_signal_np, opt_total_noise_np, opt_coords_np, extra_info


#
# @nvtx.annotate('perform_opt_photometry_optimized', category='phot.photo_gpu')
# def perform_opt_photometry_optimized(img: cp.ndarray, back: cp.ndarray, conv_ima_sigma: cp.ndarray,
#                                      source_coord: cp.ndarray, isolated_coord: cp.ndarray,
#                                      tile_section_psf: int, star_dataset: cp.ndarray, fwhm: float,
#                                      gain: float, rdnoise: float, n_images: int = 1, center_factor: float = 1.0,
#                                      min_conv_snr: float = 300.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
#     """
#     Optimized aperture photometry process (GPU-focused, Option A for crossmatch).
#     Minimizes GPU<->CPU transfers by adapting helper functions where possible.
#     """
#     overall_range = nvtx.start_range('perform_opt_photometry_optimized', category='phot.photo_gpu', color='cyan')
#     mempool = cp.get_default_memory_pool()
#
#     # BLOQUE 1: Selección de estrellas centrales (No changes needed, CPU ops are minor)
#     block1_range = nvtx.start_range('center_stars_selection', category='phot.photo_gpu', color='yellow')
#     center_factor = min(center_factor, 1.0)
#     h, w = img.shape[-2:]  # Handle both 2D and 3D image input shapes
#     xmin = int(w * 0.5 * (1 - center_factor))
#     xmax = int(w * 0.5 * (1 + center_factor))
#     ymin = int(h * 0.5 * (1 - center_factor))
#     ymax = int(h * 0.5 * (1 + center_factor))
#
#     # Perform mask calculation on GPU
#     center_mask = (isolated_coord[:, 0] > ymin) & (isolated_coord[:, 0] < ymax) & \
#                   (isolated_coord[:, 1] > xmin) & (isolated_coord[:, 1] < xmax)
#     nvtx.end_range(block1_range)
#
#     # BLOQUE 2: Procesamiento SNR y validación de fuentes (Includes transfers for crossmatch)
#     block2_range = nvtx.start_range('snr_source_validation', category='phot.photo_gpu', color='orange')
#     # Use cp.rint and clip for robust indexing
#     source_row_idx = cp.clip(cp.rint(source_coord[:, 0]), 0, h - 1).astype(cp.int32)
#     source_col_idx = cp.clip(cp.rint(source_coord[:, 1]), 0, w - 1).astype(cp.int32)
#     conv_snr = conv_ima_sigma[source_row_idx, source_col_idx]
#
#     pos_conv_snr_mask = conv_snr > 0
#     source_coord_filt = source_coord[pos_conv_snr_mask]  # Filtered source coords on GPU
#     conv_snr_filt = conv_snr[pos_conv_snr_mask]  # Filtered SNR on GPU
#
#     # Get isolated stars in center (GPU)
#     isolated_coord_center = isolated_coord[center_mask]
#
#     # --- Transfer Point 1 (Required by crossmatch_sources - Option A) ---
#     tx1_range = nvtx.start_range('transfer_for_crossmatch_isol', category='transfer', color='red')
#     isolated_coord_center_np = isolated_coord_center.get()
#     source_coord_filt_np = source_coord_filt.get()
#     nvtx.end_range(tx1_range)
#
#     # Call CPU crossmatch
#     cm_range = nvtx.start_range('crossmatch_sources_cpu', category='cpu_ops', color='blue')
#     # Assuming crossmatch returns (indices_in_source, indices_in_ref)
#     _, source_coords_matched_idx_np = crossmatch_sources(
#         isolated_coord_center_np, source_coord_filt_np, thres_px=3  # Match isolated TO filtered sources
#     )
#     # source_coords_matched_idx_np now holds indices into source_coord_filt_np
#     nvtx.end_range(cm_range)
#
#     # --- Transfer Point 2 (Transfer matched indices back to GPU) ---
#     tx2_range = nvtx.start_range('transfer_matched_idx_to_gpu', category='transfer', color='red')
#     source_coords_matched_idx = cp.asarray(source_coords_matched_idx_np)
#     nvtx.end_range(tx2_range)
#     # --- End Transfers for Crossmatch ---
#
#     # Check SNR for isolated stars (used for correction map generation)
#     isolated_row_idx = cp.clip(cp.rint(isolated_coord[:, 0]), 0, h - 1).astype(cp.int32)
#     isolated_col_idx = cp.clip(cp.rint(isolated_coord[:, 1]), 0, w - 1).astype(cp.int32)
#     conv_snr_isol = conv_ima_sigma[isolated_row_idx, isolated_col_idx]
#
#     # Mask for isolated stars used in correction map (center AND snr threshold)
#     conv_snr_mask_isol = conv_snr_isol > min_conv_snr
#     final_isolated_mask = center_mask & conv_snr_mask_isol  # Combined mask on GPU
#
#     num_good_isolated = cp.sum(final_isolated_mask)
#     if num_good_isolated < 10:
#         final_isolated_mask = conv_snr_isol > min_conv_snr * 0.5
#
#     num_good_isolated = cp.sum(final_isolated_mask)
#     if num_good_isolated < 3:
#         nvtx.end_range(block2_range)
#         nvtx.end_range(overall_range)
#         raise InsufficientStarsError(num_good_isolated.item())
#
#     # Filter coords and dataset for correction map (GPU)
#     isolated_coord_for_map = isolated_coord[final_isolated_mask]
#     star_dataset_for_map = star_dataset[final_isolated_mask]  # Assuming star_dataset corresponds to isolated_coord
#
#     nvtx.end_range(block2_range)
#
#     # BLOQUE 3: Configuración de radios de apertura (GPU)
#     block3_range = nvtx.start_range('aperture_radii_setup', category='phot.photo_gpu', color='yellow')
#     max_radii = float(np.ceil(7 * fwhm))  # Use float for potential non-int radii in get_aper_kernel
#     min_radii = float(np.ceil(0.75 * fwhm))
#     # Use cp.linspace or cp.arange depending if step=1 is guaranteed/desired
#     # Let's assume step=1 integer radii based on original np.arange(int, int, 1)
#     radii = cp.arange(int(min_radii), int(max_radii) + 1, 1, dtype=cp.float64)  # Use float for get_aper_kernel
#     if radii.size == 0:
#         nvtx.end_range(block3_range)
#         nvtx.end_range(overall_range)
#         raise DataValidationError(
#             f"Radii array is empty (min_radii={min_radii}, max_radii={max_radii}). Check FWHM value ({fwhm}).")
#     nvtx.end_range(block3_range)
#
#     # BLOQUE 4: Creación de mapa de correcciones (GPU/CPU mix, returns GPU)
#     block4_range = nvtx.start_range('aperture_corrections_mapping', category='phot.photo_gpu', color='purple')
#     # Calls create_aperture_corrections_map_gpu (handles internal transfers)
#     corrections, correction_errors, cluster_centers = create_aperture_corrections_map_gpu(
#         (h, w), tile_section_psf, star_dataset_for_map,
#         isolated_coord_for_map, radii
#     )
#     # corrections, errors, centers are now CuPy arrays
#
#     # Calls find_aperture_corrections_gpu (handles internal transfers for crossmatch)
#     # Pass filtered source coords; results are CuPy arrays for all sources
#     aperture_corrections, aperture_correction_errors = find_aperture_corrections_gpu(
#         source_coord_filt, corrections, correction_errors, cluster_centers
#     )
#     # aperture_corrections/_errors shape: (n_sources_filt, n_radii)
#     nvtx.end_range(block4_range)
#
#     # BLOQUE 5: Fotometría por lotes (GPU)
#     block5_range = nvtx.start_range('batch_aperture_photometry', category='phot.photo_gpu', color='green')
#     # Ensure coords passed are integers
#     source_coord_int = cp.rint(source_coord_filt).astype(cp.int32)
#     # Clip again just to be absolutely safe before indexing image
#     source_coord_int[:, 0] = cp.clip(source_coord_int[:, 0], 0, h - 1)
#     source_coord_int[:, 1] = cp.clip(source_coord_int[:, 1], 0, w - 1)
#
#     # Call adapted batch_aperture_photometry with CuPy radii
#     source_flux, back_flux, area = batch_aperture_photometry(
#         img, back, source_coord_int, radii
#     )
#     # source_flux/back_flux shape: (n_radii, n_sources_filt) [Assuming 2D image input]
#     # OR (n_images, n_radii, n_sources_filt) [Assuming 3D image input]
#     # area shape: (n_radii,)
#     # Let's assume 2D image input based on original Blk7 logic, adjust if needed
#     if img.ndim == 3:
#         # If input is 3D stack, average flux/noise over images? Or handle stack?
#         # Original code divides noise by sqrt(n_images), suggesting averaging/stacking.
#         # Let's average the flux here for simplicity, assuming background is similar.
#         # WARNING: This assumes simple averaging is appropriate.
#         logger.warning("Warning: Input image is 3D, averaging fluxes over the first dimension.")
#         source_flux = cp.mean(source_flux, axis=0)
#         if back_flux is not None:
#             back_flux = cp.mean(back_flux, axis=0)
#         # If n_images > 1, noise calculation needs adjustment later? Original code used n_images param.
#
#     nvtx.end_range(block5_range)
#
#     # BLOQUE 7: Cálculo de señal y ruido (All GPU)
#     block7_range = nvtx.start_range('photometric_parameters_calc_gpu', category='phot.photo_gpu', color='lime')
#
#     # Index the full GPU arrays using the matched indices (GPU array)
#     # source_coords_matched_idx refers to indices within source_coord_filt space
#     aperture_corrections_matched = aperture_corrections[source_coords_matched_idx]  # Shape (n_matched, n_radii)
#     aperture_errors_matched = aperture_correction_errors[source_coords_matched_idx]  # Shape (n_matched, n_radii)
#     source_flux_matched = source_flux[:, source_coords_matched_idx]  # Shape (n_radii, n_matched)
#     back_flux_matched = back_flux[:,
#                         source_coords_matched_idx] if back_flux is not None else None  # Shape (n_radii, n_matched)
#
#     # Transpose corrections/errors to match flux shape (n_radii, n_matched)
#     aperture_corrections_matched_t = aperture_corrections_matched.T
#     aperture_errors_matched_t = aperture_errors_matched.T
#
#     # --- Signal Calculation (GPU) ---
#     signal_calc_range = nvtx.start_range('signal_calc', category='phot.calc')
#     # Avoid division by zero
#     epsilon = 1e-9  # Small number to avoid division by zero
#     center_isolated_signal = cp.divide(source_flux_matched, aperture_corrections_matched_t + epsilon)
#     # Handle cases where correction was zero explicitly if needed
#     # center_isolated_signal = cp.where(aperture_corrections_matched_t != 0, source_flux_matched / aperture_corrections_matched_t, cp.nan)
#     nvtx.end_range(signal_calc_range)
#
#     # --- Noise Calculation (GPU) ---
#     noise_calc_range = nvtx.start_range('noise_calc', category='phot.calc')
#     # Ensure area has shape (n_radii, 1) for broadcasting with (n_radii, n_matched)
#     area_col = area.reshape(-1, 1)
#
#     # Background noise (Shot noise from background)
#     center_isolated_back_noise_sq = cp.zeros_like(source_flux_matched)
#     if back_flux_matched is not None:
#         # Use abs() in case background subtraction yielded negative values locally
#         center_isolated_back_noise_sq = cp.abs(back_flux_matched) * gain
#
#     # Read noise squared (constant per pixel, scaled by area)
#     center_isolated_read_noise_sq = area_col * rdnoise ** 2  # Broadcasts area to match (n_radii, n_matched)
#
#     # Source noise (Shot noise from source signal)
#     # Account for correction factor in variance propagation: Var(S/c) ~ Var(S)/c^2 = (S*gain)/c^2
#     corr_sq = aperture_corrections_matched_t ** 2
#     center_isolated_source_noise_sq = cp.divide(source_flux_matched * gain, corr_sq + epsilon)
#
#     # Correction uncertainty noise
#     # Var(S/c) due to c error: (S/c^2)^2 * Var(c) = (S/c^2)^2 * (err_c * c)^2 ? No, usually err_c is std dev.
#     # Var(f(c)) ~ (df/dc)^2 Var(c) => Var(S/c) ~ (-S/c^2)^2 Var(c) = (S/c^2)^2 * err_c^2
#     # Check formula: Original was (flux * gain * err / corr^2)**2 -- Let's match that
#     center_isolated_corr_noise_sq = cp.power(
#         cp.divide(source_flux_matched * gain * aperture_errors_matched_t, corr_sq + epsilon), 2
#     )
#
#     # Total noise calculation (Combine variances)
#     # Original code divides sum by n_images for source, back, read noise term.
#     # Assuming this means std dev scales like 1/sqrt(n_images) for these terms.
#     # Correction noise usually doesn't scale with n_images unless derived from stack.
#     total_noise_sq_sum = (center_isolated_source_noise_sq / n_images +
#                           center_isolated_back_noise_sq / n_images +
#                           center_isolated_read_noise_sq / n_images +
#                           center_isolated_corr_noise_sq)  # Correction noise not scaled by n_images
#
#     # Ensure non-negative variance before sqrt
#     total_noise_sq_sum = cp.maximum(total_noise_sq_sum, 0)
#
#     # Convert total variance to noise std dev in flux units (divide by gain^2 then sqrt)
#     center_isolated_total_noise = cp.sqrt(total_noise_sq_sum) / gain
#     nvtx.end_range(noise_calc_range)
#
#     # --- SNR Calculation (GPU) ---
#     snr_calc_range = nvtx.start_range('snr_calc', category='phot.calc')
#     center_isolated_snr = cp.divide(center_isolated_signal, center_isolated_total_noise + epsilon)
#     nvtx.end_range(snr_calc_range)
#
#     # Get corresponding conv_snr for the matched isolated stars (GPU)
#     center_conv_snr = conv_snr_filt[source_coords_matched_idx]
#
#     nvtx.end_range(block7_range)
#
#     # BLOQUE 8: Optimización de radio de apertura (GPU + CPU transfer for polyfit)
#     block8_range = nvtx.start_range('optimal_radii_calculation', category='phot.photo_gpu', color='orange')
#
#     # Find optimal radius index on GPU (handle NaNs that might arise from noise=0 or signal=0)
#     opt_radii_idx = cp.nanargmax(center_isolated_snr, axis=0)
#     opt_radii_gpu = radii[opt_radii_idx]  # Optimal radii on GPU
#
#     # --- Transfer Point 3 (Necessary for np.polyfit) ---
#     tx3_range = nvtx.start_range('transfer_for_polyfit', category='transfer', color='red')
#     center_conv_snr_np = center_conv_snr.get()
#     opt_radii_np = opt_radii_gpu.get()
#     nvtx.end_range(tx3_range)
#     # --- End Transfer ---
#
#     # Filter out potential invalid values (e.g., non-positive) before log10 on CPU
#     valid_fit_mask_np = (center_conv_snr_np > 0) & (opt_radii_np > 0) & \
#                         np.isfinite(center_conv_snr_np) & np.isfinite(opt_radii_np)
#
#     if np.sum(valid_fit_mask_np) < 2:  # Need at least 2 points for polyfit(1)
#         nvtx.end_range(block8_range)
#         nvtx.end_range(overall_range)
#         raise DataValidationError(
#             f"Not enough valid points ({np.sum(valid_fit_mask_np)}) for optimal radius fit. Check SNR calculations.")
#
#     center_conv_snr_fit = center_conv_snr_np[valid_fit_mask_np]
#     opt_radii_fit = opt_radii_np[valid_fit_mask_np]
#
#     # Perform fit on CPU using filtered NumPy data
#     polyfit_range = nvtx.start_range('polyfit_cpu', category='cpu_ops', color='blue')
#     try:
#         log10_conv_snr_fit = np.log10(center_conv_snr_fit)
#         log10_opt_radii_fit = np.log10(opt_radii_fit)
#         pov = np.polyfit(log10_conv_snr_fit, log10_opt_radii_fit, 1, cov=False)
#     except Exception as e:
#         nvtx.end_range(polyfit_range)
#         nvtx.end_range(block8_range)
#         nvtx.end_range(overall_range)
#         # Add more context to the error
#         raise DataValidationError(f"Polyfit failed: {e}. Check input values (SNR, radii).") from e
#     nvtx.end_range(polyfit_range)
#     nvtx.end_range(block8_range)
#
#     # BLOQUE 9: Cálculo de flujo y ruido óptimos (All GPU)
#     block9_range = nvtx.start_range('optimal_flux_calculation', category='phot.metadata', color='lime')
#
#     # Calculate optimal radius for *all* filtered sources on GPU
#     # Use the full conv_snr_filt (all sources passing initial SNR cut)
#     opt_rad_calc_range = nvtx.start_range('calc_all_opt_radii', category='phot.calc')
#     valid_conv_snr_mask = (conv_snr_filt > 0) & cp.isfinite(conv_snr_filt)
#     source_opt_rad = cp.full_like(conv_snr_filt, cp.nan)  # Initialize with NaN
#
#     if cp.any(valid_conv_snr_mask):  # Proceed only if some valid SNRs exist
#         log10_conv_snr_valid = cp.log10(conv_snr_filt[valid_conv_snr_mask])
#         # pov is NumPy array [slope, intercept] - use elements
#         log10_opt_rad_valid = pov[0] * log10_conv_snr_valid + pov[1]
#         opt_rad_intermediate_valid = cp.power(10, log10_opt_rad_valid)
#
#         # Clip and round on GPU
#         max_radii_val = radii[-1].item()  # Get max value from radii array (scalar)
#         min_radii_val = radii[0].item()  # Get min value
#         source_opt_rad_valid = cp.rint(
#             cp.clip(opt_rad_intermediate_valid, min_radii_val, max_radii_val)
#         ).astype(radii.dtype)  # Match radii dtype
#
#         # Place valid results back into the full array
#         source_opt_rad[valid_conv_snr_mask] = source_opt_rad_valid
#
#     nvtx.end_range(opt_rad_calc_range)
#
#     # Find corresponding indices in the 'radii' array on GPU using searchsorted
#     # Need to handle NaNs in source_opt_rad - searchsorted might behave unexpectedly
#     # Replace NaNs with a value outside the radii range temporarily? Or process only valid ones?
#     # Let's process only valid ones.
#     opt_rad_idx_calc_range = nvtx.start_range('calc_opt_radii_indices', category='phot.calc')
#     source_opt_rad_idx = cp.full_like(source_opt_rad, -1, dtype=cp.int32)  # Default index -1
#
#     valid_opt_rad_mask = ~cp.isnan(source_opt_rad)
#     if cp.any(valid_opt_rad_mask):
#         source_opt_rad_for_search = source_opt_rad[valid_opt_rad_mask]
#         # searchsorted finds where element *would be inserted* to maintain order
#         # side='left' means index of first element >= value
#         # side='right' means index of first element > value
#         # If radii are [1, 2, 3, 4] and we search for 2.1 (rounded to 2),
#         # side='left' gives 1, side='right' gives 2. We want index 1 (for radius 2).
#         # So, searchsorted(radii, rounded_rad, side='left') seems correct if radii are integers.
#         # If radii are floats, direct search might be okay. Let's stick with left for safety.
#         indices_valid = cp.searchsorted(radii, source_opt_rad_for_search, side='left')
#         # Clip indices to ensure they are within bounds [0, len(radii)-1]
#         indices_valid = cp.clip(indices_valid, 0, len(radii) - 1)
#         source_opt_rad_idx[valid_opt_rad_mask] = indices_valid
#     nvtx.end_range(opt_rad_idx_calc_range)
#
#     # Get optimal values using GPU indexing, only for sources with valid opt rad idx
#     final_selection_range = nvtx.start_range('select_optimal_values', category='phot.calc')
#     final_valid_mask = (source_opt_rad_idx != -1)  # Mask for sources where opt radius was calculated
#
#     opt_aperture_corrections = cp.full_like(source_opt_rad, cp.nan)
#     opt_aperture_correction_errors = cp.full_like(source_opt_rad, cp.nan)
#     opt_flux = cp.full_like(source_opt_rad, cp.nan)
#
#     if cp.any(final_valid_mask):
#         indices_for_final = source_opt_rad_idx[final_valid_mask]  # Indices into radii dimension
#         source_indices_final = cp.where(final_valid_mask)[0]  # Indices into source dimension
#
#         # aperture_corrections shape: (n_sources_filt, n_radii)
#         opt_aperture_corrections[final_valid_mask] = aperture_corrections[source_indices_final, indices_for_final]
#         opt_aperture_correction_errors[final_valid_mask] = aperture_correction_errors[
#             source_indices_final, indices_for_final]
#
#         # source_flux shape: (n_radii, n_sources_filt)
#         opt_flux[final_valid_mask] = source_flux[indices_for_final, source_indices_final]
#
#     # Further mask for positive flux
#     positive_flux_mask = final_valid_mask & (opt_flux > 0)
#
#     if cp.sum(positive_flux_mask) == 0:
#         nvtx.end_range(final_selection_range)
#         nvtx.end_range(block9_range)
#         nvtx.end_range(overall_range)
#         raise DataValidationError("No sources with valid optimal radius and positive flux found.")
#
#     # Filter all optimal arrays on GPU using positive_flux_mask
#     opt_flux_final = opt_flux[positive_flux_mask]
#     opt_aper_corr_final = opt_aperture_corrections[positive_flux_mask]
#     opt_corr_err_final = opt_aperture_correction_errors[positive_flux_mask]
#     source_opt_rad_final = source_opt_rad[positive_flux_mask]  # Keep optimal radii for these sources
#     source_opt_rad_idx_final = source_opt_rad_idx[positive_flux_mask]  # Keep optimal indices
#     final_source_indices = cp.where(positive_flux_mask)[0]  # Original indices within source_coord_filt
#
#     nvtx.end_range(final_selection_range)
#
#     # Calculate final signal and noise on GPU for the final set
#     final_calc_range = nvtx.start_range('final_signal_noise_calc', category='phot.calc')
#     opt_signal_final = cp.divide(opt_flux_final, opt_aper_corr_final + epsilon)
#
#     # Get corresponding back_flux and area values
#     opt_back_flux_final = cp.zeros_like(opt_flux_final)
#     if back_flux is not None:
#         # back_flux shape: (n_radii, n_sources_filt)
#         opt_back_flux_final = back_flux[source_opt_rad_idx_final, final_source_indices]
#
#     # area shape: (n_radii,)
#     opt_area_final = area[source_opt_rad_idx_final]  # Shape (n_final_sources,)
#
#     # Noise components for the final set
#     opt_back_noise_sq = cp.abs(opt_back_flux_final) * gain
#     opt_read_noise_sq = opt_area_final * rdnoise ** 2
#     opt_corr_sq = opt_aper_corr_final ** 2
#     opt_source_noise_sq = cp.divide(opt_flux_final * gain, opt_corr_sq + epsilon)
#     opt_corr_noise_sq = cp.power(
#         cp.divide(opt_flux_final * gain * opt_corr_err_final, opt_corr_sq + epsilon), 2
#     )
#
#     # Final total noise calculation
#     opt_total_noise_sq_sum = (opt_source_noise_sq / n_images +
#                               opt_back_noise_sq / n_images +
#                               opt_read_noise_sq / n_images +
#                               opt_corr_noise_sq)
#     opt_total_noise_sq_sum = cp.maximum(opt_total_noise_sq_sum, 0)
#     opt_total_noise_final = cp.sqrt(opt_total_noise_sq_sum) / gain
#
#     # Filter coordinates for the final set (indices relative to source_coord_filt)
#     opt_coords_final = source_coord_filt[final_source_indices]
#     nvtx.end_range(final_calc_range)
#     nvtx.end_range(block9_range)
#
#     # BLOQUE 10: Información adicional para encabezados (GPU + Scalar Transfers)
#     block10_range = nvtx.start_range('extra_info_generation', category='phot.metadata', color='yellow')
#     extra_info = {}
#     ref_snr_values = [10, 100, 250, 1000]
#
#     # Calculate final SNR on GPU
#     opt_final_snr = cp.divide(opt_signal_final, opt_total_noise_final + epsilon)
#
#     if opt_final_snr.size > 0:
#         for snr_ref in ref_snr_values:
#             # Find index of closest SNR on GPU
#             diff_snr = cp.abs(opt_final_snr - snr_ref)
#             ref_idx_cp = cp.argmin(diff_snr)  # Use argmin, nanargmin if NaNs were possible
#
#             # --- Transfer Point 4 (Scalar Index and Values) ---
#             tx4_range = nvtx.start_range(f'transfer_extra_info_snr{snr_ref}', category='transfer_scalar', color='pink')
#             ref_idx_np = ref_idx_cp.item()  # Transfer index
#
#             # Get corresponding radius and correction using scalar index from GPU arrays
#             try:
#                 # .item() transfers scalar value efficiently
#                 rad_val = source_opt_rad_final[ref_idx_np].item()
#                 corr_val = opt_aper_corr_final[ref_idx_np].item()
#                 extra_info[f'RAD{snr_ref}'] = int(rad_val)  # Round radius before int
#                 extra_info[f'CORR{snr_ref}'] = round(corr_val, 3)
#             except IndexError:
#                 # Handle case where index might be invalid (shouldn't happen with argmin on non-empty)
#                 extra_info[f'RAD{snr_ref}'] = -1
#                 extra_info[f'CORR{snr_ref}'] = -1.0
#             nvtx.end_range(tx4_range)
#             # --- End Transfer ---
#     else:  # Handle case with no valid final sources
#         for snr_ref in ref_snr_values:
#             extra_info[f'RAD{snr_ref}'] = -1
#             extra_info[f'CORR{snr_ref}'] = -1.0
#
#     nvtx.end_range(block10_range)
#
#     # --- Final Transfer GPU -> CPU for return values ---
#     final_transfer_range = nvtx.start_range('final_gpu_to_cpu_transfer', category='transfer', color='red')
#     opt_signal_np = opt_signal_final.get()
#     opt_total_noise_np = opt_total_noise_final.get()
#     opt_coords_np = opt_coords_final.get()
#     nvtx.end_range(final_transfer_range)
#
#     # Optional: Aggressive memory cleanup at the very end
#     del source_flux, back_flux, area, corrections, correction_errors, cluster_centers
#     del aperture_corrections, aperture_correction_errors, opt_signal_final, opt_total_noise_final
#     # ... delete other large intermediate CuPy arrays ...
#     mempool.free_all_blocks()
#     gc.collect()
#
#     nvtx.end_range(overall_range)  # End overall optimized function range
#
#     return opt_signal_np, opt_total_noise_np, opt_coords_np, extra_info


# ### # @hierarchical_debug(logger)
# @nvtx.annotate('perform_opt_photometry', category='phot.photo_gpu')
# def perform_opt_photometry(img: cp.ndarray, back: cp.ndarray, conv_ima_sigma: cp.ndarray,
#                            source_coord: cp.ndarray, isolated_coord: cp.ndarray,
#                            tile_section_psf: int, star_dataset: cp.ndarray, fwhm: float,
#                            gain: float, rdnoise: float, n_images: int = 1, center_factor: float = 1.0,
#                            min_conv_snr: float = 300.0):
#     """
#         Optimize the aperture photometry process to find the best radii for signal-to-noise ratio (SNR) for each star.
#
#         :param img: Input image.
#         :type img: cupy.ndarray
#         :param back: Background image.
#         :type back: cupy.ndarray
#         :param conv_ima_sigma: Convolution sigma map.
#         :type conv_ima_sigma: cupy.ndarray
#         :param source_coord: Coordinates of sources.
#         :type source_coord: cupy.ndarray
#         :param isolated_coord: Coordinates of isolated stars.
#         :type isolated_coord: cupy.ndarray
#         :param tile_section_psf: Size of PSF tiles.
#         :type tile_section_psf: int
#         :param star_dataset: Dataset of stars.
#         :type star_dataset: cupy.ndarray
#         :param fwhm: Full-width at half-maximum of the PSF.
#         :type fwhm: float
#         :param gain: Gain value for flux conversion.
#         :type gain: float
#         :param rdnoise: Read noise value of the detector.
#         :type rdnoise: float
#         :param n_images: Number of images (default is 1).
#         :type n_images: int
#         :param center_factor: Fraction defining the central region for selecting isolated stars (default is 1.0).
#         :type center_factor: float
#         :param min_conv_snr: Minimum convolutional SNR for selecting isolated stars (default is 300.0).
#         :type min_conv_snr: float
#         :return: Tuple containing the optimized fluxes, noise, coordinates, and extra information.
#         :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, dict)
#         :raises InsufficientStarsError: If there are not enough isolated stars for processing.
#         :raises DataValidationError: If input data is invalid or insufficient.
#         """
#     # Memory pool
#     mempool = cp.get_default_memory_pool()
#
#     # BLOQUE 1: Selección de estrellas centrales
#     # Select central stars
#     center_stars_range = nvtx.start_range('perform_opt_photometry.center_stars_selection', category='phot.photo_gpu',
#                                           color='green')
#     center_factor = np.min((center_factor, 1))
#     xmin = int(img.shape[1] * 0.5 * (1 - center_factor))
#     xmax = int(img.shape[1] * 0.5 * (1 + center_factor))
#     ymin = int(img.shape[0] * 0.5 * (1 - center_factor))
#     ymax = int(img.shape[0] * 0.5 * (1 + center_factor))
#
#     center_mask = (isolated_coord[:, 0] > ymin) & (isolated_coord[:, 0] < ymax) & \
#                   (isolated_coord[:, 1] > xmin) & (isolated_coord[:, 1] < xmax)
#     nvtx.end_range(center_stars_range)
#
#     # BLOQUE 2: Procesamiento SNR y validación de fuentes
#     # Obtain convolutional SNR
#     snr_processing_range = nvtx.start_range('perform_opt_photometry.snr_source_validation', category='phot.photo_gpu',
#                                             color='green')
#     conv_snr = conv_ima_sigma[
#         cp.round(source_coord[:, 0]).astype(cp.int32), cp.round(source_coord[:, 1]).astype(cp.int32)]
#     pos_conv_snr_mask = conv_snr > 0
#     source_coord = source_coord[pos_conv_snr_mask]
#     conv_snr = conv_snr[pos_conv_snr_mask]
#     _, source_coords_matched_idx = crossmatch_sources(
#         isolated_coord[center_mask].get(), source_coord.get(), thres_px=3
#     )
#     conv_snr_isol = conv_ima_sigma[
#         cp.round(isolated_coord[:, 0]).astype(cp.int32), cp.round(isolated_coord[:, 1]).astype(cp.int32)]
#     conv_snr_mask = conv_snr_isol > min_conv_snr
#     if cp.sum(conv_snr_mask) < 3:
#         nvtx.end_range(snr_processing_range)
#         raise InsufficientStarsError(cp.sum(conv_snr_mask))
#     nvtx.end_range(snr_processing_range)
#
#     # BLOQUE 3: Configuración de radios de apertura
#     # In case PSF is position-invariant
#     # if coeff_map is None or eigen_psfs is None or len(source_coords_matched_idx) < 10:
#     #     tile_section_psf = int(1.1 * max(img.shape))
#     aperture_setup_range = nvtx.start_range('perform_opt_photometry.aperture_radii_setup', category='phot.photo_gpu',
#                                             color='green')
#     max_radii = int(np.ceil(7 * fwhm) + 1)
#     min_radii = int(np.ceil(.75 * fwhm))
#     radii = np.arange(min_radii, max_radii, 1)
#     nvtx.end_range(aperture_setup_range)
#
#     # BLOQUE 4: Creación de mapa de correcciones de apertura
#     # Find aperture corrections
#     aper_corr_range = nvtx.start_range('perform_opt_photometry.aperture_corrections_mapping', category='phot.photo_gpu',
#                                        color='green')
#     corrections, correction_errors, cluster_centers = create_aperture_corrections_map(
#         img.shape, tile_section_psf, star_dataset[conv_snr_mask],
#         isolated_coord[conv_snr_mask].get(), radii
#     )
#     aperture_corrections, aperture_correction_errors = find_aperture_corrections(
#         source_coord, corrections, correction_errors, cluster_centers
#     )
#     nvtx.end_range(aper_corr_range)
#
#     # BLOQUE 5: Fotometría por lotes - punto crítico identificado
#     # Get batch photometry
#     batch_phot_range = nvtx.start_range('perform_opt_photometry.batch_aperture_photometry', category='phot.photo_gpu',
#                                         color='green')
#     source_flux, back_flux, area = batch_aperture_photometry(
#         img, back, cp.round(source_coord).astype(cp.int32), radii
#     )
#     nvtx.end_range(batch_phot_range)
#
#     # BLOQUE 6: Transferencia de datos GPU→CPU (potencial cudaMemcpy costoso)
#     # Calculate photometric parameteres
#     gpu_to_cpu_range = nvtx.start_range('perform_opt_photometry.gpu_to_cpu_transfer', category='phot.photo_gpu',
#                                         color='green')
#     center_isolated_flux = np.array(source_flux[:, source_coords_matched_idx].get())
#     center_isolated_aper_corr = np.array(aperture_corrections[source_coords_matched_idx].T)
#     center_isolated_aper_corr_err = np.array(aperture_correction_errors[source_coords_matched_idx].T)
#     nvtx.end_range(gpu_to_cpu_range)
#
#     # BLOQUE 7: Cálculo de señal y ruido
#     signal_noise_range = nvtx.start_range('perform_opt_photometry.photometric_parameters_calc',
#                                           category='phot.photo_gpu', color='green')
#     center_isolated_signal = center_isolated_flux / center_isolated_aper_corr
#
#     center_isolated_back_noise_sq = np.abs(back_flux[:, source_coords_matched_idx]).get() * gain
#     center_isolated_read_noise_sq = area.get().reshape(-1, 1) * rdnoise ** 2
#     center_isolated_source_noise_sq = center_isolated_flux * gain / center_isolated_aper_corr ** 2
#     center_isolated_corr_noise_sq = (
#                                             center_isolated_flux * gain * center_isolated_aper_corr_err / center_isolated_aper_corr ** 2) ** 2
#
#     center_isolated_total_noise = np.sqrt(
#         center_isolated_source_noise_sq + center_isolated_back_noise_sq + center_isolated_read_noise_sq + center_isolated_corr_noise_sq) / gain / np.sqrt(
#         n_images)
#
#     center_isolated_snr = center_isolated_signal / center_isolated_total_noise
#     center_conv_snr = conv_snr[source_coords_matched_idx].get()
#     nvtx.end_range(signal_noise_range)
#
#     # BLOQUE 8: Optimización de radio de apertura
#     # Find optimal aperture radius
#     opt_radius_range = nvtx.start_range('perform_opt_photometry.optimal_radii_calculation', category='phot.photo_gpu',
#                                         color='green')
#     opt_radii_idx = np.argmax(center_isolated_snr, axis=0)
#     opt_radii = radii[opt_radii_idx]
#
#     if len(center_conv_snr) == 0 or len(opt_radii) == 0:
#         nvtx.end_range(opt_radius_range)
#         raise DataValidationError("center_conv_snr or opt_radii is empty. Ensure valid data is provided.")
#
#     pov = np.polyfit(np.log10(center_conv_snr), np.log10(opt_radii), 1, cov=False)
#     nvtx.end_range(opt_radius_range)
#
#     # BLOQUE 9: Cálculo de flujo y ruido óptimos
#     # Calculate optimal flux and noise
#     opt_flux_range = nvtx.start_range('perform_opt_photometry.optimal_flux_calculation', category='phot.metadata',
#                                       color='green')
#     source_opt_rad = np.round(
#         np.fmax(np.fmin((10 ** (pov[0] * np.log10(conv_snr.get()) + pov[1])), max_radii), min_radii)).astype(int)
#     source_opt_rad_idx = np.searchsorted(radii, source_opt_rad, side='right') - 1
#
#     opt_aperture_corrections = np.array(aperture_corrections)[np.arange(source_coord.shape[0]), source_opt_rad_idx]
#     opt_aperture_correction_errors = np.array(aperture_correction_errors)[
#         np.arange(source_coord.shape[0]), source_opt_rad_idx]
#     opt_flux = np.array(source_flux[source_opt_rad_idx, np.arange(source_coord.shape[0])].get())
#
#     mask = opt_flux > 0
#     opt_flux = opt_flux[mask]
#     opt_aper_corr = np.array(opt_aperture_corrections)[mask]
#     opt_corr_err = np.array(opt_aperture_correction_errors)[mask]
#
#     opt_signal = opt_flux / opt_aper_corr
#
#     opt_back_noise_sq = np.abs(back_flux[source_opt_rad_idx, np.arange(source_coord.shape[0])]).get()[mask] * gain
#     opt_read_noise_sq = area[source_opt_rad_idx].get()[mask] * rdnoise ** 2
#     opt_source_noise_sq = opt_flux * gain / opt_aper_corr ** 2
#     opt_corr_noise_sq = (opt_flux * gain * opt_corr_err / opt_aper_corr ** 2) ** 2
#
#     opt_total_noise = np.sqrt(opt_source_noise_sq / n_images +
#                               opt_back_noise_sq / n_images +
#                               opt_read_noise_sq / n_images +
#                               opt_corr_noise_sq) / gain
#     opt_coords = source_coord.get()[mask]
#     nvtx.end_range(opt_flux_range)
#
#     # BLOQUE 10: Información adicional para encabezados
#     # Obtain aditional information for header purposes
#     header_info_range = nvtx.start_range('perform_opt_photometry.extra_info_generation', category='phot.metadata',
#                                          color='green')
#     ref_snr = [10, 100, 250, 1000]
#     extra_info = {}
#     for snr in ref_snr:
#         ref_idx = np.argmin(np.abs(opt_signal / opt_total_noise - snr))
#         extra_info[f'RAD{snr}'] = int(source_opt_rad[mask][ref_idx])
#         extra_info[f'CORR{snr}'] = round(opt_aperture_corrections[mask][ref_idx], 3)
#     nvtx.end_range(header_info_range)
#
#     return opt_signal, opt_total_noise, opt_coords, extra_info
#

### # @hierarchical_debug(logger)
@nvtx.annotate('batch_aperture_photometry', category='phot.photo_gpu')
def batch_aperture_photometry(img: cp.ndarray, back: cp.ndarray | None,
                              positions: cp.ndarray, radii: cp.ndarray, **kwargs) -> tuple[
    cp.ndarray, cp.ndarray | None, cp.ndarray]:
    """
    Perform aperture photometry in batch mode using CuPy FFT.
    Accepts radii as CuPy array. Calculates area vectorially.

    :param img: Image data (GPU). Can be 2D (h, w) or 3D (n_images, h, w).
    :type img: cupy.ndarray
    :param back: Background image (GPU, optional). Must match img dimensions if provided.
    :type back: cupy.ndarray or None
    :param positions: Positions of sources (GPU) (n_sources, 2) -> [[row, col], ...].
                      Must be integer type for indexing.
    :type positions: cupy.ndarray (int32/64)
    :param radii: Aperture radii (GPU).
    :type radii: cupy.ndarray
    :return: Tuple containing flux, background flux (or None), and aperture area per radius.
             Flux/BackFlux shape: (n_images, n_radii, n_sources) or (n_radii, n_sources)
             Area shape: (n_radii,)
    :rtype: tuple(cupy.ndarray, cupy.ndarray or None, cupy.ndarray)
    """
    nvtx_range = nvtx.start_range('batch_aperture_photometry', category='phot.photo_gpu', color='green')
    mempool = cp.get_default_memory_pool()

    if radii.size == 0:
        raise ValueError("Radii array cannot be empty.")

    # --- Calculate Area vectorially (GPU) ---
    # Determine max size needed for kernel based on largest radius
    max_radius = cp.max(radii).item()  # Get scalar value
    # Kernel size needs to be odd and large enough for largest radius
    kernel_size = int(2 * np.ceil(max_radius) + 1)
    if kernel_size % 2 == 0: kernel_size += 1  # Ensure odd size

    # Calculate area for each radius using the kernel generation function
    areas = cp.zeros(len(radii), dtype=cp.float64)
    # Small loop here is okay, or could vectorize kernel generation if complex
    for i, r in enumerate(radii):
        # We only need the area here, kernel is generated inside the main loop
        _, areas[i] = get_aper_kernel(r.item(), size=kernel_size)
    # --- End Area Calculation ---

    image_shape_orig = img.shape
    is_3d = img.ndim == 3
    n_images = image_shape_orig[0] if is_3d else 1
    img_h = image_shape_orig[1] if is_3d else image_shape_orig[0]
    img_w = image_shape_orig[2] if is_3d else image_shape_orig[1]

    padding = (kernel_size - 1) // 2  # Integer division

    # Calculate padded shape for FFT
    fft_shape = fill_image((img_h + 2 * padding, img_w + 2 * padding))

    # Pre-allocate output arrays
    n_radii = len(radii)
    n_positions = len(positions)
    flux_shape = (n_images, n_radii, n_positions) if is_3d else (n_radii, n_positions)
    flux = cp.zeros(flux_shape, dtype=cp.float64)  # Use float64 for precision
    back_flux = cp.zeros(flux_shape, dtype=cp.float64) if back is not None else None

    # --- FFT and Convolution Loop ---
    fft_loop_range = nvtx.start_range('fft_convolution_loop', category='phot.photo_gpu')

    # Function to process a single 2D image (or image plane)
    def process_plane(plane_idx, img_plane, back_plane):
        img_c = cp.fft.rfft2(img_plane, s=fft_shape)
        back_c = cp.fft.rfft2(back_plane, s=fft_shape) if back_plane is not None else None
        plane_flux = cp.zeros((n_radii, n_positions), dtype=cp.float64)
        plane_back_flux = cp.zeros((n_radii, n_positions), dtype=cp.float64) if back_plane is not None else None

        for i, r in enumerate(radii):
            kernel_range = nvtx.start_range(f'kernel_fft_r={r}', category='phot.conv')
            kernel, _ = get_aper_kernel(r.item(), size=kernel_size)  # Generate kernel for this radius
            # FFT kernel (conj needed for correlation via convolution theorem)
            kernel_c = cp.conj(cp.fft.rfft2(kernel, s=fft_shape))
            nvtx.end_range(kernel_range)

            conv_range = nvtx.start_range(f'ifft_roll_index_r={r}', category='phot.conv')
            # Convolve image
            convolved_img = cp.fft.irfft2(img_c * kernel_c, s=fft_shape)
            # Shift result to align, crop to original size
            convolved_img = cp.roll(convolved_img, shift=[padding, padding], axis=[0, 1])
            convolved_img = convolved_img[:img_h, :img_w]
            # Index at source positions
            plane_flux[i, :] = convolved_img[positions[:, 0], positions[:, 1]]

            # Convolve background if exists
            if back_c is not None:
                convolved_back = cp.fft.irfft2(back_c * kernel_c, s=fft_shape)
                convolved_back = cp.roll(convolved_back, shift=[padding, padding], axis=[0, 1])
                convolved_back = convolved_back[:img_h, :img_w]
                plane_back_flux[i, :] = convolved_back[positions[:, 0], positions[:, 1]]
            nvtx.end_range(conv_range)

            # Modest memory cleanup inside loop if kernels are large
            del kernel, kernel_c
            # mempool.free_all_blocks() # Maybe too frequent, causes overhead

        del img_c, back_c  # Cleanup FFTs for the plane
        # mempool.free_all_blocks()
        # gc.collect()
        return plane_flux, plane_back_flux

    if is_3d:
        for c in range(n_images):
            img_plane = img[c]
            back_plane = back[c] if back is not None else None
            plane_flux, plane_back_flux = process_plane(c, img_plane, back_plane)
            flux[c, :, :] = plane_flux
            if back_flux is not None:
                back_flux[c, :, :] = plane_back_flux
    else:
        # Process the single 2D image
        plane_flux, plane_back_flux = process_plane(0, img, back)
        flux = plane_flux
        back_flux = plane_back_flux

    nvtx.end_range(fft_loop_range)
    # --- End FFT Loop ---

    # Final cleanup (optional)
    mempool.free_all_blocks()
    gc.collect()

    nvtx.end_range(nvtx_range)  # End overall batch photometry range
    return flux, back_flux, areas  # Return calculated areas


# @contextmanager
# @nvtx.annotate('gpu_array_manager',category='phot.photo_gpu')
# def gpu_array_manager(data, mempool):
#     """Context manager para manejo optimizado de arrays temporales"""
#     arr = cp.asarray(data)
#     try:
#         yield arr
#     finally:
#         del arr
#         mempool.free_all_blocks()
#         cp.cuda.Stream.null.synchronize()
#
#
#

### # @hierarchical_debug(logger)
@nvtx.annotate('calibrate_image', category='phot.photo_gpu')
def calibrate_image(imdata: np.ndarray, filter: str, scale: float, gain: float, rdnoise: float,
                    exptime: float, satlevel: float, target_ra: float, target_dec: float = None, n_images: int = 1,
                    SP_filt: bool = True, CR_filt: bool = False, border: int = 10, center_factor: float = 0.7,
                    pca_method: bool = True, tile_section: int = 1000, max_stars_ref: int = 15, min_snr: int = 5,
                    color_range: float = 0.6, tile_section_psf: int = 2500, **kwargs):
    """
    Calibrate an image.

    :param imdata: Image data.
    :type imdata: numpy.ndarray
    :param filter: Filter used for the image.
    :type filter: str
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
    :return: Calibration dictionary, astrometry dictionary, photometry dataframe.
    :rtype: tuple(dict, dict, pandas.DataFrame)
    :raises InsufficientStarsError: If less than 5 isolated stars are detected.
    :raises MoffatFitError: If there's an error fitting Moffat to reference PSF.
    """

    mempool = cp.get_default_memory_pool()

    # with gpu_array_manager(imdata, mempool) as img_cp:
    img_cp = cp.asarray(imdata)

    # Protect bad prereduction
    img_cp[cp.isinf(img_cp) | cp.isnan(img_cp)] = 0

    # Get background
    back, _ = get_local_background_fft(img_cp, scale, get_std=False, **kwargs)

    del _

    rms = cp.sqrt(back * gain + rdnoise ** 2) / gain / cp.sqrt(n_images)
    xmin = int(imdata.shape[1] * 0.5 * (1 - 0.3))
    xmax = int(imdata.shape[1] * 0.5 * (1 + 0.3))
    ymin = int(imdata.shape[0] * 0.5 * (1 - 0.3))
    ymax = int(imdata.shape[0] * 0.5 * (1 + 0.3))
    m = cp.median(back[ymin:ymax, xmin:xmax])
    s = cp.std(back[ymin:ymax, xmin:xmax])
    mask = cp.abs(back[ymin:ymax, xmin:xmax] - m) < 3 * s
    m = cp.median(back[ymin:ymax, xmin:xmax][mask])

    del mask
    fluxsky = np.round(m.get(), 6)
    if fluxsky < 0: logger.warning('Median background flux is negative')
    dic_calib = {'FLUXSKY': fluxsky}

    # Detect isolated stars
    if CR_filt:
        img = CR_filter(img_cp - back)
    elif SP_filt:
        img = SP_filter(img_cp - back)  # change SP_filter_cupy to SP_filter
    else:
        img = img_cp - back

    # del img_cp
    maybe_free_arrays([img_cp], mempool)

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
    xmin = int(imdata.shape[1] * 0.5 * (1 - center_factor))
    xmax = int(imdata.shape[1] * 0.5 * (1 + center_factor))
    ymin = int(imdata.shape[0] * 0.5 * (1 - center_factor))
    ymax = int(imdata.shape[0] * 0.5 * (1 + center_factor))

    # Find reference psf
    unit_star_dataset = star_dataset.astype(cp.double) / scaling[:, 3][:, None, None]
    center_mask = (coord[:, 1] > xmin) & (coord[:, 1] < xmax) & (coord[:, 0] > ymin) & (coord[:, 0] < ymax)
    unit_star_dataset_stds = cp.std(unit_star_dataset, axis=(1, 2))
    mask_star_dataset = unit_star_dataset_stds < cp.percentile(cp.std(unit_star_dataset, axis=(1, 2)), 95.4)
    unit_star_dataset = unit_star_dataset[mask_star_dataset]
    coord = coord[mask_star_dataset]

    star_dataset_ref = unit_star_dataset[center_mask[mask_star_dataset]][:max_stars_ref, :, :]
    del center_mask, unit_star_dataset_stds, scaling
    if star_dataset_ref.shape[0] < 5:
        logger.error('Less than 5 isolated stars detected. Image may be too crowded or too noisy')
        raise InsufficientStarsError(num_stars=star_dataset_ref.shape[0])

    psf, _ = stack_sigmaclip(star_dataset_ref, n=2)

    del star_dataset_ref, _

    psf = psf / cp.sum(psf)
    try:
        _, _, _, fwhm, _ = fit_moffat(psf.get())
        del _
    except:
        logger.error('Error fitting Moffat to reference PSF')
        raise MoffatFitError()
    if fwhm < 2:
        logger.error('Error fitting Moffat to reference PSF')
        raise MoffatFitError()

    dic_calib['FWHM'] = fwhm
    eigen_psfs, coeff_map = None, None
    fwhm = max(fwhm, 2)

    if pca_method:
        # Create psf deviations dataset
        unit_star_dataset_dev = unit_star_dataset - psf
        unit_star_dataset_dev_norm = (unit_star_dataset_dev - cp.mean(unit_star_dataset_dev, axis=(1, 2))[:, None,
                                                              None]) / cp.std(unit_star_dataset_dev, axis=(1, 2))[:,
                                                                       None, None]

        # get eigen psfs
        eigen_psfs = get_eigen_psfs(unit_star_dataset_dev_norm, n_components=5)
        eigen_psfs = cp.asarray(eigen_psfs)
        coefficients = project_all_stars_onto_eigenpsfs(unit_star_dataset_dev, eigen_psfs)
        coeff_map = create_coeff_map(imdata.shape, coord, coefficients.T, scale, tile_section=tile_section)
        # del coord
        mempool.free_all_blocks()
        gc.collect()

        x1 = int(imdata.shape[1] * 0.25)
        x2 = int(imdata.shape[1] * 0.75)
        y1 = int(imdata.shape[0] * 0.25)
        y2 = int(imdata.shape[0] * 0.75)
        x = np.array([x1, x2, x1, x2])
        y = np.array([y1, y2, y2, y1])
        fwhm_lab = ['FWHMLL', 'FWHMLR', 'FWHMUL', 'FWHMUR']
        # fwhms = np.zeros(len(fwhm_lab))
        for point in range(len(fwhm_lab)):
            psf_l = psf + recreate_normed_star(coeff_map, eigen_psfs, (x[point], y[point]))
            try:
                _, _, _, fwhm_l, _ = fit_moffat(psf_l.get())
                fwhm_l = np.round(fwhm_l, 2)
            except:
                fwhm_l = 0
            dic_calib[fwhm_lab[point]] = fwhm_l

    del unit_star_dataset
    # Detect sources
    sources, conv_ima_sigma = detect_sources_psf(img, rms, fwhm, psf, eigen_psfs, coeff_map, min_snr=min_snr, **kwargs)
    sources = sources[(sources[:, 0] > border) & (sources[:, 0] < img.shape[0] - border) & (sources[:, 1] > border) & (
            sources[:, 1] < img.shape[1] - border)]

    del coeff_map, img, rms
    mempool.free_all_blocks()

    # Perform optimized photometry

    # optimal_flux, optimal_noise, optimal_coords, extra_info = perform_opt_photometry(img_cp - back, back,
    #                                                                                  conv_ima_sigma, sources,
    #                                                                                  coord, tile_section_psf,
    #                                                                                  star_dataset[mask_star_dataset],
    #                                                                                  fwhm, gain, n_images, rdnoise)

    # TODO: Revisar por Miguel: dejamos center_factor y min_conv_snr con valor dor defecto de la función, o por defecto de DEFAULT_PROCESSING_PARAMS

    imadata_shape = imdata.shape
    if 'img_cp' not in locals():
        img_cp = cp.asarray(imdata)
    # img_cp = cp.asarray(imdata)
    # with gpu_array_manager(imdata, mempool) as img_cp:
    optimal_flux, optimal_noise, optimal_coords, extra_info = perform_opt_photometry_optimized_gpu_crossmatch(
        img=img_cp - back,
        back=back,
        conv_ima_sigma=conv_ima_sigma,
        source_coord=sources,
        isolated_coord=coord,
        tile_section_psf=tile_section_psf,
        star_dataset=star_dataset[mask_star_dataset],
        fwhm=fwhm,
        gain=gain,
        rdnoise=rdnoise,
        n_images=n_images,
        # center_factor=kwargs.get('center_factor', 1.0),
        # min_conv_snr=kwargs.get('min_conv_snr', 300.0)
    )

    del img_cp

    del conv_ima_sigma, sources, back, mask_star_dataset, star_dataset, coord
    mempool.free_all_blocks()

    dic_calib.update(extra_info)

    dfm = pd.DataFrame({'xcentroid': optimal_coords[:, 1],
                        'ycentroid': optimal_coords[:, 0],
                        'flux': optimal_flux,
                        'noise': optimal_noise,
                        'snr': optimal_flux / optimal_noise})

    del optimal_coords, optimal_flux, optimal_noise

    dfm_ast = dfm.loc[dfm.snr > 5]
    dfm_ast = dfm_ast.sort_values('snr', ascending=False).dropna().reset_index(drop=True)

    # Astrometrize
    h_wcs = astrometrice2(dfm_ast, scale, target_ra, target_dec, imdata.shape, sip_order=1)
    del dfm_ast
    if h_wcs == {}:
        logger.error('Astrometry failed')
        return dfm, h_wcs, dic_calib

    else:
        # Photometrize
        coocenter, FOV, scale = get_astrometry_params(h_wcs, imadata_shape)
        result, catalog, ref_filter = catalog_results(coocenter, FOV / 2,
                                                      filter, maglimit=23, **kwargs)
        dic_calib['CATALOG'] = catalog
        dic_calib['CATBAND'] = ref_filter

        w = WCS(h_wcs)
        dfm.loc[:, 'RA'], dfm.loc[:, 'DEC'] = w.all_pix2world(dfm.xcentroid.values, dfm.ycentroid.values, 1)
        # dfm.loc[:, 'RA'] = ra
        # dfm.loc[:, 'DEC'] = dec
        photo_dict = get_zeropoint(result, dfm, exptime, center_lims=(xmin, xmax, ymin, ymax),
                                   solar_filter=color_range, dist_thres_px=1.5 * fwhm * scale / 3600)
        dic_calib.update(photo_dict)

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

        # Calculate limiting magnitude
        try:
            mag = dic_calib['ZP'] - 2.5 * np.log10(dfm.flux.values / exptime)
            snr = dfm.snr.values
            maglim3 = get_maglim(mag, snr, 3)
            del mag, snr
        except Exception as e:
            maglim3 = 0
            logger.warning('Error calculating limiting magnitude: {}'.format(e))
        dic_calib['MAGLIM'] = maglim3

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

    :param img: Image data.
    :type img: cupy.ndarray
    :param positions: Positions of sources.
    :type positions: cupy.ndarray
    :param aper_rad: Aperture radius.
    :type aper_rad: int
    :return: Flux and area of the aperture.
    :rtype: tuple(cupy.ndarray, float)
    """
    kernel, area = get_aper_kernel(aper_rad)
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


#
# ### # @hierarchical_debug(logger)
# def astrometrice2(dfm, head0, im_shape):
#     """
#     Perform astrometry on an image.
#
#     Parameters
#     ----------
#     dfm : pd.DataFrame
#         Dataframe containing detected sources.
#     head0 : dict
#         FITS header.
#     im_shape : tuple
#         Shape of the image.
#
#     Returns
#     -------
#     dict
#         Updated FITS header.
#     """
#     head = head0.copy()
#     arcsec_per_pixel = plate_scale_px(head[HeaderKey.PXSIZE.value], head[HeaderKey.FOCALEN.value]) * head[
#         HeaderKey.XBINNING.value]
#     signal.signal(signal.SIGALRM, handler)
#     signal.alarm(120)
#     try:
#         solution = get_solver().solve(stars_xs=dfm['xcentroid'], stars_ys=
#         dfm['ycentroid'], size_hint=astrometry.SizeHint(
#             lower_arcsec_per_pixel=arcsec_per_pixel * 0.8,
#             upper_arcsec_per_pixel=arcsec_per_pixel * 1.2), position_hint=
#                                       astrometry.PositionHint(ra_deg=head[HeaderKey.POINTRA.value] * 360 / 24,
#                                                               dec_deg=head[HeaderKey.POINTDEC.value], radius_deg=0.5),
#                                       solution_parameters=
#                                       astrometry.SolutionParameters(logodds_callback=
#                                                                     logodds_callback_100, sip_order=3))
#         nmatches = len(solution.matches)
#         logger.debug(nmatches)
#         return solution
#     except:
#         pass
#     signal.alarm(0)
#
#     return None

#
# ### # @hierarchical_debug(logger)
# def get_zeropoint(df_catalog, flux, noise, coord, exptime, solar_filter=0.3,
#                   dist_thres_px=3, N=50, plot=False):
#     """Calculate the zeropoint for photometry.
#
#     :param df_catalog: Catalog dataframe.
#     :type df_catalog: pd.DataFrame
#     :param flux: Flux values.
#     :type flux: ndarray
#     :param noise: Noise values.
#     :type noise: ndarray
#     :param coord: Coordinates of sources.
#     :type coord: ndarray
#     :param exptime: Exposure time.
#     :type exptime: float
#     :param solar_filter: Solar filter, by default 0.3.
#     :type solar_filter: float, optional
#     :param dist_thres_px: Distance threshold in pixels, by default 3.
#     :type dist_thres_px: int, optional
#     :param N: Number of brightest stars to use, by default 50.
#     :type N: int, optional
#     :param plot: Whether to plot the results, by default False.
#     :type plot: bool, optional
#
#
#     """
#     cat_coords = np.array([df_catalog['Y'], df_catalog['X']]).T
#     source_coords_matched_idx, ref_coords_matched_idx = crossmatch_sources(
#         coord, cat_coords, thres_px=dist_thres_px)
#     solar_cat_filt = np.abs(df_catalog['SOLAR'])[ref_coords_matched_idx
#                      ] < solar_filter
#     source_coords_matched_idx = source_coords_matched_idx[solar_cat_filt]
#     ref_coords_matched_idx = ref_coords_matched_idx[solar_cat_filt]
#     det_mag = -2.5 * np.log10(flux[source_coords_matched_idx] / exptime)
#     cat_mag = df_catalog['MAG'][ref_coords_matched_idx].to_numpy()
#     inf_nan_mask = np.isfinite(cat_mag) & np.isfinite(det_mag) & ~np.isnan(
#         cat_mag) & ~np.isnan(det_mag)
#     cat_mag = cat_mag[inf_nan_mask]
#     det_mag = det_mag[inf_nan_mask]
#     snrs = (flux / noise)[source_coords_matched_idx][inf_nan_mask]
#     m = np.argsort(snrs)
#     brightest = m[-N:]
#     bright_mask = np.zeros(len(cat_mag), dtype=bool)
#     bright_mask[brightest] = True
#     if len(cat_mag) <= 3:
#         zp, ezp, n, min_mag, max_mag = 0, 0, 0, 0, 0
#     else:
#         y = cat_mag[bright_mask] - det_mag[bright_mask]
#         mask = np.abs(y - np.nanmean(y)) < np.nanstd(y)
#         if np.sum(mask) > 15:
#             reg = RANSACRegressor(random_state=42, residual_threshold=0.05
#                                   ).fit(det_mag[bright_mask].reshape([-1, 1])[mask], cat_mag[
#                 bright_mask].reshape([-1, 1])[mask])
#             inlier = reg.inlier_mask_
#             if np.sum(inlier) > 10 and np.abs(np.mean(y[mask][inlier]) - np
#                     .mean(y[mask])) < np.std(y[mask]):
#                 zp = np.mean(y[mask][inlier])
#                 n = np.sum(inlier)
#                 ezp = np.std(y[mask][inlier]) / np.sqrt(n)
#                 min_mag = np.min(cat_mag[bright_mask][mask][inlier])
#                 max_mag = np.max(cat_mag[bright_mask][mask][inlier])
#             else:
#                 zp = np.mean(y[mask])
#                 n = np.sum(mask)
#                 ezp = np.std(y[mask]) / np.sqrt(n)
#                 min_mag = np.min(cat_mag[bright_mask][mask])
#                 max_mag = np.max(cat_mag[bright_mask][mask])
#         else:
#             zp = np.mean(y[mask])
#             n = np.sum(mask)
#             ezp = np.std(y[mask]) / np.sqrt(n)
#             min_mag = np.min(cat_mag[bright_mask][mask])
#             max_mag = np.max(cat_mag[bright_mask][mask])
#     if plot:
#         plt.figure(figsize=(8, 8))
#         ax = plt.subplot(111)
#         ax.plot(cat_mag, cat_mag - det_mag - zp, 'k.', alpha=0.6)
#         if np.sum(mask) > 15:
#             ax.plot(cat_mag[bright_mask][mask], cat_mag[bright_mask][mask] -
#                     det_mag[bright_mask][mask] - zp, 'b.', alpha=0.1)
#             ax.plot(cat_mag[bright_mask][mask][inlier], cat_mag[bright_mask
#             ][mask][inlier] - det_mag[bright_mask][mask][inlier] - zp,
#                     'r.', label='zp = {:.3f} +/- {:.3f} (n={})'.format(zp, ezp,
#                                                                        n), alpha=0.5)
#         else:
#             ax.plot(cat_mag[bright_mask][mask], cat_mag[bright_mask][mask] -
#                     det_mag[bright_mask][mask] - zp, 'r.', label=
#                     'zp = {:.3f} +/- {:.3f} (n={})'.format(zp, ezp, n), alpha=0.5)
#         ax.set_xlabel('catalog magnitude')
#         ax.set_ylabel('error magnitude')
#         ax.legend(frameon=False)
#         ax.set_ylim(-0.5, 0.5)
#         plt.show()
#
#     return zp, ezp, n, min_mag, max_mag


# ### # @hierarchical_debug(logger)
# def delete_header_from(header, val):
#     """Delete a section from the FITS header.
#
#     :param header: FITS header.
#     :type header: dict
#     :param val: Value to delete.
#     :type val: str
#
#
#     """
#     for i, v in enumerate(header.values()):
#         if val in str(v):
#             idx = i - 1
#     for i in range(len(header) - idx):
#         del header[idx]
#
#     return header


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


# ### # @hierarchical_debug(logger)
# def handler(signum, frame):
#     """
#     Timeout handler for astrometry.
#
#     Parameters
#     ----------
#     signum : int
#         Signal number.
#     frame : frame
#         Stack frame.
#     """
#     logger.error('Astrometrization timeout!')
#
#     raise Exception('end of time')


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


### # @hierarchical_debug(logger)
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


# ### # @hierarchical_debug(logger)
# def logodds_callback_100(logodds):
#     """
#     Callback function for astrometry.
#
#     Parameters
#     ----------
#     logodds : float
#         Log odds value.
#
#     Returns
#     -------
#     astrometry.Action
#         Action to take (CONTINUE or STOP).
#     """
#     if (logodds[0] > 100.0) | (len(logodds) > 2):
#
#         return astrometry.Action.STOP
#     else:
#
#         return astrometry.Action.CONTINUE


if __name__ == '__main__':
    from astropy.io import fits
    import os

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
