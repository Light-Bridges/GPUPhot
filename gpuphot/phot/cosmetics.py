# SPDX-License-Identifier: MIT
"""
Cosmetic image filters for gpuphot.phot.

Contains small pre-processing filters such as salt-and-pepper removal (SP_filter)
and cosmic ray filtering (CR_filter). Implementations prefer CuPy for GPU
acceleration where available.
"""

### # @hierarchical_debug(logger)
from __future__ import annotations


import cupy as cp
import nvtx
# import tensorflow as tf
from cupyx.scipy.ndimage import convolve, median_filter, laplace, binary_dilation

from .conv import fill_nan_fft
from ..logger.hierarchical_logging import setup_logger
from ..utils.gpu import free_gpu_mem

logger = setup_logger(__name__)

@nvtx.annotate('SP_filter', category='phot.photo_gpu')
def SP_filter(img, filter_size=3, high_threshold_factor=5,
              low_threshold_factor=5, scaling_factor=1.4826, **kwargs):
    """
    Apply a median filter to remove salt-and-pepper noise.

    :param img: Input image.
    :type img: cupy.ndarray
    :param filter_size: Size of the median filter (default is 3).
    :type filter_size: int, optional
    :param high_threshold_factor: Factor to determine the high threshold for noise detection (default is 5).
    :type high_threshold_factor: float, optional
    :param low_threshold_factor: Factor to determine the low threshold for noise detection (default is 5).
    :type low_threshold_factor: float, optional
    :param scaling_factor: Scaling factor for estimating the standard deviation from MAD (default is 1.4826).
    :type scaling_factor: float, optional
    :return: Filtered image.
    :rtype: cupy.ndarray
    """
    logger.info("Applying SP_filter...")
    med_filter = median_filter(img, size=filter_size)
    dif = img - med_filter
    med = cp.nanmedian(dif)
    ms = scaling_factor * cp.nanmedian(cp.abs(dif - med))
    mask = (dif > med + high_threshold_factor * ms) | (dif < med -
                                                       low_threshold_factor * ms)
    kernel = cp.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=cp.uint8)
    neighbor_count = convolve(mask.astype(cp.uint8), kernel, mode='constant', cval=0.0)
    mask = (mask & (neighbor_count < 2))
    img[mask] = med_filter[mask]
    logger.info(f"SP_filter applied: {cp.sum(mask)} pixels corrected.")
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

    mask = laplace(img)
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