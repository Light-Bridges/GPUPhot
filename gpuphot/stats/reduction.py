# SPDX-License-Identifier: MIT
"""
Reduction utilities and helpers for gpuphot.stats.

This module contains routines for subpixel registration via phase
cross-correlation, sigma-clipped stacking, frame registration helpers and
small statistical utilities used across the photometry pipeline.

Bibliographic references present in the original code are preserved in the
function docstrings where relevant.
"""

from typing import Union

import numpy as np
from astropy.io import fits

try:
    import nvtx
except ImportError:

    class nvtx:
        @staticmethod
        def annotate(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

try:
    import cupy as cp
    from cupyx.scipy.ndimage import binary_erosion, shift

    ArrayType = Union[cp.ndarray, np.ndarray]
except ImportError:
    cp = None
    ArrayType = np.ndarray


    def binary_erosion(*args, **kwargs):
        raise ImportError("CuPy no instalado")


    def shift(*args, **kwargs):
        raise ImportError("CuPy no instalado")

try:
    from ..phot.cosmetics import SP_filter
    from ..stats.subpixel import phase_cross_correlation as phase_cross_correlation_gpu
except ImportError:
    pass

from ..logger.hierarchical_logging import setup_logger  # , hierarchical_debug

logger = setup_logger(__name__)


@nvtx.annotate('center', category='stats.reduction')
def center(im: Union[cp.ndarray, np.ndarray], size: int) -> Union[cp.ndarray, np.ndarray]:
    """
    Center the image to the given size.

    Parameters
    ----------
    im : array_like (NumPy or CuPy)
        Input image.
    size : int
        Desired centered window size.

    Returns
    -------
    array_like
        Cropped image centered to `size` (or original image if smaller).
    """
    h, w = im.shape[0], im.shape[1]  # Funciona para np y cp

    if h > size:
        c0_start = (h - size) // 2
        c0_end = c0_start + size
    else:
        c0_start = 0
        c0_end = h

    if w > size:
        c1_start = (w - size) // 2
        c1_end = c1_start + size
    else:
        c1_start = 0
        c1_end = w

    return im[c0_start:c0_end, c1_start:c1_end]


### # @hierarchical_debug(logger)
@nvtx.annotate('register_shift', category='stats.reduction')
def register_shift(fc, uf=100, n=1000):
    """
    Register and shift an image stack using phase cross-correlation.

    The routine recenters each frame to a window of size `n`, computes a
    binary mask via erosion and uses the GPU-enabled phase_cross_correlation
    implementation to estimate subpixel shifts.

    Parameters
    ----------
    fc : array_like
        Stack of images to register (frames on axis 0).
    uf : int, optional
        Upsampling factor for subpixel precision (default: 100).
    n : int, optional
        Size for centering the images prior to registration (default: 1000).

    Returns
    -------
    array_like
        Registered and shifted image stack.
    """
    fc1 = fc.copy()
    im0 = center(fc[0], n)
    im0 = binary_erosion(im0 > im0.mean() + im0.std())
    for i in np.arange(1, fc.shape[0]):
        im1 = center(fc[i], n)
        im1 = binary_erosion(im1 > im1.mean() + im1.std())
        shifted, _, _ = phase_cross_correlation_gpu(im0.get(), im1.get(),
                                                    upsample_factor=uf)
        if (np.abs(shifted[0]) > 300) | (np.abs(shifted[1]) > 300):
            shifted = 0, 0
        fc1[i] = shift(cp.asarray(fc[i]), shift=(shifted[0], shifted[1]),
                       order=1, mode='constant').get()
        logger.debug(f'Detected subpixel offset (y, x): {shifted}')

        del im1, shifted
    return fc1


### # @hierarchical_debug(logger)
@nvtx.annotate('stack_sigmaclip', category='stats.reduction')
def stack_sigmaclip(data, it=5, n=3):
    """
    Stack images using iterative sigma-clipping.

    Parameters
    ----------
    data : cupy.ndarray
        Input image stack (N, H, W).
    it : int, optional
        Number of sigma-clipping iterations (default: 5).
    n : int, optional
        Sigma threshold for clipping (default: 3).

    Returns
    -------
    tuple
        (mean, std) of the stacked images. If fewer than 3 images are provided,
        returns the simple mean and None.
    """
    nim = data.shape[0]
    if nim < 3:
        return cp.asarray(data.mean(axis=0)), None
    delta0 = -1

    im0 = data[0]

    iplus = cp.zeros_like(im0, dtype=cp.float32) + 1.e10
    iminu = cp.zeros_like(im0, dtype=cp.float32) - 1.e10
    avg = cp.zeros_like(im0, dtype=cp.float32)
    std = cp.zeros_like(im0, dtype=cp.float32)
    for j in range(it):
        center = cp.zeros_like(im0, dtype=cp.double)
        sigma = cp.zeros_like(im0, dtype=cp.double)
        mask = cp.zeros_like(im0, dtype=cp.float32)
        delta = 0

        for i in range(data.shape[0]):
            im = data[i, :]

            mk = (im >= iminu)
            mk = mk * (im <= iplus)
            im = im * mk
            center = center + im
            sigma = sigma + im * im
            delta = delta + cp.sum(mk == 0)
            mask = mask + mk

        center = center / mask
        sigma = sigma / mask
        sigma = cp.sqrt(sigma - center * center)
        if delta == delta0:
            break
        delta0 = delta
        iplus = center + n * sigma
        iminu = center - n * sigma
        avg[mask >= 3] = center[mask >= 3]
        std[mask >= 3] = sigma[mask >= 3]

    del iplus, iminu, im0, delta0, mask, delta, center, sigma
    return avg, std


### # @hierarchical_debug(logger)
@nvtx.annotate('register_shift_frames', category='stats.reduction')
def register_shift_frames(frames_list, upsample_factor=100, center_size=6000,
                          shift_limit_pix=300, SP_filt=False):
    """
    Register and shift frames from a list of file paths.

    Parameters
    ----------
    frames_list : list
        List of frame file paths (FITS or other formats readable by astropy).
    upsample_factor : int, optional
        Upsample factor for subpixel refinement (default: 100).
    center_size : int, optional
        Center window size used for registration (default: 6000).
    shift_limit_pix : int, optional
        Maximum allowed shift in pixels; shifts larger than this are treated
        as invalid and replaced by (0, 0).
    SP_filt : bool, optional
        Apply spatial filter if True.

    Returns
    -------
    cupy.ndarray
        Registered image stack.
    """
    fc0 = cp.asarray(fits.getdata(frames_list[0]), dtype=cp.float32)
    if SP_filt: fc0 = SP_filter(fc0)
    im0 = center(fc0, center_size)
    im0 = binary_erosion(im0 > im0.mean() + im0.std())
    fc = cp.zeros((len(frames_list), fc0.shape[0], fc0.shape[1]), dtype=cp.
                  float32)
    fc[0, :] = fc0
    del fc0
    for i in np.arange(1, len(frames_list)):
        fc1 = cp.asarray(fits.getdata(frames_list[i]), dtype=cp.float32)
        if SP_filt: fc1 = SP_filter(fc1)
        im1 = center(fc1, center_size)
        im1 = binary_erosion(im1 > im1.mean() + im1.std())
        shifted, _, _ = phase_cross_correlation_gpu(im0, im1,
                                                    upsample_factor=upsample_factor)
        if (np.abs(shifted[0]) > shift_limit_pix) | (np.abs(shifted[1]) > shift_limit_pix):
            shifted = 0, 0
        logger.debug(f'Detected subpixel offset (y, x): {shifted}')
        fc[i, :] = shift(fc1, shift=(shifted[0], shifted[1]), order=1, mode
        ='constant')

    return fc


### # @hierarchical_debug(logger)
@nvtx.annotate('weighted_mean_std', category='stats.reduction')
def weighted_mean_std(data, errors):
    """
    Compute weighted mean and weighted standard deviation.

    Parameters
    ----------
    data : array_like
        Input data values.
    errors : array_like
        Corresponding uncertainties (sigma) for the data.

    Returns
    -------
    tuple
        (weighted_mean, weighted_std).
    """
    weights = 1 / errors ** 2
    weighted_mean = np.sum(weights * data) / np.sum(weights)
    weighted_std = np.sqrt(np.sum(weights * (data - weighted_mean) ** 2) / np.sum(weights))
    return weighted_mean, weighted_std
