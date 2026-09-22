# SPDX-License-Identifier: MIT
"""
Masked normalized cross-correlation implementation.

Based on:
D. Padfield. Masked Object Registration in the Fourier Domain.
IEEE Transactions on Image Processing (2012)

This module implements the masked normalized cross-correlation algorithm and
follows the original MATLAB reference implementation by Dirk Padfield
(see http://www.dirkpadfield.com/). Bibliographic references are preserved
in function docstrings to keep scientific attribution.
"""

from functools import partial
import nvtx
import cupy as cp
from cupyx.scipy import fft as fftmodule
from cupyx.scipy.fft import next_fast_len

from ..logger.hierarchical_logging import setup_logger, hierarchical_debug

logger = setup_logger(__name__)


### # @hierarchical_debug(logger)
@nvtx.annotate('_masked_phase_cross_correlation', category='stats.subpixel_masked')
def _masked_phase_cross_correlation(reference_image, moving_image,
                                    reference_mask, moving_mask=None, overlap_ratio=0.3):
    """
    Perform masked image translation registration using masked normalized cross-correlation.

    Parameters
    ----------
    reference_image : ndarray
        Reference image.
    moving_image : ndarray
        Image to register. May be different size from the reference but must be
        compatible when masks are provided.
    reference_mask : ndarray
        Boolean mask for the reference image (True for valid pixels).
    moving_mask : ndarray or None
        Boolean mask for the moving image. If None, reference_mask is used.
    overlap_ratio : float, optional
        Minimum allowed overlap ratio between images; results with less overlap
        will be ignored.

    Returns
    -------
    ndarray
        Shift vector (in pixels) required to align moving_image to reference_image.

    Raises
    ------
    ValueError
        If input shapes are incompatible with masks.

    References
    ----------
    .. [1] Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5), pp. 2706-2718 (2012).
           :DOI:`10.1109/TIP.2011.2181402`
    .. [2] D. Padfield. "Masked FFT registration". In Proc. CVPR (2010).
    """
    if moving_mask is None:
        if reference_image.shape != moving_image.shape:
            raise ValueError(
                'Input images have different shapes, moving_mask must be explicitly set.'
            )
        moving_mask = reference_mask.astype(bool)
    for im, mask in [(reference_image, reference_mask), (moving_image,
                                                         moving_mask)]:
        if im.shape != mask.shape:
            raise ValueError(
                'Image sizes must match their respective mask sizes.')
    xcorr = cross_correlate_masked(moving_image, reference_image,
                                   moving_mask, reference_mask, axes=tuple(range(moving_image.ndim)),
                                   mode='full', overlap_ratio=overlap_ratio)
    maxima = cp.stack(cp.nonzero(xcorr == xcorr.max()), axis=1)
    center = cp.mean(maxima, axis=0)
    shifts = center - cp.array(reference_image.shape) + 1
    size_mismatch = cp.array(moving_image.shape) - cp.array(reference_image
                                                            .shape)

    return -shifts + size_mismatch / 2


### # @hierarchical_debug(logger)
@nvtx.annotate('cross_correlate_masked', category='stats.subpixel_masked')
def cross_correlate_masked(arr1, arr2, m1, m2, mode='full', axes=(-2, -1),
                           overlap_ratio=0.3):
    """
    Compute masked normalized cross-correlation between arrays.

    Parameters
    ----------
    arr1, arr2 : ndarray
        Arrays to be correlated.
    m1, m2 : ndarray
        Boolean masks for the inputs (True = valid pixels).
    mode : {'full', 'same'}, optional
        Output mode for the correlation.
    axes : tuple, optional
        Axes along which to compute the correlation.
    overlap_ratio : float, optional
        Minimum allowed overlap ratio.

    Returns
    -------
    ndarray
        Masked normalized cross-correlation result.

    Raises
    ------
    ValueError
        If mode is invalid or non-transformation axes are not consistent.
    """
    if mode not in {'full', 'same'}:
        raise ValueError(f"Correlation mode '{mode}' is not valid.")
    fixed_image = cp.asarray(arr1)
    moving_image = cp.asarray(arr2)
    float_dtype = cp.float32
    fixed_image = fixed_image.astype(float_dtype)
    fixed_mask = cp.array(m1, dtype=bool)
    moving_image = moving_image.astype(float_dtype)
    moving_mask = cp.array(m2, dtype=bool)
    eps = cp.finfo(float_dtype).eps
    all_axes = set(range(fixed_image.ndim))
    for axis in (all_axes - set(axes)):
        if fixed_image.shape[axis] != moving_image.shape[axis]:
            raise ValueError(
                f'Array shapes along non-transformation axes should be equal, but dimensions along axis {axis} are not.'
            )
    final_shape = list(arr1.shape)
    for axis in axes:
        final_shape[axis] = fixed_image.shape[axis] + moving_image.shape[axis
        ] - 1
    final_shape = tuple(final_shape)
    final_slice = tuple([slice(0, int(sz)) for sz in final_shape])
    fast_shape = tuple([next_fast_len(final_shape[ax]) for ax in axes])
    fft = partial(fftmodule.fftn, s=fast_shape, axes=axes)
    _ifft = partial(fftmodule.ifftn, s=fast_shape, axes=axes)

    def ifft(x):
        """Helper: inverse FFT returning the real part."""
        return _ifft(x).real

    fixed_image[cp.logical_not(fixed_mask)] = 0.0
    moving_image[cp.logical_not(moving_mask)] = 0.0
    rotated_moving_image = _flip(moving_image, axes=axes)
    rotated_moving_mask = _flip(moving_mask, axes=axes)
    fixed_fft = fft(fixed_image)
    rotated_moving_fft = fft(rotated_moving_image)
    fixed_mask_fft = fft(fixed_mask.astype(float_dtype))
    rotated_moving_mask_fft = fft(rotated_moving_mask.astype(float_dtype))
    number_overlap_masked_px = ifft(rotated_moving_mask_fft * fixed_mask_fft)
    number_overlap_masked_px[:] = cp.round(number_overlap_masked_px)
    number_overlap_masked_px[:] = cp.fmax(number_overlap_masked_px, eps)
    masked_correlated_fixed_fft = ifft(rotated_moving_mask_fft * fixed_fft)
    masked_correlated_rotated_moving_fft = ifft(fixed_mask_fft *
                                                rotated_moving_fft)
    numerator = ifft(rotated_moving_fft * fixed_fft)
    numerator -= (masked_correlated_fixed_fft *
                  masked_correlated_rotated_moving_fft / number_overlap_masked_px)
    fixed_squared_fft = fft(cp.square(fixed_image))
    fixed_denom = ifft(rotated_moving_mask_fft * fixed_squared_fft)
    fixed_denom -= cp.square(masked_correlated_fixed_fft
                             ) / number_overlap_masked_px
    fixed_denom[:] = cp.fmax(fixed_denom, 0.0)
    rotated_moving_squared_fft = fft(cp.square(rotated_moving_image))
    moving_denom = ifft(fixed_mask_fft * rotated_moving_squared_fft)
    moving_denom -= cp.square(masked_correlated_rotated_moving_fft
                              ) / number_overlap_masked_px
    moving_denom[:] = cp.fmax(moving_denom, 0.0)
    denom = cp.sqrt(fixed_denom * moving_denom)
    numerator = numerator[final_slice]
    denom = denom[final_slice]
    number_overlap_masked_px = number_overlap_masked_px[final_slice]
    if mode == 'same':
        _centering = partial(_centered, newshape=fixed_image.shape, axes=axes)
        denom = _centering(denom)
        numerator = _centering(numerator)
        number_overlap_masked_px = _centering(number_overlap_masked_px)
    tol = 1000.0 * eps * cp.max(cp.abs(denom), axis=axes, keepdims=True)
    nonzero_indices = denom > tol
    out = cp.zeros_like(denom, dtype=float_dtype)
    out[nonzero_indices] = numerator[nonzero_indices] / denom[nonzero_indices]
    cp.clip(out, a_min=-1, a_max=1, out=out)
    number_px_threshold = overlap_ratio * cp.max(number_overlap_masked_px,
                                                 axis=axes, keepdims=True)
    out[number_overlap_masked_px < number_px_threshold] = 0.0

    return out


### # @hierarchical_debug(logger)
@nvtx.annotate('_flip', category='stats.subpixel_masked')
def _flip(arr, axes=None):
    """
    Reverse array over many axes. Generalization of arr[::-1] for many dimensions.

    Parameters
    ----------
    arr : ndarray
        Input array to be flipped.
    axes : tuple of ints or None, optional
        Axes over which to flip the array. If None, flips over all axes.

    Returns
    -------
    ndarray
        Flipped array.
    """

    if axes is None:
        reverse = [slice(None, None, -1)] * arr.ndim
    else:
        reverse = [slice(None, None, None)] * arr.ndim
        for axis in axes:
            reverse[axis] = slice(None, None, -1)

    return arr[tuple(reverse)]


### # @hierarchical_debug(logger)
@nvtx.annotate('_centered', category='stats.subpixel_masked')
def _centered(arr, newshape, axes):
    """
    Return the center `newshape` portion of `arr`, leaving axes not in `axes` untouched.

    Parameters
    ----------
    arr : ndarray
        Input array.
    newshape : tuple of ints
        Shape of the centered output array.
    axes : tuple of ints
        Axes along which to center the array.

    Returns
    -------
    ndarray
        Centered array.
    """
    newshape = cp.asarray(newshape)
    currshape = cp.array(arr.shape)
    slices = [slice(None, None)] * arr.ndim
    for ax in axes:
        startind = (currshape[ax] - newshape[ax]) // 2
        endind = startind + newshape[ax]
        slices[ax] = slice(startind, endind)

    return arr[tuple(slices)]
