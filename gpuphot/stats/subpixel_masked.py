"""
Implementation of the masked normalized cross-correlation.
Based on the following publication:
D. Padfield. Masked object registration in the Fourier domain.
IEEE Transactions on Image Processing (2012)
and the author's original MATLAB implementation, available on this website:
http://www.dirkpadfield.com/
"""
from functools import partial

import cupy as cp
from cupyx.scipy import fft as fftmodule
from cupyx.scipy.fft import next_fast_len

from ..logger.hierarchical_logging import setup_logger, hierarchical_debug

logger = setup_logger(__name__)


@hierarchical_debug(logger)
def _masked_phase_cross_correlation(reference_image, moving_image,
                                    reference_mask, moving_mask=None, overlap_ratio=0.3):
    """
    Perform masked image translation registration by masked normalized cross-correlation.

    :param reference_image: Reference image.
    :type reference_image: ndarray
    :param moving_image: Image to register. Must be the same dimensionality as `reference_image`,
                         but not necessarily the same size.
    :type moving_image: ndarray
    :param reference_mask: Boolean mask for `reference_image`. The mask should evaluate to `True`
                           (or 1) on valid pixels. `reference_mask` should have the same shape as `reference_image`.
    :type reference_mask: ndarray
    :param moving_mask: Boolean mask for `moving_image`. The mask should evaluate to `True`
                        (or 1) on valid pixels. `moving_mask` should have the same shape
                        as `moving_image`. If `None`, `reference_mask` will be used.
    :type moving_mask: ndarray or None, optional
    :param overlap_ratio: Minimum allowed overlap ratio between images. The correlation for
                          translations corresponding with an overlap ratio lower than this
                          threshold will be ignored.
    :type overlap_ratio: float, optional
    :return: Shift vector (in pixels) required to register `moving_image`
             with `reference_image`. Axis ordering is consistent with
             numpy (e.g. Z, Y, X).
    :rtype: ndarray
    :raises ValueError: If input images have different shapes and moving_mask is not explicitly set,
                        or if image sizes don't match their respective mask sizes.


    References
    ----------
    .. [1] Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5),
           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
    .. [2] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
           Pattern Recognition, pp. 2918-2925 (2010).
           :DOI:`10.1109/CVPR.2010.5540032`
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


@hierarchical_debug(logger)
def cross_correlate_masked(arr1, arr2, m1, m2, mode='full', axes=(-2, -1),
                           overlap_ratio=0.3):
    """
    Compute masked normalized cross-correlation between arrays.

    :param arr1: First array.
    :type arr1: ndarray
    :param arr2: Second array. The dimensions of `arr2` along axes that are not
                 transformed should be equal to that of `arr1`.
    :type arr2: ndarray
    :param m1: Mask of `arr1`. The mask should evaluate to `True`
               (or 1) on valid pixels. `m1` should have the same shape as `arr1`.
    :type m1: ndarray
    :param m2: Mask of `arr2`. The mask should evaluate to `True`
               (or 1) on valid pixels. `m2` should have the same shape as `arr2`.
    :type m2: ndarray
    :param mode: {'full', 'same'}, optional
                 'full': Returns the convolution at each point of overlap.
                 'same': The output is the same size as `arr1`, centered with respect
                         to the 'full' output.
    :type mode: str, optional
    :param axes: Axes along which to compute the cross-correlation.
    :type axes: tuple of ints, optional
    :param overlap_ratio: Minimum allowed overlap ratio between images.
    :type overlap_ratio: float, optional
    :return: Masked normalized cross-correlation.
    :rtype: ndarray
    :raises ValueError: If correlation `mode` is not valid, or array dimensions along
                        non-transformation axes are not equal.

    References
    ----------
    .. [1] Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5),
           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
    .. [2] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
           Pattern Recognition, pp. 2918-2925 (2010).
           :DOI:`10.1109/CVPR.2010.5540032`
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
        """

        :param x: 

        """
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


@hierarchical_debug(logger)
def _flip(arr, axes=None):
    """
    Reverse array over many axes. Generalization of arr[::-1] for many dimensions.

    :param arr: Input array to be flipped.
    :type arr: ndarray
    :param axes: Axes over which to flip the array. If None, flips over all axes.
    :type axes: tuple of ints or None, optional
    :return: Flipped array.
    :rtype: ndarray
    """

    if axes is None:
        reverse = [slice(None, None, -1)] * arr.ndim
    else:
        reverse = [slice(None, None, None)] * arr.ndim
        for axis in axes:
            reverse[axis] = slice(None, None, -1)

    return arr[tuple(reverse)]


@hierarchical_debug(logger)
def _centered(arr, newshape, axes):
    """
    Return the center `newshape` portion of `arr`, leaving axes not in `axes` untouched.

    :param arr: Input array.
    :type arr: ndarray
    :param newshape: Shape of the centered output array.
    :type newshape: tuple of ints
    :param axes: Axes along which to center the array.
    :type axes: tuple of ints
    :return: Centered array.
    :rtype: ndarray
    """
    newshape = cp.asarray(newshape)
    currshape = cp.array(arr.shape)
    slices = [slice(None, None)] * arr.ndim
    for ax in axes:
        startind = (currshape[ax] - newshape[ax]) // 2
        endind = startind + newshape[ax]
        slices[ax] = slice(startind, endind)

    return arr[tuple(slices)]
