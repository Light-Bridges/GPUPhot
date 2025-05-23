"""
Port of Manuel Guizar's code from:
https://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation
Cupyfication from skimage/registration/_phase_cross_correlation.py
"""
import cupy as cp
import numpy as np
from cupy.fft import fftn, ifftn, fftfreq
import nvtx
from .subpixel_masked import _masked_phase_cross_correlation
from ..logger.hierarchical_logging import setup_logger, hierarchical_debug

logger = setup_logger(__name__)


### # @hierarchical_debug(logger)
@nvtx.annotate('phase_cross_correlation',category='stats.subpixel')
def phase_cross_correlation(reference_image, moving_image, *,
                            upsample_factor=100, space='real', return_error=True, reference_mask=None,
                            moving_mask=None, overlap_ratio=0.3, normalization='phase'):
    """
     Perform efficient subpixel image translation registration by cross-correlation.

    This function provides the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.[1]_.


    :param reference_image: Reference image.
    :type reference_image: array
    :param moving_image: Image to register. Must be same dimensionality as ``reference_image``.
    :type moving_image: array
    :param upsample_factor: Upsampling factor. Images will be registered to within ``1 / upsample_factor`` of a pixel.
                            Default is 100 (upsampling). Not used if any of ``reference_mask`` or ``moving_mask`` is not None.
    :type upsample_factor: int, optional
    :param space: Defines how the algorithm interprets input data. "real" means data will be FFT'd to compute the correlation,
                  while "fourier" data will bypass FFT of input data. Case insensitive. Not used if any of ``reference_mask``
                  or ``moving_mask`` is not None.
    :type space: str, optional
    :param return_error: Returns error and phase difference if True, otherwise only shifts are returned.
                         Has no effect if any of ``reference_mask`` or ``moving_mask`` is not None.
    :type return_error: bool, optional
    :param reference_mask: Boolean mask for ``reference_image``. The mask should evaluate to ``True`` (or 1) on valid pixels.
                           ``reference_mask`` should have the same shape as ``reference_image``.
    :type reference_mask: ndarray, optional
    :param moving_mask: Boolean mask for ``moving_image``. The mask should evaluate to ``True`` (or 1) on valid pixels.
                        ``moving_mask`` should have the same shape as ``moving_image``. If ``None``, ``reference_mask`` will be used.
    :type moving_mask: ndarray or None, optional
    :param overlap_ratio: Minimum allowed overlap ratio between images. The correlation for translations corresponding with an overlap
                          ratio lower than this threshold will be ignored. Used only if one of ``reference_mask`` or ``moving_mask`` is not None.
    :type overlap_ratio: float, optional
    :param normalization: The type of normalization to apply to the cross-correlation. This parameter is unused when masks are provided.
    :type normalization: {"phase", None}, optional
    :return: A tuple containing:
             - shifts: Shift vector (in pixels) required to register ``moving_image`` with ``reference_image``.
             - error: Registration error (if ``return_error=True``)
             - phasediff: Global phase difference between the two images (if ``return_error=True``)
    :rtype: tuple
    :raises ValueError: If images are not the same shape, or if ``space`` is not "real" or "fourier",
                        or if ``normalization`` is not "phase" or None, or if NaN values are found in the input data.


    Notes
    -----
    The use of cross-correlation to estimate image translation has a long
    history dating back to at least [2]_. The "phase correlation"
    method (selected by ``normalization="phase"``) was first proposed in [3]_.
    Publications [1]_ and [2]_ use an unnormalized cross-correlation
    (``normalization=None``). Which form of normalization is better is
    application-dependent. For example, the phase correlation method works
    well in registering images under different illumination, but is not very
    robust to noise. In a high noise scenario, the unnormalized method may be
    preferable.
    When masks are provided, a masked normalized cross-correlation algorithm is
    used [5]_, [6]_.
    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008). :DOI:`10.1364/OL.33.000156`
    .. [2] P. Anuta, Spatial registration of multispectral and multitemporal
           digital imagery using fast Fourier transform techniques, IEEE Trans.
           Geosci. Electron., vol. 8, no. 4, pp. 353–368, Oct. 1970.
           :DOI:`10.1109/TGE.1970.271435`.
    .. [3] C. D. Kuglin D. C. Hines. The phase correlation image alignment
           method, Proceeding of IEEE International Conference on Cybernetics
           and Society, pp. 163-165, New York, NY, USA, 1975, pp. 163–165.
    .. [4] James R. Fienup, "Invariant error metrics for image reconstruction"
           Optics Letters 36, 8352-8357 (1997). :DOI:`10.1364/AO.36.008352`
    .. [5] Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5),
           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
    .. [6] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
           Pattern Recognition, pp. 2918-2925 (2010).
           :DOI:`10.1109/CVPR.2010.5540032`
    """
    if reference_mask is not None or moving_mask is not None:
        return _masked_phase_cross_correlation(reference_image,
                                               moving_image, reference_mask, moving_mask, overlap_ratio), 0, 0
    if reference_image.shape != moving_image.shape:
        raise ValueError('images must be same shape')
    if space.lower() == 'fourier':
        src_freq = reference_image
        target_freq = moving_image
    elif space.lower() == 'real':
        imr = cp.asarray(reference_image, dtype=cp.float32)
        imv = cp.asarray(moving_image, dtype=cp.float32)
        src_freq = fftn(cp.asarray(imr))
        target_freq = fftn(cp.asarray(imv))
        del (imr, imv)
    else:

        raise ValueError('space argument must be "real" of "fourier"')
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if normalization == 'phase':
        eps = cp.finfo(image_product.real.dtype).eps
        image_product /= np.maximum(np.abs(image_product), 100 * eps)
    elif normalization is not None:

        raise ValueError('normalization must be either phase or None')
    cross_correlation = ifftn(image_product)
    maxima = cp.unravel_index(np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape)
    midpoints = cp.array([cp.fix(axis_size / 2) for axis_size in shape])
    float_dtype = image_product.real.dtype
    shifts = cp.stack(maxima).astype(float_dtype, copy=False)
    shifts[shifts > midpoints] -= cp.array(shape)[shifts > midpoints]
    if upsample_factor == 1:
        if return_error:
            src_amp = cp.sum(np.real(src_freq * src_freq.conj()))
            src_amp /= src_freq.size
            target_amp = cp.sum(cp.real(target_freq * target_freq.conj()))
            target_amp /= target_freq.size
            CCmax = cross_correlation[maxima]
    else:
        upsample_factor = cp.array(upsample_factor, dtype=float_dtype)
        shifts = cp.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = cp.ceil(upsample_factor * 1.5)
        dftshift = cp.fix(upsampled_region_size / 2.0)
        sample_region_offset = dftshift - shifts * upsample_factor
        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size, upsample_factor, sample_region_offset).conj(
        )
        maxima = cp.unravel_index(cp.argmax(cp.abs(cross_correlation)),
                                  cross_correlation.shape)
        CCmax = cross_correlation[maxima]
        maxima = cp.stack(maxima).astype(float_dtype, copy=False)
        maxima -= dftshift
        shifts += maxima / upsample_factor
        if return_error:
            src_amp = cp.sum(np.real(src_freq * src_freq.conj()))
            target_amp = cp.sum(np.real(target_freq * target_freq.conj()))

    del target_freq, image_product
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0
    if return_error:
        if np.isnan(CCmax) or np.isnan(src_amp) or np.isnan(target_amp):
            raise ValueError(
                'NaN values found, please remove NaNs from your input data or use the `reference_mask`/`moving_mask` keywords, eg: phase_cross_correlation(reference_image, moving_image, reference_mask=~np.isnan(reference_image), moving_mask=~np.isnan(moving_image))'
            )

        return shifts, _compute_error(CCmax, src_amp, target_amp
                                      ), _compute_phasediff(CCmax)
    else:

        return shifts, 0, 0


### # @hierarchical_debug(logger)
@nvtx.annotate('_upsampled_dft',category='stats.subpixel')
def _upsampled_dft(data, upsampled_region_size, upsample_factor=1,
                   axis_offsets=None):
    """
    Perform an upsampled Discrete Fourier Transform (DFT) by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.


    This function computes the DFT in the output array without the need to zero-pad,
    achieving faster and more memory-efficient results compared to a zero-padded FFT
    approach when the upsampled region size is much smaller than ``data.size * upsample_factor``.

    :param data: The input data array (DFT of original data) to upsample.
    :type data: cupy.ndarray
    :param upsampled_region_size: The size of the region to be sampled. If one integer is provided,
                                   it is duplicated up to the dimensionality of ``data``.
    :type upsampled_region_size: int or tuple of int
    :param upsample_factor: The upsampling factor. Defaults to 1.
    :type upsample_factor: int, optional
    :param axis_offsets: Offsets for each axis in the output region. Defaults to None.
                         If None, offsets are set to 0 for all axes.
    :type axis_offsets: tuple of int, optional
    :return: The upsampled DFT result.
    :rtype: cupy.ndarray
    :raises ValueError: If the number of axis offsets does not match the number of dimensions in ``data``.
    """
    upsampled_region_size = [upsampled_region_size] * data.ndim
    if axis_offsets is None:
        axis_offsets = [0] * data.ndim
    elif len(axis_offsets) != data.ndim:
        raise ValueError(
            "number of axis offsets must be equal to input data's number of dimensions."
        )
    im2pi = 1.0j * 2 * np.pi
    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))
    for n_items, ups_size, ax_offset in dim_properties[::-1]:
        kernel = (cp.arange(ups_size) - ax_offset)[:, None] * fftfreq(n_items,
                                                                      upsample_factor)
        kernel = cp.exp(-im2pi * kernel)
        kernel = kernel.astype(data.dtype, copy=False)
        data = cp.tensordot(kernel, data, axes=(1, -1))

    return data


### # @hierarchical_debug(logger)
@nvtx.annotate('_compute_error',category='stats.subpixel')
def _compute_error(cross_correlation_max, src_amp, target_amp):
    """
    Compute the RMS error metric between two images based on their cross-correlation.

    :param cross_correlation_max: The complex value of the cross-correlation at its maximum point.
    :type cross_correlation_max: complex
    :param src_amp: The normalized average image intensity of the source image.
    :type src_amp: float
    :param target_amp: The normalized average image intensity of the target image.
    :type target_amp: float
    :return: The computed RMS error metric.
    :rtype: cupy.ndarray
    """
    error = 1.0 - cross_correlation_max * cross_correlation_max.conj() / (
            src_amp * target_amp)

    return cp.sqrt(np.abs(error))

@nvtx.annotate('_compute_phasediff',category='stats.subpixel')
def _compute_phasediff(cross_correlation_max):
    """
    Compute the global phase difference between two images.

    This value should be zero if both images are non-negative.

    :param cross_correlation_max: The complex value of the cross-correlation at its maximum point.
    :type cross_correlation_max: complex
    :return: The global phase difference in radians.
    :rtype: cupy.ndarray
    """

    return cp.arctan2(cross_correlation_max.imag, cross_correlation_max.real)
