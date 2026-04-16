# SPDX-License-Identifier: MIT
"""
Convolution utilities for gpuphot.phot.

Contains FFT-based convolution helpers, kernel generators (Gaussian and
aperture kernels), and small utilities used across the photometry pipeline.
The functions prefer CuPy arrays for GPU acceleration and fall back to NumPy
via cupynumeric where necessary.
"""

from __future__ import annotations

import cupy as cp
try:
    import cupynumeric as np
except ImportError:
    import numpy as np

import nvtx

from ..logger.hierarchical_logging import setup_logger

logger = setup_logger(__name__)


### # @hierarchical_debug(logger)
@nvtx.annotate('convolve_fft', category='phot.conv')
def convolve_fft(image: cp.ndarray, kernel: cp.ndarray, do_pad: bool = True, **kwargs) -> cp.ndarray:
    """
    Convolve an image with a kernel using FFT.

    :param image: Image array to be processed.
    :type image: cupy.ndarray
    :param kernel: Kernel to be used in the convolution.
    :type kernel: cupy.ndarray
    :param do_pad: Whether to pad the image.
    :type do_pad: bool
    :return: The convolved image.
    :rtype: cupy.ndarray
    """
    nvtx_range = nvtx.start_range('initialization', category='phot.conv', color='blue')
    image_shape = image.shape
    kernel_shape = kernel.shape
    padding = int((kernel_shape[0] - 1) / 2)
    nvtx.end_range(nvtx_range)

    if do_pad:
        nvtx_range = nvtx.start_range('padding', category='phot.conv', color='yellow')
        image = (
            cp.pad(image, pad_width=padding, mode='reflect')
        )  # NOTE: this padding can cause a large memory increase; monitor memory use when calling with large images
        nvtx.end_range(nvtx_range)

    nvtx_range = nvtx.start_range('fft_calculation', category='phot.conv', color='green')
    new_image_shape = image.shape
    F_image = cp.fft.rfft2(image, s=new_image_shape)
    F_kernel = cp.fft.rfft2(kernel, s=new_image_shape)
    nvtx.end_range(nvtx_range)

    del image, kernel

    nvtx_range = nvtx.start_range('frequency_multiplication', category='phot.conv', color='red')
    cp.multiply(F_image, F_kernel.conj(), out=F_image)
    nvtx.end_range(nvtx_range)

    del F_kernel

    nvtx_range = nvtx.start_range('inverse_fft_and_postprocessing', category='phot.conv', color='magenta')
    F_image = cp.fft.irfft2(F_image, s=new_image_shape)
    F_image = cp.roll(F_image, shift=[padding, padding], axis=[0, 1])
    nvtx.end_range(nvtx_range)

    if do_pad:
        nvtx_range = nvtx.start_range('crop_padding', category='phot.conv', color='purple')
        F_image = F_image[padding:padding + image_shape[0], padding:padding + image_shape[1]]
        nvtx.end_range(nvtx_range)

    # cp.get_default_memory_pool().free_all_blocks()

    return F_image


### # @hierarchical_debug(logger)
@nvtx.annotate('get_mean_std', category='phot.conv')
def get_mean_std(im_g: cp.ndarray, lk: int, std: bool = True, **kwargs) -> tuple:
    """
    Calculate the mean and standard deviation of an image using FFT convolution.

    :param im_g: Image array to be processed.
    :type im_g: cupy.ndarray
    :param lk: Length of the kernel.
    :type lk: int
    :param std: Whether to calculate the standard deviation.
    :type std: bool
    :return: A tuple containing the mean and standard deviation of the image.
    :rtype: tuple
    """
    im_g = cp.asarray(im_g, dtype=cp.float64)
    k_app = gen_apm_filter(lk)
    fot_m = convolve_fft(im_g, k_app, **kwargs)
    if std:
        fot_m2 = convolve_fft(im_g * im_g, k_app, **kwargs)
        # fot_m2 = cp.sqrt(fot_m2 - fot_m * fot_m)
        cp.subtract(fot_m2, fot_m * fot_m, out=fot_m2)
        cp.sqrt(fot_m2, out=fot_m2)
    else:
        fot_m2 = None
    del k_app
    return fot_m, fot_m2


### # @hierarchical_debug(logger)
@nvtx.annotate('gaussian_kernel', category='phot.conv')
def gaussian_kernel(lk: int, sigma: int, **kwargs) -> cp.ndarray:
    """
    Generate a 2D Gaussian kernel.

    :param lk: Length of the kernel.
    :type lk: int
    :param sigma: Standard deviation of the Gaussian kernel.
    :type sigma: int
    :return: A 2D Gaussian kernel.
    :rtype: cupy.ndarray
    """
    k_dim = (2 * lk + 1, 2 * lk + 1)
    x = cp.linspace(-lk, lk, k_dim[0])
    y = cp.linspace(-lk, lk, k_dim[1])
    xx, yy = cp.meshgrid(x, y)
    kernel = cp.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    del x, y, xx, yy
    return kernel / cp.sum(kernel)


# @nvtx.annotate('get_aper_kernel', category='phot.conv')
# def get_aper_kernel(radius: int, size: int = None, **kwargs) -> tuple:
#     """
#     Generate a circular kernel for aperture photometry.
#
#     :param radius: Radius of the circular kernel.
#     :type radius: int
#     :param size: Size of the kernel. Default is 2*radius+1.
#     :type size: int or None
#     :return: A circular kernel and the area of the kernel.
#     :rtype: tuple
#     """
#     # if size is None:
#     #     size = 2 * radius + 1
#     # kernel = cp.zeros((size, size))
#     # y, x = cp.indices(kernel.shape)
#     # mask = (x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2 <= radius ** 2
#     # kernel[mask] = 1
#     # area = cp.sum(kernel)
#     # return kernel, area
#
#     if size is None:
#         size = 2 * radius + 1
#     center = (size - 1) / 2.0
#     y, x = cp.indices((size, size), dtype=cp.float64)
#     mask = (x - center) ** 2 + (y - center) ** 2 <= float(radius) ** 2
#     kernel = mask.astype(cp.float64)
#     area = cp.sum(kernel)
#     return kernel, area
@nvtx.annotate('get_aper_kernel', category='phot.conv')
def get_aper_kernel(radius: float, size: int) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Generate a circular kernel for aperture photometry using CuPy.
    Radius can be float. Size determines the array dimensions.

    :param radius: Radius of the circular kernel.
    :type radius: float
    :param size: Size of the kernel array (must be odd).
    :type size: int
    :return: A circular kernel (cupy array) and the area (sum of kernel, cupy scalar).
    :rtype: tuple(cupy.ndarray, cupy.ndarray)
    """
    if size % 2 == 0:
        # FFT convolution kernels often work best with odd sizes for centering
        # Or adjust center calculation if even size is needed. Assume odd for simplicity.
        raise ValueError("Kernel size must be odd for simple centering.")
    center = (size - 1) / 2.0
    y, x = cp.indices((size, size), dtype=cp.float64)
    # Use squared radius comparison
    mask = (x - center) ** 2 + (y - center) ** 2 <= float(radius) ** 2
    kernel = mask.astype(cp.float64)  # Use float64 for precision if needed
    area = cp.sum(kernel)
    # Normalize kernel? Usually not for aperture sum, but depends on convention. Assume sum=area.
    return kernel, area


@nvtx.annotate('fill_image', category='phot.conv')
def fill_image(image_shape: tuple) -> tuple[int, int]:
    """
    Calculate the new image shape rounding up to the next power of 2.

    :param image_shape: Original image shape.
    :type image_shape: tuple
    :return: A tuple containing the new height and width values.
    :rtype: tuple
    """
    h, w = image_shape
    new_height = 2 ** int(np.ceil(np.log2(h)))
    new_width = 2 ** int(np.ceil(np.log2(w)))
    return new_height, new_width


### # @hierarchical_debug(logger)
@nvtx.annotate('gen_apm_filter', category='phot.conv')
def gen_apm_filter(lk: int, li: int = 0, norm: bool = True, **kwargs) -> cp.ndarray:
    """
    Generate an aperture filter for the detection of sources in an image.

    :param lk: Radius of the kernel.
    :type lk: int
    :param li: Radius of the inner kernel.  Default is 0 (no inner kernel).
    :type li: int
    :param norm: Whether to normalize the kernel.
    :type norm: bool
    :return: An aperture filter.
    :rtype: cupy.ndarray
    """

    k_dim = (2 * lk + 1, 2 * lk + 1)
    indi = cp.indices(k_dim, dtype=cp.float64)  # Use float64 for consistency
    dist_sq = (lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2

    # Create the outer circle mask directly
    mask_outer = dist_sq <= lk ** 2

    # Create the inner circle mask (if li != 0) and combine
    if li != 0:
        mask_inner = dist_sq < (li ** 2)
        mask = mask_outer & ~mask_inner  # Combine using boolean logic
    else:
        mask = mask_outer

    # Create the kernel from the combined mask
    k_app = mask.astype(cp.float64)

    if norm:
        kernel_sum = cp.sum(k_app)  # Calculate sum only if normalization is needed
        if kernel_sum != 0:
            k_app = k_app / kernel_sum

    return k_app


### # @hierarchical_debug(logger)
@nvtx.annotate('batch_aper_kernel', category='phot.conv')
def batch_aper_kernel(radius, **kwargs):
    """
    Generate a batch of aperture kernels.

    :param radius: Radius or list of radii for the kernels.
    :type radius: int or list
    :return: A tuple containing the kernel and its area.
    :rtype: tuple
    """
    # Ensure radius is a CuPy array for efficient calculations
    if isinstance(radius, list) or isinstance(radius, np.ndarray):
        radius = cp.array(radius, dtype=cp.float64)  # Convert list/np.ndarray to cp.ndarray
    elif isinstance(radius, int):
        radius = cp.array([radius], dtype=cp.float64)  # Convert to array
    elif not isinstance(radius, cp.ndarray):
        raise TypeError("radius must be an int, list, or NumPy/CuPy array")

    # Handle single radius case to avoid issues with indexing
    if radius.ndim == 0:  # Scalar case
        size = int(2 * radius + 1)
        center = float(radius)
        y, x = cp.indices((size, size), dtype=cp.float64)
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        kernel = mask.astype(cp.float64)
        area = cp.sum(kernel)

    else:  # radius is an array
        max_radius = int(cp.max(radius))  # Find the maximum radius
        size = 2 * max_radius + 1
        center = float(max_radius)
        y, x = cp.indices((size, size), dtype=cp.float64)

        # Broadcasting to create masks for all radii at once
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius.reshape(-1, 1,
                                                                       1) ** 2  # radius[:, None, None] is also valid

        # The entire mask array serves as the kernel (no need for cp.zeros)
        kernel = mask.astype(cp.float64)  # Convert boolean mask to float64
        area = cp.sum(kernel, axis=(1, 2))  # Sum across rows and columns for each kernel

    return kernel, area


### # @hierarchical_debug(logger)
@nvtx.annotate('fill_nan_fft', category='phot.conv')
def fill_nan_fft(image: cp.ndarray, lk: int, li: int = 0, min_neighbors: int = 5, **kwargs) -> cp.ndarray:
    """
    Fill NaN values in an image using FFT convolution.

    :param image: Image array to be processed.
    :type image: cupy.ndarray
    :param lk: Length of the kernel.
    :type lk: int
    :param li: Length of the inner kernel.
    :type li: int
    :param min_neighbors: Minimum number of valid neighbors.
    :type min_neighbors: int
    :return: The image array with NaN values filled.
    :rtype: cupy.ndarray
    """
    k_app = gen_apm_filter(lk, li=li, norm=False)
    image = image.astype(cp.double)
    not_nan_mask = (~cp.isnan(image)).astype(cp.double)
    valid_neighbors = convolve_fft(not_nan_mask, k_app, **kwargs)
    del not_nan_mask
    image_zeroed = cp.where(cp.isnan(image), 0, image)
    neighbor_sum = convolve_fft(image_zeroed, k_app, **kwargs)
    del image_zeroed, k_app
    result = cp.where((valid_neighbors >= min_neighbors) & (cp.isnan(image)), neighbor_sum / valid_neighbors, image)
    del valid_neighbors, neighbor_sum
    return result
