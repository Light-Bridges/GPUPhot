import logging
import time

import cupy as cp
import numpy as np

from ..logger.hierarchical_logging import setup_logger, hierarchical_debug
logger = setup_logger(__name__)

@hierarchical_debug(logger)
def convolve_fft(image, kernel):
    logger.debug(
        f'Iniciando función convolve_fft(image={image}, kernel={kernel})')
    start_time = time.time()
    """
    Perform convolution of an image with a kernel using FFT.

    Parameters
    ----------
    image : ndarray
        Input image.
    kernel : ndarray
        Convolution kernel.

    Returns
    -------
    ndarray
        Convolved image.
    """
    image_shape = image.shape
    kernel_shape = kernel.shape
    padding = int((kernel_shape[0] - 1) / 2)
    new_image_shape = fill_image((image_shape[0] + 2 * padding, image_shape
    [1] + 2 * padding))
    padded_image = cp.pad(image, ((padding, padding), (padding, padding)))
    padded_kernel = cp.pad(kernel, ((0, new_image_shape[0] - kernel_shape[0
    ]), (0, new_image_shape[1] - kernel_shape[1])))
    F_image = cp.fft.rfft2(padded_image, s=new_image_shape)
    F_kernel = cp.fft.rfft2(padded_kernel, s=new_image_shape)
    F_convolved = F_image * F_kernel
    convolved = cp.fft.irfft2(F_convolved, s=new_image_shape)
    convolved = convolved[padding:padding + image_shape[0], padding:padding +
                                                                    image_shape[1]]
    logger.debug(
        f'Función convolve_fft completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return convolved

@hierarchical_debug(logger)
def get_mean_std(im_g, lk, std=True):
    logger.debug(
        f'Iniciando función get_mean_std(im_g={im_g}, lk={lk}, std={std})')
    start_time = time.time()
    """
    Calculate the mean and standard deviation of an image using a Gaussian kernel.

    Parameters
    ----------
    im_g : ndarray
        Input image.
    lk : int
        Kernel size.
    std : bool, optional
        If True, calculate the standard deviation, by default True.

    Returns
    -------
    tuple
        (mean, std) where mean is the mean image and std is the standard deviation image.
    """
    k_app = gen_apm_filter(lk)
    fot_m = convolve_fft(im_g, k_app)
    if std:
        fot_m2 = convolve_fft(im_g * im_g, k_app)
        fot_m2 = cp.sqrt(fot_m2 - fot_m * fot_m)
    else:
        fot_m2 = None
    del k_app
    logger.debug(
        f'Función get_mean_std completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return fot_m, fot_m2

@hierarchical_debug(logger)
def gaussian_kernel(lk, sigma):
    logger.debug(f'Iniciando función gaussian_kernel(lk={lk}, sigma={sigma})')
    start_time = time.time()
    """
    Generate a Gaussian kernel.

    Parameters
    ----------
    lk : int
        Kernel size.
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    ndarray
        Gaussian kernel.
    """
    k_dim = 2 * lk + 1, 2 * lk + 1
    x = cp.linspace(-lk, lk, k_dim[0])
    y = cp.linspace(-lk, lk, k_dim[1])
    xx, yy = cp.meshgrid(x, y)
    kernel = cp.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    del x, y, xx, yy
    logger.debug(
        f'Función gaussian_kernel completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return kernel / cp.sum(kernel)

@hierarchical_debug(logger)
def get_aper_kernel(radius, size=None):
    logger.debug(
        f'Iniciando función get_aper_kernel(radius={radius}, size={size})')
    start_time = time.time()
    """
    Generate an aperture kernel.

    Parameters
    ----------
    radius : int
        Radius of the aperture.
    size : int, optional
        Size of the kernel, by default None.

    Returns
    -------
    tuple
        (kernel, area) where kernel is the aperture kernel and area is the area of the aperture.
    """
    if size is None:
        size = 2 * radius + 1
    kernel = cp.zeros((size, size))
    y, x = cp.indices(kernel.shape)
    mask = (x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2 <= radius ** 2
    kernel[mask] = 1
    area = cp.sum(kernel)
    logger.debug(
        f'Función get_aper_kernel completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return kernel, area

@hierarchical_debug(logger)
def fill_image(image_shape):
    logger.debug(f'Iniciando función fill_image(image_shape={image_shape})')
    start_time = time.time()
    """
    Calculate the shape for zero-padding an image to the next power of 2.

    Parameters
    ----------
    image_shape : tuple of int
        Shape of the input image.

    Returns
    -------
    tuple of int
        Shape of the padded image.
    """
    h, w = image_shape
    new_height = 2 ** int(np.ceil(np.log2(h)))
    new_width = 2 ** int(np.ceil(np.log2(w)))
    logger.debug(
        f'Función fill_image completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return new_height, new_width

@hierarchical_debug(logger)
def gen_apm_filter(lk, li=0, norm=True):
    logger.debug(
        f'Iniciando función gen_apm_filter(lk={lk}, li={li}, norm={norm})')
    start_time = time.time()
    """
    Generate an aperture mask filter.

    Parameters
    ----------
    lk : int
        Kernel size.
    li : int, optional
        Inner radius to exclude from the mask, by default 0.
    norm : bool, optional
        If True, normalize the kernel, by default True.

    Returns
    -------
    ndarray
        Aperture mask filter.
    """
    k_dim = 2 * lk + 1, 2 * lk + 1
    indi = cp.indices(k_dim)
    fw2 = lk ** 2
    k_app = cp.zeros(k_dim)
    struc = cp.where((lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2 <
                     fw2)
    k_app[struc] = 1

    if li != 0:
        struc = cp.where((lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) **
                         2 < fw2 - li ** 2)
        k_app[struc] = 0
    if norm:
        k_app = k_app / k_app.sum()
    logger.debug(
        f'Función gen_apm_filter completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return k_app

@hierarchical_debug(logger)
def batch_aper_kernel(radius):
    logger.debug(f'Iniciando función batch_aper_kernel(radius={radius})')
    start_time = time.time()
    kernel = cp.zeros((2 * radius[-1] + 1, 2 * radius[-1] + 1))
    y, x = cp.indices(kernel.shape)
    mask = (x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2
    kernel[mask] = 1
    area = cp.sum(kernel)
    logger.debug(
        f'Función batch_aper_kernel completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return kernel, area
