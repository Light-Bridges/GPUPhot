import cupy as cp
import numpy as np

from ..logger.hierarchical_logging import setup_logger, hierarchical_debug

logger = setup_logger(__name__)


@hierarchical_debug(logger)
def convolve_fft(image: cp.ndarray, kernel: cp.ndarray, **kwargs) -> cp.ndarray:
    """Convolve an image with a kernel using FFT.

    :param image: Image array to be processed.
    :param kernel: Kernel to be used in the convolution.
    :return: The convolved image.
    """
    image_shape = image.shape
    kernel_shape = kernel.shape
    padding = int((kernel_shape[0] - 1) / 2)

    do_pad = kwargs.get('do_pad', True)

    if do_pad: image = cp.pad(image, pad_width=padding,
                              mode='reflect')  # esto está provocando un aumento terrible de memoria
    new_image_shape = image.shape
    F_image = cp.fft.rfft2(image, s=new_image_shape)
    F_kernel = cp.fft.rfft2(kernel, s=new_image_shape)
    F_kernel = cp.conj(F_kernel)
    convolved = F_image * F_kernel
    convolved = cp.fft.irfft2(convolved, s=new_image_shape)
    convolved = cp.roll(convolved, shift=[padding, padding], axis=[0, 1])
    if do_pad: convolved = convolved[padding:padding + image_shape[0], padding:padding + image_shape[1]]
    del F_image, F_kernel
    return convolved


@hierarchical_debug(logger)
def get_mean_std(im_g: cp.ndarray, lk: int, std: bool = True, **kwargs) -> tuple:
    """Calculates the mean and standard deviation of an image using FFT convolution.

    :param im_g: Image array to be processed.
    :param lk: Length of the kernel.
    :param std: Whether to calculate the standard deviation.
    :return: A tuple containing the mean and standard deviation of the image.
    """
    im_g = cp.asarray(im_g, dtype=cp.float64)
    k_app = gen_apm_filter(lk)
    fot_m = convolve_fft(im_g, k_app, **kwargs)
    if std:
        fot_m2 = convolve_fft(im_g * im_g, k_app, **kwargs)
        fot_m2 = cp.sqrt(fot_m2 - fot_m * fot_m)
    else:
        fot_m2 = None
    del k_app
    return fot_m, fot_m2


@hierarchical_debug(logger)
def gaussian_kernel(lk: int, sigma: int, **kwargs) -> cp.ndarray:
    """Generates a 2D Gaussian kernel.

    :param lk: Length of the kernel.
    :param sigma: Standard deviation of the Gaussian kernel.
    :return: A 2D Gaussian kernel.
    """
    k_dim = (2 * lk + 1, 2 * lk + 1)
    x = cp.linspace(-lk, lk, k_dim[0])
    y = cp.linspace(-lk, lk, k_dim[1])
    xx, yy = cp.meshgrid(x, y)
    kernel = cp.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    del x, y, xx, yy
    return kernel / cp.sum(kernel)


@hierarchical_debug(logger)
def get_aper_kernel(radius: int, size: int = None, **kwargs) -> tuple:
    """Generates a circular kernel for aperture photometry.

    :param radius: Radius of the circular kernel.
    :param size: Size of the kernel. Default is 2*radius+1.
    :return: A circular kernel and the area of the kernel.
    """
    if size is None:
        size = 2 * radius + 1
    kernel = cp.zeros((size, size))
    y, x = cp.indices(kernel.shape)
    mask = (x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2 <= radius ** 2
    kernel[mask] = 1
    area = cp.sum(kernel)
    return kernel, area


@hierarchical_debug(logger)
def fill_image(image_shape: tuple, **kwargs) -> tuple:
    """Calculates the new image shape rounding up to the next power of 2.

    :param image_shape: Original image shape.
    :return: A tuple containing the new height and width values.
    """
    h, w = image_shape
    new_height = 2 ** int(np.ceil(np.log2(h)))
    new_width = 2 ** int(np.ceil(np.log2(w)))
    return new_height, new_width


@hierarchical_debug(logger)
def gen_apm_filter(lk: int, li: int = 0, norm: bool = True, **kwargs) -> cp.ndarray:
    """Generates an aperture filter for the detection of sources in an image.

    :param lk: Length of the kernel.
    :param li: Length of the inner kernel. Default is 0 (no inner kernel).
    :param norm: Whether to normalize the kernel.
    :return: An aperture filter.
    """
    k_dim = (2 * lk + 1, 2 * lk + 1)
    indi = cp.indices(k_dim)
    fw2 = lk ** 2
    k_app = cp.zeros(k_dim)
    dist_sq = (lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2
    struc = cp.where(dist_sq <= fw2)
    k_app[struc] = 1

    # if li is not 0, it creates a circle with radius li and sets the values inside to 0
    if li != 0:
        struc_inner = cp.where(dist_sq < (li ** 2))
        k_app[struc_inner] = 0
    if norm and k_app.sum() != 0: k_app = k_app / k_app.sum()
    return k_app


@hierarchical_debug(logger)
def batch_aper_kernel(radius, **kwargs):
    """

    :param radius: 

    """

    kernel = cp.zeros((2 * radius[-1] + 1, 2 * radius[-1] + 1))
    y, x = cp.indices(kernel.shape)
    mask = (x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2
    kernel[mask] = 1
    area = cp.sum(kernel)

    return kernel, area


@hierarchical_debug(logger)
def fill_nan_fft(image: cp.ndarray, lk: int, li: int = 0, min_neighbors: int = 5, **kwargs) -> cp.ndarray:
    """Fills NaN values in an image using FFT convolution.

    :param image: Image array to be processed.
    :param lk: Length of the kernel.
    :param li: Length of the inner kernel.
    :param min_neighbors: Minimum number of valid neighbors.
    :return: The image array with NaN values filled.
    """
    k_app = gen_apm_filter(lk, li=li, norm=False)
    image = image.astype(cp.double)
    not_nan_mask = (~cp.isnan(image)).astype(cp.double)
    valid_neighbors = convolve_fft(not_nan_mask, k_app, **kwargs)
    del not_nan_mask
    image_zeroed = cp.where(cp.isnan(image), 0, image)
    neighbor_sum = convolve_fft(image_zeroed, k_app, **kwargs)
    del image_zeroed
    result = cp.where((valid_neighbors >= min_neighbors) & (cp.isnan(image)), neighbor_sum / valid_neighbors, image)
    del valid_neighbors
    return result
