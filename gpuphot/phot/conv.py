import cupy as cp
import numpy as np

from ..logger.hierarchical_logging import setup_logger, hierarchical_debug

logger = setup_logger(__name__)


@hierarchical_debug(logger)
def convolve_fft(image: cp.ndarray, kernel: cp.ndarray) -> cp.ndarray:
    """Convolve an image with a kernel using FFT.

    :param image: Image array to be processed.
    :param kernel: Kernel to be used in the convolution.
    :return: The convolved image.
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
    del F_image, F_kernel
    convolved = cp.fft.irfft2(F_convolved, s=new_image_shape)
    convolved = convolved[padding:padding + image_shape[0], padding:padding +
                                                                    image_shape[1]]

    return convolved


@hierarchical_debug(logger)
def get_mean_std(im_g: cp.ndarray, lk: int, std: bool = True) -> tuple:
    """Calculates the mean and standard deviation of an image using FFT convolution.

    :param im_g: Image array to be processed.
    :param lk: Length of the kernel.
    :param std: Whether to calculate the standard deviation.
    :return: A tuple containing the mean and standard deviation of the image.
    """
    k_app = gen_apm_filter(lk)
    fot_m = convolve_fft(im_g, k_app)
    if std:
        fot_m2 = convolve_fft(im_g * im_g, k_app)
        fot_m2 = cp.sqrt(fot_m2 - fot_m * fot_m)
    else:
        fot_m2 = None
    del k_app
    return fot_m, fot_m2


@hierarchical_debug(logger)
def gaussian_kernel(lk: int, sigma: int) -> cp.ndarray:
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
def get_aper_kernel(radius: int, size: int = None) -> tuple:
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
def fill_image(image_shape: tuple) -> tuple:
    """Calculates the new image shape rounding up to the next power of 2.

    :param image_shape: Original image shape.
    :return: A tuple containing the new height and width values.
    """
    h, w = image_shape
    new_height = 2 ** int(np.ceil(np.log2(h)))
    new_width = 2 ** int(np.ceil(np.log2(w)))
    return new_height, new_width


@hierarchical_debug(logger)
def gen_apm_filter(lk: int, li: int = 0, norm: bool = True) -> cp.ndarray:
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
    struc = cp.where(((lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2) < fw2)
    k_app[struc] = 1

    # if li is not 0, it creates a circle with radius li and sets the values inside to 0
    if li != 0:
        struc = cp.where(((lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2) < (fw2 - li ** 2))
        k_app[struc] = 0
    if norm: k_app = k_app / k_app.sum()
    return (k_app)


@hierarchical_debug(logger)
def batch_aper_kernel(radius):
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
def fill_nan_fft(image: cp.ndarray, lk: int, li: int = 0, min_neighbors: int = 5, pad=301) -> cp.ndarray:
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
    valid_neighbors = convolve_fft(not_nan_mask, k_app)
    del not_nan_mask
    image_zeroed = cp.where(cp.isnan(image), 0, image)
    neighbor_sum = convolve_fft(image_zeroed, k_app)
    del image_zeroed
    # fill the NaN values with the local mean
    result = cp.where((valid_neighbors >= min_neighbors) & (cp.isnan(image)), neighbor_sum / valid_neighbors, image)
    del valid_neighbors
    return result
