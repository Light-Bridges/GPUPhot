import cupy as cp
import numpy as np

from ..logger.hierarchical_logging import setup_logger, hierarchical_debug

logger = setup_logger(__name__)

@hierarchical_debug(logger)
def convolve_fft(image, kernel):


    """Perform convolution of an image with a kernel using FFT.

    :param image: Input image.
    :type image: ndarray
    :param kernel: Convolution kernel.
    :type kernel: ndarray

    
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

    return convolved

@hierarchical_debug(logger)
def get_mean_std(im_g, lk, std=True):


    """Calculate the mean and standard deviation of an image using a Gaussian kernel.

    :param im_g: Input image.
    :type im_g: ndarray
    :param lk: Kernel size.
    :type lk: int
    :param std: If True, calculate the standard deviation, by default True.
    :type std: bool, optional

    
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
def gaussian_kernel(lk, sigma):

    """Generate a Gaussian kernel.

    :param lk: Kernel size.
    :type lk: int
    :param sigma: Standard deviation of the Gaussian.
    :type sigma: float

    
    """
    k_dim = 2 * lk + 1, 2 * lk + 1
    x = cp.linspace(-lk, lk, k_dim[0])
    y = cp.linspace(-lk, lk, k_dim[1])
    xx, yy = cp.meshgrid(x, y)
    kernel = cp.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    del x, y, xx, yy

    return kernel / cp.sum(kernel)

@hierarchical_debug(logger)
def get_aper_kernel(radius, size=None):


    """Generate an aperture kernel.

    :param radius: Radius of the aperture.
    :type radius: int
    :param size: Size of the kernel, by default None.
    :type size: int, optional

    
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
def fill_image(image_shape):

    """Calculate the shape for zero-padding an image to the next power of 2.

    :param image_shape: Shape of the input image.
    :type image_shape: tuple of int

    
    """
    h, w = image_shape
    new_height = 2 ** int(np.ceil(np.log2(h)))
    new_width = 2 ** int(np.ceil(np.log2(w)))

    return new_height, new_width

@hierarchical_debug(logger)
def gen_apm_filter(lk, li=0, norm=True):


    """Generate an aperture mask filter.

    :param lk: Kernel size.
    :type lk: int
    :param li: Inner radius to exclude from the mask, by default 0.
    :type li: int, optional
    :param norm: If True, normalize the kernel, by default True.
    :type norm: bool, optional

    
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

    return k_app

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
