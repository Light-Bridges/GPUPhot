import cupy as cp
import numpy as np
from astropy.io import fits
from cupyx.scipy.ndimage import binary_erosion, shift

from ..logger.hierarchical_logging import setup_logger, hierarchical_debug
from ..stats.subpixel import phase_cross_correlation as phase_cross_correlation_gpu

logger = setup_logger(__name__)


@hierarchical_debug(logger)
def center(im, size):
    """Center the image to the given size.

    :param im: Input image.
    :type im: ndarray
    :param size: Desired size for centering.
    :type size: int

    
    """
    if im.shape[0] > size:
        c0 = int((im.shape[0] - size) / 2)
    else:
        c0 = 0
    if im.shape[1] > size:
        c1 = int((im.shape[1] - size) / 2)
    else:
        c1 = 0
    return im[c0:-c0, c1:-c1]


@hierarchical_debug(logger)
def register_shift(fc, uf=100, n=1000):
    """Register and shift image stack based on phase cross-correlation.

    :param fc: Stack of images to be registered.
    :type fc: ndarray
    :param uf: Upsample factor for subpixel precision, by default 100.
    :type uf: int, optional
    :param n: Size for centering the images, by default 1000.
    :type n: int, optional

    
    """
    fc1 = fc.copy()
    im0 = center(fc[0], n)
    im0 = binary_erosion(im0 > im0.mean() + im0.std())
    for i in np.arange(1, fc.shape[0]):
        im1 = center(fc[i], n)
        im1 = binary_erosion(im1 > im1.mean() + im1.std())
        shifted, _, _ = phase_cross_correlation_gpu(im0.get(), im1.get(),
                                                    upsample_factor=uf)
        if (np.abs(shifted[0]) > 300) | np.abs(shifted[1] > 300):
            shifted = 0, 0
        fc1[i] = shift(cp.asarray(fc[i]), shift=(shifted[0], shifted[1]),
                       order=1, mode='constant').get()
        logger.debug(f'Detected subpixel offset (y, x): {shifted}')

    return fc1


@hierarchical_debug(logger)
def stack_sigmaclip(data, it=5, n=3):
    """Stack images with sigma clipping.

    :param data: Stack of images to be processed.
    :type data: ndarray
    :param it: Number of iterations for sigma clipping, by default 5.
    :type it: int, optional
    :param n: Sigma clipping threshold, by default 3.
    :type n: int, optional

    
    """
    nim = data.shape[0]
    if nim < 3:
        return cp.asarray(data.mean(axis=0)), None
    delta0 = -1
    iplus = cp.zeros_like(im0, dtype=cp.float32)+1.e10
    iminu = cp.zeros_like(im0, dtype=cp.float32)-1.e10
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
            mk = mk*(im <= iplus)
            im = im * mk
            center = center + im
            sigma = sigma + im*im
            delta = delta + cp.sum(mk == 0)
            mask = mask + mk

        center = center/mask
        sigma = sigma/mask
        sigma = cp.sqrt(sigma-center*center)
        if delta == delta0:
            break
        delta0 = delta
        iplus = center + n * sigma
        iminu = center - n * sigma
        avg[mask >=3] = center[mask >=3]
        std[mask >=3] = sigma[mask >=3]
        
    del iplus, iminu, im0, delta0, mask, delta, center, sigma
    return avg, std


@hierarchical_debug(logger)
def register_shift_frames(frames_list, upsample_factor=100, center_size=
1000, shift_limit_pix=300):
    """

    :param frames_list: 
    :param upsample_factor:  (Default value = 100)
    :param center_size:  (Default value = 1000)
    :param shift_limit_pix:  (Default value = 300)

    """

    fc0 = cp.asarray(fits.getdata(frames_list[0]), dtype=cp.float32)
    im0 = center(fc0, center_size)
    im0 = binary_erosion(im0 > im0.mean() + im0.std())
    fc = cp.zeros((len(frames_list), fc0.shape[0], fc0.shape[1]), dtype=cp.
                  float32)
    fc[0, :] = fc0
    del fc0
    for i in np.arange(1, len(frames_list)):
        fc1 = cp.asarray(fits.getdata(frames_list[i]), dtype=cp.float32)
        im1 = center(fc1, center_size)
        im1 = binary_erosion(im1 > im1.mean() + im1.std())
        shifted, _, _ = phase_cross_correlation_gpu(im0, im1,
                                                    upsample_factor=upsample_factor)
        if (np.abs(shifted[0]) > shift_limit_pix) | np.abs(shifted[1] >
                                                           shift_limit_pix):
            shifted = 0, 0
        logger.warning(f'Detected subpixel offset (y, x): {shifted}')
        fc[i, :] = shift(fc1, shift=(shifted[0], shifted[1]), order=1, mode
        ='constant')

    return fc
