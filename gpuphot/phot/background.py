from __future__ import annotations

import cupy as cp
import nvtx
from cupyx.scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation

from .conv import get_mean_std, fill_nan_fft
from .utils import decompose_into_tiles, calculate_tile_percentiles, recompose_from_percentiles
from ..logger.hierarchical_logging import setup_logger
from ..utils.gpu import free_gpu_mem

logger = setup_logger(__name__)


### # @hierarchical_debug(logger)
@nvtx.annotate('get_local_background_fft', category='phot.background')
def get_local_background_fft(image, pxscale: float, qt: float = 75, fill_aper: int = 100,
                             avg_aper: int = 20, tile_section: int = 500, ks: int = 2,
                             get_std: bool = False, **kwargs) -> tuple:
    """
    Obtain the local background using FFT-based convolution.

    :param image: Image array to be processed.
    :type image: numpy.ndarray or cupy.ndarray
    :param pxscale: Pixel scale in arcsec/pixel.
    :type pxscale: float
    :param qt: Percentile for local background.
    :type qt: float
    :param fill_aper: Aperture size for filling NaNs, in arcsec.
    :type fill_aper: int
    :param avg_aper: Aperture size for calculating the mean, in arcsec.
    :type avg_aper: int
    :param tile_section: Tile size for calculating the local percentile, in pixels.
    :type tile_section: int
    :param ks: Aperture size for dilation, in arcsec.
    :type ks: int
    :param get_std: Whether to return standard deviation image.
    :type get_std: bool
    :param kwargs: Additional keyword arguments.
    :return: A tuple containing the local background of an image and the standard deviation.
    :rtype: tuple
    """
    if type(image) != cp.ndarray: image = cp.array(image)

    image_cp_conv = gaussian_filter(image, sigma=2)
    _, img_std_conv = get_mean_std(image_cp_conv, max(int(1 / pxscale + 1), 3), **kwargs)
    del image_cp_conv, _
    free_gpu_mem()

    tiles = decompose_into_tiles(img_std_conv, tile_section)
    percentiles = calculate_tile_percentiles(tiles, qt=qt)
    per = recompose_from_percentiles(percentiles, image.shape, tile_section)

    del tiles, percentiles
    free_gpu_mem()

    mask = img_std_conv > per

    del per, img_std_conv
    free_gpu_mem()

    mask = binary_erosion(mask, cp.ones((max(int(ks / pxscale + 1), 2), max(int(ks / pxscale + 1), 2))))
    mask = binary_dilation(mask, cp.ones((max(int(ks / pxscale + 1), 2), max(int(ks / pxscale + 1), 2))))
    mask = gaussian_filter(cp.array(mask, dtype=cp.float32), sigma=max(int(ks / pxscale + 1), 3)) > 0.05
    img_filled = image.copy()
    img_filled[mask] = cp.nan
    del mask, image
    free_gpu_mem()

    lk = max(int(fill_aper / pxscale + 1), 5)
    li = int(lk / 1.8 + 1)
    while cp.sum(cp.isnan(img_filled)) > 0:
        img_filled = fill_nan_fft(img_filled, lk, li, min_neighbors=5, **kwargs)

    img_filled_m, img_filled_2 = get_mean_std(img_filled, max(int(avg_aper / pxscale + 1), 3), std=get_std,
                                              **kwargs)

    del img_filled
    free_gpu_mem()
    return img_filled_m, img_filled_2


if __name__ == '__main__':
    logger.info(get_local_background_fft(cp.random.rand(100, 100), 1))
