import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation

from .conv import get_mean_std, fill_nan_fft
from .utils import decompose_into_tiles, calculate_tile_percentiles, recompose_from_percentiles
from ..logger.hierarchical_logging import setup_logger, hierarchical_debug

logger = setup_logger(__name__)


@hierarchical_debug(logger)
def get_local_background_fft(image, pxscale: float, qt: float = 60, fill_aper: float = 15,
                             avg_aper: float = 15, tile_px: int = 300, ks: float = 2, get_std: bool = False):
    """Obtains the local background using FFT-based convolution.

    :param image: Image array to be processed.
    :param pxscale: Pixel scale in arcsec/pixel.
    :param qt: Percentile for local background.
    :param fill_aper: Aperture size for filling NaNs, in arcsec.
    :param avg_aper: Aperture size for calculating the mean, in arcsec.
    :param tile_px: Tile size for calculating the local percentile, in pixels.
    :param ks: Aperture size for dilation, in arcsec.
    :param get_std: Whether to return standard deviation image.
    :return: A tuple containing the local background of an image and the standard deviation.
    """
    if type(image) != cp.ndarray: image = cp.array(image)
    mempool = cp.get_default_memory_pool()

    # calculate the standard deviation of the image by fft convolution
    image_cp_conv = gaussian_filter(image, sigma=2)
    _, img_std_conv = get_mean_std(image_cp_conv, max(int(1 / pxscale + 1), 3))
    del image_cp_conv
    mempool.free_all_blocks()

    # calculate the local percentile
    tile_shape = tile_px
    tiles = decompose_into_tiles(img_std_conv, tile_shape)
    percentiles = calculate_tile_percentiles(tiles, qt=qt)
    per = recompose_from_percentiles(percentiles, image.shape, tile_shape)
    del tiles, percentiles
    mempool.free_all_blocks()

    # create a mask to obtain the local background
    mask = img_std_conv > per
    mask = binary_erosion(mask, cp.ones((max(int(2 / pxscale + 1), 2), max(int(2 / pxscale + 1), 2))))
    mask = binary_dilation(mask, cp.ones((max(int(ks / pxscale + 1), 2), max(int(ks / pxscale + 1), 2))))
    mask = binary_dilation(mask, cp.ones((max(int(ks / pxscale + 1), 2), max(int(ks / pxscale + 1), 2))))
    img_filled = image.copy()
    img_filled[mask] = cp.nan
    del img_std_conv, mask
    mempool.free_all_blocks()

    # fill the NaNs in the image with the local mean by fft convolution
    lk = max(int(fill_aper / pxscale + 1), 5)
    li = int(lk / 1.2 + 1)
    while cp.sum(cp.isnan(img_filled)) > 0:
        img_filled = fill_nan_fft(img_filled, lk, li)
    # calculate the mean and standard deviation of the image by fft convolution
    img_filled_m, img_filled_2 = get_mean_std(img_filled, max(int(avg_aper / pxscale + 1), 3), std=get_std)

    del img_filled
    mempool.free_all_blocks()
    return img_filled_m, img_filled_2


if __name__ == '__main__':
    logger.info(get_local_background_fft(cp.random.rand(100, 100), 1))
