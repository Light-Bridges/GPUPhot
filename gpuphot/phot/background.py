import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation

from cv.gpuphot.gpuphot.phot.convo import get_mean_std
from cv.gpuphot.gpuphot.phot.utils import decompose_into_tiles, calculate_tile_percentiles, recompose_from_percentiles, \
    fill_nan_fft


def get_local_background_fft(image, pxscale, qt=60, fill_aper=15, avg_aper=15, tile_px=300, ks=2, get_std=False):
    '''
    Get local background using FFT-based convolution

    Parameters
    ----------
    image : cupy.ndarray
        Image to be processed
    pxscale : float
        Pixel scale in arcsec/pixel
    qt : float
        Percentile for local background
    fill_aper : float
        Aperture size for filling NaNs, in arcsec
    avg_aper : float
        Aperture size for calculating mean, in arcsec
    tile_asec : float
        Tile size for calculating local percentile, in arcsec
    ks : float
        Aperture size for dilation, in arcsec
    get_std : bool
        Whether to return standard deviation image

    Returns
    -------
    img_filled_m : cupy.ndarray
        Image with local background subtracted
    '''
    if type(image) != cp.ndarray: image = cp.array(image)
    mempool = cp.get_default_memory_pool()

    tile_shape = tile_px
    image_cp_conv = gaussian_filter(image, sigma=2)
    _, img_std_conv = get_mean_std(image_cp_conv, max(int(1 / pxscale + 1), 3))
    del image_cp_conv
    mempool.free_all_blocks()

    tiles = decompose_into_tiles(img_std_conv, tile_shape)
    percentiles = calculate_tile_percentiles(tiles, qt=qt)
    per = recompose_from_percentiles(percentiles, image.shape, tile_shape)
    del tiles, percentiles
    mempool.free_all_blocks()

    mask = img_std_conv > per
    mask = binary_erosion(mask, cp.ones((max(int(2 / pxscale + 1), 2), max(int(2 / pxscale + 1), 2))))
    mask = binary_dilation(mask, cp.ones((max(int(ks / pxscale + 1), 2), max(int(ks / pxscale + 1), 2))))
    mask = binary_dilation(mask, cp.ones((max(int(ks / pxscale + 1), 2), max(int(ks / pxscale + 1), 2))))
    img_filled = image.copy()
    img_filled[mask] = cp.nan
    del img_std_conv, mask
    mempool.free_all_blocks()

    lk = max(int(fill_aper / pxscale + 1), 5)
    li = int(lk / 1.2 + 1)
    while cp.sum(cp.isnan(img_filled)) > 0:
        img_filled = fill_nan_fft(img_filled, lk, li)

    img_filled_m, img_filled_2 = get_mean_std(img_filled, max(int(avg_aper / pxscale + 1), 3), std=get_std)

    del img_filled
    mempool.free_all_blocks()
    return img_filled_m, img_filled_2
