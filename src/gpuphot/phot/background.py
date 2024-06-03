import gc

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter, convolve, binary_erosion, binary_dilation


# Calculate tile-wise percentiles
def decompose_into_tiles(image, block_size):
    h, w = image.shape
    num_tiles_y = h // block_size
    num_tiles_x = w // block_size
    tiles = cp.empty((num_tiles_y * num_tiles_x, block_size, block_size))
    idx = 0
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            tiles[idx] = image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            idx += 1
    return tiles


def calculate_tile_percentiles(tiles, qt=70):
    return cp.percentile(tiles, qt, axis=(1, 2))


def recompose_from_percentiles(percentiles, original_shape, block_size):
    h, w = original_shape
    num_tiles_y = h // block_size
    num_tiles_x = w // block_size
    recomposed = cp.empty(original_shape)
    idx = 0
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            recomposed[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = percentiles[idx]
            idx += 1
    recomposed[block_size * num_tiles_y:, :] = recomposed[block_size * num_tiles_y - 1, :]
    recomposed[:, block_size * num_tiles_x:] = recomposed[:,
                                               2 * (block_size * num_tiles_x - w):block_size * num_tiles_x - w]
    return recomposed


# additional conv functions
def gen_apm_filter(lk, li=0, norm=True):
    k_dim = (2 * lk + 1, 2 * lk + 1)
    indi = cp.indices(k_dim)
    fw2 = lk ** 2
    k_app = cp.zeros(k_dim)
    struc = cp.where(((lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2) < fw2)
    k_app[struc] = 1
    if li != 0:
        struc = cp.where(((lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2) < (fw2 - li ** 2))
        k_app[struc] = 0
    if norm: k_app = k_app / k_app.sum()
    return (k_app)


def get_mean_std(im_g, lk, std=True):
    k_app = gen_apm_filter(lk)
    fot_m = convolve(im_g, k_app, origin=(0, 0))
    if std:
        fot_m2 = convolve(im_g * im_g, k_app, origin=(0, 0))
        fot_m2 = cp.sqrt(fot_m2 - fot_m * fot_m)
    else:
        fot_m2 = None
    del k_app
    gc.collect()
    return fot_m, fot_m2


# FFT-based convolution
def fill_image(image_shape):
    h, w = image_shape
    new_height = 2 ** int(np.ceil(np.log2(h)))
    new_width = 2 ** int(np.ceil(np.log2(w)))
    return (new_height, new_width)


def convolve_fft(image, kernel):
    image_shape = image.shape
    kernel_shape = kernel.shape
    padding = int((kernel_shape[0] - 1) / 2)
    new_image_shape = fill_image((image_shape[0] + 2 * padding, image_shape[1] + 2 * padding))
    F_image = cp.fft.rfft2(image, s=new_image_shape)
    F_kernel = cp.fft.rfft2(kernel, s=new_image_shape)
    F_convolved = F_image * cp.conj(F_kernel)
    convolved = cp.fft.irfft2(F_convolved, s=new_image_shape)
    convolved = cp.roll(convolved, shift=[padding, padding], axis=[0, 1])
    convolved = convolved[:image_shape[0], :image_shape[1]]
    return convolved


def fill_nan_fft(image, lk, li=0, min_neighbors=5, pad=301):
    k_app = gen_apm_filter(lk, li=li, norm=False)
    image = image.astype(cp.double)
    not_nan_mask = (~cp.isnan(image)).astype(cp.double)
    valid_neighbors = convolve_fft(not_nan_mask, k_app)
    image_zeroed = cp.where(cp.isnan(image), 0, image)
    neighbor_sum = convolve_fft(image_zeroed, k_app)
    result = cp.where((valid_neighbors >= min_neighbors) & (cp.isnan(image)), neighbor_sum / valid_neighbors, image)
    return result


def get_local_background_fft(image, pxscale, qt=60, fill_aper=15, avg_aper=15, tile_px=300, ks=2):
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

    Returns
    -------
    img_filled_m : cupy.ndarray
        Image with local background subtracted
    '''
    if type(image) != cp.ndarray: image = cp.array(image)

    tile_shape = tile_px
    image_cp_conv = gaussian_filter(image, sigma=2)
    _, img_std_conv = get_mean_std(image_cp_conv, max(int(1 / pxscale + 1), 3))

    tiles = decompose_into_tiles(img_std_conv, tile_shape)
    percentiles = calculate_tile_percentiles(tiles, qt=qt)
    per = recompose_from_percentiles(percentiles, image.shape, tile_shape)

    mask = img_std_conv > per
    mask = binary_erosion(mask, cp.ones((max(int(2 / pxscale + 1), 2), max(int(2 / pxscale + 1), 2))))
    mask = binary_dilation(mask, cp.ones((max(int(ks / pxscale + 1), 2), max(int(ks / pxscale + 1), 2))))
    mask = binary_dilation(mask, cp.ones((max(int(ks / pxscale + 1), 2), max(int(ks / pxscale + 1), 2))))
    img_filled = cp.copy(image)
    img_filled[mask] = cp.nan

    lk = max(int(fill_aper / pxscale + 1), 5)
    li = int(lk / 1.2 + 1)
    while cp.sum(cp.isnan(img_filled)) > 0:
        img_filled = fill_nan_fft(img_filled, lk, li)

    img_filled_m, _ = get_mean_std(img_filled, max(int(avg_aper / pxscale + 1), 3), std=False)

    del image_cp_conv, img_std_conv, tiles, percentiles, per, mask, img_filled
    gc.collect()

    return img_filled_m
