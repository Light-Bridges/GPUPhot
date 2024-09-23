import logging
import time

import cupy as cp

from .convo import gen_apm_filter, convolve_fft

from ..logger.hierarchical_logging import setup_logger, hierarchical_debug
logger = setup_logger(__name__)

@hierarchical_debug(logger)
def decompose_into_tiles(image, block_size):
    logger.debug(
        f'Iniciando función decompose_into_tiles(image={image}, block_size={block_size})'
    )
    start_time = time.time()
    """
    Decompose an image into smaller tiles of specified block size.

    Parameters
    ----------
    image : ndarray
        The input image to be decomposed.
    block_size : int
        The size of each block (tile).

    Returns
    -------
    ndarray
        A stack of tiles extracted from the image.
    """
    h, w = image.shape
    num_tiles_y = h // block_size
    num_tiles_x = w // block_size
    tiles = cp.empty((num_tiles_y * num_tiles_x, block_size, block_size))
    idx = 0
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            tiles[idx] = image[i * block_size:(i + 1) * block_size, j *
                                                                    block_size:(j + 1) * block_size]
            idx += 1
    logger.debug(
        f'Función decompose_into_tiles completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return tiles

@hierarchical_debug(logger)
def calculate_tile_percentiles(tiles, qt=70):
    logger.debug(
        f'Iniciando función calculate_tile_percentiles(tiles={tiles}, qt={qt})'
    )
    start_time = time.time()
    """
    Calculate the percentiles of tiles along specified axes.

    Parameters
    ----------
    tiles : ndarray
        Stack of image tiles.
    qt : int, optional
        Percentile to compute, by default 70.

    Returns
    -------
    ndarray
        Percentile values of the tiles.
    """
    logger.debug(
        f'Función calculate_tile_percentiles completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return cp.percentile(tiles, qt, axis=(1, 2))

@hierarchical_debug(logger)
def recompose_from_percentiles(percentiles, original_shape, block_size):
    logger.debug(
        f'Iniciando función recompose_from_percentiles(percentiles={percentiles}, original_shape={original_shape}, block_size={block_size})'
    )
    start_time = time.time()
    """
    Recompose an image from its percentile values.

    Parameters
    ----------
    percentiles : ndarray
        Percentile values of the tiles.
    original_shape : tuple of int
        Shape of the original image.
    block_size : int
        Size of each block (tile).

    Returns
    -------
    ndarray
        Reconstructed image from the percentiles.
    """
    h, w = original_shape
    num_tiles_y = h // block_size
    num_tiles_x = w // block_size
    recomposed = cp.empty(original_shape)
    idx = 0
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            recomposed[i * block_size:(i + 1) * block_size, j * block_size:
                                                            (j + 1) * block_size] = percentiles[idx]
            idx += 1
    recomposed[block_size * num_tiles_y:, :] = recomposed[block_size *
                                                          num_tiles_y - 1, :]
    recomposed[:, block_size * num_tiles_x:] = recomposed[:, 2 * (
            block_size * num_tiles_x - w):block_size * num_tiles_x - w]
    logger.debug(
        f'Función recompose_from_percentiles completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return recomposed

@hierarchical_debug(logger)
def fill_nan_fft(image, lk, li=0, min_neighbors=5, pad=301):
    logger.debug(
        f'Iniciando función fill_nan_fft(image={image}, lk={lk}, li={li}, min_neighbors={min_neighbors}, pad={pad})'
    )
    start_time = time.time()
    """
    Fill NaN values in an image using FFT-based convolution with a specified filter.

    Parameters
    ----------
    image : ndarray
        Input image with NaN values.
    lk : int
        Kernel size for the filter.
    li : int, optional
        Additional parameter for the filter generation, by default 0.
    min_neighbors : int, optional
        Minimum number of valid neighbors required to replace a NaN value, by default 5.
    pad : int, optional
        Padding size for the convolution, by default 301.

    Returns
    -------
    ndarray
        Image with NaN values filled.
    """
    k_app = gen_apm_filter(lk, li=li, norm=False)
    image = image.astype(cp.double)
    not_nan_mask = (~cp.isnan(image)).astype(cp.double)
    valid_neighbors = convolve_fft(not_nan_mask, k_app)
    del not_nan_mask
    image_zeroed = cp.where(cp.isnan(image), 0, image)
    neighbor_sum = convolve_fft(image_zeroed, k_app)
    del image_zeroed
    result = cp.where((valid_neighbors >= min_neighbors) & cp.isnan(image),
                      neighbor_sum / valid_neighbors, image)
    del valid_neighbors
    logger.debug(
        f'Función fill_nan_fft completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return result

@hierarchical_debug(logger)
def calculate_tile_nanmean(tiles):
    logger.debug(f'Iniciando función calculate_tile_nanmean(tiles={tiles})')
    start_time = time.time()
    """
    Calculate the mean of tiles ignoring NaN values.

    Parameters
    ----------
    tiles : ndarray
        Stack of image tiles.

    Returns
    -------
    ndarray
        Mean values of the tiles, ignoring NaN values.
    """
    logger.debug(
        f'Función calculate_tile_nanmean completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return cp.nanmean(tiles, axis=(1, 2))
