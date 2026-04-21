# SPDX-License-Identifier: MIT
"""
Utility helpers for gpuphot.phot.

This module provides small utilities for tiling images, computing tile-based
statistics (percentiles, nan-mean, sigma-clipped stats) and recomposing an
image from tile statistics. Implementations use CuPy for GPU-accelerated
array operations.
"""

import cupy as cp
import nvtx
from ..logger.hierarchical_logging import setup_logger, hierarchical_debug

logger = setup_logger(__name__)


@nvtx.annotate('decompose_into_tiles',category='phot.utils')
def decompose_into_tiles(image: cp.ndarray, block_size: int) -> cp.ndarray:
    """
    Decompose an image into tiles of a specific size.

    :param image: Image array to be decomposed.
    :type image: cupy.ndarray
    :param block_size: Size of the tiles.
    :type block_size: int
    :return: An array containing the tiles.
    :rtype: cupy.ndarray
    """
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


### # @hierarchical_debug(logger)
@nvtx.annotate('calculate_tile_percentiles',category='phot.utils')
def calculate_tile_percentiles(tiles: cp.ndarray, qt: float = 70) -> cp.ndarray:
    """
    Calculate the percentiles of an array of tiles.

    :param tiles: Array of tiles.
    :type tiles: cupy.ndarray
    :param qt: Percentile to be calculated.
    :type qt: float
    :return: An array containing the percentiles for each tile.
    :rtype: cupy.ndarray
    """

    tiles = tiles[~cp.isnan(tiles).any(axis=(1, 2))]
    return cp.percentile(tiles, qt, axis=(1, 2))


### # @hierarchical_debug(logger)
@nvtx.annotate('calculate_tile_nanmean',category='phot.utils')
def calculate_tile_nanmean(tiles: cp.ndarray) -> cp.ndarray:
    """
    Calculate the mean of an array of tiles.

    :param tiles: Array of tiles.
    :type tiles: cupy.ndarray
    :return: An array containing the mean for each tile.
    :rtype: cupy.ndarray
    """
    return cp.nanmean(tiles, axis=(1, 2))


### # @hierarchical_debug(logger)
@nvtx.annotate('calculate_tile_nanmean_sigclip',category='phot.utils')
def calculate_tile_nanmean_sigclip(tiles: cp.ndarray, nsigma: float = 2) -> cp.ndarray:
    """
    Calculate the mean of an array of tiles using sigma clipping.

    :param tiles: Array of tiles.
    :type tiles: cupy.ndarray
    :param nsigma: Number of sigmas to be clipped.
    :type nsigma: float
    :return: A tuple containing the mean and standard deviation of the tiles.
    :rtype: tuple(cupy.ndarray, cupy.ndarray)
    """
    m = cp.nanmedian(tiles, axis=(1, 2))
    s = cp.nanstd(tiles, axis=(1, 2))
    mask = (cp.abs(tiles - m[:, None, None]) < nsigma * s[:, None, None]) | (s[:, None, None] == 0)
    tiles[~mask] = cp.nan
    return cp.nanmean(tiles, axis=(1, 2)), cp.nanstd(tiles, axis=(1, 2))

### # @hierarchical_debug(logger)
@nvtx.annotate('calculate_tile_nanmedian_sigclip',category='phot.utils')
def calculate_tile_nanmedian_sigclip(tiles: cp.ndarray, nsigma: float = 2) -> cp.ndarray:
    """
    Calculate the median of an array of tiles using sigma clipping.

    :param tiles: Array of tiles.
    :type tiles: cupy.ndarray
    :param nsigma: Number of sigmas to be clipped.
    :type nsigma: float
    :return: A tuple containing the median and standard deviation of the tiles.
    :rtype: tuple(cupy.ndarray, cupy.ndarray)
    """
    m = cp.nanmedian(tiles, axis=(1, 2))
    s = cp.nanstd(tiles, axis=(1, 2))
    mask = cp.abs(tiles - m[:, None, None]) < nsigma * s[:, None, None]
    tiles[~mask] = cp.nan
    return cp.nanmedian(tiles, axis=(1, 2)), cp.nanstd(tiles, axis=(1, 2))


@nvtx.annotate('recompose_from_percentiles',category='phot.utils')
def recompose_from_percentiles(percentiles: cp.ndarray, original_shape: tuple, block_size: int) -> cp.ndarray:
    """
    Recompose an image from its percentiles.

    :param percentiles: Array containing the percentiles of the tiles.
    :type percentiles: cupy.ndarray
    :param original_shape: Shape of the original image.
    :type original_shape: tuple
    :param block_size: Size of the tiles.
    :type block_size: int
    :return: An array containing the recomposed image.
    :rtype: cupy.ndarray
    """
    h, w = original_shape
    num_tiles_y = h // block_size
    num_tiles_x = w // block_size
    recomposed = cp.empty(original_shape)

    # fill the recomposed image with the percentiles
    idx = 0
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            recomposed[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = percentiles[idx]
            idx += 1

    # fill in the borders of the recomposed image
    recomposed[block_size * num_tiles_y:, :] = recomposed[block_size * num_tiles_y - 1, :]
    recomposed[:, block_size * num_tiles_x:] = recomposed[:,
                                               2 * (block_size * num_tiles_x - w):block_size * num_tiles_x - w]
    return recomposed
