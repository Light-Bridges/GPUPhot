import cupy as cp

from ..logger.hierarchical_logging import setup_logger, hierarchical_debug

logger = setup_logger(__name__)


def decompose_into_tiles(image: cp.ndarray, block_size: int) -> cp.ndarray:
    """Decomposes an image into tiles of a specific size.

    :param image: Image array to be decomposed.
    :param block_size: Size of the tiles.
    :return: An array containing the tiles.
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


@hierarchical_debug(logger)
def calculate_tile_percentiles(tiles: cp.ndarray, qt: float = 70) -> cp.ndarray:
    """Calculates the percentiles of an array of tiles.

    :param tiles: Array of tiles.
    :param qt: Percentile to be calculated.
    :return: An array containing the percentiles for each tile.
    """
    tiles = tiles[~cp.isnan(tiles).any(axis=(1, 2))]
    return cp.percentile(tiles, qt, axis=(1, 2))


@hierarchical_debug(logger)
def calculate_tile_nanmean(tiles: cp.ndarray) -> cp.ndarray:
    """Calculates the mean of an array of tiles.

    :param tiles: Array of tiles.
    :return: An array containing the mean for each tile.
    """
    return cp.nanmean(tiles, axis=(1, 2))


@hierarchical_debug(logger)
def calculate_tile_nanmean_sigclip(tiles: cp.ndarray, nsigma: float = 2) -> cp.ndarray:
    """Calculates the mean of an array of tiles.

    :param tiles: Array of tiles.
    :param nsigma: Number of sigmas to be clipped.
    :return: A tuple containing the mean and standard deviation of the tiles.
    """
    m = cp.nanmedian(tiles, axis=(1, 2))
    s = cp.nanstd(tiles, axis=(1, 2))
    mask = cp.abs(tiles - m[:, None, None]) < nsigma * s[:, None, None]
    tiles[~mask] = cp.nan
    return cp.nanmean(tiles, axis=(1, 2)), cp.nanstd(tiles, axis=(1, 2))

@hierarchical_debug(logger)
def calculate_tile_nanmedian_sigclip(tiles: cp.ndarray, nsigma: float = 2) -> cp.ndarray:
    """Calculates the mean of an array of tiles.

    :param tiles: Array of tiles.
    :param nsigma: Number of sigmas to be clipped.
    :return: A tuple containing the mean and standard deviation of the tiles.
    """
    m = cp.nanmedian(tiles, axis=(1, 2))
    s = cp.nanstd(tiles, axis=(1, 2))
    mask = cp.abs(tiles - m[:, None, None]) < nsigma * s[:, None, None]
    tiles[~mask] = cp.nan
    return cp.nanmedian(tiles, axis=(1, 2)), cp.nanstd(tiles, axis=(1, 2))


def recompose_from_percentiles(percentiles: cp.ndarray, original_shape: tuple, block_size: int) -> cp.ndarray:
    """Recomposes an image from its percentiles.

    :param percentiles: Array containing the percentiles of the tiles.
    :param original_shape: Shape of the original image.
    :param block_size: Size of the tiles.
    :return: An array containing the recomposed image.
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
