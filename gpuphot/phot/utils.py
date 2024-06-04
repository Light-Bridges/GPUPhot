import cupy as cp

from cv.gpuphot.gpuphot.phot.convo import gen_apm_filter, convolve_fft


def decompose_into_tiles(image, block_size):
    h, w = image.shape
    num_tiles_y = h // block_size
    num_tiles_x = w // block_size
    tiles = cp.empty((num_tiles_y * num_tiles_x, block_size, block_size))
    idx = 0
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            tiles[idx] = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            idx += 1
    return tiles


def calculate_tile_percentiles(tiles, qt = 70):
    return cp.percentile(tiles, qt, axis=(1, 2))


def recompose_from_percentiles(percentiles, original_shape, block_size):
    h, w = original_shape
    num_tiles_y = h // block_size
    num_tiles_x = w // block_size
    recomposed = cp.empty(original_shape)
    idx = 0
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            recomposed[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = percentiles[idx]
            idx += 1
    recomposed[block_size*num_tiles_y:, :] = recomposed[block_size*num_tiles_y-1, :]
    recomposed[:, block_size*num_tiles_x:] = recomposed[:, 2*(block_size*num_tiles_x-w):block_size*num_tiles_x-w]
    return recomposed


def fill_nan_fft(image, lk, li=0, min_neighbors=5, pad = 301):
    k_app = gen_apm_filter(lk, li=li, norm=False)
    image = image.astype(cp.double)
    not_nan_mask = (~cp.isnan(image)).astype(cp.double)
    valid_neighbors = convolve_fft(not_nan_mask, k_app)
    del not_nan_mask
    image_zeroed = cp.where(cp.isnan(image), 0, image)
    neighbor_sum = convolve_fft(image_zeroed, k_app)
    del image_zeroed
    result = cp.where((valid_neighbors >= min_neighbors) & (cp.isnan(image)), neighbor_sum / valid_neighbors, image)
    del valid_neighbors
    return result


def calculate_tile_nanmean(tiles):
    return cp.nanmean(tiles, axis=(1, 2))
