import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import minimum_filter, label, sum as nd_sum, laplace, gaussian_filter
from lmfit import Model
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

from gpuphot.phot.convo import gaussian_kernel, convolve_fft
from gpuphot.phot.utils import decompose_into_tiles, recompose_from_percentiles, fill_nan_fft, \
    calculate_tile_nanmean


def detect_isolated_stars(img, rms, pxscale, sat_lim=50000, min_snr=10, dist_asec=20):
    mempool = cp.get_default_memory_pool()
    dist_px = max(dist_asec / pxscale, 20)
    border = 2 * dist_px

    kernel = gaussian_kernel(int(np.max((5 * 2 + 1, 10 / pxscale))), 2)
    kernel = (kernel - cp.mean(kernel)) / cp.std(kernel)
    conv_ima = convolve_fft(img, kernel)
    conv_sigma = conv_ima / rms / cp.sqrt(kernel.shape[0] * kernel.shape[1])
    del kernel, conv_ima
    mask = minimum_filter((conv_sigma > 5), 2).astype(cp.int32)
    mask[-border:, :] = 0
    mask[:border, :] = 0
    mask[:, -border:] = 0
    mask[:, :border] = 0
    mempool.free_all_blocks()

    lbs = label(mask)
    ids = cp.asarray([range(lbs[1] + 1)])
    # npix = nd_sum(mask, lbs[0], ids)
    idx = cp.indices(img.shape, dtype=cp.int32)
    im1 = img * idx
    ids = ids[0]
    del mask

    # find the peak
    s = nd_sum(img, lbs[0], ids)
    x = (nd_sum(im1[0, :, :], lbs[0], ids) / s)
    y = (nd_sum(im1[1, :, :], lbs[0], ids) / s)

    del im1, idx, lbs, ids, s
    mempool.free_all_blocks()

    coor = cp.asarray((cp.round(x).astype(cp.int32), cp.round(y).astype(cp.int32))).T
    coor_f = filter_centroids_kdtree(coor.get(), dist_px)
    snr = conv_sigma[coor_f[:, 0], coor_f[:, 1]]
    peak = img[coor_f[:, 0], coor_f[:, 1]]
    m = (snr > min_snr) & (peak < sat_lim)
    if cp.sum(m) == 0:
        m = (snr > 3) & (peak < sat_lim)
    if cp.sum(m) == 0:
        print('No stars found')

    coor_f = cp.asarray(coor_f)[m]
    del coor, snr, peak, m
    mempool.free_all_blocks()
    return coor_f


def create_star_dataset(img, coords, pxscale, N=3000, CR_filter=False, CR_thres=30):
    f = max(int(5 / pxscale), 6)
    n = max(int(1 / pxscale), 2)

    star_dataset = cp.zeros((len(coords), 2 * f + 1, 2 * f + 1), dtype=cp.float32)
    scaling_dataset = cp.zeros((len(coords), 4), dtype=cp.float32)

    idx = cp.arange(len(coords))
    for i, (y, x) in enumerate(coords):
        x_min = x - f
        x_max = x + f + 1
        y_min = y - f
        y_max = y + f + 1

        if x_min < 0 or y_min < 0 or x_max > img.shape[1] or y_max > img.shape[0]:
            idx = idx[idx != i]
            continue

        subima = img[y_min:y_max, x_min:x_max][::-1, :]
        peak_pos = cp.unravel_index(cp.argmax(subima), subima.shape)

        # continue is the peak is separated from the center by more than n pixel
        if abs(peak_pos[0] - f) > n or abs(peak_pos[1] - f) > n:
            idx = idx[idx != i]
            continue
        else:
            peak = subima[peak_pos]

        if CR_filter == True:
            if cp.std(laplace(subima)) > CR_thres:
                idx = idx[idx != i]
                continue

        star_dataset[i, :] = subima
        scaling_dataset[i, 0] = peak
        scaling_dataset[i, 1] = cp.mean(subima)
        scaling_dataset[i, 2] = cp.var(subima)
        scaling_dataset[i, 3] = cp.sum(subima)

    N = min(N, len(idx))
    idx = idx[cp.argsort(scaling_dataset[:, 3])[-N:]]

    del subima, peak_pos, peak, x_min, x_max, y_min, y_max, f, n

    return star_dataset[idx], coords[idx], scaling_dataset[idx]


def get_eigen_psfs(normed_star_dataset, n_components=5):
    pca = PCA(n_components=n_components)
    starset_flattened = normed_star_dataset.reshape(normed_star_dataset.shape[0],
                                                    normed_star_dataset.shape[1] ** 2).get()
    # pca_result = pca.fit_transform(starset_flattened)
    eigen_psfs = pca.components_.reshape(-1, normed_star_dataset.shape[1], normed_star_dataset.shape[2])
    return eigen_psfs


def project_all_stars_onto_eigenpsfs(normed_star_dataset, eigen_psfs):
    num_stars = normed_star_dataset.shape[0]
    flattened_star_dim = normed_star_dataset.shape[1] * normed_star_dataset.shape[2]
    stars_matrix = normed_star_dataset.reshape((num_stars, flattened_star_dim))
    eigen_matrix = eigen_psfs.reshape((eigen_psfs.shape[0], flattened_star_dim)).T
    coefficients_matrix = cp.dot(stars_matrix, eigen_matrix)
    del stars_matrix, eigen_matrix, flattened_star_dim
    return coefficients_matrix


def create_coeff_map(img_shape, positions, coefficients, pxscale, tile_section=None, smooth=False):
    coeff_map = cp.nan * cp.ones((coefficients.shape[0], img_shape[0], img_shape[1]), dtype=cp.float32)
    coeff_map[:, positions[:, 0], positions[:, 1]] = coefficients

    if tile_section is None:
        block_size = int(200 / pxscale)
    else:
        block_size = int(tile_section)

    for c in range(coefficients.shape[0]):
        tiles = decompose_into_tiles(coeff_map[c, :, :], block_size)
        tiles = calculate_tile_nanmean(tiles)
        tiles = tiles.reshape((img_shape[0] // block_size, img_shape[1] // block_size))
        while cp.sum(cp.isnan(tiles)) > 0:
            tiles = fill_nan_fft(tiles, 2, 0, min_neighbors=2, pad=1)

        if smooth == True:
            tiles = gaussian_filter(tiles, min(1, 2 / pxscale))

        coeff_map[c, :, :] = recompose_from_percentiles(tiles.reshape(-1), img_shape, block_size)

    del tiles
    return coeff_map


def detect_sources_pca(img, rms, pxscale, eigen_psfs, coeff_map, min_snr=5):
    mempool = cp.get_default_memory_pool()
    min_size = int(max(1 / pxscale, 2))
    conv_ima_pca = cp.zeros_like(img)
    for e in range(eigen_psfs.shape[0]):
        flipped_psf = cp.flip(eigen_psfs[e], (0, 1))
        conv_ima_pca += convolve_fft(img, flipped_psf) * coeff_map[e, :, :]
    conv_ima_sigma = conv_ima_pca / rms / cp.sqrt(eigen_psfs.shape[1] * eigen_psfs.shape[2])
    lbs = label(minimum_filter((conv_ima_sigma > min_snr), min_size).astype(cp.int32))
    mempool.free_all_blocks()
    ids = cp.asarray([range(lbs[1] + 1)])[0]
    # im1 = img * cp.indices(img.shape, dtype=cp.int32)

    # find the peak of convolved image
    im1 = conv_ima_pca * cp.indices(conv_ima_pca.shape, dtype=cp.int32)
    x = (nd_sum(im1[0, :, :], lbs[0], ids) / nd_sum(conv_ima_pca, lbs[0], ids))
    y = (nd_sum(im1[1, :, :], lbs[0], ids) / nd_sum(conv_ima_pca, lbs[0], ids))
    coor = cp.asarray((x, y)).T
    del im1, lbs, ids, conv_ima_pca
    mempool.free_all_blocks()
    return coor, conv_ima_sigma


def detect_sources_kernel(img, rms, kernel, pxscale, min_snr=5):
    mempool = cp.get_default_memory_pool()
    min_size = int(max(1 / pxscale, 2))
    conv_ima = convolve_fft(img, cp.flip(kernel, (0, 1)))
    conv_sigma = conv_ima / rms / cp.sqrt(kernel.shape[0] * kernel.shape[1])
    lbs = label(minimum_filter((conv_sigma > min_snr), min_size).astype(cp.int32))
    # del conv_sigma
    mempool.free_all_blocks()
    ids = cp.asarray([range(lbs[1] + 1)])[0]
    # im1 = img * cp.indices(img.shape, dtype=cp.int32)

    # find the peak of convolved image
    im1 = conv_ima * cp.indices(conv_ima.shape, dtype=cp.int32)
    x = (nd_sum(im1[0, :, :], lbs[0], ids) / nd_sum(conv_ima, lbs[0], ids))
    y = (nd_sum(im1[1, :, :], lbs[0], ids) / nd_sum(conv_ima, lbs[0], ids))
    coor = cp.asarray((x, y)).T
    del im1, lbs, ids, conv_ima
    mempool.free_all_blocks()
    return coor, conv_sigma


def recreate_normed_star(coeff_map, eigen_psfs, coords):
    x, y = coords
    coeff = coeff_map[:, y, x]
    kernel = cp.dot(coeff, eigen_psfs.reshape((eigen_psfs.shape[0], -1)))
    kernel = kernel.reshape((eigen_psfs.shape[1], eigen_psfs.shape[2]))
    return kernel


def fit_moffat(star_data):
    # Reshape the 2D image to 1D arrays
    ax, ay = np.meshgrid(np.arange(star_data.shape[1]), np.arange(star_data.shape[0]))
    X, Y, Z = ax.ravel(), ay.ravel(), star_data.ravel()

    # Transform to radial coordiantes
    center = (np.array([X[np.argmax(Z)], Y[np.argmax(Z)]])).astype(int)
    r = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    # Remove the sky from the outer part of the star
    sky = np.median(Z[r > np.percentile(r, 0.7)])
    Z = Z.astype(np.float32) - sky

    peak = np.max(Z)
    fwhm = r[np.argmin(np.abs(Z - (np.max(Z) - np.min(Z)) / 2))]

    mask = r < 5 * fwhm
    r = r[mask]
    Z = Z[mask]

    # Fit the Moffat profile
    model = Model(moffat, independent_vars=['r'])
    params = model.make_params(r0=0, A=peak)
    result = model.fit(Z, r=r, params=params)

    # Obtain the FWHM and uncertainty
    if result.params['R'].stderr is None or result.params['R'].stderr is None:
        fwhm, fwhm_err = moffat_fwhm(result.params['R'].value, result.params['B'].value,
                                     1e30, 1e30)
    else:
        fwhm, fwhm_err = moffat_fwhm(result.params['R'].value, result.params['B'].value,
                                     result.params['R'].stderr, result.params['B'].stderr)

    return r, Z, result, fwhm, fwhm_err


def filter_centroids_kdtree(centroids, min_distance):
    tree = KDTree(centroids)
    filtered = []
    for centroid in centroids:
        dist, _ = tree.query(centroid, k=2)
        if dist[1] >= min_distance:
            filtered.append(centroid)
    return np.array(filtered)


def moffat(r, A=1., r0=0., B=1., R=1.):
    """
    https://nbviewer.org/github/ysbach/AO_2017/blob/master/04_Ground_Based_Concept.ipynb#1.2.-Moffat
    """
    return A * (1 + ((r - r0) / R) ** 2) ** (-B)


def moffat_fwhm(R, B, R_err, B_err):
    FWHM = 2 * R * np.sqrt(2 ** (1 / B) - 1)
    FWHM_err = 2 * R_err * np.sqrt(2 ** (1 / B) - 1) + 2 * R * B_err * \
               (np.log(2) * 2 ** ((1 / B) - 1)) / (B ** 2 * np.sqrt(2 ** (1 / B) - 1))
    return FWHM, FWHM_err
