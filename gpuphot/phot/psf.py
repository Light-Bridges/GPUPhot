import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import maximum_filter
from lmfit import Model
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

from .conv import gaussian_kernel, convolve_fft, fill_nan_fft
from .utils import calculate_tile_nanmean_sigclip, decompose_into_tiles, recompose_from_percentiles
from ..logger.hierarchical_logging import setup_logger, hierarchical_debug

logger = setup_logger(__name__)


@hierarchical_debug(logger)
def find_local_max(image: cp.ndarray, min_distance: int, threshold_abs: float) -> cp.ndarray:
    """Calculate local maxima in an image.

    :param image: Image array to be processed.
    :param min_distance: Minimum distance between peaks.
    :param threshold_abs: Absolute threshold for peaks.
    :return: Array of detected peaks.
    """
    max_mask = image == maximum_filter(image, size=min_distance)
    threshold_mask = image > threshold_abs
    peaks = cp.logical_and(max_mask, threshold_mask)
    return cp.argwhere(peaks)


@hierarchical_debug(logger)
def find_local_centroid(image: cp.ndarray, peaks: cp.ndarray, window_size: int = 5) -> cp.ndarray:
    """Calculate centroids of detected peaks in an image.
    
    :param image: Image where the peaks are located.
    :param peaks: Coordinates of the peaks.
    :param window_size: Size of the square neighborhood around the peak.
    :return: Array of calculated centroids.
    """

    if window_size % 2 == 0: window_size += 1

    # Create offset indices for the neighborhood
    half_size = window_size // 2
    y_indices = cp.arange(-half_size, half_size + 1)
    x_indices = cp.arange(-half_size, half_size + 1)
    y_offsets = peaks[:, 0, None, None] + y_indices[None, :, None]
    x_offsets = peaks[:, 1, None, None] + x_indices[None, None, :]
    y_offsets = cp.clip(y_offsets, 0, image.shape[0] - 1)
    x_offsets = cp.clip(x_offsets, 0, image.shape[1] - 1)

    # Extract neighborhoods using advanced indexing
    neighborhoods = image[y_offsets, x_offsets]
    y_coords, x_coords = cp.meshgrid(y_indices, x_indices, indexing='ij')

    # Calculate moments to find the centroid
    total_intensity = cp.sum(neighborhoods, axis=(1, 2))
    total_intensity = cp.where(total_intensity == 0, 1, total_intensity)  # Avoid division by zero
    y_centroid_offset = cp.sum(y_coords[None, :, :] * neighborhoods, axis=(1, 2)) / total_intensity
    x_centroid_offset = cp.sum(x_coords[None, :, :] * neighborhoods, axis=(1, 2)) / total_intensity

    # Calculate absolute positions of the centroids
    y_centroid = peaks[:, 0] + y_centroid_offset
    x_centroid = peaks[:, 1] + x_centroid_offset

    return cp.stack((y_centroid, x_centroid), axis=1)


@hierarchical_debug(logger)
def detect_isolated_stars(img: cp.ndarray, rms: cp.ndarray, pxscale: float, sat_lim: int = 50000, min_snr: float = 10,
                          dist_asec: float = 10, sort: bool = True, **kwargs) -> cp.array:
    """Detects isolated stars in an image using a fft convolution kernel.
        The stars are detected by convolving the image with a Gaussian kernel and filtered by a minimum signal-to-noise ratio.

    :param img: Image array to be processed.
    :param rms: RMS of the image.
    :param pxscale: Pixel scale in arcsec/pixel.
    :param sat_lim: Saturation limit.
    :param min_snr: Minimum signal-to-noise ratio.
    :param dist_asec: Minimum distance in arcsec.
    :param sort: Whether to sort the detected stars by signal-to-noise ratio and distance.
    :return: An array containing the coordinates of the detected stars.
    """

    mempool = cp.get_default_memory_pool()
    dist_px = max(dist_asec / pxscale, 20)
    border = 2 * dist_px
    kernel = gaussian_kernel(int(np.max((5 * 2 + 1, 10 / pxscale))), 2)
    kernel = (kernel - cp.mean(kernel)) / cp.std(kernel)
    conv_ima = convolve_fft(img, kernel, **kwargs)
    conv_sigma = conv_ima / rms / cp.sqrt(kernel.shape[0] * kernel.shape[1])
    del kernel, conv_ima
    conv_sigma[:border, :] = 0
    conv_sigma[-border:, :] = 0
    conv_sigma[:, :border] = 0
    conv_sigma[:, -border:] = 0
    coor_f = find_local_max(conv_sigma, min_distance=int(3 / pxscale), threshold_abs=min_snr)
    dist = get_centroids_distance_kdtree(coor_f.get())
    dist_mask = dist > dist_px
    coor_f = coor_f[dist_mask]
    dist = dist[dist_mask]
    snr = conv_sigma[coor_f[:, 0], coor_f[:, 1]]
    peak = img[coor_f[:, 0], coor_f[:, 1]]
    m = (snr > min_snr) & (peak < sat_lim)
    if cp.sum(m) == 0:
        m = (snr > 3) & (peak < sat_lim)
    if cp.sum(m) == 0:
        logger.error('Less than 5 isolated stars detected. Image may be too crowded or too noisy')
        raise ValueError('Less than 5 isolated stars detected. Image may be too crowded or too noisy')
    coor_f = cp.asarray(coor_f)[m]
    coor_f = find_local_centroid(conv_sigma, coor_f, int(3 / pxscale))
    if sort:
        sort_metric = snr[m].get() + dist[m.get()]
        idx = cp.argsort(np.max(sort_metric) - sort_metric)
        coor_f = coor_f[idx]
    del snr, peak, m, dist, dist_mask, sort_metric, idx
    mempool.free_all_blocks()

    return coor_f


@hierarchical_debug(logger)
def create_star_dataset(img: cp.ndarray, coords: cp.array, pxscale: float, N: int = 1000) -> tuple:
    """Creates a dataset of stars from an image and a list of coordinates.

    :param img: Image array to be processed.
    :param coords: Array containing the coordinates of the stars.
    :param pxscale: Pixel scale in arcsec/pixel.
    :param N: Number of stars to be selected.
    :param CR_filter: Whether to apply cosmic ray filtering.
    :param CR_thres: Cosmic ray threshold.
    :return: A tuple containing the star dataset, the coordinates of the stars, and the scaling dataset."""

    f = max(int(10 / pxscale), 6)
    n = max(int(1 / pxscale), 2)
    star_dataset = cp.zeros((len(coords), 2 * f + 1, 2 * f + 1), dtype=cp.
                            float32)
    scaling_dataset = cp.zeros((len(coords), 4), dtype=cp.float32)
    idx = cp.arange(len(coords))
    for i, (y, x) in enumerate(coords):
        x_min = x - f
        x_max = x + f + 1
        y_min = y - f
        y_max = y + f + 1
        if x_min < 0 or y_min < 0 or x_max > img.shape[1] or y_max > img.shape[
            0]:
            idx = idx[idx != i]
            continue
        subima = img[y_min:y_max, x_min:x_max][::-1, :]
        peak_pos = cp.unravel_index(cp.argmax(subima), subima.shape)
        if abs(peak_pos[0] - f) > n or abs(peak_pos[1] - f) > n:
            idx = idx[idx != i]
            continue
        else:
            peak = subima[peak_pos]
        star_dataset[i, :] = subima
        scaling_dataset[i, 0] = peak
        scaling_dataset[i, 1] = cp.mean(subima)
        scaling_dataset[i, 2] = cp.var(subima)
        scaling_dataset[i, 3] = cp.sum(subima)
    N = min(N, len(idx))
    idx = idx[:N]
    try:
        del subima, peak_pos, peak, x_min, x_max, y_min, y_max, f, n
    except Exception as e:
        logger.error(e)
    return star_dataset[idx], coords[idx], scaling_dataset[idx]


@hierarchical_debug(logger)
def get_eigen_psfs(normed_star_dataset: cp.array, n_components: int = 5) -> cp.array:
    """Calculates the eigen PSFs from a dataset of normalized stars.

    :param normed_star_dataset: Dataset of normalized stars.
    :param n_components: Number of components to be used.
    :return: An array containing the eigen PSFs.
    """

    pca = PCA(n_components=n_components)
    starset_flattened = normed_star_dataset.reshape(normed_star_dataset.shape[0],
                                                    normed_star_dataset.shape[1] ** 2).get()
    pca_result = pca.fit_transform(starset_flattened)
    eigen_psfs = pca.components_.reshape(-1, normed_star_dataset.shape[1], normed_star_dataset.shape[2])
    return eigen_psfs


@hierarchical_debug(logger)
def project_all_stars_onto_eigenpsfs(normed_star_dataset: cp.array, eigen_psfs: cp.array) -> cp.array:
    """Projects all stars onto the eigen PSFs.

    :param normed_star_dataset: Dataset of normalized stars.
    :param eigen_psfs: Array containing the eigen PSFs.
    :return: An array containing the coefficients of the stars projected onto the eigen PSFs.
    """

    num_stars = normed_star_dataset.shape[0]
    flattened_star_dim = normed_star_dataset.shape[1] * normed_star_dataset.shape[2]
    stars_matrix = normed_star_dataset.reshape((num_stars, flattened_star_dim))
    eigen_matrix = eigen_psfs.reshape((eigen_psfs.shape[0], flattened_star_dim)).T
    coefficients_matrix = cp.dot(stars_matrix, eigen_matrix)
    del stars_matrix, eigen_matrix, flattened_star_dim
    return coefficients_matrix


@hierarchical_debug(logger)
def create_coeff_map(img_shape: tuple, positions: cp.array, coefficients: cp.array, pxscale: float,
                     tile_section: int = None, env_factor: float = 3) -> cp.ndarray:
    """Creates a coefficient map from a list of positions and coefficients.

    :param img_shape: Dimensions of the image.
    :param positions: Array containing the positions of the coefficients.
    :param coefficients: Array containing the coefficients.
    :param pxscale: Pixel scale in arcsec/pixel.
    :param tile_section: Size of the tile section.
    :param env_factor: Factor for the environment check for outlier detection.
    :return: A coefficient map.
    """

    coeff_map = cp.nan * cp.ones((coefficients.shape[0], img_shape[0], img_shape[1]), dtype=cp.float32)
    coeff_map[:, cp.round(positions[:, 0]).astype(int), cp.round(positions[:, 1]).astype(int)] = coefficients
    if tile_section is None:
        block_size = int(200 / pxscale)
    else:
        block_size = int(tile_section)

    for c in range(coefficients.shape[0]):
        tiles = decompose_into_tiles(coeff_map[c, :, :], block_size)
        tiles, _ = calculate_tile_nanmean_sigclip(tiles)
        tiles = tiles.reshape((img_shape[0] // block_size, img_shape[1] // block_size))
        s = cp.sum(cp.isnan(tiles))
        lk = 2
        while s > 0:
            tiles = fill_nan_fft(tiles, lk, 0, min_neighbors=2)
            if cp.sum(cp.isnan(tiles)) == s: lk += 1
            s = cp.sum(cp.isnan(tiles))

        coeff_map[c, :, :] = recompose_from_percentiles(tiles.reshape(-1), img_shape, block_size)

    del tiles
    return coeff_map


@hierarchical_debug(logger)
def calculate_kernel_area(img_shape: tuple, psf: cp.ndarray, coeff_map: cp.ndarray = None,
                          eigen_psfs: cp.ndarray = None):
    """
    Calculate the area of the kernel.

    Parameters
    ----------

    psf : ndarray
        Point spread function.
    coeff_map : ndarray, optional
        Coefficient map, by default None.
    eigen_psfs : ndarray, optional
        Eigen PSFs, by default None.
    
    Returns
    -------
    ndarray
        Map of the kernel area.
    """

    A = cp.zeros(img_shape, dtype=cp.float32)
    A += cp.sum(psf * psf)
    if coeff_map is not None and eigen_psfs is not None:
        eigen_psfs = eigen_psfs.astype(cp.float32)
        A += cp.sum(coeff_map ** 2 * cp.sum(eigen_psfs ** 2, axis=(1, 2))[:, cp.newaxis, cp.newaxis], axis=0)
        k = coeff_map.shape[0]
        for i in range(k):
            for j in range(i + 1, k):
                A += coeff_map[i] * coeff_map[j] * cp.sum(eigen_psfs[i] * eigen_psfs[j])
    return A


@hierarchical_debug(logger)
def detect_sources_psf(img: cp.ndarray, rms: cp.ndarray, fwhm: float, psf: cp.array,
                       eigen_psfs: cp.ndarray = None, coeff_map: cp.ndarray = None,
                       min_snr: int = 5, **kwargs) -> cp.ndarray:
    """
    Detect sources using PCA.

    Parameters
    ----------
    img : ndarray
        Input image.
    rms : float
        Root mean square noise level.
    fwhm : float
        Full width at half maximum.
    psf : ndarray
        Reference point spread function.
    eigen_psfs : ndarray
        Eigen PSFs.
    coeff_map : ndarray
        Coefficient map.
    min_snr : float, optional
        Minimum signal-to-noise ratio, by default 5.

    Returns
    -------
    tuple
        (coordinates, conv_ima_sigma) where coordinates are the detected sources and conv_ima_sigma is the convolved image with PCA.
    """
    mempool = cp.get_default_memory_pool()
    flipped_psf = cp.flip(psf, (0, 1))
    conv_ima_pca = convolve_fft(img, flipped_psf, **kwargs)
    if coeff_map is not None and eigen_psfs is not None:
        for e in range(eigen_psfs.shape[0]):
            flipped_psf = cp.flip(eigen_psfs[e], (0, 1))
            conv_ima_pca += convolve_fft(img, flipped_psf, **kwargs) * coeff_map[e, :, :]
    A = calculate_kernel_area(img.shape, psf, coeff_map, eigen_psfs)
    conv_ima_sigma = conv_ima_pca / rms / cp.sqrt(A)
    coor = find_local_max(conv_ima_sigma, min_distance=int(np.ceil(2 * fwhm)), threshold_abs=min_snr)
    coor = find_local_centroid(conv_ima_sigma, coor, np.round(np.max((fwhm, 5))).astype(int))
    del flipped_psf, conv_ima_pca, A
    mempool.free_all_blocks()
    return coor, conv_ima_sigma


def recreate_normed_star(coeff_map: cp.ndarray, eigen_psfs: cp.array, coords: cp.array) -> cp.ndarray:
    """Recreates a normalized star from the coefficient map.

    :param coeff_map: Coefficient map.
    :param eigen_psfs: Array containing the eigen PSFs.
    :param coords: Tuple containing the coordinates of the star.
    :return: The recreated normalized star.
    """
    x, y = coords
    coeff = coeff_map[:, y, x]
    kernel = cp.dot(coeff, eigen_psfs.reshape((eigen_psfs.shape[0], -1)))
    kernel = kernel.reshape((eigen_psfs.shape[1], eigen_psfs.shape[2]))
    return kernel


@hierarchical_debug(logger)
def recreate_normed_star_vectorized(coeff_map: cp.ndarray, eigen_psfs: cp.array, xs: cp.array, ys: cp.array,
                                    **kwargs) -> cp.ndarray:
    """Recreates a set of normalized stars from the coefficient map.

    :param coeff_map: Coefficient map.
    :param eigen_psfs: Array containing the eigen PSFs.
    :param xs: Array containing the x-coordinates of the stars.
    :param ys: Array containing the y-coordinates of the stars.
    :return: An array containing the recreated normalized stars.
    """
    coeffs = coeff_map[:, ys, xs]
    coeffs = coeffs.reshape(-1, coeffs.shape[-1])
    reshaped_eigen_psfs = eigen_psfs.reshape(eigen_psfs.shape[0], -1)
    kernels = cp.dot(coeffs.T, reshaped_eigen_psfs)
    kernels = kernels.reshape(xs.size, eigen_psfs.shape[1], eigen_psfs.shape[2])
    del coeffs, reshaped_eigen_psfs
    return kernels


def recreate_normed_stars_batch(coeff_map: cp.ndarray, eigen_psfs: cp.ndarray, coords: cp.ndarray,
                                **kwargs) -> cp.ndarray:
    """Recreates normalized stars for a batch of coordinates.

    :param coeff_map: Coefficient map of shape (num_coeffs, height, width).
    :param eigen_psfs: Array containing the eigen PSFs of shape (num_coeffs, psf_height, psf_width).
    :param coords: Array of coordinates of shape (num_points, 2), where each row is (x, y).
    :return: Array of recreated normalized stars of shape (num_points, psf_height, psf_width).
    """
    xs, ys = coords[:, 0], coords[:, 1]
    coeffs = coeff_map[:, ys, xs]
    eigen_psfs_reshaped = eigen_psfs.reshape(eigen_psfs.shape[0], -1)
    kernels = cp.dot(coeffs.T, eigen_psfs_reshaped)
    kernels = kernels.reshape(coords.shape[0], eigen_psfs.shape[1], eigen_psfs.shape[2])
    return kernels


@hierarchical_debug(logger)
def fit_moffat(star_data: np.ndarray) -> tuple:
    """Fits a Moffat profile to a star.

    :param star_data: Image array containing the star data.
    :return: A tuple containing the radial coordinates, the intensity profile, the fit result, the FWHM, and the FWHM uncertainty.
    """

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


def filter_centroids_kdtree(centroids: np.array, min_distance: float) -> np.array:
    """ Filters centroids using a KDTree.

    :param centroids: Array containing the centroids to be filtered.
    :param min_distance: Minimum distance between centroids.
    :return: An array containing the filtered centroids.
    """
    distances = get_centroids_distance_kdtree(centroids)
    mask = distances >= min_distance
    return centroids[mask]


@hierarchical_debug(logger)
def get_centroids_distance_kdtree(centroids: np.array):
    """Find the distance between centroid and its nearest neighbor using KDTree.
    
    :param centroids: Array of centroids.
    :return: Array of distances.
    """

    tree = KDTree(centroids)
    dist, _ = tree.query(centroids, k=2)
    return dist[:, 1]


def moffat(r: np.array, A: float = 1., r0: float = 0., B: float = 1., R: float = 1.) -> np.array:
    """ Moffat profile function.
    https://nbviewer.org/github/ysbach/AO_2017/blob/master/04_Ground_Based_Concept.ipynb#1.2.-Moffat

    :param r: Radial coordinates.
    :param A: Peak intensity.
    :param r0: Central position.
    :param B: Power index.
    :param R: Scale factor.
    :return: The Moffat profile.
    """
    return A * (1 + ((r - r0) / R) ** 2) ** (-B)


def moffat_fwhm(R: float, B: float, R_err: float, B_err: float) -> tuple:
    """Calculates the FWHM of a Moffat profile.

    :param R: Scale factor.
    :param B: Power index.
    :param R_err: Scale factor uncertainty.
    :param B_err: Power index uncertainty.
    :return: A tuple containing the FWHM and the FWHM uncertainty.
    """
    FWHM = 2 * R * np.sqrt(2 ** (1 / B) - 1)
    FWHM_err = 2 * R_err * np.sqrt(2 ** (1 / B) - 1) + 2 * R * B_err * \
               (np.log(2) * 2 ** ((1 / B) - 1)) / (B ** 2 * np.sqrt(2 ** (1 / B) - 1))
    return FWHM, FWHM_err
