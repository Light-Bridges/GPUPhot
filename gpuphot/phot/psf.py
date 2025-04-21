from __future__ import annotations

import cupy as cp
import numpy as np
import nvtx
from cupyx.scipy.ndimage import maximum_filter
from lmfit import Model
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

try:
    from cuml import AgglomerativeClustering
except ImportError:
    from sklearn.cluster import AgglomerativeClustering

try:
    from cuml.decomposition import PCA
except ImportError:
    from sklearn.decomposition import PCA
try:
    import cuml
    from cuml.cluster import AgglomerativeClustering as cuAgglomerativeClustering
    from cuml.metrics import pairwise_distances as cu_pairwise_distances

    CUML_CLUSTERING_AVAILABLE = True
    # logger.debug("RAPIDS cuML Clustering & Metrics found.")
except ImportError:
    # logger.warning("Warning: RAPIDS cuML Clustering/Metrics not found. Grouping will use CPU (sklearn/scipy).")
    CUML_CLUSTERING_AVAILABLE = False

# Import CPU libraries unconditionally for fallback
from sklearn.cluster import AgglomerativeClustering as skAgglomerativeClustering

from .conv import gaussian_kernel, convolve_fft, fill_nan_fft
from .utils import calculate_tile_nanmean_sigclip, decompose_into_tiles, recompose_from_percentiles
from ..exceptions import InsufficientStarsError
from ..logger.hierarchical_logging import setup_logger

logger = setup_logger(__name__)


### # @hierarchical_debug(logger)
@nvtx.annotate('find_local_max', category='phot.psf')
def find_local_max(image: cp.ndarray, min_distance: int, threshold_abs: float) -> cp.ndarray:
    """
    Calculate local maxima in an image.

    :param image: Image array to be processed.
    :type image: cupy.ndarray
    :param min_distance: Minimum distance between peaks.
    :type min_distance: int
    :param threshold_abs: Absolute threshold for peaks.
    :type threshold_abs: float
    :return: Array of detected peaks.
    :rtype: cupy.ndarray
    """
    max_mask = image == maximum_filter(image, size=min_distance)
    threshold_mask = image > threshold_abs
    peaks = cp.logical_and(max_mask, threshold_mask)
    return cp.argwhere(peaks)


### # @hierarchical_debug(logger)
@nvtx.annotate('find_local_centroid', category='phot.psf')
def find_local_centroid(image: cp.ndarray, peaks: cp.ndarray, window_size: int = 5) -> cp.ndarray:
    """
    Calculate centroids of detected peaks in an image.

    :param image: Image where the peaks are located.
    :type image: cupy.ndarray
    :param peaks: Coordinates of the peaks.
    :type peaks: cupy.ndarray
    :param window_size: Size of the square neighborhood around the peak.
    :type window_size: int
    :return: Array of calculated centroids.
    :rtype: cupy.ndarray
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


### # @hierarchical_debug(logger)
@nvtx.annotate('detect_isolated_stars', category='phot.psf')
def detect_isolated_stars(img: cp.ndarray, rms: cp.ndarray, pxscale: float, sat_lim: int = 50000, min_snr: float = 10,
                          dist_asec: float = 10, sort: bool = True, **kwargs) -> cp.array:
    """
    Detect isolated stars in an image using a fft convolution kernel.
    The stars are detected by convolving the image with a Gaussian kernel and filtered by a minimum signal-to-noise ratio.

    :param img: Image array to be processed.
    :type img: cupy.ndarray
    :param rms: RMS of the image.
    :type rms: cupy.ndarray
    :param pxscale: Pixel scale in arcsec/pixel.
    :type pxscale: float
    :param sat_lim: Saturation limit.
    :type sat_lim: int
    :param min_snr: Minimum signal-to-noise ratio.
    :type min_snr: float
    :param dist_asec: Minimum distance in arcsec.
    :type dist_asec: float
    :param sort: Whether to sort the detected stars by signal-to-noise ratio and distance.
    :type sort: bool
    :return: An array containing the coordinates of the detected stars.
    :rtype: cupy.ndarray
    :raises InsufficientStarsError: If less than 5 isolated stars are detected.
    """

    mempool = cp.get_default_memory_pool()
    dist_px = max(dist_asec / pxscale, 20)
    border = 2 * dist_px
    kernel = gaussian_kernel(int(np.max((5 * 2 + 1, 10 / pxscale))), 2)
    kernel = (kernel - cp.mean(kernel)) / cp.std(kernel)

    # Usar un contexto para conv_ima y conv_sigma
    with cp.cuda.Stream():  # Asegura la ejecución asíncrona y la liberación de recursos
        conv_ima = convolve_fft(img, kernel, **kwargs)
        conv_sigma = conv_ima / rms / cp.sqrt(kernel.shape[0] * kernel.shape[1])
        del kernel, conv_ima  # Liberar kernel y conv_ima tan pronto como sea posible
        mempool.free_all_blocks()  # Asegurar liberación

        conv_sigma[:border, :] = 0
        conv_sigma[-border:, :] = 0
        conv_sigma[:, :border] = 0
        conv_sigma[:, -border:] = 0
        coor_f = find_local_max(conv_sigma, min_distance=int(3 / pxscale), threshold_abs=min_snr)
        dist = get_centroids_distance_kdtree(coor_f.get())
        dist_mask = dist > dist_px
        coor_f = coor_f[dist_mask]
        dist = dist[dist_mask]  # Actualizar dist después del filtrado
        snr = conv_sigma[coor_f[:, 0], coor_f[:, 1]]
        peak = img[coor_f[:, 0], coor_f[:, 1]]
        m = (snr > min_snr) & (peak < sat_lim)
        if cp.sum(m) == 0:
            m = (snr > 3) & (peak < sat_lim)
        if cp.sum(m) == 0:
            logger.error('Less than 5 isolated stars detected. Image may be too crowded or too noisy')
            raise InsufficientStarsError(num_stars=0)

        coor_f = cp.asarray(coor_f)[m]
        coor_f = find_local_centroid(conv_sigma, coor_f, int(3 / pxscale))

        del conv_sigma  # Liberar antes del sort

        if sort:
            # Calcular sort_metric en la GPU si es posible
            sort_metric = snr[m].get() + dist[m.get()]  # Ahora dist ya ha sido filtrado.
            idx = cp.argsort(np.max(
                sort_metric) - sort_metric)  # Se calcula con numpy ya que la cantidad de datos a ordenar es pequeña
            coor_f = coor_f[idx]

        # Liberación de memoria
        del snr, peak, m, dist, dist_mask, sort_metric, idx

    mempool.free_all_blocks()
    return coor_f


### # @hierarchical_debug(logger)
@nvtx.annotate('create_star_dataset', category='phot.psf')
def create_star_dataset(img: cp.ndarray, coords: cp.array, pxscale: float, N: int = 1000) -> tuple:
    """
    Create a dataset of stars from an image and a list of coordinates.

    :param img: Image array to be processed.
    :type img: cupy.ndarray
    :param coords: Array containing the coordinates of the stars.
    :type coords: cupy.ndarray
    :param pxscale: Pixel scale in arcsec/pixel.
    :type pxscale: float
    :param N: Number of stars to be selected.
    :type N: int
    :return: A tuple containing the star dataset, the coordinates of the stars, and the scaling dataset.
    :rtype: tuple(cupy.ndarray, cupy.ndarray, cupy.ndarray)
    """
    f = max(int(10 / pxscale), 6)
    n = max(int(1 / pxscale), 2)
    star_dataset = cp.zeros((len(coords), 2 * f + 1, 2 * f + 1), dtype=cp.float32)
    scaling_dataset = cp.zeros((len(coords), 4), dtype=cp.float32)  # Solo necesitamos el pico
    valid_coords = []  # Lista para almacenar las coordenadas válidas

    for i, (y, x) in enumerate(coords):
        x_min = int(x - f)
        x_max = int(x + f + 1)
        y_min = int(y - f)
        y_max = int(y + f + 1)

        # Comprobar límites *antes* de acceder a la imagen.
        if 0 <= x_min < x_max <= img.shape[1] and 0 <= y_min < y_max <= img.shape[0]:
            subima = img[y_min:y_max, x_min:x_max]
            peak_pos = cp.unravel_index(cp.argmax(subima), subima.shape)
            if abs(peak_pos[0] - f) <= n and abs(peak_pos[1] - f) <= n:
                peak = subima[peak_pos]
                star_dataset[i, :] = subima / peak  # Normalizar por el pico
                scaling_dataset[i, 0] = peak
                scaling_dataset[i, 1] = cp.mean(subima)
                scaling_dataset[i, 2] = cp.var(subima)
                scaling_dataset[i, 3] = cp.sum(subima)
                valid_coords.append(coords[i])  # Usamos una lista

    valid_coords = cp.array(valid_coords)  # Convertimos a array
    N = min(N, len(valid_coords))
    # Usar slicing para seleccionar los primeros N elementos *después* de filtrar.
    return star_dataset[:N], valid_coords[:N], scaling_dataset[:N]


@nvtx.annotate('_group_star_dataset_cpu_impl', category='phot.psf_cpu')
def _group_star_dataset_cpu_impl(coords: np.ndarray, avg_group_size: int = 10, min_group_size: int = 5) -> np.ndarray:
    """CPU implementation using sklearn/scipy."""
    if avg_group_size <= 0 and min_group_size <= 0:
        # raise InvalidGroupSizeError(avg_group_size, min_group_size)
        raise ValueError("avg_group_size and min_group_size must be positive")

    n_stars = len(coords)
    if n_stars == 0: return np.array([], dtype=int)
    if n_stars <= min_group_size: return np.zeros(n_stars, dtype=int)

    effective_avg_group_size = max(min_group_size, avg_group_size)
    num_clusters = max(1, n_stars // effective_avg_group_size)
    num_clusters = min(num_clusters, n_stars)

    # Initial Clustering (Scikit-learn)
    clustering = skAgglomerativeClustering(n_clusters=num_clusters)
    labels = clustering.fit_predict(coords)

    unique_labels_initial, counts = np.unique(labels, return_counts=True)

    # Identify valid/small groups
    valid_mask = counts >= min_group_size
    valid_group_ids_list = unique_labels_initial[valid_mask].tolist()
    small_group_labels = unique_labels_initial[~valid_mask]

    # Check if any valid groups exist
    if not valid_group_ids_list:
        # logger.warning(f"Warning: No CPU clusters met min_group_size ({min_group_size}). Assigning all stars to group 0.")
        return np.zeros(n_stars, dtype=int)

    # Get coords and original indices for small groups
    small_groups_mask = np.isin(labels, small_group_labels)
    small_groups_coords = coords[small_groups_mask]
    small_groups_orig_indices = np.where(small_groups_mask)[0]

    # Only proceed if there are stars to reassign
    if small_groups_coords.shape[0] == 0:
        # No small groups, initial labels are final (maybe renumber)
        pass  # Skip reassignment if no small groups
    else:
        # Calculate centroids ONCE
        valid_centroids = np.array([np.mean(coords[labels == gid], axis=0) for gid in valid_group_ids_list])

        # Calculate distances (SciPy cdist)
        # Note: cdist calculates all pairs, slightly different from pairwise_distances
        distances = cdist(small_groups_coords, valid_centroids)  # Shape (n_small, n_valid)

        # Find closest valid centroid indices
        closest_valid_idx = np.argmin(distances, axis=1)  # Index into valid_centroids/valid_group_ids_list

        # Get the actual labels of the closest groups
        closest_group_ids = np.array(valid_group_ids_list)[closest_valid_idx]

        # Update labels
        final_labels = labels.copy()
        final_labels[small_groups_orig_indices] = closest_group_ids
        labels = final_labels  # Use updated labels from here

    # Renumber labels (Optional but good practice)
    unique_final_labels = np.unique(labels)
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_final_labels)}
    final_labels_contiguous = np.array([label_map[l] for l in labels], dtype=int)

    return final_labels_contiguous


# --- Implementación GPU ---
@nvtx.annotate('_group_star_dataset_gpu_impl', category='phot.psf_gpu')
def _group_star_dataset_gpu_impl(coords: cp.ndarray, avg_group_size: int = 10, min_group_size: int = 5) -> cp.ndarray:
    """GPU implementation using CuPy/cuML."""
    if avg_group_size <= 0 and min_group_size <= 0:
        # raise InvalidGroupSizeError(avg_group_size, min_group_size)
        raise ValueError("avg_group_size and min_group_size must be positive")

    n_stars = len(coords)
    if n_stars == 0: return cp.array([], dtype=cp.int32)
    if n_stars <= min_group_size: return cp.zeros(n_stars, dtype=cp.int32)

    # Ensure coords are float32 for cuML
    coords_f32 = coords.astype(cp.float32, copy=False)

    effective_avg_group_size = max(min_group_size, avg_group_size)
    num_clusters = max(1, n_stars // effective_avg_group_size)
    num_clusters = min(num_clusters, n_stars)

    # Initial Clustering (cuML)
    clustering = cuAgglomerativeClustering(n_clusters=num_clusters)
    labels = clustering.fit_predict(coords_f32)
    labels = labels.astype(cp.int32, copy=False)  # Ensure int type

    unique_labels_initial, counts = cp.unique(labels, return_counts=True)

    # Identify valid/small groups
    valid_mask = counts >= min_group_size
    valid_group_ids = unique_labels_initial[valid_mask]  # Keep on GPU
    small_group_labels = unique_labels_initial[~valid_mask]

    # Check if any valid groups exist
    if valid_group_ids.size == 0:
        # logger.warning(f"Warning: No GPU clusters met min_group_size ({min_group_size}). Assigning all stars to group 0.")
        return cp.zeros(n_stars, dtype=cp.int32)

    # Get coords and original indices for small groups
    small_groups_mask = cp.isin(labels, small_group_labels)
    small_groups_coords = coords_f32[small_groups_mask]
    small_groups_orig_indices = cp.where(small_groups_mask)[0]

    # Only proceed if there are stars to reassign
    if small_groups_coords.shape[0] == 0:
        pass  # Skip reassignment if no small groups
    else:
        # Calculate centroids ONCE (GPU)
        # Loop might be necessary here, or more complex vectorized approach
        valid_centroids_list = []
        for gid in valid_group_ids:
            valid_centroids_list.append(cp.mean(coords_f32[labels == gid], axis=0))
        valid_centroids = cp.stack(valid_centroids_list)  # Shape (n_valid, n_features)

        # Calculate distances (cuML pairwise_distances)
        distances = cu_pairwise_distances(small_groups_coords, valid_centroids)  # Shape (n_small, n_valid)

        # Find closest valid centroid indices
        closest_valid_idx = cp.argmin(distances, axis=1)  # Index into valid_centroids/valid_group_ids

        # Get the actual labels of the closest groups
        closest_group_ids = valid_group_ids[closest_valid_idx]

        # Update labels
        final_labels = labels.copy()
        final_labels[small_groups_orig_indices] = closest_group_ids
        labels = final_labels  # Use updated labels from here

    # Renumber labels (GPU version)
    unique_final_labels = cp.unique(labels)
    # Creating map on CPU might be easier unless n_clusters is huge
    label_map_cpu = {old_label.item(): new_label for new_label, old_label in enumerate(unique_final_labels)}
    # Apply map using CuPy (can be slow if map is large and called elementwise)
    # A more advanced approach might use cp.searchsorted or custom kernels
    # Simple approach (potentially slow for millions of stars/labels):
    final_labels_contiguous = cp.array([label_map_cpu[l.item()] for l in labels], dtype=cp.int32)

    return final_labels_contiguous


### # @hierarchical_debug(logger)
# --- Wrapper Function ---
@nvtx.annotate('group_star_dataset', category='phot.psf')
def group_star_dataset(coords, avg_group_size: int = 10, min_group_size: int = 5) -> np.ndarray | cp.ndarray:
    """
    Wrapper to group stars, attempting GPU (cuML) first if available and input is CuPy,
    falling back to CPU (sklearn/scipy).

    :param coords: Coordinates of the stars (n_stars, n_features).
    :type coords: numpy.ndarray
    :param avg_group_size: Desired average group size.
    :type avg_group_size: int
    :param min_group_size: Minimum allowed size for a group.
    :type min_group_size: int
    :return: Array of labels indicating the cluster each star belongs to.
    :rtype: numpy.ndarray
    :raises InvalidGroupSizeError: If both avg_group_size and min_group_size are less than or equal to zero.
    """
    nvtx_range = nvtx.start_range('group_star_dataset_wrapper', category='phot.psf', color='cyan')

    is_gpu_input = isinstance(coords, cp.ndarray)

    if is_gpu_input and CUML_CLUSTERING_AVAILABLE:
        # --- Attempt GPU Path ---
        nvtx_gpu_attempt = nvtx.start_range('attempt_gpu_grouping', category='phot.psf_gpu')
        # logger.debug("Attempting GPU grouping.")
        try:
            result = _group_star_dataset_gpu_impl(coords, avg_group_size, min_group_size)
            # logger.debug("GPU grouping successful.")
            nvtx.end_range(nvtx_gpu_attempt)
            nvtx.end_range(nvtx_range)
            return result  # Return CuPy array
        except Exception as gpu_e:
            # logger.warning(f"GPU grouping failed: {gpu_e}. Falling back to CPU.", exc_info=False)
            nvtx.end_range(nvtx_gpu_attempt)
            # Fallback happens below

    # --- CPU Path (Fallback or Original Input) ---
    nvtx_cpu_path = nvtx.start_range('cpu_grouping_path', category='phot.psf_cpu')
    # logger.debug("Using CPU grouping.")

    # Ensure coords are NumPy for CPU implementation
    if is_gpu_input:  # This means GPU failed, need to transfer
        tx_range = nvtx.start_range('transfer_gpu_to_cpu_group_fallback', category='transfer', color='red')
        coords_np = coords.get()
        nvtx.end_range(tx_range)
    else:  # Input was already NumPy
        coords_np = coords

    # Call CPU implementation
    try:
        result_np = _group_star_dataset_cpu_impl(coords_np, avg_group_size, min_group_size)
        # logger.debug("CPU grouping successful.")
    except Exception as cpu_e:
        # logger.error(f"CPU grouping failed: {cpu_e}")
        nvtx.end_range(nvtx_cpu_path)
        nvtx.end_range(nvtx_range)
        raise  # Re-raise the CPU exception

    nvtx.end_range(nvtx_cpu_path)

    # Determine return type based on ORIGINAL input type
    if is_gpu_input:  # Original was GPU, but we used CPU (fallback)
        tx_back_range = nvtx.start_range('transfer_cpu_to_gpu_group_fallback_result', category='transfer', color='red')
        # logger.debug("Transferring CPU fallback result back to GPU.")
        result_cp = cp.asarray(result_np)
        nvtx.end_range(tx_back_range)
        nvtx.end_range(nvtx_range)
        return result_cp  # Return CuPy array
    else:  # Original was CPU, return NumPy
        nvtx.end_range(nvtx_range)
        return result_np  # Return NumPy array


### # @hierarchical_debug(logger)
@nvtx.annotate('get_eigen_psfs', category='phot.psf')
def get_eigen_psfs(normed_star_dataset: cp.array, n_components: int = 5) -> cp.array:
    """
    Calculate the eigen PSFs from a dataset of normalized stars.

    :param normed_star_dataset: Dataset of normalized stars.
    :type normed_star_dataset: cupy.ndarray
    :param n_components: Number of components to be used.
    :type n_components: int
    :return: An array containing the eigen PSFs.
    :rtype: cupy.ndarray
    """
    # pca = PCA(n_components=n_components)
    # starset_flattened = normed_star_dataset.reshape(normed_star_dataset.shape[0],
    #                                                 normed_star_dataset.shape[1] ** 2).get()
    # pca.fit_transform(starset_flattened)
    # eigen_psfs = pca.components_.reshape(-1, normed_star_dataset.shape[1], normed_star_dataset.shape[2])
    # return eigen_psfs

    # Instanciar PCA con el número de componentes deseado
    pca = PCA(n_components=n_components)
    # Aplanar cada estrella (manteniendo datos en GPU; no se usa .get())
    # (N, H, W) a (N, H*W). Usar -1 en reshape
    # deja que Cupy calcule H*W sin suponer que W == H.
    starset_flattened = normed_star_dataset.reshape(normed_star_dataset.shape[0], -1).get()
    # Ajustar PCA directamente en GPU
    pca.fit(starset_flattened)
    # Obtener los eigen PSFs reestructurando los componentes principales al tamaño original de la imagen
    eigen_psfs = pca.components_.reshape(-1, normed_star_dataset.shape[1], normed_star_dataset.shape[2])
    return eigen_psfs


### # @hierarchical_debug(logger)
@nvtx.annotate('project_all_stars_onto_eigenpsfs', category='phot.psf')
def project_all_stars_onto_eigenpsfs(normed_star_dataset: cp.array, eigen_psfs: cp.array) -> cp.array:
    """
    Project all stars onto the eigen PSFs.

    :param normed_star_dataset: Dataset of normalized stars.
    :type normed_star_dataset: cupy.ndarray
    :param eigen_psfs: Array containing the eigen PSFs.
    :type eigen_psfs: cupy.ndarray
    :return: An array containing the coefficients of the stars projected onto the eigen PSFs.
    :rtype: cupy.ndarray
    """
    num_stars = normed_star_dataset.shape[0]
    flattened_star_dim = normed_star_dataset.shape[1] * normed_star_dataset.shape[2]
    stars_matrix = normed_star_dataset.reshape((num_stars, flattened_star_dim))
    eigen_matrix = eigen_psfs.reshape((eigen_psfs.shape[0], flattened_star_dim)).T
    coefficients_matrix = cp.dot(stars_matrix, eigen_matrix)
    del stars_matrix, eigen_matrix, flattened_star_dim
    return coefficients_matrix


### # @hierarchical_debug(logger)
@nvtx.annotate('create_coeff_map', category='phot.psf')
def create_coeff_map(img_shape: tuple, positions: cp.array, coefficients: cp.array, pxscale: float,
                     tile_section: int = None, env_factor: float = 3) -> cp.ndarray:
    """
    Create a coefficient map from a list of positions and coefficients.

    :param img_shape: Dimensions of the image.
    :type img_shape: tuple
    :param positions: Array containing the positions of the coefficients.
    :type positions: cupy.ndarray
    :param coefficients: Array containing the coefficients.
    :type coefficients: cupy.ndarray
    :param pxscale: Pixel scale in arcsec/pixel.
    :type pxscale: float
    :param tile_section: Size of the tile section.
    :type tile_section: int or None
    :param env_factor: Factor for the environment check for outlier detection.
    :type env_factor: float
    :return: A coefficient map.
    :rtype: cupy.ndarray
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


### # @hierarchical_debug(logger)
@nvtx.annotate('calculate_kernel_area', category='phot.psf')
def calculate_kernel_area(img_shape: tuple, psf: cp.ndarray, coeff_map: cp.ndarray = None,
                          eigen_psfs: cp.ndarray = None):
    """
    Calculate the area of the kernel.

    :param img_shape: Shape of the image.
    :type img_shape: tuple
    :param psf: Point spread function.
    :type psf: cupy.ndarray
    :param coeff_map: Coefficient map, by default None.
    :type coeff_map: cupy.ndarray or None
    :param eigen_psfs: Eigen PSFs, by default None.
    :type eigen_psfs: cupy.ndarray or None
    :return: Map of the kernel area.
    :rtype: cupy.ndarray
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


### # @hierarchical_debug(logger)
@nvtx.annotate('detect_sources_psf', category='phot.psf')
def detect_sources_psf(img: cp.ndarray, rms: cp.ndarray, fwhm: float, psf: cp.array,
                       eigen_psfs: cp.ndarray = None, coeff_map: cp.ndarray = None,
                       min_snr: int = 5, **kwargs) -> cp.ndarray:
    """
    Detect sources using PCA.

    :param img: Input image.
    :type img: cupy.ndarray
    :param rms: Root mean square noise level.
    :type rms: cupy.ndarray
    :param fwhm: Full width at half maximum.
    :type fwhm: float
    :param psf: Reference point spread function.
    :type psf: cupy.ndarray
    :param eigen_psfs: Eigen PSFs.
    :type eigen_psfs: cupy.ndarray or None
    :param coeff_map: Coefficient map.
    :type coeff_map: cupy.ndarray or None
    :param min_snr: Minimum signal-to-noise ratio, by default 5.
    :type min_snr: int
    :return: Tuple containing coordinates of detected sources and the convolved image with PCA.
    :rtype: tuple(cupy.ndarray, cupy.ndarray)
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


@nvtx.annotate('recreate_normed_star', category='phot.psf')
def recreate_normed_star(coeff_map: cp.ndarray, eigen_psfs: cp.array, coords: cp.array) -> cp.ndarray:
    """
    Recreate a normalized star from the coefficient map.

    :param coeff_map: Coefficient map.
    :type coeff_map: cupy.ndarray
    :param eigen_psfs: Array containing the eigen PSFs.
    :type eigen_psfs: cupy.ndarray
    :param coords: Tuple containing the coordinates of the star (x,y).
    :type coords: tuple
    :return: The recreated normalized star.
    :rtype: cupy.ndarray
    """
    x, y = coords
    coeff = coeff_map[:, y, x]
    kernel = cp.dot(coeff, eigen_psfs.reshape((eigen_psfs.shape[0], -1)))
    kernel = kernel.reshape((eigen_psfs.shape[1], eigen_psfs.shape[2]))
    return kernel


### # @hierarchical_debug(logger)
@nvtx.annotate('recreate_normed_star_vectorized', category='phot.psf')
def recreate_normed_star_vectorized(coeff_map: cp.ndarray, eigen_psfs: cp.array, xs: cp.array, ys: cp.array,
                                    **kwargs) -> cp.ndarray:
    """
    Recreate a set of normalized stars from the coefficient map.

    :param coeff_map: Coefficient map.
    :type coeff_map: cupy.ndarray
    :param eigen_psfs: Array containing the eigen PSFs.
    :type eigen_psfs: cupy.ndarray
    :param xs: Array containing the x-coordinates of the stars.
    :type xs: cupy.ndarray
    :param ys: Array containing the y-coordinates of the stars.
    :type ys: cupy.ndarray
    :return: An array containing the recreated normalized stars.
    :rtype: cupy.ndarray
    """

    coeffs = coeff_map[:, ys, xs]
    coeffs = coeffs.reshape(-1, coeffs.shape[-1])
    reshaped_eigen_psfs = eigen_psfs.reshape(eigen_psfs.shape[0], -1)
    kernels = cp.dot(coeffs.T, reshaped_eigen_psfs)
    kernels = kernels.reshape(xs.size, eigen_psfs.shape[1], eigen_psfs.shape[2])
    del coeffs, reshaped_eigen_psfs
    return kernels


@nvtx.annotate('recreate_normed_stars_batch', category='phot.psf')
def recreate_normed_stars_batch(coeff_map: cp.ndarray, eigen_psfs: cp.ndarray, coords: cp.ndarray,
                                **kwargs) -> cp.ndarray:
    """
    Recreate normalized stars for a batch of coordinates.

    :param coeff_map: Coefficient map of shape (num_coeffs, height, width).
    :type coeff_map: cupy.ndarray
    :param eigen_psfs: Array containing the eigen PSFs of shape (num_coeffs, psf_height, psf_width).
    :type eigen_psfs: cupy.ndarray
    :param coords: Array of coordinates of shape (num_points, 2), where each row is (x, y).
    :type coords: cupy.ndarray
    :return: Array of recreated normalized stars of shape (num_points, psf_height, psf_width).
    :rtype: cupy.ndarray
    """
    xs, ys = coords[:, 0], coords[:, 1]
    coeffs = coeff_map[:, ys, xs]
    eigen_psfs_reshaped = eigen_psfs.reshape(eigen_psfs.shape[0], -1)
    kernels = cp.dot(coeffs.T, eigen_psfs_reshaped)
    kernels = kernels.reshape(coords.shape[0], eigen_psfs.shape[1], eigen_psfs.shape[2])
    return kernels


### # @hierarchical_debug(logger)
@nvtx.annotate('fit_moffat', category='phot.psf')
def fit_moffat(star_data: np.ndarray) -> tuple:
    """
    Fit a Moffat profile to a star.

    :param star_data: Image array containing the star data.
    :type star_data: numpy.ndarray
    :return: A tuple containing the radial coordinates, the intensity profile, the fit result, the FWHM, and the FWHM uncertainty.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, lmfit.model.ModelResult, float, float)
    """

    # Reshape the 2D image to 1D arrays
    ax, ay = np.meshgrid(np.arange(star_data.shape[1]), np.arange(star_data.shape[0]))
    X, Y, Z = ax.ravel(), ay.ravel(), star_data.ravel()

    # Transform to radial coordiantes
    center = (np.array([X[np.argmax(Z)], Y[np.argmax(Z)]])).astype(int)
    r = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    del X, Y

    # Remove the sky from the outer part of the star
    sky = np.median(Z[r > np.percentile(r, 0.7)])
    Z = Z.astype(np.float32) - sky

    peak = np.max(Z)
    fwhm = r[np.argmin(np.abs(Z - (np.max(Z) - np.min(Z)) / 2))]

    mask = r < 5 * fwhm
    r = r[mask]
    Z = Z[mask]
    del mask

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


@nvtx.annotate('filter_centroids_kdtree', category='phot.psf')
def filter_centroids_kdtree(centroids: np.array, min_distance: float) -> np.array:
    """
    Filter centroids using a KDTree.

    :param centroids: Array containing the centroids to be filtered.
    :type centroids: numpy.ndarray
    :param min_distance: Minimum distance between centroids.
    :type min_distance: float
    :return: An array containing the filtered centroids.
    :rtype: numpy.ndarray
    """
    distances = get_centroids_distance_kdtree(centroids)
    mask = distances >= min_distance
    return centroids[mask]


### # @hierarchical_debug(logger)
@nvtx.annotate('get_centroids_distance_kdtree', category='phot.psf')
def get_centroids_distance_kdtree(centroids: np.array):
    """
    Find the distance between centroid and its nearest neighbor using KDTree.

    :param centroids: Array of centroids.
    :type centroids: numpy.ndarray
    :return: Array of distances.
    :rtype: numpy.ndarray
    """

    tree = KDTree(centroids)
    dist, _ = tree.query(centroids, k=2)
    return dist[:, 1]


@nvtx.annotate('moffat', category='phot.psf')
def moffat(r: np.array, A: float = 1., r0: float = 0., B: float = 1., R: float = 1.) -> np.array:
    """
    Moffat profile function.
    https://nbviewer.org/github/ysbach/AO_2017/blob/master/04_Ground_Based_Concept.ipynb#1.2.-Moffat

    :param r: Radial coordinates.
    :type r: numpy.ndarray
    :param A: Peak intensity.
    :type A: float
    :param r0: Central position.
    :type r0: float
    :param B: Power index.
    :type B: float
    :param R: Scale factor.
    :type R: float
    :return: The Moffat profile.
    :rtype: numpy.ndarray
    """
    return A * (1 + ((r - r0) / R) ** 2) ** (-B)


@nvtx.annotate('moffat_fwhm', category='phot.psf')
def moffat_fwhm(R: float, B: float, R_err: float, B_err: float) -> tuple:
    """
    Calculate the FWHM of a Moffat profile.

    :param R: Scale factor.
    :type R: float
    :param B: Power index.
    :type B: float
    :param R_err: Scale factor uncertainty.
    :type R_err: float
    :param B_err: Power index uncertainty.
    :type B_err: float
    :return: A tuple containing the FWHM and the FWHM uncertainty.
    :rtype: tuple(float, float)
    """
    FWHM = 2 * R * np.sqrt(2 ** (1 / B) - 1)
    FWHM_err = 2 * R_err * np.sqrt(2 ** (1 / B) - 1) + 2 * R * B_err * \
               (np.log(2) * 2 ** ((1 / B) - 1)) / (B ** 2 * np.sqrt(2 ** (1 / B) - 1))
    return FWHM, FWHM_err
