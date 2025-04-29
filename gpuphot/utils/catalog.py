from __future__ import annotations

import time

import cupy as cp
import numpy as np
import nvtx
import pandas as pd
from astropy import units as u
from astroquery.vizier import Vizier
from scipy.spatial import KDTree

from .timeout import TimeoutExecutor
from ..logger.hierarchical_logging import setup_logger

logger = setup_logger(__name__)

# Import cuml. Comprobar si está instalado.
try:
    import cuml
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

    CUML_AVAILABLE = True
    logger.debug("RAPIDS cuML found. Using GPU for crossmatch.")
except ImportError:
    logger.warning("Warning: RAPIDS cuML not found. Falling back to CPU crossmatch or GPU crossmatch will fail.")
    CUML_AVAILABLE = False


# --- GPU Implementation Detail ---
@nvtx.annotate('crossmatch_sources_gpu_impl', category='utils.catalog_gpu')
def _crossmatch_sources_gpu_impl(source_coords: cp.ndarray, ref_coords: cp.ndarray,
                                 thres_px: float = 2.0) -> tuple[cp.ndarray, cp.ndarray]:
    """GPU implementation using cuML (Internal use)."""
    # (Código de crossmatch_sources_gpu anterior, sin la comprobación CUML_AVAILABLE)
    nvtx_range = nvtx.start_range('_crossmatch_sources_gpu_impl', category='utils.catalog_gpu', color='magenta')

    if not isinstance(source_coords, cp.ndarray) or not isinstance(ref_coords, cp.ndarray):
        nvtx.end_range(nvtx_range)
        raise TypeError("Inputs must be CuPy arrays for GPU impl.")
    if source_coords.ndim != 2 or ref_coords.ndim != 2:
        nvtx.end_range(nvtx_range)
        raise ValueError("Input arrays must be 2D for GPU impl.")
    n_sources, n_refs = source_coords.shape[0], ref_coords.shape[0]
    if n_sources == 0 or n_refs == 0:
        nvtx.end_range(nvtx_range)
        return cp.array([], dtype=cp.int32), cp.array([], dtype=cp.int32)

    source_coords_f32 = source_coords.astype(cp.float32, copy=False)
    ref_coords_f32 = ref_coords.astype(cp.float32, copy=False)

    try:
        knn_range = nvtx.start_range('cuml_knn', category='cuml')
        nn = cuNearestNeighbors(n_neighbors=1, algorithm='rbc')
        nn.fit(ref_coords_f32)
        distances, indices = nn.kneighbors(source_coords_f32)
        nvtx.end_range(knn_range)

        filter_range = nvtx.start_range('filter_results_gpu', category='utils.catalog_gpu')
        distances_sq = distances.squeeze()
        ref_indices_all = indices.squeeze()
        thres_px_sq = thres_px ** 2
        mask = distances_sq < thres_px_sq
        source_coords_matched_idx = cp.arange(n_sources, dtype=cp.int32)[mask]
        ref_coords_matched_idx = ref_indices_all[mask]
        nvtx.end_range(filter_range)

    except Exception as e:
        logger.error(f"Error during cuML NearestNeighbors operation: {e}")
        nvtx.end_range(nvtx_range)
        raise RuntimeError(f"cuML crossmatch failed: {e}") from e

    nvtx.end_range(nvtx_range)
    return source_coords_matched_idx, ref_coords_matched_idx


# --- CPU Implementation Detail ---
@nvtx.annotate('crossmatch_sources_cpu_impl', category='utils.catalog_cpu')
def _crossmatch_sources_cpu_impl(source_coords: np.ndarray, ref_coords: np.ndarray,
                                 thres_px: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """CPU implementation using KDTree (Internal use)."""
    # (Código de crossmatch_sources_cpu anterior)
    nvtx_range = nvtx.start_range('_crossmatch_sources_cpu_impl', category='utils.catalog_cpu', color='blue')

    if not isinstance(source_coords, np.ndarray) or not isinstance(ref_coords, np.ndarray):
        nvtx.end_range(nvtx_range)
        raise TypeError("Inputs must be NumPy arrays for CPU impl.")
    if source_coords.ndim != 2 or ref_coords.ndim != 2:
        nvtx.end_range(nvtx_range)
        raise ValueError("Input arrays must be 2D for CPU impl.")
    if source_coords.shape[0] == 0 or ref_coords.shape[0] == 0:
        nvtx.end_range(nvtx_range)
        return np.array([], dtype=int), np.array([], dtype=int)

    try:
        kdtree_range = nvtx.start_range('scipy_kdtree', category='scipy')
        tree = KDTree(ref_coords)
        dist, idx = tree.query(source_coords, k=1)
        nvtx.end_range(kdtree_range)

        filter_range = nvtx.start_range('filter_results_cpu', category='cpu_ops')
        mask = dist < thres_px
        source_coords_matched_idx = np.arange(len(source_coords), dtype=int)[mask]
        ref_coords_matched_idx = idx[mask]
        nvtx.end_range(filter_range)

    except Exception as e:
        logger.error(f"Error during KDTree operation: {e}")
        nvtx.end_range(nvtx_range)
        raise RuntimeError(f"KDTree crossmatch failed: {e}") from e

    nvtx.end_range(nvtx_range)
    return source_coords_matched_idx, ref_coords_matched_idx


# --- Wrapper Function (La que se debe llamar desde fuera) ---
@nvtx.annotate('crossmatch_sources', category='utils.catalog')
def crossmatch_sources(source_coords, ref_coords, thres_px: float = 2.0):
    """
    Cross-match source coordinates with reference coordinates.
    Automatically attempts GPU (cuML) acceleration if available and inputs are CuPy arrays.
    Falls back to CPU (SciPy KDTree) if cuML is unavailable, inputs are NumPy arrays,
    or if the GPU method fails.

    :param source_coords: Array of source coordinates (N, D). Can be NumPy or CuPy.
    :param ref_coords: Array of reference coordinates (M, D). Can be NumPy or CuPy.
                       Must be same type as source_coords.
    :param thres_px: Threshold distance in pixels for matching.
    :type thres_px: float
    :return: Tuple of matched source indices and matched reference indices.
             Return type (NumPy or CuPy) matches the input type.
    :rtype: tuple
    """
    nvtx_range = nvtx.start_range('crossmatch_sources_wrapper', category='utils.catalog', color='gray')

    is_gpu_input = isinstance(source_coords, cp.ndarray)
    is_cpu_input = isinstance(source_coords, np.ndarray)

    # Verify input types consistency
    if type(source_coords) != type(ref_coords):
        nvtx.end_range(nvtx_range)
        raise TypeError(
            f"source_coords ({type(source_coords)}) and ref_coords ({type(ref_coords)}) must be of the same type (both NumPy or both CuPy).")

    if not is_gpu_input and not is_cpu_input:
        nvtx.end_range(nvtx_range)
        raise TypeError("Inputs must be NumPy or CuPy arrays.")

    use_gpu_attempt = False  # Flag to track if we even try GPU
    if CUML_AVAILABLE and is_gpu_input:
        use_gpu_attempt = True
        logger.debug("Attempting GPU crossmatch.")
        try:
            result = _crossmatch_sources_gpu_impl(source_coords, ref_coords, thres_px)
            logger.debug("GPU crossmatch successful.")
            nvtx.end_range(nvtx_range)
            return result  # Return GPU result directly
        except Exception as gpu_e:
            logger.warning(f"GPU crossmatch failed: {gpu_e}. Falling back to CPU.", exc_info=False)
            # Fallback will happen below, no need to set use_gpu_attempt back to False

    # --- CPU Path (if GPU not attempted, GPU failed, or input was CPU) ---
    # This block executes if:
    # 1. CUML_AVAILABLE is False
    # 2. is_gpu_input is False
    # 3. GPU was attempted but failed (Exception caught above)
    logger.debug("Using CPU crossmatch.")
    # Prepare NumPy arrays for CPU implementation
    if is_gpu_input:
        # Need to transfer data from GPU to CPU for fallback
        transfer_range = nvtx.start_range('transfer_gpu_to_cpu_fallback', category='transfer', color='red')
        source_np = source_coords.get()
        ref_np = ref_coords.get()
        nvtx.end_range(transfer_range)
    else:  # Input was already CPU
        source_np = source_coords
        ref_np = ref_coords

    # Call CPU implementation
    try:
        result_np_src, result_np_ref = _crossmatch_sources_cpu_impl(source_np, ref_np, thres_px)
        logger.debug("CPU crossmatch successful.")
    except Exception as cpu_e:
        logger.error(f"CPU crossmatch failed: {cpu_e}")
        nvtx.end_range(nvtx_range)
        raise  # Re-raise the exception from the CPU implementation

    # Determine final return type based on ORIGINAL input type
    if is_gpu_input:
        # Original input was GPU, transfer results back
        transfer_back_range = nvtx.start_range('transfer_cpu_to_gpu_fallback_result', category='transfer', color='red')
        result_cp_src = cp.asarray(result_np_src)
        result_cp_ref = cp.asarray(result_np_ref)
        nvtx.end_range(transfer_back_range)
        logger.debug("Returning fallback results transferred back to GPU.")
        nvtx.end_range(nvtx_range)
        return result_cp_src, result_cp_ref
    else:
        # Original input was CPU, return NumPy results
        logger.debug("Returning CPU results (original input was CPU).")
        nvtx.end_range(nvtx_range)
        return result_np_src, result_np_ref


### # @hierarchical_debug(logger)
@nvtx.annotate('__getVizier', category='utils.catalog')
def __getVizier(catalog, coocenter, radii, maglimit, ref_filter,
                vizier_timeout=60, vizier_row_limit=-1,
                vizier_cache=True, custom_vizier_search_func=None,
                expected_columns=None, **kwargs):
    """
    Retrieve astronomical data from the Vizier catalog based on specified parameters.

    :param catalog: The name of the Vizier catalog to query.
    :type catalog: str
    :param coocenter: The coordinates (center) around which to search for objects.
    :type coocenter: astropy.coordinates.SkyCoord
    :param radii: The radius within which to search for objects (in degrees).
    :type radii: float
    :param maglimit: The magnitude limit for filtering results.
    :type maglimit: float
    :param ref_filter: The reference filter used for magnitude filtering.
    :type ref_filter: str
    :param vizier_timeout: Timeout duration for the query (default is 60 seconds).
    :type vizier_timeout: int
    :param vizier_row_limit: Maximum number of rows to return from the query (default is -1, which means no limit).
    :type vizier_row_limit: int
    :param vizier_cache: Boolean flag to enable or disable caching of results (default is True).
    :type vizier_cache: bool
    :param custom_vizier_search_func: Optional custom function for querying the Vizier catalog.
                                       This function must accept the following parameters:
                                       - coocenter: The coordinates around which to search.
                                       - catalog: The name of the Vizier catalog to query.
                                       - radius: The search radius in degrees.
                                       - mag_limit: The magnitude limit for filtering results.
                                       - ref_filter: The reference filter used for magnitude filtering.
                                       - row_limit: Maximum number of rows to return from the query.
                                       - expected_columns: List of expected column names in the results.
    :type custom_vizier_search_func: callable or None
    :param expected_columns: List of expected column names in the results.
    :type expected_columns: list or None
    :return: A DataFrame containing the results of the query, filtered by the specified parameters.
    :rtype: pandas.DataFrame
    """
    if custom_vizier_search_func is not None:
        try:
            executor = TimeoutExecutor(timeout=vizier_timeout)
            result = executor.execute(
                custom_vizier_search_func,
                coocenter=coocenter,
                catalog=catalog,
                radius=float(radii),
                mag_limit=float(maglimit),
                ref_filter=ref_filter,
                row_limit=int(vizier_row_limit),
                expected_columns=list(set(expected_columns))
            )
            return result
        except TimeoutExecutor.TimeoutError as e:
            logger.error(f'Timeout error in custom Vizier search function: {e}')
            return None
        except Exception as e:
            logger.error(f'Error in custom Vizier search function: {e}')
            return None

    Vizier.ROW_LIMIT = vizier_row_limit
    timeout = vizier_timeout
    vizier_results = Vizier(timeout=timeout, row_limit=vizier_row_limit) \
        .query_region(coocenter, radius=radii * u.deg, catalog=catalog,
                      column_filters={ref_filter: '<%.1f ' % maglimit}, cache=vizier_cache)

    return vizier_results[0]


def calculate_gaia_solar_index(bprp_color: pd.Series) -> pd.Series:
    """
    Calculate a solar index using a specific quadratic transformation
    from Gaia DR3 BP-RP color.

    :param bprp_color: Gaia DR3 BP-RP color index (BPmag - RPmag).
    :type bprp_color: pd.Series
    :return: Calculated solar index based on the transformation.
    :rtype: pd.Series
    """

    solar_index = (0.01760 - 0.003226) + \
                  (0.3833 + 0.00686) * bprp_color + \
                  (-0.1345 + 0.1732) * bprp_color ** 2 - 0.36

    return solar_index


def calculate_ps1_solar_index(gmag: pd.Series, rmag: pd.Series) -> pd.Series:
    """
    Calculate the solar color index using Pan-STARRS photometric transformations.

    :param gmag: PS1 g-band magnitudes (AB system).
    :type gmag: pd.Series
    :param rmag: PS1 r-band magnitudes (AB system).
    :type rmag: pd.Series
    :return: Solar color index (B - V - 0.65) using transformations from Tonry et al. 2012.
    :rtype: pd.Series

    Reference:
    Tonry J.L. et al. (2012), "The Pan-STARRS1 Photometric System",
    arXiv:1706.06147 [astro-ph.IM]. URL: https://arxiv.org/abs/1706.06147
    """
    # Calculate color term (g - r)
    color = gmag - rmag

    # Transform to Johnson-Cousins B and V magnitudes (Eq.13-14 from paper)
    B = gmag + 0.194 + 0.561 * color  # B = g + 0.194 + 0.561(g - r)
    V = gmag - 0.017 - 0.508 * color  # V = g - 0.017 - 0.508(g - r)

    # Compute solar index relative to solar color (B - V)_sun = 0.65
    return B - V - 0.65


### # @hierarchical_debug(logger)
@nvtx.annotate('catalog_results', category='utils.catalog')
def catalog_results(coocenter, radius, filter, maglimit=23, **kwargs):
    """
    Process astronomical data to calculate magnitudes and other parameters for stars based on various filters and models.

    :param coocenter: The coordinates (center) around which to search for objects.
    :type coocenter: astropy.coordinates.SkyCoord
    :param radius: The radius within which to search for objects (in degrees).
    :type radius: float
    :param filter: The specific filter type used to determine which catalog to query.
    :type filter: str
    :param maglimit: The magnitude limit for filtering results (default is 23).
    :type maglimit: float
    :param kwargs: Additional keyword arguments passed to __getVizier, including luminosity coefficients.
    :return: A tuple containing:
             - result: A DataFrame with calculated magnitudes and associated parameters.
             - catalog: The name of the catalog used in the query.
             - ref_filter: The reference filter used in calculations.
    :rtype: tuple(pandas.DataFrame, str, str)
    """
    logger.info(f'Attempting to retrieve data from catalog.')
    start_time = time.time()
    if coocenter.dec.deg < -30:
        def get_filter(_filter):
            filter_map = {
                'SDSSg': 'gPSF',
                'SDSSr': 'rPSF',
                'SDSSi': 'iPSF',
                'SDSSzs': 'zPSF',
                'SDSSu': 'uPSF',
            }
            return filter_map.get(_filter, 'gPSF')

        catalog = 'II/379'  # SkyMapper Southern Sky Survey. DR4 : II/379
        ref_filter = 'gPSF'
        final_ref_filter = get_filter(filter)
        expected_columns = ['SMSS', 'RAICRS', 'DEICRS', 'gPSF', 'rPSF', final_ref_filter]  # Columnas para SkyMapper
        vizier_results = __getVizier(catalog, coocenter, radius, maglimit, ref_filter,
                                     expected_columns=expected_columns, **kwargs)

        solar_index = calculate_ps1_solar_index(vizier_results['gPSF'], vizier_results['rPSF'])

        result = pd.DataFrame({'ID': vizier_results['SMSS'],
                               'RA': vizier_results['RAICRS'],
                               'DEC': vizier_results['DEICRS'],
                               'MAG': vizier_results[final_ref_filter],
                               'MAGERR': vizier_results[final_ref_filter] * 0,
                               'SOLAR': solar_index})
        ref_filter = final_ref_filter[:-3]

    elif filter == 'Open' or filter == 'OPEN':
        catalog = 'I/355/gaiadr3'  # Gaia DR3 Part 1. Main source : I/355
        ref_filter = 'BPmag'
        expected_columns = ['Source', 'RAJ2000', 'DEJ2000', 'BP-RP', f'F{ref_filter[:2]}',
                            f'e_F{ref_filter[:2]}', ref_filter]  # Columnas para Gaia DR3
        vizier_results = __getVizier(catalog, coocenter, radius, maglimit, ref_filter,
                                     expected_columns=expected_columns, **kwargs)
        # solar_index = 0.01760 - 0.003226 + (0.3833 + 0.00686) * vizier_results['BP-RP'] + (-0.1345 + 0.1732) * \
        #               vizier_results['BP-RP'] ** 2 - 0.36
        solar_index = calculate_gaia_solar_index(vizier_results['BP-RP'])
        magerr = -2.5 * np.log10(vizier_results['F' + ref_filter[:2]] / (
                vizier_results['F' + ref_filter[:2]] + vizier_results['e_F' + ref_filter[:2]]))
        result = pd.DataFrame({'ID': vizier_results['Source'],
                               'RA': vizier_results['RAJ2000'],
                               'DEC': vizier_results['DEJ2000'],
                               'MAG': vizier_results[ref_filter],
                               'MAGERR': magerr,
                               'SOLAR': solar_index})
        ref_filter = ref_filter[:-3]

    elif filter == 'SDSSu':
        catalog = 'I/353/gsc242'
        ref_filter = 'umag'
        expected_columns = ['GSC2', 'RA_ICRS', 'DE_ICRS', ref_filter]  # Columnas para GSC2
        vizier_results = __getVizier(catalog, coocenter, radius, maglimit, ref_filter,
                                     expected_columns=expected_columns, **kwargs)
        result = pd.DataFrame({'ID': vizier_results['GSC2'],
                               'RA': vizier_results['RA_ICRS'],
                               'DEC': vizier_results['DE_ICRS'],
                               'MAG': vizier_results[ref_filter],
                               'MAGERR': np.zeros(len(vizier_results['RA_ICRS'])),  # uncertainty unknown
                               'SOLAR': np.zeros(
                                   len(vizier_results['RA_ICRS']))})  # for u filter no solar colors are applied

    elif filter == 'Lum' or filter == 'w':
        catalog = 'II/349/ps1'  # The Pan-STARRS release 1 (PS1) Survey - DR1 : II/349
        ref_filter = 'gmag'
        expected_columns = ['objID', 'RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'e_gmag',
                            'e_rmag']  # Columnas para Pan-STARRS
        vizier_results = __getVizier(catalog, coocenter, radius, maglimit, ref_filter,
                                     expected_columns=expected_columns, **kwargs)

        solar_index = calculate_ps1_solar_index(vizier_results['gmag'], vizier_results['rmag'])

        lum_gmag_coeff = kwargs.get('lum_gmag_coeff', 0.5)
        lum_rmag_coeff = kwargs.get('lum_rmag_coeff', 0.5)

        mag = lum_gmag_coeff * vizier_results['gmag'] + lum_rmag_coeff * vizier_results['rmag']
        magerr = lum_gmag_coeff * vizier_results['e_gmag'] + lum_rmag_coeff * vizier_results['e_rmag']
        ref_filter = f'{lum_gmag_coeff}*g+{lum_rmag_coeff}*r'

        result = pd.DataFrame({'ID': vizier_results['objID'],
                               'RA': vizier_results['RAJ2000'],
                               'DEC': vizier_results['DEJ2000'],
                               'MAG': mag,
                               'MAGERR': magerr,
                               'SOLAR': solar_index})

    else:
        def get_filter(_filter):
            filter_map = {
                'SDSSg': 'gmag',
                'SDSSr': 'rmag',
                'SDSSi': 'imag',
                'SDSSzs': 'zmag'
            }
            return filter_map.get(_filter, 'gmag')

        catalog = 'II/349/ps1'  # The Pan-STARRS release 1 (PS1) Survey - DR1 : II/349
        ref_filter = 'gmag'
        final_ref_filter = get_filter(filter)
        expected_columns = ['objID', 'RAJ2000', 'DEJ2000', 'gmag', 'rmag', final_ref_filter,
                            f'e_{final_ref_filter}']  # Columnas por defecto para Pan-STARRS
        vizier_results = __getVizier(catalog, coocenter, radius, maglimit, ref_filter,
                                     expected_columns=expected_columns, **kwargs)

        solar_index = calculate_ps1_solar_index(vizier_results['gmag'], vizier_results['rmag'])

        result = pd.DataFrame({'ID': vizier_results['objID'],
                               'RA': vizier_results['RAJ2000'],
                               'DEC': vizier_results['DEJ2000'],
                               'MAG': vizier_results[final_ref_filter],
                               'MAGERR': vizier_results['e_' + final_ref_filter],
                               'SOLAR': solar_index})
        ref_filter = final_ref_filter[:-3]

    logger.info(f"Catalog query completed in {time.time() - start_time:.2f} seconds.")
    return result, catalog, ref_filter
