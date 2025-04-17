import numpy as np
import pandas as pd
from astropy import units as u
from astroquery.vizier import Vizier
from scipy.spatial import KDTree

from .timeout import TimeoutExecutor
from ..logger.hierarchical_logging import setup_logger, hierarchical_debug

logger = setup_logger(__name__)


def crossmatch_sources(source_coords, ref_coords, thres_px=2):
    """
    Cross-match source coordinates with reference coordinates.

    :param source_coords: Array of source coordinates.
    :type source_coords: numpy.ndarray
    :param ref_coords: Array of reference coordinates.
    :type ref_coords: numpy.ndarray
    :param thres_px: Threshold distance in pixels for matching, default is 2.
    :type thres_px: float
    :return: Tuple of matched source and reference indices.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    tree = KDTree(ref_coords)
    dist, idx = tree.query(source_coords, k=1)
    mask = dist < thres_px
    source_coords_matched_idx = np.arange(len(source_coords))[mask]
    ref_coords_matched_idx = idx[mask]
    return source_coords_matched_idx, ref_coords_matched_idx


def _is_valid_result(result: pd.DataFrame, expected_columns: list):
    """
    Validate the result of a catalog query.

    :param result: The query result to validate.
    :type result: pandas.DataFrame or None
    :param expected_columns: List of expected column names in the result.
    :type expected_columns: list or None
    :return: True if the result is valid, False otherwise.
    :rtype: bool
    """
    if result is None:
        return False
    if isinstance(result, pd.DataFrame):
        checks = [
            not result.empty,
            expected_columns is None or all(col in result.columns for col in expected_columns)
        ]
        return all(checks)
    return False


@hierarchical_debug(logger)
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
                                       This function should return a DataFrame or None.
    :type custom_vizier_search_func: callable or None
    :param expected_columns: List of expected column names in the results.
    :type expected_columns: list or None
    :return: A DataFrame containing the results of the query, filtered by the specified parameters.
    :rtype: pandas.DataFrame
    """
    if custom_vizier_search_func is not None:
        try:
            logger.debug(
                f'Using custom Vizier search function: {custom_vizier_search_func.__name__} for catalog: {catalog}')
            if expected_columns is None:
                expected_columns = []
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
            if _is_valid_result(result, expected_columns):
                return result

            logger.warning(f'Custom Vizier search function returned invalid result for catalog: {catalog}')
        except TimeoutExecutor.TimeoutError as e:
            logger.error(f'Timeout error in custom Vizier search function: {e}')
            # return None
        except Exception as e:
            logger.error(f'Error in custom Vizier search function: {e}')
            # return None

    logger.debug(f'Using Vizier catalog: {catalog}')
    Vizier.ROW_LIMIT = vizier_row_limit
    timeout = vizier_timeout
    vizier_results = Vizier(timeout=timeout, row_limit=vizier_row_limit) \
        .query_region(coocenter, radius=radii * u.deg, catalog=catalog,
                      column_filters={ref_filter: '<%.1f ' % maglimit}, cache=vizier_cache)

    return vizier_results[0]


def calculate_solar_index(gmag: pd.Series, rmag: pd.Series) -> pd.Series:
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


@hierarchical_debug(logger)
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
    :return:
        - result (pandas.DataFrame): A DataFrame with calculated magnitudes and associated parameters.
        - catalog (str): The name of the catalog used in the query.
        - ref_filter (str): The reference filter used in calculations.
    :rtype:
        tuple(pandas.DataFrame, str, str)
    """

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

        solar_index = calculate_solar_index(vizier_results['gPSF'], vizier_results['rPSF'])

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
        solar_index = 0.01760 - 0.003226 + (0.3833 + 0.00686) * vizier_results['BP-RP'] + (-0.1345 + 0.1732) * \
                      vizier_results['BP-RP'] ** 2 - 0.36
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

        solar_index = calculate_solar_index(vizier_results['gmag'], vizier_results['rmag'])

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

        solar_index = calculate_solar_index(vizier_results['gmag'], vizier_results['rmag'])

        result = pd.DataFrame({'ID': vizier_results['objID'],
                               'RA': vizier_results['RAJ2000'],
                               'DEC': vizier_results['DEJ2000'],
                               'MAG': vizier_results[final_ref_filter],
                               'MAGERR': vizier_results['e_' + final_ref_filter],
                               'SOLAR': solar_index})
        ref_filter = final_ref_filter[:-3]

    return result, catalog, ref_filter
