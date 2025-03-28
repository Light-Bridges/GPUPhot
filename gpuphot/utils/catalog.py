import numpy as np
import nvtx
import pandas as pd
from astropy import units as u
from astroquery.vizier import Vizier
from scipy.spatial import KDTree

from ..logger.hierarchical_logging import setup_logger

logger = setup_logger(__name__)


@nvtx.annotate('crossmatch_sources', category='utils.catalog')
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
                                       - catalog: The name of the Vizier catalog to query.
                                       - coocenter: The coordinates around which to search.
                                       - radii: The search radius in degrees.
                                       - maglimit: The magnitude limit for filtering results.
                                       - ref_filter: The reference filter used for magnitude filtering.
                                       - expected_columns: List of expected column names in the results.
    :type custom_vizier_search_func: callable or None
    :param expected_columns: List of expected column names in the results.
    :type expected_columns: list or None
    :return: A DataFrame containing the results of the query, filtered by the specified parameters.
    :rtype: pandas.DataFrame
    """
    if custom_vizier_search_func is not None:
        return custom_vizier_search_func(catalog, coocenter, radii, maglimit, ref_filter, expected_columns)

    Vizier.ROW_LIMIT = vizier_row_limit
    timeout = vizier_timeout
    vizier_results = Vizier(timeout=timeout, row_limit=vizier_row_limit) \
        .query_region(coocenter, radius=radii * u.deg, catalog=catalog,
                      column_filters={ref_filter: '<%.1f ' % maglimit}, cache=vizier_cache)

    return vizier_results[0]


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
    if coocenter.dec.deg < -30:
        catalog = 'II/379'  # SkyMapper Southern Sky Survey. DR4 : II/379
        ref_filter = 'gPSF'
        expected_columns = ['SMSS', 'RAICRS', 'DEICRS', 'gPSF', 'rPSF']  # Columnas para SkyMapper
        vizier_results = __getVizier(catalog, coocenter, radius, maglimit, ref_filter,
                                     expected_columns=expected_columns, **kwargs)
        color = vizier_results['gPSF'] - vizier_results['rPSF']
        B = vizier_results['gPSF'] + 0.194 + 0.561 * color
        V = vizier_results['gPSF'] - 0.017 - 0.508 * color
        solar_index = B - V - 0.65

        if filter == 'SDSSg':
            ref_filter = 'gPSF'
        elif filter == 'SDSSr':
            ref_filter = 'rPSF'
        elif filter == 'SDSSi':
            ref_filter = 'iPSF'
        elif filter == 'SDSSzs':
            ref_filter = 'zPSF'
        elif filter == 'SDSSu':
            ref_filter = 'uPSF'

        result = pd.DataFrame({'ID': vizier_results['SMSS'],
                               'RA': vizier_results['RAICRS'],
                               'DEC': vizier_results['DEICRS'],
                               'MAG': vizier_results[ref_filter],
                               'MAGERR': vizier_results[ref_filter] * 0,
                               'SOLAR': solar_index})
        ref_filter = ref_filter[:-3]

    elif filter == 'Open' or filter == 'OPEN':
        catalog = 'I/355/gaiadr3'  # Gaia DR3 Part 1. Main source : I/355
        ref_filter = 'BPmag'
        expected_columns = ['Source', 'RAJ2000', 'DEJ2000', 'BP-RP', f'F{ref_filter[:2]}',
                            f'e_F{ref_filter[:2]}']  # Columnas para Gaia DR3
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
        expected_columns = ['objID', 'RAJ2000', 'DEJ2000', 'gmag', 'rmag', f'e_{ref_filter}',
                            f'e_rmag']  # Columnas para Pan-STARRS
        vizier_results = __getVizier(catalog, coocenter, radius, maglimit, ref_filter,
                                     expected_columns=expected_columns, **kwargs)
        color = vizier_results['gmag'] - vizier_results['rmag']
        # B = vizier_results['gmag'] + 0.213 + 0.587 * color
        # V = vizier_results['rmag'] + 0.006 + 0.474 * color
        # https://arxiv.org/pdf/1706.06147.pdf
        B = vizier_results['gmag'] + 0.194 + 0.561 * color
        V = vizier_results['gmag'] - 0.017 - 0.508 * color
        solar_index = B - V - 0.65

        lum_gmag_coeff = kwargs.get('lum_gmag_coeff', 0.5)
        lum_rmag_coeff = kwargs.get('lum_rmag_coeff', 0.5)

        mag = lum_gmag_coeff * vizier_results['gmag'] + lum_rmag_coeff * vizier_results['rmag']
        magerr = lum_gmag_coeff * vizier_results['e_gmag'] + lum_rmag_coeff * vizier_results['e_rmag']
        ref_filter = f'{lum_gmag_coeff}*g+{lum_rmag_coeff}*r'

        # if inmodel == 'iKon936':
        #     mag = 0.46872 * vizier_results['gmag'] + 0.53127 * vizier_results['rmag']
        #     magerr = 0.46872 * vizier_results['e_gmag'] + 0.53127 * vizier_results['e_rmag']
        #     ref_filter = '0.46872*g+0.53127*r'
        #
        # elif inmodel == 'QHY411MERIS':
        #     mag = 0.51595 * vizier_results['gmag'] + 0.48404 * vizier_results['rmag']
        #     magerr = 0.51595 * vizier_results['e_gmag'] + 0.48404 * vizier_results['e_rmag']
        #     ref_filter = '0.51595*g+0.48404*r'
        # else:
        #     mag = 0.5 * vizier_results['gmag'] + 0.5 * vizier_results['rmag']
        #     magerr = 0.5 * vizier_results['e_gmag'] + 0.5 * vizier_results['e_rmag']
        #     ref_filter = '0.5*g+0.5*r'

        result = pd.DataFrame({'ID': vizier_results['objID'],
                               'RA': vizier_results['RAJ2000'],
                               'DEC': vizier_results['DEJ2000'],
                               'MAG': mag,
                               'MAGERR': magerr,
                               'SOLAR': solar_index})

    else:
        catalog = 'II/349/ps1'  # The Pan-STARRS release 1 (PS1) Survey - DR1 : II/349
        ref_filter = 'gmag'
        expected_columns = ['objID', 'RAJ2000', 'DEJ2000', ref_filter,
                            f'e_{ref_filter}']  # Columnas por defecto para Pan-STARRS
        vizier_results = __getVizier(catalog, coocenter, radius, maglimit, ref_filter,
                                     expected_columns=expected_columns, **kwargs)
        color = vizier_results['gmag'] - vizier_results['rmag']
        # B = vizier_results['gmag'] + 0.213 + 0.587 * color
        # V = vizier_results['rmag'] + 0.006 + 0.474 * color
        # https://arxiv.org/pdf/1706.06147.pdf
        B = vizier_results['gmag'] + 0.194 + 0.561 * color
        V = vizier_results['gmag'] - 0.017 - 0.508 * color
        solar_index = B - V - 0.65

        if filter == 'SDSSg':
            ref_filter = 'gmag'
        elif filter == 'SDSSr':
            ref_filter = 'rmag'
        elif filter == 'SDSSi':
            ref_filter = 'imag'
        elif filter == 'SDSSzs':
            ref_filter = 'zmag'

        # añadir Johnson

        result = pd.DataFrame({'ID': vizier_results['objID'],
                               'RA': vizier_results['RAJ2000'],
                               'DEC': vizier_results['DEJ2000'],
                               'MAG': vizier_results[ref_filter],
                               'MAGERR': vizier_results['e_' + ref_filter],
                               'SOLAR': solar_index})
        ref_filter = ref_filter[:-3]

    return result, catalog, ref_filter
