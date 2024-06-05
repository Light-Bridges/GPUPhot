import logging

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


def cat_input_from_header(header):
    """
    Extract catalog input parameters from FITS header.

    Parameters
    ----------
    header : dict
        FITS header containing observation metadata.

    Returns
    -------
    tuple
        (coocenter, FOV, filter, scale, inmodel) where:
        - coocenter : SkyCoord
            Sky coordinates of the image center.
        - FOV : float
            Field of view in degrees.
        - filter : str
            Filter used for the observation.
        - scale : float
            Pixel scale in arcseconds per pixel.
        - inmodel : str
            Instrument model used for the observation.
    """
    try:
        coocenter = SkyCoord(ra=header['RA'], dec=header['DEC'], unit=(u.deg, u.deg), frame='icrs')
    except KeyError:
        coocenter = SkyCoord(ra=header['POINTRA'], dec=header['POINTDEC'], unit=(u.deg, u.deg), frame='icrs')
    scale = header['SCALE']
    FOV = np.sqrt(header['NAXIS1'] ** 2 + header['NAXIS2'] ** 2) * scale / 3600
    filter = header['FILTER']
    inmodel = header['INMODEL']
    return coocenter, FOV, filter, scale, inmodel


def catalog_results(coocenter, radius, filter, inmodel, maglimit=22):
    """
    Retrieve catalog results from Vizier based on input parameters.

    Parameters
    ----------
    coocenter : SkyCoord
        Sky coordinates of the image center.
    radius : float
        Search radius in degrees.
    filter : str
        Filter used for the observation.
    inmodel : str
        Instrument model used for the observation.
    maglimit : float, optional
        Magnitude limit for the search, by default 22.

    Returns
    -------
    tuple
        (result, catalog, ref_filter) where:
        - result : DataFrame
            DataFrame containing catalog results.
        - catalog : str
            Catalog ID used for the search.
        - ref_filter : str
            Reference filter used for the magnitude calculation.
    """
    if coocenter.dec.deg < -30:
        catalog = 'II/379'  # PANSTAR catalog
        ref_filter = 'gPSF'
        vizier_results = __getVizier(catalog, coocenter, radius, maglimit, ref_filter)
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

    elif filter in ['Open', 'OPEN', 'SDSSu']:
        catalog = 'I/355/gaiadr3'  # GAIA catalog
        ref_filter = 'BPmag'
        vizier_results = __getVizier(catalog, coocenter, radius, maglimit, ref_filter)
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

    elif filter in ['Lum', 'w']:
        catalog = 'II/349/ps1'  # PANSTAR catalog
        ref_filter = 'gmag'
        vizier_results = __getVizier(catalog, coocenter, radius, maglimit, ref_filter)
        color = vizier_results['gmag'] - vizier_results['rmag']
        B = vizier_results['gmag'] + 0.194 + 0.561 * color
        V = vizier_results['gmag'] - 0.017 - 0.508 * color
        solar_index = B - V - 0.65

        if inmodel == 'iKon936':
            mag = 0.46872 * vizier_results['gmag'] + 0.53127 * vizier_results['rmag']
            magerr = 0.46872 * vizier_results['e_gmag'] + 0.53127 * vizier_results['e_rmag']
            ref_filter = '0.46872*g+0.53127*r'

        elif inmodel == 'QHY411MERIS':
            mag = 0.51595 * vizier_results['gmag'] + 0.48404 * vizier_results['rmag']
            magerr = 0.51595 * vizier_results['e_gmag'] + 0.48404 * vizier_results['e_rmag']
            ref_filter = '0.51595*g+0.48404*r'

        else:
            mag = 0.5 * vizier_results['gmag'] + 0.5 * vizier_results['rmag']
            magerr = 0.5 * vizier_results['e_gmag'] + 0.5 * vizier_results['e_rmag']
            ref_filter = '0.5*g+0.5*r'

        result = pd.DataFrame({'ID': vizier_results['objID'],
                               'RA': vizier_results['RAJ2000'],
                               'DEC': vizier_results['DEJ2000'],
                               'MAG': mag,
                               'MAGERR': magerr,
                               'SOLAR': solar_index})

    else:
        catalog = 'II/349/ps1'  # PANSTAR catalog
        ref_filter = 'gmag'
        vizier_results = __getVizier(catalog, coocenter, radius, maglimit, ref_filter)
        color = vizier_results['gmag'] - vizier_results['rmag']
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

        result = pd.DataFrame({'ID': vizier_results['objID'],
                               'RA': vizier_results['RAJ2000'],
                               'DEC': vizier_results['DEJ2000'],
                               'MAG': vizier_results[ref_filter],
                               'MAGERR': vizier_results['e_' + ref_filter],
                               'SOLAR': solar_index})
        ref_filter = ref_filter[:-3]

    return result, catalog, ref_filter


def crossmatch_sources(source_coords, ref_coords, thres_px=2):
    """
    Cross-match source coordinates with reference coordinates.

    Parameters
    ----------
    source_coords : ndarray
        Coordinates of the sources.
    ref_coords : ndarray
        Coordinates of the reference catalog.
    thres_px : float, optional
        Threshold distance in pixels for matching, by default 2.

    Returns
    -------
    tuple
        (source_coords_matched_idx, ref_coords_matched_idx) where:
        - source_coords_matched_idx : ndarray
            Indices of matched source coordinates.
        - ref_coords_matched_idx : ndarray
            Indices of matched reference coordinates.
    """
    tree = KDTree(ref_coords)
    dist, idx = tree.query(source_coords, k=1)
    mask = dist < thres_px
    source_coords_matched_idx = np.arange(len(source_coords))[mask]
    ref_coords_matched_idx = idx[mask]
    return source_coords_matched_idx, ref_coords_matched_idx


def __getVizier(catalog, coocenter, radii, maglimit, ref_filter):
    """
    Query the Vizier catalog for objects within a specified region.

    Parameters
    ----------
    catalog : str
        Catalog ID to query.
    coocenter : SkyCoord
        Center coordinates for the search.
    radii : float
        Search radius in degrees.
    maglimit : float
        Magnitude limit for the search.
    ref_filter : str
        Reference filter for magnitude selection.

    Returns
    -------
    Table
        Vizier query results.
    """
    Vizier.ROW_LIMIT = -1
    timeout = 60
    vizier_results = Vizier(timeout=timeout, row_limit=-1) \
        .query_region(coocenter, radius=radii * u.deg, catalog=catalog,
                      column_filters={ref_filter: '<%.1f ' % maglimit}, cache=True)
    return vizier_results[0]
