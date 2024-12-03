
import numpy as np
import pandas as pd
from astropy import units as u
from astroquery.vizier import Vizier
from scipy.spatial import KDTree
from ..logger.hierarchical_logging import setup_logger, hierarchical_debug

logger = setup_logger(__name__)

def crossmatch_sources(source_coords, ref_coords, thres_px=2):
    tree = KDTree(ref_coords)
    dist, idx = tree.query(source_coords, k=1)
    mask = dist < thres_px
    source_coords_matched_idx = np.arange(len(source_coords))[mask]
    ref_coords_matched_idx = idx[mask]
    return source_coords_matched_idx, ref_coords_matched_idx

@hierarchical_debug(logger)
def __getVizier(catalog, coocenter, radii, maglimit, ref_filter):
    Vizier.ROW_LIMIT = -1
    timeout = 60
    vizier_results = Vizier(timeout=timeout, row_limit=-1) \
        .query_region(coocenter, radius=radii*u.deg, catalog=catalog, column_filters={ref_filter: '<%.1f ' % maglimit} , cache=True)
    return vizier_results[0]

@hierarchical_debug(logger)
def catalog_results(coocenter, radius, filter, inmodel, maglimit=23):

    if coocenter.dec.deg < -30:
        catalog='II/379' # SkyMapper Southern Sky Survey. DR4 : II/379
        ref_filter = 'gPSF'
        vizier_results = __getVizier(catalog, coocenter, radius, maglimit, ref_filter)
        color = vizier_results['gPSF'] - vizier_results['rPSF']
        B = vizier_results['gPSF'] + 0.194 + 0.561 * color
        V = vizier_results['gPSF'] - 0.017 - 0.508 * color
        solar_index = B - V - 0.65

        if filter == 'SDSSg': ref_filter = 'gPSF'
        elif filter == 'SDSSr': ref_filter = 'rPSF'
        elif filter == 'SDSSi': ref_filter = 'iPSF'
        elif filter == 'SDSSzs': ref_filter = 'zPSF'
        elif filter == 'SDSSu': ref_filter = 'uPSF'


        result = pd.DataFrame({'ID': vizier_results['SMSS'],
                               'RA': vizier_results['RAICRS'],
                               'DEC': vizier_results['DEICRS'],
                               'MAG': vizier_results[ref_filter],
                               'MAGERR': vizier_results[ref_filter]*0,
                               'SOLAR': solar_index})
        ref_filter = ref_filter[:-3]

    elif filter=='Open' or filter=='OPEN':
        catalog='I/355/gaiadr3' # Gaia DR3 Part 1. Main source : I/355
        ref_filter = 'BPmag'
        vizier_results = __getVizier(catalog, coocenter, radius, maglimit, ref_filter)
        solar_index = 0.01760 - 0.003226 + (0.3833+0.00686) * vizier_results['BP-RP'] + (-0.1345+0.1732) * vizier_results['BP-RP']**2 - 0.36
        magerr = -2.5 * np.log10 (vizier_results['F'+ref_filter[:2]] / (vizier_results['F'+ref_filter[:2]] + vizier_results['e_F'+ref_filter[:2]]))
        result = pd.DataFrame({'ID': vizier_results['Source'],
                                'RA': vizier_results['RAJ2000'],
                                'DEC': vizier_results['DEJ2000'],
                                'MAG': vizier_results[ref_filter],
                                'MAGERR': magerr,
                                'SOLAR': solar_index})
        ref_filter = ref_filter[:-3]

    elif filter == 'SDSSu':
        catalog='I/353/gsc242'
        ref_filter = 'umag'
        vizier_results = __getVizier(catalog, coocenter, radius, maglimit, ref_filter)
        result = pd.DataFrame({'ID': vizier_results['GSC2'],
                               'RA': vizier_results['RA_ICRS'],
                               'DEC': vizier_results['DE_ICRS'],
                               'MAG': vizier_results[ref_filter],
                               'MAGERR': np.zeros(len(vizier_results['RA_ICRS'])), # uncertainty unknown
                               'SOLAR': np.zeros(len(vizier_results['RA_ICRS']))}) # for u filter no solar colors are applied

    elif filter == 'Lum' or filter == 'w':
        catalog='II/349/ps1' # The Pan-STARRS release 1 (PS1) Survey - DR1 : II/349
        ref_filter = 'gmag'
        vizier_results = __getVizier(catalog, coocenter, radius, maglimit, ref_filter)
        color = vizier_results['gmag'] - vizier_results['rmag']
        # B = vizier_results['gmag'] + 0.213 + 0.587 * color
        # V = vizier_results['rmag'] + 0.006 + 0.474 * color
        # https://arxiv.org/pdf/1706.06147.pdf
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
        catalog='II/349/ps1' # The Pan-STARRS release 1 (PS1) Survey - DR1 : II/349
        ref_filter = 'gmag'
        vizier_results = __getVizier(catalog, coocenter, radius, maglimit, ref_filter)
        color = vizier_results['gmag'] - vizier_results['rmag']
        # B = vizier_results['gmag'] + 0.213 + 0.587 * color
        # V = vizier_results['rmag'] + 0.006 + 0.474 * color
        # https://arxiv.org/pdf/1706.06147.pdf
        B = vizier_results['gmag'] + 0.194 + 0.561 * color
        V = vizier_results['gmag'] - 0.017 - 0.508 * color
        solar_index = B - V - 0.65

        if filter == 'SDSSg': ref_filter = 'gmag'
        elif filter == 'SDSSr': ref_filter = 'rmag'
        elif filter == 'SDSSi': ref_filter = 'imag'
        elif filter == 'SDSSzs': ref_filter = 'zmag'

        # añadir Johnson

        result = pd.DataFrame({'ID': vizier_results['objID'],
                               'RA': vizier_results['RAJ2000'],
                               'DEC': vizier_results['DEJ2000'],
                               'MAG': vizier_results[ref_filter],
                               'MAGERR': vizier_results['e_'+ref_filter],
                               'SOLAR': solar_index})
        ref_filter = ref_filter[:-3]

    return result, catalog, ref_filter
