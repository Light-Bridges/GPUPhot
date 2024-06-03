# -*- coding: utf-8 -*-

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, angles
from astroquery.vizier import Vizier

from src.astrometry.utils import px_to_wcs, wcs_to_px


def __getVizierALL(catalog):
    Vizier.ROW_LIMIT = -1
    results = Vizier.get_catalogs([catalog])
    return results[0]


def __getVizier(catalog, radec, radii, maglimit):
    Vizier.ROW_LIMIT = -1
    timeout = 60
    row_limit = -1
    cache = True
    radius = radii * u.arcsec

    if catalog == 'I/345/gaia2':
        columns = ['DR2Name', 'RA_ICRS', 'DE_ICRS', 'Gmag', 'e_Gmag', 'BP-G']
        column_filters = {'Gmag': '<' + str(maglimit), 'Gmag': '!= ' + '', 'BP-G': '!= ' + ''}

    elif catalog == 'II/349/ps1':
        columns = ['objID', 'RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000', 'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag',
                   'e_imag', 'zmag', 'e_zmag']
        column_filters = {'gmag': '<' + str(maglimit), 'rmag': '!= ' + '', 'gmag': '!= ' + '', 'imag': '!= ' + '',
                          'zmag': '!= ' + ''}

    elif catalog == 'II/183A/table2':
        columns = ['objID', 'RAJ2000', 'DEJ2000', 'Vmag', 'e_Vmag', 'B-V', 'V-R', 'R-I']
        column_filters = {'Vmag': '<' + str(maglimit), 'B-V': '!= ' + '', 'V-R': '!= ' + '', 'R-I': '!= ' + ''}
        radius = radii * u.deg

    else:
        return None

    vizier_results = Vizier(
        timeout=timeout,
        columns=columns,
        row_limit=row_limit,
    ).query_region(
        radec,
        radius=radius,
        catalog=catalog,
        column_filters=column_filters,
        cache=cache
    )

    return vizier_results[0]


def catalog_results(coocenter, FOV, filter, maglimit=23):
    labels = {}
    radii_int = int(np.sqrt(FOV) * 1800)  # internal radius
    radii_ext = int(np.sqrt(2) * radii_int)  # external radius
    radii = radii_ext
    if filter == 'Open' or filter == 'OPEN' or filter == 'Lum' or filter == 'w':
        catalog = 'I/345/gaia2'  # GAIA catalog
        vizier_results = __getVizier(catalog, coocenter, radii, maglimit)
        vizier_results.rename_column('RA_ICRS', 'RA')
        vizier_results.rename_column('DE_ICRS', 'DEC')
        vizier_results.rename_column('BP-G', 'color')
        vizier_results.rename_column('Gmag', 'rmag')
        vizier_results.rename_column('DR2Name', 'objID')
        mask_solar = (vizier_results['color'] < 0.9) & (vizier_results['color'] > 0.2)  # solar like stars

    else:
        catalog = 'II/349/ps1'  # PANSTAR catalog
        vizier_results = __getVizier(catalog, coocenter, radii, maglimit)
        vizier_results.rename_column('RAJ2000', 'RA')
        vizier_results.rename_column('DEJ2000', 'DEC')
        vizier_results['color'] = vizier_results['gmag'] - vizier_results['rmag']
        vizier_results['color1'] = vizier_results['rmag'] - vizier_results['imag']
        vizier_results['color2'] = vizier_results['imag'] - vizier_results['zmag']
        mask_solar = (vizier_results['color'] < 0.64) & (vizier_results['color'] > 0.24)

        # BVRI transformations from Jordi et al. (2005)
        # http://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php#Jordi2006
        # B-g   =     (0.313 ± 0.003)*(g-r)  + (0.219 ± 0.002)
        # V-g   =     (-0.565 ± 0.001)*(g-r) - (0.016 ± 0.001)
        # R-r   =     (-0.153 ± 0.003)*(r-i) - (0.117 ± 0.003)
        # I-i   =     (-0.386 ± 0.004)*(i-z) - (0.397 ± 0.001)
        if filter == 'U' or filter == 'B' or filter == 'V' or filter == 'R' or filter == 'I':
            vizier_results['B'] = vizier_results['gmag'] + 0.313 * vizier_results['color'] + 0.219
            vizier_results['V'] = vizier_results['gmag'] - 0.565 * vizier_results['color'] - 0.016
            vizier_results['R'] = vizier_results['rmag'] - 0.153 * vizier_results['color1'] - 0.117
            vizier_results['I'] = vizier_results['imag'] - 0.386 * vizier_results['color2'] - 0.397
            vizier_results['color3'] = vizier_results['B'] - vizier_results['V']
            vizier_results['color4'] = vizier_results['V'] - vizier_results['R']
            vizier_results['color5'] = vizier_results['R'] - vizier_results['I']
    print('Input parameters for standard catalog search: FOV={:.2f}'.format(FOV),
          '(degrees**2), radius(int/ext/user defined)=', radii_int, radii_ext, radii, ' (arcsec)')
    print('Total ' + catalog + ' catalog stars found', len(vizier_results),
          ' for ' + filter + ' magnitud < {:.2f}'.format(maglimit))
    vizier_results['mask_solar'] = mask_solar
    return vizier_results, catalog, labels


def catalog_match(sources, cat_sources, header, fwhm):
    # xy centroids to radec
    print(sources)
    ra, dec = px_to_wcs(sources['xcentroid'], sources['ycentroid'], header)
    xcat, ycat = wcs_to_px(cat_sources, header)
    sources["ra"], sources["dec"] = ra, dec
    cat_sources['xcat'], cat_sources['ycat'] = xcat, ycat
    det_stars = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    cat_stars = SkyCoord(ra=cat_sources['RA'], dec=cat_sources['DEC'], unit=(u.degree, u.degree))

    # matching
    idx_cat, d2d, d3d = det_stars.match_to_catalog_sky(cat_stars, nthneighbor=1)

    # filter by distance
    sep_angles = (angles.Angle(d2d)).arcsec
    idx_cat_sep = sep_angles < fwhm * 2

    # merge dataframes
    df_sources = sources[idx_cat_sep]
    df_sources.index = idx_cat[idx_cat_sep]
    df_cat = cat_sources.to_pandas()
    df_cat['objID'] = df_cat['objID'].astype(str)
    df = df_sources.join(df_cat)

    return df, sources
