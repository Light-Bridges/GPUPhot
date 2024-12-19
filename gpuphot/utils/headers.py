from datetime import datetime

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from .astro import date_to_jd, get_scale, get_ccw, radec_to_altaz, radec_to_ecl, radec_to_gal, radec_to_moon_sun
from ..logger.hierarchical_logging import setup_logger

logger = setup_logger(__name__)


def delete_header_from(header, val):
    """Delete a section from the FITS header.

    :param header: FITS header.
    :type header: dict
    :param val: Value to delete.
    :type val: str

    """
    for i, v in enumerate(header.values()):
        if val in str(v):
            idx = i - 1
    for i in range(len(header) - idx):
        del header[idx]

    return header


def get_if_header_already_post_processed(header, postprocess):
    comments = list()
    for comment in header["COMMENT"]:
        comments.append(comment.strip("   "))
    comments = set(comments)
    return postprocess in comments


def deg_to_hms(RA, DEC):
    coords_deg = SkyCoord(RA * u.deg, DEC * u.deg, frame='icrs', unit='deg')
    ra_hms = '%02d:%02d:%.6f' % coords_deg.ra.hms
    if DEC > 0:
        dec_dms = '%02d:%02d:%02.6f' % coords_deg.dec.dms
    else:
        coords_deg = SkyCoord(RA * u.deg, -1 * DEC * u.deg, frame='icrs', unit='deg')
        dec_dms = '-%02d:%02d:%02.6f' % coords_deg.dec.dms
    return ra_hms, dec_dms


def update_header_with_astrometry(imheader, h_wcs, site_latitude, site_longitude, site_elevation, date_obs,
                                  header_descriptions=None):
    if header_descriptions is None:
        header_descriptions = {}
    astro_exists = get_if_header_already_post_processed(imheader, "ASTROMETRY")
    if astro_exists:
        imheader = delete_header_from(imheader, 'ASTROMETRY')

    imheader['COMINIT'] = 'e'
    imheader.insert('COMINIT', ('COMMENT', '***************************'))
    imheader.insert('COMINIT', ('COMMENT', '       ASTROMETRY          '))
    imheader.insert('COMINIT', ('COMMENT', '***************************'))

    for v in h_wcs:
        imheader[v] = h_wcs[v]

    w = WCS(h_wcs)
    ra, dec = w.wcs_pix2world(imheader['NAXIS1'] // 2, imheader['NAXIS2'] // 2, 1)
    ra = ra.tolist()
    dec = dec.tolist()

    ra_hms, dec_dms = deg_to_hms(ra, dec)
    az, alt, airmass, zd = radec_to_altaz(ra, dec, site_latitude, site_longitude, site_elevation, date_obs)
    moon_alt, moon_az, distance_to_moon, moon_phase, sun_alt, sun_az = radec_to_moon_sun(
        ra, dec, site_latitude, site_longitude, site_elevation, date_obs)
    longal, latgal = radec_to_gal(ra, dec)
    lonecl, latecl = radec_to_ecl(ra, dec)
    scale = np.round(get_scale(imheader), 3)
    ccw = np.round(get_ccw(imheader), 1)
    fovx = np.round(imheader['NAXIS1'] * scale / 60, 2)
    fovy = np.round(imheader['NAXIS2'] * scale / 60, 2)

    # Usar .get() para evitar KeyError
    imheader.insert('COMINIT', ('RA', ra, header_descriptions.get('RA', '')))
    imheader.insert('COMINIT', ('DEC', dec, header_descriptions.get('DEC', '')))
    imheader.insert('COMINIT', ('RAhms', ra_hms, header_descriptions.get('RAhms', '')))
    imheader.insert('COMINIT', ('DECdms', dec_dms, header_descriptions.get('DECdms', '')))
    imheader.insert('COMINIT', ('FOVX', fovx, header_descriptions.get('FOVX', '')))
    imheader.insert('COMINIT', ('FOVY', fovy, header_descriptions.get('FOVY', '')))
    imheader.insert('COMINIT', ('SCALE', scale, header_descriptions.get('SCALE', '')))
    imheader.insert('COMINIT', ('CCW', ccw, header_descriptions.get('CCW', '')))
    imheader.insert('COMINIT', ('AZ', az, header_descriptions.get('AZ', '')))
    imheader.insert('COMINIT', ('ALT', alt, header_descriptions.get('ALT', '')))
    imheader.insert('COMINIT', ('ZD', zd, header_descriptions.get('ZD', '')))
    imheader.insert('COMINIT', ('AIRMASS', airmass, header_descriptions.get('AIRMASS', '')))
    imheader.insert('COMINIT', ('LONGAL', longal, header_descriptions.get('LONGAL', '')))
    imheader.insert('COMINIT', ('LATGAL', latgal, header_descriptions.get('LATGAL', '')))
    imheader.insert('COMINIT', ('LONECL', lonecl, header_descriptions.get('LONECL', '')))
    imheader.insert('COMINIT', ('LATECL', latecl, header_descriptions.get('LATECL', '')))
    imheader.insert('COMINIT', ('MOONALT', moon_alt, header_descriptions.get('MOONALT', '')))
    imheader.insert('COMINIT', ('MOONAZ', moon_az, header_descriptions.get('MOONAZ', '')))
    imheader.insert('COMINIT', ('MOONILUM', moon_phase, header_descriptions.get('MOONILUM', '')))
    imheader.insert('COMINIT', ('MOONDIST', distance_to_moon, header_descriptions.get('MOONDIST', '')))
    imheader.insert('COMINIT', ('SUNALT', sun_alt, header_descriptions.get('SUNALT', '')))
    imheader.insert('COMINIT', ('SUNAZ', sun_az, header_descriptions.get('SUNAZ', '')))

    if 'JD' not in imheader:
        try:
            jd, mjd = date_to_jd(imheader['DATE-OBS'])
            imheader.insert('PCDATE', ('JD-OBS', jd, header_descriptions.get('JD-OBS', '')))
            imheader.insert('PCDATE', ('MJD-OBS', mjd, header_descriptions.get('MJD-OBS', '')))
        except:
            pass

    del imheader['COMINIT']

    return imheader


def update_header_with_photometry(imheader, dic_calib, header_descriptions=None):
    if header_descriptions is None:
        header_descriptions = {}
    phot_exists = get_if_header_already_post_processed(imheader, "PHOTOMETRY")
    if phot_exists:
        imheader = delete_header_from(imheader, 'PHOTOMETRY')

    try:
        del imheader['FWHM']
    except KeyError:
        pass  # Es mejor capturar KeyError específicamente

    imheader['COMINIT'] = 'e'
    imheader.insert('COMINIT', ('COMMENT', '***************************'))
    imheader.insert('COMINIT', ('COMMENT', '       PHOTOMETRY          '))
    imheader.insert('COMINIT', ('COMMENT', '***************************'))

    for v in dic_calib.keys():
        # Usar .get() para evitar KeyError
        description_value = header_descriptions.get(v, '')
        imheader.insert('COMINIT', (v, dic_calib[v], description_value))

    # Usar .get() para la fecha de procesamiento
    dateproc_value = header_descriptions.get("DATEPROC", '')

    # Inserta la fecha de procesamiento
    imheader.insert(
        'COMINIT',
        ('DATEPROC',
         datetime.utcnow().isoformat()[:-7],
         dateproc_value)
    )

    del imheader['COMINIT']

    return imheader
