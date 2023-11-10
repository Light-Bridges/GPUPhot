from datetime import datetime

import numpy as np
import pandas as pd
import pytz
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astroquery.jplhorizons import Horizons


def px_to_wcs(x, y, header):
    w = WCS(header=header)
    return w.all_pix2world(x, y, 0)


def wcs_to_px(cat, header):
    w = WCS(header)
    x, y = w.all_world2pix(cat['RA'], cat['DEC'], 0, quiet=True)
    return x, y


def get_target_ephemeris(target_name, epochs, site_lat, site_lon, site_elev):
    target = Horizons(
        id=target_name,
        location={
            "lat": site_lat,
            "lon": site_lon,
            "elevation": site_elev / 1e3,
        },
        epochs=epochs
    )
    ephem = target.ephemerides()
    return ephem["RA"].value.data[0] * 24 / 360, ephem["DEC"].value.data[0]


def get_target_ra_dec(target_name: str, moving_target: bool, header, ra_dec_dict=None):
    """Function that returns the ra, dec from the object at the mean time of observation."""

    if moving_target:
        if ra_dec_dict is None:
            date_obs = pd.Timestamp(header["UTOBS"]).to_julian_date()
            site_lat = header["SITE_LAT"]
            site_lon = header["SITE_LON"]
            site_elev = header["SITEELEV"]
            return get_target_ephemeris(target_name, date_obs, site_lat, site_lon, site_elev)
        else:
            date_obs = datetime.strptime(header["UTOBS"], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=pytz.UTC)
            ra = np.interp(date_obs.timestamp(), [t.timestamp() for t in ra_dec_dict["time_index"]], ra_dec_dict["ra"])
            dec = np.interp(date_obs.timestamp(), [t.timestamp() for t in ra_dec_dict["time_index"]],
                            ra_dec_dict["dec"])
            return ra, dec
    else:
        return header["POINTRA"], header["POINTDEC"]


def get_target_pix(ra, dec, header):
    df = pd.DataFrame({"RA": [ra / 24 * 360], "DEC": [dec]})
    x, y = wcs_to_px(df, header)
    return x[0], y[0]


def get_ccw(header):
    cd11 = header['CD1_1']
    cd12 = header['CD1_2']
    cd21 = header['CD2_1']
    cd22 = header['CD2_2']

    det = cd11 * cd22 - cd12 * cd21
    if det >= 0:
        parity = 1.
    else:
        parity = -1.
    T = parity * cd11 + cd22
    A = parity * cd21 - cd12
    return -np.degrees(np.arctan2(A, T))


def get_if_header_already_post_processed(header, postprocessing):
    comments = list()
    for comment in header["COMMENT"]:
        comments.append(comment.strip("   "))
    comments = set(comments)
    return postprocessing in comments


def cat_input_from_header(header):
    coocenter = SkyCoord(ra=header['RA'], dec=header['DEC'], unit=(u.deg, u.deg), frame='icrs')
    # coocenter = SkyCoord(ra=header['CRVAL1'], dec=header['CRVAL2'], unit=(u.deg, u.deg), frame='icrs')
    scale = header['SCALEORI']
    # npx = np.array([header['NAXIS1'], header['NAXIS2']])
    # FOV = np.sqrt(np.sum((scale * npx / 3600)**2))
    npx = max(header['NAXIS1'], header['NAXIS2'])
    FOV = 2 * np.sqrt(2 * ((scale * npx / 3600) ** 2))
    filter = header['FILTER']
    return coocenter, FOV, filter, scale


if __name__ == "__main__":
    _site_lat = +28.300332
    _site_lon = -16.512206
    _site_elev = 2390
