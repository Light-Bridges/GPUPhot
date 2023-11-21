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
    x, y = w.all_world2pix(cat["RA"], cat["DEC"], 0, quiet=True)
    return x, y


def get_target_ephemeris(target_name, epochs, site_lat, site_lon, site_elev):
    target = Horizons(
        id=target_name,
        location={
            "lat": site_lat,
            "lon": site_lon,
            "elevation": site_elev / 1e3,
        },
        epochs=epochs,
    )
    ephem = target.ephemerides()
    return ephem["RA"].value.data[0] * 24 / 360, ephem["DEC"].value.data[0]


def get_target_ra_dec(target_name: str, moving_target: bool, header, ra_dec_dict=None):
    """Function that returns the ra, dec from the object at the mean time of observation."""

    if moving_target:
        if ra_dec_dict is None:
            date_obs = pd.Timestamp(header["UTOBS"]).to_julian_date()
            site_lat = header["SITELAT"]
            site_lon = header["SITELONG"]
            site_elev = header["SITEALT"]
            return get_target_ephemeris(
                target_name, date_obs, site_lat, site_lon, site_elev
            )
        else:
            date_obs = datetime.strptime(
                header["UTOBS"], "%Y-%m-%dT%H:%M:%S.%f"
            ).replace(tzinfo=pytz.UTC)
            ra = np.interp(
                date_obs.timestamp(),
                [t.timestamp() for t in ra_dec_dict["time_index"]],
                ra_dec_dict["ra"],
            )
            dec = np.interp(
                date_obs.timestamp(),
                [t.timestamp() for t in ra_dec_dict["time_index"]],
                ra_dec_dict["dec"],
            )
            return ra, dec
    else:
        return header["POINTRA"], header["POINTDEC"]


def get_target_pix(ra, dec, header):
    df = pd.DataFrame({"RA": [ra / 24 * 360], "DEC": [dec]})
    x, y = wcs_to_px(df, header)
    return x[0], y[0]


def get_ccw(header):
    cd11 = header["CD1_1"]
    cd12 = header["CD1_2"]
    cd21 = header["CD2_1"]
    cd22 = header["CD2_2"]

    det = cd11 * cd22 - cd12 * cd21
    if det >= 0:
        parity = 1.0
    else:
        parity = -1.0
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
    coocenter = SkyCoord(
        ra=header["RA"], dec=header["DEC"], unit=(u.deg, u.deg), frame="icrs"
    )
    # coocenter = SkyCoord(ra=header['CRVAL1'], dec=header['CRVAL2'], unit=(u.deg, u.deg), frame='icrs')
    scale = header["SCALEORI"]
    # npx = np.array([header['NAXIS1'], header['NAXIS2']])
    # FOV = np.sqrt(np.sum((scale * npx / 3600)**2))
    npx = max(header["NAXIS1"], header["NAXIS2"])
    FOV = 2 * np.sqrt(2 * ((scale * npx / 3600) ** 2))
    filter = header["FILTER"]
    return coocenter, FOV, filter, scale


if __name__ == "__main__":
    _site_lat = +28.300332
    _site_lon = -16.512206
    _site_elev = 2390

    _header1 = {
        "SIMPLE": True,
        "BITPIX": -32,
        "NAXIS": 2,
        "NAXIS1": 2048,
        "NAXIS2": 2048,
        "TELESCOP": "TTT1",
        "SITELAT": 28.29871121,
        "SITELONG": -16.50956893,
        "SITEALT": 2359.12,
        "DIAMETER": 800.0,
        "FOCAL": 6.85,
        "FOCALEN": 5480.0,
        "TRACK": 1,
        "SHMODE": "Mechanical shutter",
        "DRMODE": None,
        "COMODE": None,
        "RDMODE": "Multi track",
        "RDNOISE": 7.0,
        "INTEMP": -55.883,
        "SATLEVEL": 65536,
        "GAIN": 1.0,
        "DC": 0.02,
        "SCALEORI": 0.508,
        "PARITY": "pos",
        "XBINNING": 1,
        "YBINNING": 1,
        "EFFECTSX": 0,
        "EFFECTNX": 2048,
        "EFFECTSY": 0,
        "EFFECTNY": 2048,
        "STATSX": 0,
        "STATNX": 2048,
        "STATSY": 0,
        "STATNY": 2048,
        "INSTRUME": "iKon936-TTT1",
        "INMODEL": "iKon936",
        "INSERIAL": 27305,
        "INSDK": "2.104",
        "OFFSET": None,
        "PXSIZE": 13.5,
        "ORISIZEX": 2048,
        "ORISIZEY": 2048,
        "ALISIZEX": 2048,
        "ALISIZEY": 2048,
        "FINSIZEX": 2048,
        "FINSIZEY": 2048,
        "CAMERA": "iKon936-1",
        "INFIRMW": "20.12",
        "HSSPEED": 1.0,
        "VSSPEED": 38.549999,
        "PREAMPGA": 4.0,
        "DATE-OBS": "2023-02-08T20:53:25.452554",
        "JD-OBS": 2459984.370433479,
        "MJD-OBS": 59983.87043347863,
        "PCDATE": "2023-02-08T20:53:25.452554",
        "COOLERST": "DRV_TEMP_NOT_REACHED",
        "EXPTIME": 20.022,
        "WINDOWSX": 0,
        "WINDOWNX": 2048,
        "WINDOWSY": 0,
        "WINDOWNY": 2048,
        "OBJECT": "Kellyoconnor",
        "FILTER": "SDSSi",
        "POINTRA": 3.652383384165774,
        "POINTDEC": 11.85168222874598,
        "UT1": "2023-02-08T20:53:25.452554",
        "PCDAT1": "2023-02-08T20:53:25.452554",
        "EXPT1": 20.022,
        "TEMP1": -55.883,
        "FOCUS": 12190,
        "UTOBS": "2023-02-08T20:53:25.452554",
        "INTEGT": 20.0,
        "TOTIMA": 1,
        "OBLINEID": 81158,
        "HUMIDITY": 4.2,
        "MIRRHUM": 15.612,
        "PRESSURE": 766.0,
        "AMBTEMP": 3.41,
        "MIRRTEMP": 0.96,
        "CLOUD": 0.0,
        "ILLUMINA": 1.6,
        "WINDDIR": 106.0,
        "WINDVEL": 2.33,
        "DUSTPLA": 0.003,
        "DUSTPM1": 0.04,
        "DUSTPM10": 0.04,
        "DUSTPM25": 0.04,
        "PWV": 2.04,
        "TESSMAG": 20.84,
        "SKYIRTEM": -49.75,
        "BIASCORR": "bias+dark",
        "BIASMEAN": 268.871,
        "BIASSTD": 6.412,
        "BIASN": 21,
        "BIASTEMP": -53.287,
        "BIASDATE": "2023-03-01T07:39:43.292806",
        "DARKMEAN": 0.662,
        "DARKSTD": 0.287,
        "DARKN": 21,
        "DARKTEMP": -53.287,
        "DARKDATE": "2023-03-01T07:44:13.077313",
        "FLATMEAN": 14836.159,
        "FLATSTD": 539.274,
        "FLATN": 11,
        "FLATTEMP": -55.883,
        "FLATDATE": "2023-02-09T08:48:35.819472",
        "DATE": "2023-04-15T03:26:03",
        "RA": 54.78902141028366,
        "DEC": 11.85320287613557,
        "RAHMS": "03:39:9.365138",
        "DECDMS": "11:51:11.530354",
        "AZ": 233.523691,
        "ALT": 64.850553,
        "ZD": 25.149447,
        "AIRMASS": 1.104725,
        "LONGAL": 174.721683,
        "LATGAL": -33.665416,
        "LONECL": 55.312877,
        "LATECL": -7.447191,
        "WCSAXES": 2,
        "EQUINOX": 2000.0,
        "LONPOLE": 180.0,
        "LATPOLE": 0.0,
        "CRVAL1": 54.650272494715,
        "CRVAL2": 11.85584513410472,
        "CRPIX1": 212.0882167819279,
        "CRPIX2": 1545.511680545574,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CD1_1": 0.000116910747794951,
        "CD1_2": -7.836721585484e-05,
        "CD2_1": -7.8326675648825e-05,
        "CD2_2": -0.00011694055432154,
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "A_ORDER": 3,
        "A_0_0": 0.0,
        "A_0_1": 0.0,
        "A_0_2": 7.85130602880151e-07,
        "A_0_3": 4.87714430139195e-10,
        "A_1_0": 0.0,
        "A_1_1": -5.2398680309053e-07,
        "A_1_2": 1.06085839651212e-10,
        "A_2_0": -7.7585895073241e-07,
        "A_2_1": 3.74421642914737e-10,
        "A_3_0": 6.63685497970571e-10,
        "B_ORDER": 3,
        "B_0_0": 0.0,
        "B_0_1": 0.0,
        "B_0_2": 2.33951184393364e-07,
        "B_0_3": 2.58285053217136e-10,
        "B_1_0": 0.0,
        "B_1_1": 2.09009183645733e-07,
        "B_1_2": 1.53541176428838e-10,
        "B_2_0": 5.12947344849658e-07,
        "B_2_1": 2.46888610282133e-11,
        "B_3_0": -2.1443451661585e-10,
        "AP_ORDER": 3,
        "AP_0_0": 1.5219546870604e-05,
        "AP_0_1": -2.6448675612831e-08,
        "AP_0_2": -7.8428516167927e-07,
        "AP_0_3": -4.870677038752e-10,
        "AP_1_0": 5.00186329487163e-07,
        "AP_1_1": 5.24466855304315e-07,
        "AP_1_2": -1.0535107700489e-10,
        "AP_2_0": 7.73085658503646e-07,
        "AP_2_1": -3.7411460141753e-10,
        "AP_3_0": -6.6162747930699e-10,
        "BP_ORDER": 3,
        "BP_0_0": -4.8112570978542e-05,
        "BP_0_1": -1.2852388760091e-08,
        "BP_0_2": -2.3355636615901e-07,
        "BP_0_3": -2.57929193847e-10,
        "BP_1_0": -2.3905053194158e-07,
        "BP_1_1": -2.0901938882307e-07,
        "BP_1_2": -1.5314954797012e-10,
        "BP_2_0": -5.1235862137302e-07,
        "BP_2_1": -2.4328225981164e-11,
        "BP_3_0": 2.14212738169679e-10,
        "FOVX": 0.2889955555555556,
        "FOVY": 0.2889955555555556,
        "ZP": 0.0,
        "EZP": 0.0,
        "FWHM": 2.96001935005188,
        "EFWHM": 0.3019448220729828,
        "M_LIM": 0.0,
        "M_SKY": -5.202066868645168,
        "SKY": 160.2217864990234,
        "ESKY": 16.04984092712402,
        "SCALE": 0.508,
        "CCW": -146.1756287619255,
        "AP_SNR": 49,
        "AP_FLUX": 49,
    }

    x = 100
    y = 150
    header = _header1  # Función para crear un encabezado de ejemplo
    wcs_coords = px_to_wcs(x, y, header)
    print(wcs_coords[0])
