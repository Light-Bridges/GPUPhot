import os
import signal
from datetime import datetime

import astrometry
import ephem
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.wcs import WCS
from astropy.time import Time
from sklearn.linear_model import RANSACRegressor

from .catalog import crossmatch_sources
from ..logger.hierarchical_logging import setup_logger, hierarchical_debug
from ..stats.reduction import weighted_mean_std

logger = setup_logger(__name__)

import threading

class SingletonSolver:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.debug("Creando nueva instancia del solver")
                    cls._instance = super().__new__(cls)
                    cls._instance.initialize_solver()
        return cls._instance

    def initialize_solver(self):
        if os.path.exists('/data'):
            cache = '/data/astrometry_cache'
        else:
            cache = '/mnt/data/astrometry_cache'



        self.solver = astrometry.Solver(
            astrometry.series_5200.index_files(cache_directory=cache, scales={0, 1, 2, 3, 4, 5, 6}) +
            astrometry.series_4100.index_files(cache_directory=cache, scales={7, 8, 9, 10, 11})
        )

@hierarchical_debug(logger)
def get_solver():
    """
    Get the astrometry solver with index files.

    Returns
    -------
    astrometry.Solver
        Configured astrometry solver instance.
    """
    return SingletonSolver().solver
    # if os.path.exists('/data'):
    #     cache = '/data/astrometry_cache'
    # else:
    #     cache = '/mnt/data/astrometry_cache'
    #
    # solver = (astrometry.Solver(
    #     astrometry.series_5200.index_files(cache_directory=cache, scales={0, 1, 2, 3, 4, 5, 6}) +
    #     astrometry.series_4100.index_files(cache_directory=cache, scales={7, 8, 9, 10, 11}))
    # )
    #
    # return solver


def get_astrometry_params(h_wcs, image_shape):
    w = WCS(h_wcs)
    ra, dec = w.all_pix2world(image_shape[1] // 2, image_shape[0] // 2, 1)
    cd11 = float(h_wcs['CD1_1'][0])
    cd12 = float(h_wcs['CD1_2'][0])
    scale = np.sqrt(cd11 ** 2 + cd12 ** 2) * 3600
    coocenter = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    npx = max(image_shape)
    FOV = 2 * np.sqrt(2 * ((scale * npx / 3600) ** 2))
    return coocenter, FOV, scale


def logodds_callback_100(logodds):
    # print(logodds)
    if (logodds[0] > 100.0) | (len(logodds) > 2):
        return astrometry.Action.STOP
    else:
        return astrometry.Action.CONTINUE


def handler(signum, frame):
    print("Astrometrization timeout!")
    raise Exception("end of time")


@hierarchical_debug(logger)
def astrometrice2(df: pd.DataFrame, scale: float,
                  central_ra: float, central_dec: float,
                  sip_order: int = 3) -> dict:
    """
    Perform astrometry on an image.

    Parameters
    ----------
    dfm : pd.DataFrame
        Dataframe containing detected sources.
    head0 : dict
        FITS header.
    im_shape : tuple
        Shape of the image.

    Returns
    -------
    dict
        Updated FITS header.
    """

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(60)

    try:
        solution = get_solver().solve(
            stars_xs=df['xcentroid'],
            stars_ys=df['ycentroid'],
            size_hint=astrometry.SizeHint(
                lower_arcsec_per_pixel=scale * 0.8,
                upper_arcsec_per_pixel=scale * 1.2)
            ,
            position_hint=astrometry.PositionHint(
                ra_deg=central_ra,
                dec_deg=central_dec,
                radius_deg=0.5, )
            ,
            solution_parameters=astrometry.SolutionParameters(
                logodds_callback=logodds_callback_100,
                sip_order=sip_order)
        )
        nmatches = len(solution.matches)
        logger.debug(f'Total matches: {nmatches}')
        if nmatches > 0:
            h_wcs = solution.best_match().wcs_fields
        else:
            logger.warning('No matches found. Trying without position hint.')
            solution = get_solver().solve(
                stars_xs=df['xcentroid'],
                stars_ys=df['ycentroid'],
                size_hint=None
                ,
                position_hint=None
                ,
                solution_parameters=astrometry.SolutionParameters(
                    logodds_callback=logodds_callback_100,
                    sip_order=sip_order)
            )
            nmatches = len(solution.matches)
            logger.debug(f'Total matches: {nmatches}')
            if nmatches > 0:
                h_wcs = solution.best_match().wcs_fields
            else:
                h_wcs = {}
                logger.warning('No matches found.')
    except Exception as e:
        logger.error(e)
        h_wcs = {}
    signal.alarm(0)
    return h_wcs


def get_zeropoint(df_catalog, df_sources, exptime, center_lims=None, N=50,
                  solar_filter=0.6, dist_thres_px=3, min_snr=30, max_snr=300,
                  plot=False):
    """Calculate the zeropoint for photometry.

    :param df_catalog: Catalog dataframe.
    :type df_catalog: pd.DataFrame
    :param flux: Flux values.
    :type flux: ndarray
    :param noise: Noise values.
    :type noise: ndarray
    :param coord: Coordinates of sources.
    :type coord: ndarray
    :param exptime: Exposure time.
    :type exptime: float
    :param solar_filter: Solar filter, by default 0.3.
    :type solar_filter: float, optional
    :param dist_thres_px: Distance threshold in pixels, by default 3.
    :type dist_thres_px: int, optional
    :param N: Number of brightest stars to use, by default 50.
    :type N: int, optional
    :param plot: Whether to plot the results, by default False.
    :type plot: bool, optional


    """
    source_coords_matched_idx, ref_coords_matched_idx = crossmatch_sources(df_sources[['RA', 'DEC']].values,
                                                                           df_catalog[['RA', 'DEC']].values,
                                                                           thres_px=dist_thres_px)

    solar_cat_filt = np.abs(df_catalog['SOLAR'])[ref_coords_matched_idx
                     ] < solar_filter
    source_coords_matched_idx = source_coords_matched_idx[solar_cat_filt]
    ref_coords_matched_idx = ref_coords_matched_idx[solar_cat_filt]
    det_mag = -2.5 * np.log10(df_sources.flux[source_coords_matched_idx].values / exptime)
    cat_mag = df_catalog['MAG'][ref_coords_matched_idx].to_numpy()
    inf_nan_mask = np.isfinite(cat_mag) & np.isfinite(det_mag) & ~np.isnan(
        cat_mag) & ~np.isnan(det_mag)
    cat_mag = cat_mag[inf_nan_mask]
    det_mag = det_mag[inf_nan_mask]
    snr = df_sources['snr'][source_coords_matched_idx][inf_nan_mask].values
    bright_mask = (snr > min_snr) & (snr < max_snr)

    if center_lims is None:
        xmin = np.floor(df_sources['xcentroid'].min())
        xmax = np.ceil(df_sources['xcentroid'].max())
        ymin = np.floor(df_sources['ycentroid'].min())
        ymax = np.ceil(df_sources['ycentroid'].max())
    else:
        xmin, xmax, ymin, ymax = center_lims

    bright_mask = bright_mask & (df_sources['xcentroid'][source_coords_matched_idx][inf_nan_mask].values >= xmin) & \
                  (df_sources['xcentroid'][source_coords_matched_idx][inf_nan_mask].values <= xmax) & \
                  (df_sources['ycentroid'][source_coords_matched_idx][inf_nan_mask].values >= ymin) & \
                  (df_sources['ycentroid'][source_coords_matched_idx][inf_nan_mask].values <= ymax)

    if len(cat_mag[bright_mask]) > N:
        indices_original = np.where(bright_mask)[0]
        m = np.argsort(snr[bright_mask])
        brightest_original = indices_original[m[-N:]]
        brightest_mask = np.zeros(len(bright_mask), dtype=bool)
        brightest_mask[brightest_original] = True
        bright_mask = bright_mask & brightest_mask

    if len(cat_mag[bright_mask]) <= 3:
        zp, ezp, n, min_mag, max_mag = 0, 0, 0, 0, 0

    else:
        y = cat_mag[bright_mask] - det_mag[bright_mask]
        mask = np.abs(y - np.nanmean(y)) < np.nanstd(y)
        if np.sum(mask) > 15:
            reg = RANSACRegressor(random_state=42, residual_threshold=0.05
                                  ).fit(det_mag[bright_mask].reshape([-1, 1])[mask], cat_mag[
                bright_mask].reshape([-1, 1])[mask])
            inlier = reg.inlier_mask_
            if np.sum(inlier) > 10 and np.abs(np.mean(y[mask][inlier]) - np
                    .mean(y[mask])) < np.std(y[mask]):
                zp = np.mean(y[mask][inlier])
                n = np.sum(inlier)
                ezp = np.std(y[mask][inlier]) / np.sqrt(n)
                min_mag = np.min(cat_mag[bright_mask][mask][inlier])
                max_mag = np.max(cat_mag[bright_mask][mask][inlier])
            else:
                zp = np.mean(y[mask])
                n = np.sum(mask)
                ezp = np.std(y[mask]) / np.sqrt(n)
                min_mag = np.min(cat_mag[bright_mask][mask])
                max_mag = np.max(cat_mag[bright_mask][mask])
        else:
            zp = np.mean(y[mask])
            n = np.sum(mask)
            ezp = np.std(y[mask]) / np.sqrt(n)
            min_mag = np.min(cat_mag[bright_mask][mask])
            max_mag = np.max(cat_mag[bright_mask][mask])

    params = {
        'ZP': np.round(zp, 4),
        'EZP': np.round(ezp, 4),
        'CATNSTAR': n,
        'ZPMINMAG': np.round(min_mag, 2),
        'ZPMAXMAG': np.round(max_mag, 2),
        'BVMIN': np.round(0.65 - solar_filter / 2, 2),
        'BVMAX': np.round(0.65 + solar_filter / 2, 2),
    }

    if plot:
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111)
        ax.plot(cat_mag, cat_mag - det_mag - zp, 'k.', alpha=0.6)
        if np.sum(mask) > 15:
            ax.plot(cat_mag[bright_mask][mask], cat_mag[bright_mask][mask] -
                    det_mag[bright_mask][mask] - zp, 'b.', alpha=0.1)
            ax.plot(cat_mag[bright_mask][mask][inlier], cat_mag[bright_mask
            ][mask][inlier] - det_mag[bright_mask][mask][inlier] - zp,
                    'r.', label='zp = {:.3f} +/- {:.3f} (n={})'.format(zp, ezp,
                                                                       n), alpha=0.5)
        else:
            ax.plot(cat_mag[bright_mask][mask], cat_mag[bright_mask][mask] -
                    det_mag[bright_mask][mask] - zp, 'r.', label=
                    'zp = {:.3f} +/- {:.3f} (n={})'.format(zp, ezp, n), alpha=0.5)
        ax.set_xlabel('catalog magnitude')
        ax.set_ylabel('error magnitude')
        ax.legend(frameon=False)
        ax.set_ylim(-1.5, 1.5)
        plt.show()

    return params


def get_target_snr(dfm: pd.DataFrame, target_ra: float, target_dec: float, dist_thres_px=3) -> float:
    """
    Get the SNR of the target.

    :param dfm: Dataframe with photometry data.
    :param target_ra: Target right ascension, in degrees.
    :param target_dec: Target declination, in degrees.
    :param dist_thres_px: Distance threshold in pixels, by default 3.
    :return: SNR of the target.
    """
    _, idx = crossmatch_sources([[target_ra, target_dec]],
                                dfm[['RA', 'DEC']].values,
                                thres_px=dist_thres_px)
    if idx.size > 0:
        target_snr = dfm.snr[idx[0]]
    else:
        target_snr = 0
    return target_snr


def get_maglim(mag: np.ndarray, snr: np.ndarray, snr_lim: float) -> float:
    """
    Get the limiting magnitude.

    :param mag: Magnitude values.
    :param snr: SNR values.
    :param snr_lim: SNR limit.
    :return: Limiting magnitude.
    """
    mask = (snr > 2) & (snr < 15)
    if np.sum(mask) < 3:
        maglim = 0
        logger.warning('Not enough stars to calculate the limiting magnitude.')
    else:
        p = np.polyfit(mag[mask], np.log10(snr[mask]), 1)
        x = np.linspace(12, 24, 100)
        fitted_snr = np.polyval(p, x)
        maglim = x[np.argmin(np.abs(fitted_snr - np.log10(snr_lim)))]
    return np.round(maglim, 2)


def radec_to_moon_sun(ra, dec, site_latitude, site_longitude, site_elevation, date_obs):
    observer = ephem.Observer()
    observer.lat = np.radians(site_latitude)
    observer.lon = np.radians(site_longitude)
    observer.elevation = site_elevation
    observer.date = datetime.strptime(date_obs, '%Y-%m-%dT%H:%M:%S.%f')
    target = ephem.FixedBody()
    target._ra = np.radians(ra)
    target._dec = np.radians(dec)
    target.compute(observer)

    moon = ephem.Moon()
    sun = ephem.Sun()
    moon.compute(observer)
    sun.compute(observer)
    moon.compute(observer)
    moon_alt = np.degrees(moon.alt)
    moon_az = np.degrees(moon.az)
    distance_to_moon = np.degrees(ephem.separation((target.ra, target.dec), (moon.ra, moon.dec)))
    moon_phase = moon.phase

    sun_alt = np.degrees(sun.alt)
    sun_az = np.degrees(sun.az)

    return round(moon_alt, 2), round(moon_az, 2), round(distance_to_moon, 2), round(moon_phase, 3), round(sun_alt,
                                                                                                          2), round(
        sun_az, 2)


def radec_to_altaz(RA, DEC, SITELAT, SITELON, SITEELEV, Date):
    coords_deg = SkyCoord(RA * u.deg, DEC * u.deg, frame='icrs', unit='deg')
    Observatory = EarthLocation(lat=SITELAT * u.deg, lon=SITELON * u.deg, height=SITEELEV * u.m)
    aa = AltAz(location=Observatory, obstime=Date)
    coords_altaz = coords_deg.transform_to(aa)
    airmass = float(coords_altaz.secz)
    zen = coords_altaz.zen
    return round(coords_altaz.az.deg, 6), round(coords_altaz.alt.deg, 6), round(airmass, 6), round(zen.deg, 6)


def radec_to_gal(RA, DEC):
    coords_gal = SkyCoord(RA * u.deg, DEC * u.deg, frame='icrs', unit='deg').galactic
    return round(coords_gal.l.deg, 6), round(coords_gal.b.deg, 6)


def radec_to_ecl(RA, DEC):
    coords_gal = SkyCoord(RA * u.deg, DEC * u.deg, frame='icrs', unit='deg').barycentricmeanecliptic
    return round(coords_gal.lon.deg, 6), round(coords_gal.lat.deg, 6)

def date_to_jd(dateobs):
    Date = Time(dateobs, scale='utc')
    return Date.jd, Date.mjd

def get_ccw(hwcs):
    cd11 = hwcs['CD1_1']
    cd12 = hwcs['CD1_2']
    cd21 = hwcs['CD2_1']
    cd22 = hwcs['CD2_2']
    det = cd11 * cd22 - cd12 * cd21
    if det >= 0:
        parity = 1.
    else:
        parity = -1.
    T = parity * cd11 + cd22
    A = parity * cd21 - cd12
    return -np.degrees(np.arctan2(A, T))


def get_scale(hwcs):
    cd11 = hwcs['CD1_1']
    cd12 = hwcs['CD1_2']
    cd21 = hwcs['CD2_1']
    cd22 = hwcs['CD2_2']
    return np.sqrt(cd11 ** 2 + cd12 ** 2) * 3600


def plate_scale_px(microns, focal):
    # pixel size in microns
    return plate_scale_mm(focal) * microns / 1000  # arcsec/px


def plate_scale_mm(focal):
    # focal length in mm
    return 206265 / focal  # arcsec/mm
