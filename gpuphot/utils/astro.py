import inspect
import os
import re
import signal
import tempfile
import threading
from datetime import datetime

import astrometry
import ephem
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy.wcs import WCS
from sklearn.linear_model import RANSACRegressor

from .catalog import crossmatch_sources
from ..exceptions import AstrometrizationTimeoutError
from ..instrument_config_parser import HeaderKey
from ..logger.hierarchical_logging import setup_logger, hierarchical_debug
from astroquery.astrometry_net import AstrometryNet

logger = setup_logger(__name__)


class SingletonSolver:
    """
    A singleton class for managing the astrometry solver.
    """
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
        """
        Initialize the astrometry solver with appropriate index files.
        """

        default_cache = '/data/astrometry_cache'
        env_cache = os.getenv('ASTROMETRY_CACHE_PATH')
        fallback_cache = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'astrometry_cache')

        if os.path.exists(default_cache):
            cache = default_cache
        elif env_cache:
            cache = env_cache
        else:
            cache = fallback_cache

        if not os.path.exists(cache):
            try:
                os.makedirs(cache, exist_ok=True)
                logger.warning(f"Cache directory not found. Created directory: {cache}")
            except PermissionError:
                logger.error(f"Unable to create directory {cache}. Check permissions.")
                cache = tempfile.mkdtemp(prefix='astrometry_cache_')
                logger.warning(f"Using temporary directory as cache: {cache}")

        logger.debug(f"Using cache directory: {cache}")

        index_files_exist = self.check_index_files_exist(cache)
        if not index_files_exist:
            logger.warning(
                "Unable to locate astrometry index files. Starting the download of index files now. "
                "This process may take up to an hour, depending on your internet speed. "
                "If calling this from within a Celery worker with a soft or hard time limit, this may cause issues. "
                "It is recommended to call `from gpuphot.utils.astro import get_solver; solver = get_solver()` "
                "outside the worker to download the index files beforehand."
            )

        try:
            self.solver = astrometry.Solver(
                astrometry.series_5200.index_files(cache_directory=cache, scales={0, 1, 2, 3, 4, 5, 6}) +
                astrometry.series_4100.index_files(cache_directory=cache, scales={7, 8, 9, 10, 11})
            )
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error initializing the solver: {error_message}")

            match = re.search(r'loading "(.*?)" failed', error_message)
            if match:
                problematic_file = match.group(1)
                logger.warning(f"Attempting to remove problematic file: {problematic_file}")
                try:
                    os.remove(problematic_file)
                    logger.debug(f"Removed file: {problematic_file}")
                    self.initialize_solver()
                except OSError as remove_error:
                    logger.error(f"Error removing file: {remove_error}")

    @staticmethod
    def check_index_files_exist(directory):
        """
        Check if astrometry index files exist in the given directory.

        :param directory: Directory to check for index files.
        :type directory: str
        :return: True if index files exist, False otherwise.
        :rtype: bool
        """
        fits_count = 0
        download_count = 0

        for root, dirs, files in os.walk(directory):
            fits_count += len([f for f in files if f.endswith('.fits')])
            download_count += len([f for f in files if f.endswith('.download')])

        return fits_count > 0 and download_count == 0


@hierarchical_debug(logger)
def get_solver():
    """
    Get the astrometry solver with index files.

    :return: Configured astrometry solver instance.
    :rtype: astrometry.Solver
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
    """
    Get astrometry parameters from WCS header.

    :param h_wcs: WCS header.
    :type h_wcs: dict
    :param image_shape: Shape of the image.
    :type image_shape: tuple
    :return: Center coordinates, field of view, and scale.
    :rtype: tuple
    """
    w = WCS(h_wcs)
    ra, dec = w.all_pix2world(image_shape[1] // 2, image_shape[0] // 2, 1)
    try: cd11 = float(h_wcs['CD1_1'][0])
    except: cd11 = float(h_wcs['CD1_1'])
    try: cd12 = float(h_wcs[HeaderKey.CD1_2.value][0])
    except: cd12 = float(h_wcs[HeaderKey.CD1_2.value])
    scale = np.sqrt(cd11 ** 2 + cd12 ** 2) * 3600
    coocenter = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    npx = max(image_shape)
    FOV = 2 * np.sqrt(2 * ((scale * npx / 3600) ** 2))
    return coocenter, FOV, scale


def logodds_callback_100(logodds):
    """
    Callback function for astrometry solver.

    :param logodds: Log-odds of the solution.
    :type logodds: list
    :return: Action to take (STOP or CONTINUE).
    :rtype: astrometry.Action
    """
    # print(logodds)
    if (logodds[0] > 100.0) | (len(logodds) > 2):
        return astrometry.Action.STOP
    else:
        return astrometry.Action.CONTINUE


def handler(signum, frame):
    """
    Signal handler for astrometry timeout.

    :param signum: Signal number.
    :type signum: int
    :param frame: Current stack frame.
    :type frame: frame
    :raises AstrometrizationTimeoutError: If astrometry times out.
    """
    logger.info("Astrometrization timeout!")
    raise AstrometrizationTimeoutError("End of time for astrometrization")


@hierarchical_debug(logger)
def astrometrice2(df: pd.DataFrame, scale: float,
                  central_ra: float, central_dec: float,
                  image_shape: tuple,
                  sip_order: int = 3, n_max=500) -> dict:
    """
    Perform astrometry on an image.

    :param df: Dataframe containing detected sources.
    :type df: pd.DataFrame
    :param scale: Image scale in arcseconds per pixel.
    :type scale: float
    :param central_ra: Central right ascension in degrees.
    :type central_ra: float
    :param central_dec: Central declination in degrees.
    :type central_dec: float
    :param sip_order: SIP (Simple Imaging Polynomial) order, default is 3.
    :type sip_order: int
    :return: Updated WCS header.
    :rtype: dict
    """

    try:
        df = df.head(n_max)

        solver = get_solver()
        solve_params = inspect.signature(solver.solve).parameters

        if 'stars_xs' in solve_params and 'stars_ys' in solve_params:
            star_data = {'stars_xs': df['xcentroid'], 'stars_ys': df['ycentroid']}
        elif 'stars' in solve_params:
            star_data = {'stars': df[['xcentroid', 'ycentroid']].values.tolist()}
        else:
            raise AstrometrizationTimeoutError("Unexpected solver.solve() signature")

        common_params = {
            'solution_parameters': astrometry.SolutionParameters(
                logodds_callback=logodds_callback_100,
                sip_order=sip_order
            )
        }

        # Try solving locally with position hint
        try:
            logger.info("Starting local astrometry with position hint")
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(60)
            solution = solver.solve(
                **star_data,
                size_hint=astrometry.SizeHint(
                    lower_arcsec_per_pixel=scale * 0.8,
                    upper_arcsec_per_pixel=scale * 1.
                ),
                position_hint=astrometry.PositionHint(
                    ra_deg=central_ra,
                    dec_deg=central_dec,
                    radius_deg=1,
                ),
                **common_params
            )
            if solution.matches:
                return solution.best_match().wcs_fields
        except Exception as e:
            logger.warning(f"Local astrometry with position hint failed: {e}")


        # Try solving locally without position hint
        try:
            # xmin, xmax = int(image_shape[1] * 0.25), int(image_shape[1] * 0.75)
            # ymin, ymax = int(image_shape[0] * 0.25), int(image_shape[0] * 0.75)
            # df_trim = df[(df['xcentroid'] > xmin) & (df['xcentroid'] < xmax) & (df['ycentroid'] > ymin) & (df['ycentroid'] < ymax)]
            # star_data = {'stars_xs': df_trim['xcentroid'], 'stars_ys': df_trim['ycentroid']}

            logger.info("Starting local astrometry without position hint")
            signal.alarm(60)
            solution = solver.solve(
                **star_data,
                size_hint=None,
                position_hint=None,
                **common_params
            )
            if solution.matches:
                return solution.best_match().wcs_fields
        except Exception as e:
            logger.warning(f"Local astrometry without position hint failed: {e}")

        # Try solving online
        try:
            logger.info("Starting online astrometry")
            signal.alarm(60)
            logger.info("Starting online astrometry with AstrometryNet")
            ast = AstrometryNet()
            ast.api_key = 'ruavrmwepqfvhdqm'
            image_width, image_height = image_shape
            h_wcs = ast.solve_from_source_list(star_data['stars_xs'], star_data['stars_ys'],
                                               image_width, image_height,
                                               solve_timeout=90)
            if h_wcs:
                return h_wcs
        except Exception as e:
            logger.error(f"Online astrometry failed: {e}")
            raise AstrometrizationTimeoutError("Astrometry failed")

    finally:
        signal.alarm(0)  # Disable alarm

    return {}


def get_zeropoint(df_catalog, df_sources, exptime, center_lims=None, N=50,
                  solar_filter=0.6, dist_thres_px=3, min_snr=30, max_snr=300,
                  plot=False):
    """
    Calculate the zeropoint for photometry.

    :param df_catalog: Catalog dataframe.
    :type df_catalog: pd.DataFrame
    :param df_sources: Sources dataframe.
    :type df_sources: pd.DataFrame
    :param exptime: Exposure time.
    :type exptime: float
    :param center_lims: Limits for center region, optional.
    :type center_lims: tuple or None
    :param N: Number of brightest stars to use, default is 50.
    :type N: int
    :param solar_filter: Solar filter value, default is 0.6.
    :type solar_filter: float
    :param dist_thres_px: Distance threshold in pixels, default is 3.
    :type dist_thres_px: int
    :param min_snr: Minimum signal-to-noise ratio, default is 30.
    :type min_snr: int
    :param max_snr: Maximum signal-to-noise ratio, default is 300.
    :type max_snr: int
    :param plot: Whether to plot the results, default is False.
    :type plot: bool
    :return: Dictionary of zeropoint parameters.
    :rtype: dict
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
    :type dfm: pd.DataFrame
    :param target_ra: Target right ascension, in degrees.
    :type target_ra: float
    :param target_dec: Target declination, in degrees.
    :type target_dec: float
    :param dist_thres_px: Distance threshold in pixels, default is 3.
    :type dist_thres_px: int
    :return: SNR of the target.
    :rtype: float
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
    :type mag: numpy.ndarray
    :param snr: SNR values.
    :type snr: numpy.ndarray
    :param snr_lim: SNR limit.
    :type snr_lim: float
    :return: Limiting magnitude.
    :rtype: float
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
    """
    Calculate moon and sun positions relative to a target.

    :param ra: Right ascension of target in degrees.
    :type ra: float
    :param dec: Declination of target in degrees.
    :type dec: float
    :param site_latitude: Latitude of observation site in degrees.
    :type site_latitude: float
    :param site_longitude: Longitude of observation site in degrees.
    :type site_longitude: float
    :param site_elevation: Elevation of observation site in meters.
    :type site_elevation: float
    :param date_obs: Date and time of observation.
    :type date_obs: str
    :return: Tuple of moon altitude, azimuth, distance to target, phase, and sun altitude and azimuth.
    :rtype: tuple
    """
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
    """
    Convert RA/Dec to altitude and azimuth.

    :param RA: Right ascension in degrees.
    :type RA: float
    :param DEC: Declination in degrees.
    :type DEC: float
    :param SITELAT: Site latitude in degrees.
    :type SITELAT: float
    :param SITELON: Site longitude in degrees.
    :type SITELON: float
    :param SITEELEV: Site elevation in meters.
    :type SITEELEV: float
    :param Date: Observation date and time.
    :type Date: str
    :return: Tuple of azimuth, altitude, airmass, and zenith distance.
    :rtype: tuple
    """
    coords_deg = SkyCoord(RA * u.deg, DEC * u.deg, frame='icrs', unit='deg')
    Observatory = EarthLocation(lat=SITELAT * u.deg, lon=SITELON * u.deg, height=SITEELEV * u.m)
    aa = AltAz(location=Observatory, obstime=Date)
    coords_altaz = coords_deg.transform_to(aa)
    airmass = float(coords_altaz.secz)
    zen = coords_altaz.zen
    return round(coords_altaz.az.deg, 6), round(coords_altaz.alt.deg, 6), round(airmass, 6), round(zen.deg, 6)


def radec_to_gal(RA, DEC):
    """
    Convert RA/Dec to Galactic coordinates.

    :param RA: Right ascension in degrees.
    :type RA: float
    :param DEC: Declination in degrees.
    :type DEC: float
    :return: Tuple of Galactic longitude and latitude.
    :rtype: tuple
    """
    coords_gal = SkyCoord(RA * u.deg, DEC * u.deg, frame='icrs', unit='deg').galactic
    return round(coords_gal.l.deg, 6), round(coords_gal.b.deg, 6)


def radec_to_ecl(RA, DEC):
    """
    Convert RA/Dec to Ecliptic coordinates.

    :param RA: Right ascension in degrees.
    :type RA: float
    :param DEC: Declination in degrees.
    :type DEC: float
    :return: Tuple of Ecliptic longitude and latitude.
    :rtype: tuple
    """
    coords_gal = SkyCoord(RA * u.deg, DEC * u.deg, frame='icrs', unit='deg').barycentricmeanecliptic
    return round(coords_gal.lon.deg, 6), round(coords_gal.lat.deg, 6)


def date_to_jd(dateobs):
    """
    Convert date to Julian Date.

    :param dateobs: Observation date and time.
    :type dateobs: str
    :return: Tuple of Julian Date and Modified Julian Date.
    :rtype: tuple
    """
    Date = Time(dateobs, scale='utc')
    return Date.jd, Date.mjd


def get_ccw(hwcs):
    """
    Get the counter-clockwise rotation angle from WCS header.

    :param hwcs: WCS header.
    :type hwcs: dict
    :return: Counter-clockwise rotation angle in degrees.
    :rtype: float
    """
    cd11 = hwcs[HeaderKey.CD1_1.value]
    cd12 = hwcs[HeaderKey.CD1_2.value]
    cd21 = hwcs[HeaderKey.CD2_1.value]
    cd22 = hwcs[HeaderKey.CD2_2.value]
    det = cd11 * cd22 - cd12 * cd21
    if det >= 0:
        parity = 1.
    else:
        parity = -1.
    T = parity * cd11 + cd22
    A = parity * cd21 - cd12
    return -np.degrees(np.arctan2(A, T))


def get_scale(hwcs):
    """
    Get the image scale from WCS header.

    :param hwcs: WCS header.
    :type hwcs: dict
    :return: Image scale in arcseconds per pixel.
    :rtype: float
    """
    cd11 = hwcs[HeaderKey.CD1_1.value]
    cd12 = hwcs[HeaderKey.CD1_2.value]
    cd21 = hwcs[HeaderKey.CD2_1.value]
    cd22 = hwcs[HeaderKey.CD2_2.value]
    return np.sqrt(cd11 ** 2 + cd12 ** 2) * 3600


def plate_scale_px(microns, focal):
    """
    Calculate plate scale in arcseconds per pixel.

    :param microns: Pixel size in microns.
    :type microns: float
    :param focal: Focal length in mm.
    :type focal: float
    :return: Plate scale in arcseconds per pixel.
    :rtype: float
    """

    # pixel size in microns
    return plate_scale_mm(focal) * microns / 1000  # arcsec/px


def plate_scale_mm(focal):
    """
    Calculate plate scale in arcseconds per mm.

    :param focal: Focal length in mm.
    :type focal: float
    :return: Plate scale in arcseconds per mm.
    :rtype: float
    """
    # focal length in mm
    return 206265 / focal  # arcsec/mm
