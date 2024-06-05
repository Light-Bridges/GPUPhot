from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time


def plate_scale_px(microns, focal):
    """
    Calculate the plate scale in arcseconds per pixel.

    Parameters
    ----------
    microns : float
        Pixel size in micrometers.
    focal : float
        Focal length in millimeters.

    Returns
    -------
    float
        Plate scale in arcseconds per pixel.
    """
    return plate_scale_mm(focal) * microns / 1000  # arcsec/px


def deg_to_hms(RA, DEC):
    """
    Convert coordinates from degrees to hours, minutes, seconds (HMS) and degrees, minutes, seconds (DMS).

    Parameters
    ----------
    RA : float
        Right Ascension in degrees.
    DEC : float
        Declination in degrees.

    Returns
    -------
    tuple
        (ra_hms, dec_dms) where ra_hms is the Right Ascension in HMS format and dec_dms is the Declination in DMS format.
    """
    coords_deg = SkyCoord(RA * u.deg, DEC * u.deg, frame='icrs', unit='deg')
    ra_hms = '%02d:%02d:%.6f' % coords_deg.ra.hms
    if DEC > 0:
        dec_dms = '%02d:%02d:%02.6f' % coords_deg.dec.dms
    else:
        coords_deg = SkyCoord(RA * u.deg, -1 * DEC * u.deg, frame='icrs', unit='deg')
        dec_dms = '-%02d:%02d:%02.6f' % coords_deg.dec.dms
    return ra_hms, dec_dms


def radec_to_altaz(RA, DEC, SITELAT, SITELON, SITEELEV, date):
    """
    Convert RA/DEC coordinates to Alt/Az coordinates.

    Parameters
    ----------
    RA : float
        Right Ascension in degrees.
    DEC : float
        Declination in degrees.
    SITELAT : float
        Site latitude in degrees.
    SITELON : float
        Site longitude in degrees.
    SITEELEV : float
        Site elevation in meters.
    date : str
        Observation date and time in ISO format (YYYY-MM-DD HH:MM:SS).

    Returns
    -------
    tuple
        (az, alt, airmass, zen) where az is azimuth in degrees, alt is altitude in degrees,
        airmass is the airmass, and zen is the zenith angle in degrees.
    """
    coords_deg = SkyCoord(RA * u.deg, DEC * u.deg, frame='icrs', unit='deg')
    Observatory = EarthLocation(lat=SITELAT * u.deg, lon=SITELON * u.deg, height=SITEELEV * u.m)
    aa = AltAz(location=Observatory, obstime=date)
    coords_altaz = coords_deg.transform_to(aa)
    airmass = float(coords_altaz.secz)
    zen = coords_altaz.zen
    return round(coords_altaz.az.deg, 6), round(coords_altaz.alt.deg, 6), round(airmass, 6), round(zen.deg, 6)


def radec_to_gal(RA, DEC):
    """
    Convert RA/DEC coordinates to galactic coordinates.

    Parameters
    ----------
    RA : float
        Right Ascension in degrees.
    DEC : float
        Declination in degrees.

    Returns
    -------
    tuple
        (l, b) where l is galactic longitude in degrees and b is galactic latitude in degrees.
    """
    coords_gal = SkyCoord(RA * u.deg, DEC * u.deg, frame='icrs', unit='deg').galactic
    return round(coords_gal.l.deg, 6), round(coords_gal.b.deg, 6)


def radec_to_ecl(RA, DEC):
    """
    Convert RA/DEC coordinates to ecliptic coordinates.

    Parameters
    ----------
    RA : float
        Right Ascension in degrees.
    DEC : float
        Declination in degrees.

    Returns
    -------
    tuple
        (lon, lat) where lon is ecliptic longitude in degrees and lat is ecliptic latitude in degrees.
    """
    coords_gal = SkyCoord(RA * u.deg, DEC * u.deg, frame='icrs', unit='deg').barycentricmeanecliptic
    return round(coords_gal.lon.deg, 6), round(coords_gal.lat.deg, 6)


def date_to_jd(dateobs):
    """
    Convert an observation date to Julian Date (JD) and Modified Julian Date (MJD).

    Parameters
    ----------
    dateobs : str
        Observation date and time in ISO format (YYYY-MM-DD HH:MM:SS).

    Returns
    -------
    tuple
        (jd, mjd) where jd is Julian Date and mjd is Modified Julian Date.
    """
    date = Time(dateobs, scale='utc')
    return date.jd, date.mjd


def plate_scale_mm(focal):
    """
    Calculate the plate scale in arcseconds per millimeter.

    Parameters
    ----------
    focal : float
        Focal length in millimeters.

    Returns
    -------
    float
        Plate scale in arcseconds per millimeter.
    """
    return 206265 / focal  # arcsec/mm
