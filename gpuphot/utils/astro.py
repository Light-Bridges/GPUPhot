from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time


def plate_scale_px(microns, focal):
    # pixel size in microns
    return plate_scale_mm(focal) * microns / 1000  # arcsec/px


def deg_to_hms(RA, DEC):
    coords_deg = SkyCoord(RA * u.deg, DEC * u.deg, frame='icrs', unit='deg')
    ra_hms = '%02d:%02d:%.6f' % coords_deg.ra.hms
    if DEC > 0:
        dec_dms = '%02d:%02d:%02.6f' % coords_deg.dec.dms
    else:
        coords_deg = SkyCoord(RA * u.deg, -1 * DEC * u.deg, frame='icrs', unit='deg')
        dec_dms = '-%02d:%02d:%02.6f' % coords_deg.dec.dms
    return ra_hms, dec_dms


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


def plate_scale_mm(focal):
    # focal length in mm
    return 206265 / focal  # arcsec/mm
