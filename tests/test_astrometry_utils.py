from unittest import TestCase

from gpuphot.astrometry.utils import get_target_ephemeris


class Test(TestCase):
    def test_get_target_ephemeris(self):
        _site_lat = +28.300332
        _site_lon = -16.512206
        _site_elev = 2390
        _target_name = 'Ceres'
        _date_obs = 2458133.33546
        _ephems = (9.434766666666667, 27.97686)

        ephems = get_target_ephemeris(_target_name, _date_obs, _site_lat, _site_lon, _site_elev)

        self.assertEquals(ephems, _ephems)
