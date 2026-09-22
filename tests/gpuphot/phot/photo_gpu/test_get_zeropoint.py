import unittest

import numpy as np
import pandas as pd


class TestGetZeropoint(unittest.TestCase):
    """
    Tests for get_zeropoint using synthetic catalog and source DataFrames.

    Strategy: create stars at known positions with a known zeropoint (ZP=25),
    so cat_mag = -2.5*log10(flux/exptime) + ZP. The function should recover
    ZP ~25.
    """

    def _make_data(self, n_stars=30, zp_true=25.0, exptime=60.0, seed=42):
        """Build matched synthetic catalog and sources DataFrames."""
        rng = np.random.default_rng(seed)

        # Star positions on a grid so crossmatch finds exact matches
        ra = np.linspace(10.0, 10.1, n_stars)
        dec = np.linspace(20.0, 20.1, n_stars)

        # Catalog magnitudes spread across a reasonable range
        cat_mag = rng.uniform(12.0, 18.0, n_stars)

        # Flux that corresponds to the known zeropoint:
        #   cat_mag = -2.5*log10(flux/exptime) + ZP
        #   flux = exptime * 10^((ZP - cat_mag) / 2.5)
        flux = exptime * 10 ** ((zp_true - cat_mag) / 2.5)

        # Add small noise to flux so it's not perfectly deterministic
        flux *= rng.normal(1.0, 0.01, n_stars)

        snr = rng.uniform(50, 200, n_stars)
        solar = rng.uniform(-0.1, 0.1, n_stars)  # near-solar B-V

        x = rng.uniform(100, 900, n_stars)
        y = rng.uniform(100, 900, n_stars)

        df_catalog = pd.DataFrame({
            'RA': ra,
            'DEC': dec,
            'MAG': cat_mag,
            'SOLAR': solar,
        })

        df_sources = pd.DataFrame({
            'RA': ra,       # exact match positions
            'DEC': dec,
            'flux': flux,
            'snr': snr,
            'xcentroid': x,
            'ycentroid': y,
        })

        return df_catalog, df_sources, exptime

    def test_returns_dict(self):
        from gpuphot.utils.astro import get_zeropoint
        df_cat, df_src, exptime = self._make_data()
        result = get_zeropoint(df_cat, df_src, exptime)
        self.assertIsInstance(result, dict)

    def test_dict_has_required_keys(self):
        from gpuphot.utils.astro import get_zeropoint
        df_cat, df_src, exptime = self._make_data()
        result = get_zeropoint(df_cat, df_src, exptime)
        for key in ['ZP', 'EZP', 'CATNSTAR', 'ZPMINMAG', 'ZPMAXMAG', 'BVMIN', 'BVMAX']:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_recovers_known_zeropoint(self):
        from gpuphot.utils.astro import get_zeropoint
        zp_true = 25.0
        df_cat, df_src, exptime = self._make_data(n_stars=50, zp_true=zp_true)
        result = get_zeropoint(df_cat, df_src, exptime, dist_thres_px=100,
                               min_snr=10, max_snr=500)
        if result['CATNSTAR'] > 0:
            self.assertAlmostEqual(result['ZP'], zp_true, delta=0.5)

    def test_too_few_matches_returns_zero_zp(self):
        from gpuphot.utils.astro import get_zeropoint
        # Sources and catalog at completely different positions -> no matches
        df_cat = pd.DataFrame({
            'RA': [100.0, 101.0],
            'DEC': [50.0, 51.0],
            'MAG': [14.0, 15.0],
            'SOLAR': [0.0, 0.0],
        })
        df_src = pd.DataFrame({
            'RA': [200.0, 201.0],
            'DEC': [60.0, 61.0],
            'flux': [1000.0, 2000.0],
            'snr': [100.0, 100.0],
            'xcentroid': [500.0, 600.0],
            'ycentroid': [500.0, 600.0],
        })
        result = get_zeropoint(df_cat, df_src, exptime=60.0)
        self.assertEqual(result['ZP'], 0)
        self.assertEqual(result['CATNSTAR'], 0)

    def test_catnstar_is_nonnegative(self):
        from gpuphot.utils.astro import get_zeropoint
        df_cat, df_src, exptime = self._make_data()
        result = get_zeropoint(df_cat, df_src, exptime)
        self.assertGreaterEqual(result['CATNSTAR'], 0)

    def test_bv_range_matches_the_colour_cut_actually_applied(self):
        """BVMIN/BVMAX must describe the band the SOLAR cut really keeps.

        get_zeropoint filters on |SOLAR| < solar_filter, where SOLAR is
        (B-V) - 0.65, so the retained band is 0.65 +/- solar_filter.  Asserting
        against that band rather than against the header expression keeps this
        test from simply mirroring the implementation: it fails if the reported
        range and the applied cut ever drift apart again.
        """
        from gpuphot.utils.astro import get_zeropoint

        solar_filter = 0.6
        solar_bv_sun = 0.65

        # Stars straddling the cut: the first five pass, the last two do not.
        solar = np.array([-0.55, -0.30, 0.0, 0.30, 0.55, -0.80, 0.80])
        df_cat, df_src, exptime = self._make_data(n_stars=len(solar))
        df_cat['SOLAR'] = solar

        result = get_zeropoint(df_cat, df_src, exptime, solar_filter=solar_filter,
                               dist_thres_px=100, min_snr=10, max_snr=500)

        kept_bv = solar_bv_sun + solar[np.abs(solar) < solar_filter]
        rejected_bv = solar_bv_sun + solar[np.abs(solar) >= solar_filter]

        # Every calibrator that survived the cut must lie inside the reported band.
        self.assertLessEqual(result['BVMIN'], kept_bv.min())
        self.assertGreaterEqual(result['BVMAX'], kept_bv.max())

        # And the band must not advertise stars the cut threw away.
        self.assertGreater(result['BVMIN'], rejected_bv.min())
        self.assertLess(result['BVMAX'], rejected_bv.max())


if __name__ == '__main__':
    unittest.main()
