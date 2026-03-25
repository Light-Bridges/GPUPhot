import unittest
import warnings

import pytest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from ...conftest import NVRTC_WORKS


@unittest.skipUnless(HAS_CUPY, "CuPy required")
@pytest.mark.skipif(not NVRTC_WORKS, reason="CuPy NVRTC compilation broken")
class TestAperturePhotometry(unittest.TestCase):

    def setUp(self):
        from .....gpuphot.phot.photo_gpu import aperture_photometry
        self.aperture_photometry = aperture_photometry

    def _make_image(self, size=128, value=1000.0):
        import numpy as np
        rng = np.random.default_rng(42)
        data = rng.normal(loc=value, scale=20.0, size=(size, size)).astype(np.float32)
        return cp.asarray(data)

    def test_emits_deprecation_warning(self):
        img = self._make_image()
        positions = cp.array([[64, 64]], dtype=cp.float32)
        with self.assertWarns(DeprecationWarning):
            self.aperture_photometry(img, positions, aper_rad=5)

    def test_returns_two_tuple(self):
        img = self._make_image()
        positions = cp.array([[64, 64]], dtype=cp.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = self.aperture_photometry(img, positions, aper_rad=5)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_flux_is_cupy_array(self):
        img = self._make_image()
        positions = cp.array([[64, 64]], dtype=cp.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            flux, area = self.aperture_photometry(img, positions, aper_rad=5)
        self.assertIsInstance(flux, cp.ndarray)

    def test_flux_length_matches_positions(self):
        img = self._make_image()
        positions = cp.array([[30, 30], [64, 64], [90, 90]], dtype=cp.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            flux, area = self.aperture_photometry(img, positions, aper_rad=5)
        self.assertEqual(len(flux), 3)

    def test_area_is_positive(self):
        img = self._make_image()
        positions = cp.array([[64, 64]], dtype=cp.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            flux, area = self.aperture_photometry(img, positions, aper_rad=5)
        self.assertGreater(float(area.get()), 0.0)


if __name__ == '__main__':
    unittest.main()
