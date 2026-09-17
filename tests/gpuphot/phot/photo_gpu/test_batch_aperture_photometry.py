import unittest

import pytest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from ...conftest import NVRTC_WORKS


@unittest.skipUnless(HAS_CUPY, "CuPy required")
@pytest.mark.skipif(not NVRTC_WORKS, reason="CuPy NVRTC compilation broken")
class TestBatchAperturePhotometry(unittest.TestCase):

    def setUp(self):
        from gpuphot.phot.photo_gpu import batch_aperture_photometry
        self.batch_aperture_photometry = batch_aperture_photometry

    def _make_image(self, size=128, value=1000.0):
        import numpy as np
        rng = np.random.default_rng(42)
        data = rng.normal(loc=value, scale=20.0, size=(size, size)).astype(np.float32)
        return cp.asarray(data)

    def test_returns_three_tuple(self):
        img = self._make_image()
        positions = cp.array([[64, 64]], dtype=cp.int32)
        radii = cp.array([3.0, 5.0, 7.0], dtype=cp.float32)
        result = self.batch_aperture_photometry(img, None, positions, radii)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_raises_value_error_on_empty_radii(self):
        img = self._make_image()
        positions = cp.array([[64, 64]], dtype=cp.int32)
        radii = cp.array([], dtype=cp.float32)
        with self.assertRaises(ValueError):
            self.batch_aperture_photometry(img, None, positions, radii)

    def test_area_output_length_matches_radii(self):
        img = self._make_image()
        positions = cp.array([[40, 40], [64, 64]], dtype=cp.int32)
        radii = cp.array([3.0, 5.0, 8.0], dtype=cp.float32)
        flux, back_flux, area = self.batch_aperture_photometry(img, None, positions, radii)
        self.assertEqual(len(area), 3)

    def test_area_values_positive(self):
        img = self._make_image()
        positions = cp.array([[64, 64]], dtype=cp.int32)
        radii = cp.array([4.0, 6.0], dtype=cp.float32)
        flux, back_flux, area = self.batch_aperture_photometry(img, None, positions, radii)
        area_np = area.get()
        self.assertTrue((area_np > 0).all())

    def test_back_flux_is_none_when_no_background(self):
        img = self._make_image()
        positions = cp.array([[64, 64]], dtype=cp.int32)
        radii = cp.array([5.0], dtype=cp.float32)
        flux, back_flux, area = self.batch_aperture_photometry(img, None, positions, radii)
        self.assertIsNone(back_flux)

    def test_flux_is_cupy_array(self):
        img = self._make_image()
        positions = cp.array([[64, 64]], dtype=cp.int32)
        radii = cp.array([5.0], dtype=cp.float32)
        flux, back_flux, area = self.batch_aperture_photometry(img, None, positions, radii)
        self.assertIsInstance(flux, cp.ndarray)


if __name__ == '__main__':
    unittest.main()
