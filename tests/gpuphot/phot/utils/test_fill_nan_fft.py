import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestFillNanFft(unittest.TestCase):

    def test_image_without_nan_unchanged(self):
        from .....gpuphot.phot.conv import fill_nan_fft
        image = cp.ones((64, 64), dtype=cp.float64) * 100.0
        result = fill_nan_fft(image, lk=3)
        np.testing.assert_allclose(result.get(), 100.0, atol=1e-8)

    def test_fills_interior_nan(self):
        from .....gpuphot.phot.conv import fill_nan_fft
        image = cp.ones((64, 64), dtype=cp.float64) * 50.0
        image[30, 30] = cp.nan
        result = fill_nan_fft(image, lk=3, min_neighbors=1)
        self.assertFalse(bool(cp.isnan(result[30, 30])))

    def test_output_shape_matches(self):
        from .....gpuphot.phot.conv import fill_nan_fft
        image = cp.ones((50, 70), dtype=cp.float64)
        image[10, 10] = cp.nan
        result = fill_nan_fft(image, lk=3, min_neighbors=1)
        self.assertEqual(result.shape, (50, 70))

    def test_filled_value_reasonable(self):
        from .....gpuphot.phot.conv import fill_nan_fft
        image = cp.ones((64, 64), dtype=cp.float64) * 200.0
        image[32, 32] = cp.nan
        result = fill_nan_fft(image, lk=5, min_neighbors=1)
        filled_val = float(result[32, 32])
        self.assertAlmostEqual(filled_val, 200.0, delta=10.0)

    def test_returns_cupy_array(self):
        from .....gpuphot.phot.conv import fill_nan_fft
        image = cp.ones((32, 32), dtype=cp.float64)
        result = fill_nan_fft(image, lk=2)
        self.assertIsInstance(result, cp.ndarray)


if __name__ == '__main__':
    unittest.main()
