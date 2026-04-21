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
class TestCovNan(unittest.TestCase):

    def setUp(self):
        from .....gpuphot.phot.photo_gpu import cov_nan
        self.cov_nan = cov_nan

    def _uniform_image(self, size=100, value=500.0):
        import numpy as np
        data = np.full((size, size), value, dtype=np.float32)
        return cp.asarray(data)

    def test_no_nan_in_output(self):
        img = self._uniform_image()
        # Inject NaNs
        img[10:15, 10:15] = cp.nan
        result = self.cov_nan(img, nc=10)
        nan_count = int(cp.sum(cp.isnan(result)).get())
        self.assertEqual(nan_count, 0)

    def test_image_without_nan_is_unchanged(self):
        import numpy as np
        rng = np.random.default_rng(7)
        data = rng.normal(500.0, 10.0, (100, 100)).astype(np.float32)
        img_orig = cp.asarray(data.copy())
        img_input = cp.asarray(data.copy())
        result = self.cov_nan(img_input, nc=10)
        self.assertTrue(bool(cp.allclose(result, img_orig).get()))

    def test_fills_nan_with_reasonable_values(self):
        # All values are ~500; NaN-filled pixels should also be near 500
        img = self._uniform_image(value=500.0)
        img[40:60, 40:60] = cp.nan
        result = self.cov_nan(img, nc=10)
        filled_region = result[40:60, 40:60]
        mean_val = float(cp.mean(filled_region).get())
        self.assertGreater(mean_val, 400.0)
        self.assertLess(mean_val, 600.0)

    def test_returns_cupy_array(self):
        img = self._uniform_image()
        img[5, 5] = cp.nan
        result = self.cov_nan(img, nc=10)
        self.assertIsInstance(result, cp.ndarray)

    def test_output_shape_unchanged(self):
        img = self._uniform_image(size=120)
        img[10, 10] = cp.nan
        result = self.cov_nan(img, nc=10)
        self.assertEqual(result.shape, (120, 120))


if __name__ == '__main__':
    unittest.main()
