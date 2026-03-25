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
class TestSPFilter(unittest.TestCase):

    def setUp(self):
        from .....gpuphot.phot.cosmetics import SP_filter
        self.SP_filter = SP_filter

    def _smooth_image(self, size=128, value=1000.0):
        import numpy as np
        rng = np.random.default_rng(42)
        data = rng.normal(loc=value, scale=5.0, size=(size, size)).astype(np.float32)
        return cp.asarray(data)

    def test_output_shape_matches_input(self):
        img = self._smooth_image(size=128)
        result = self.SP_filter(img)
        self.assertEqual(result.shape, (128, 128))

    def test_returns_cupy_array(self):
        img = self._smooth_image()
        result = self.SP_filter(img)
        self.assertIsInstance(result, cp.ndarray)

    def test_removes_spike_noise(self):
        # Inject obvious salt-and-pepper spikes into a uniform background
        img = self._smooth_image(value=1000.0)
        # Place strong spikes far above background level
        img[30, 30] = 1.0e6
        img[60, 60] = -1.0e6
        result = self.SP_filter(img)
        # After filtering spikes should be gone (values near background level)
        val_high = float(result[30, 30].get())
        val_low = float(result[60, 60].get())
        self.assertLess(val_high, 1.0e5)
        self.assertGreater(val_low, -1.0e5)

    def test_smooth_image_mostly_unchanged(self):
        import numpy as np
        img_orig = self._smooth_image(value=1000.0)
        img_copy = img_orig.copy()
        result = self.SP_filter(img_copy)
        # For a smooth image, the vast majority of pixels should be near original
        diff = cp.abs(result - img_orig)
        frac_changed = float((cp.sum(diff > 50.0) / diff.size).get())
        self.assertLess(frac_changed, 0.05)

    def test_no_nan_in_output(self):
        img = self._smooth_image()
        result = self.SP_filter(img)
        nan_count = int(cp.sum(cp.isnan(result)).get())
        self.assertEqual(nan_count, 0)


if __name__ == '__main__':
    unittest.main()
