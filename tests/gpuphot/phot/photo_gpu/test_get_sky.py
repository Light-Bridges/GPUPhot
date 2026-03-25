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
class TestGetSky(unittest.TestCase):

    def setUp(self):
        from .....gpuphot.phot.photo_gpu import get_sky
        self.get_sky = get_sky

    def _make_image(self, size=256, value=1000.0):
        import numpy as np
        rng = np.random.default_rng(42)
        data = rng.normal(loc=value, scale=10.0, size=(size, size)).astype(np.float32)
        return cp.asarray(data)

    def test_returns_three_tuple(self):
        img = self._make_image()
        result = self.get_sky(img, fw=4.0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_sky_is_cupy_array(self):
        img = self._make_image()
        sky, rms, mem = self.get_sky(img, fw=4.0)
        self.assertIsInstance(sky, cp.ndarray)

    def test_rms_is_cupy_array(self):
        img = self._make_image()
        sky, rms, mem = self.get_sky(img, fw=4.0)
        self.assertIsInstance(rms, cp.ndarray)

    def test_sky_and_rms_have_same_shape_as_input(self):
        img = self._make_image(size=256)
        sky, rms, mem = self.get_sky(img, fw=4.0)
        self.assertEqual(sky.shape, (256, 256))
        self.assertEqual(rms.shape, (256, 256))

    def test_mem_bytes_is_numeric(self):
        img = self._make_image()
        sky, rms, mem = self.get_sky(img, fw=4.0)
        self.assertIsInstance(mem, (int, float))

    def test_sky_values_reasonable_for_uniform_image(self):
        img = self._make_image(value=1000.0)
        sky, rms, mem = self.get_sky(img, fw=4.0)
        sky_mean = float(cp.nanmean(sky).get())
        self.assertGreater(sky_mean, 500.0)
        self.assertLess(sky_mean, 2000.0)

    def test_rms_values_nonnegative(self):
        img = self._make_image()
        sky, rms, mem = self.get_sky(img, fw=4.0)
        min_rms = float(cp.nanmin(rms).get())
        self.assertGreaterEqual(min_rms, 0.0)


if __name__ == '__main__':
    unittest.main()
