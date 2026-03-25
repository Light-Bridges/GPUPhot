import unittest

import pytest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np

from ...conftest import NVRTC_WORKS


@unittest.skipUnless(HAS_CUPY, "CuPy required")
@pytest.mark.skipif(not NVRTC_WORKS, reason="CuPy NVRTC compilation broken")
class TestStackSigmaclip(unittest.TestCase):

    def test_less_than_3_images_returns_none_std(self):
        from .....gpuphot.stats.reduction import stack_sigmaclip
        data = cp.ones((2, 64, 64), dtype=cp.float32) * 100.0
        avg, std = stack_sigmaclip(data)
        self.assertIsNone(std)

    def test_output_shape(self):
        from .....gpuphot.stats.reduction import stack_sigmaclip
        data = cp.ones((5, 32, 32), dtype=cp.float32)
        avg, std = stack_sigmaclip(data)
        self.assertEqual(avg.shape, (32, 32))
        self.assertEqual(std.shape, (32, 32))

    def test_uniform_stack_low_std(self):
        from .....gpuphot.stats.reduction import stack_sigmaclip
        data = cp.ones((5, 32, 32), dtype=cp.float32) * 500.0
        avg, std = stack_sigmaclip(data)
        np.testing.assert_allclose(avg.get(), 500.0, atol=1.0)
        np.testing.assert_allclose(std.get(), 0.0, atol=1.0)

    def test_returns_cupy_arrays(self):
        from .....gpuphot.stats.reduction import stack_sigmaclip
        data = cp.ones((4, 16, 16), dtype=cp.float32)
        avg, std = stack_sigmaclip(data)
        self.assertIsInstance(avg, cp.ndarray)
        self.assertIsInstance(std, cp.ndarray)

    def test_clips_outliers(self):
        from .....gpuphot.stats.reduction import stack_sigmaclip
        data = cp.ones((10, 32, 32), dtype=cp.float32) * 100.0
        # Add an extreme outlier in one frame
        data[0, :, :] = 100000.0
        avg, std = stack_sigmaclip(data, n=2)
        # Average should be near 100, not pulled to 100000
        mean_val = float(cp.mean(avg).get())
        self.assertLess(mean_val, 1000.0)


if __name__ == '__main__':
    unittest.main()
