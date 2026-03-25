import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestComputeError(unittest.TestCase):

    def test_perfect_correlation_zero_error(self):
        from .....gpuphot.stats.subpixel import _compute_error
        # cc_max = sqrt(src_amp * target_amp) => error = 0
        src_amp = cp.array(100.0)
        target_amp = cp.array(100.0)
        cc_max = cp.array(100.0 + 0j)  # perfect correlation
        result = _compute_error(cc_max, src_amp, target_amp)
        np.testing.assert_allclose(float(result), 0.0, atol=1e-6)

    def test_error_is_nonnegative(self):
        from .....gpuphot.stats.subpixel import _compute_error
        cc_max = cp.array(50.0 + 10j)
        src_amp = cp.array(200.0)
        target_amp = cp.array(200.0)
        result = _compute_error(cc_max, src_amp, target_amp)
        self.assertGreaterEqual(float(result), 0.0)

    def test_returns_cupy_array(self):
        from .....gpuphot.stats.subpixel import _compute_error
        cc_max = cp.array(10.0 + 0j)
        src_amp = cp.array(100.0)
        target_amp = cp.array(100.0)
        result = _compute_error(cc_max, src_amp, target_amp)
        self.assertIsInstance(result, cp.ndarray)

    def test_low_correlation_high_error(self):
        from .....gpuphot.stats.subpixel import _compute_error
        cc_max = cp.array(1.0 + 0j)
        src_amp = cp.array(1000.0)
        target_amp = cp.array(1000.0)
        result = _compute_error(cc_max, src_amp, target_amp)
        self.assertGreater(float(result), 0.9)


if __name__ == '__main__':
    unittest.main()
