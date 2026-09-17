import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestComputePhasediff(unittest.TestCase):

    def test_real_positive_zero_phase(self):
        from gpuphot.stats.subpixel import _compute_phasediff
        cc_max = cp.array(10.0 + 0j)
        result = _compute_phasediff(cc_max)
        np.testing.assert_allclose(float(result), 0.0, atol=1e-10)

    def test_pure_imaginary_pi_over_2(self):
        from gpuphot.stats.subpixel import _compute_phasediff
        cc_max = cp.array(0.0 + 1.0j)
        result = _compute_phasediff(cc_max)
        np.testing.assert_allclose(float(result), np.pi / 2, atol=1e-6)

    def test_negative_real_pi(self):
        from gpuphot.stats.subpixel import _compute_phasediff
        cc_max = cp.array(-5.0 + 0j)
        result = _compute_phasediff(cc_max)
        np.testing.assert_allclose(float(result), np.pi, atol=1e-6)

    def test_returns_cupy_array(self):
        from gpuphot.stats.subpixel import _compute_phasediff
        cc_max = cp.array(1.0 + 1.0j)
        result = _compute_phasediff(cc_max)
        self.assertIsInstance(result, cp.ndarray)


if __name__ == '__main__':
    unittest.main()
