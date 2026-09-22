import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestCrossCorrelateMasked(unittest.TestCase):

    def test_invalid_mode_raises(self):
        from gpuphot.stats.subpixel_masked import cross_correlate_masked
        arr = cp.ones((16, 16), dtype=cp.float32)
        mask = cp.ones((16, 16), dtype=bool)
        with self.assertRaises(ValueError):
            cross_correlate_masked(arr, arr, mask, mask, mode='invalid')

    def test_identical_arrays_high_correlation(self):
        from gpuphot.stats.subpixel_masked import cross_correlate_masked
        arr = cp.random.rand(32, 32).astype(cp.float32)
        mask = cp.ones((32, 32), dtype=bool)
        result = cross_correlate_masked(arr, arr, mask, mask, mode='full')
        max_corr = float(cp.max(result))
        self.assertGreater(max_corr, 0.5)

    def test_output_shape_full_mode(self):
        from gpuphot.stats.subpixel_masked import cross_correlate_masked
        arr1 = cp.ones((16, 16), dtype=cp.float32)
        arr2 = cp.ones((16, 16), dtype=cp.float32)
        mask = cp.ones((16, 16), dtype=bool)
        result = cross_correlate_masked(arr1, arr2, mask, mask, mode='full')
        # Full mode: shape = (n1+n2-1, n1+n2-1)
        self.assertEqual(result.shape, (31, 31))

    def test_returns_cupy_array(self):
        from gpuphot.stats.subpixel_masked import cross_correlate_masked
        arr = cp.ones((8, 8), dtype=cp.float32)
        mask = cp.ones((8, 8), dtype=bool)
        result = cross_correlate_masked(arr, arr, mask, mask)
        self.assertIsInstance(result, cp.ndarray)

    def test_values_bounded(self):
        from gpuphot.stats.subpixel_masked import cross_correlate_masked
        arr = cp.random.rand(16, 16).astype(cp.float32)
        mask = cp.ones((16, 16), dtype=bool)
        result = cross_correlate_masked(arr, arr, mask, mask)
        self.assertLessEqual(float(cp.max(result)), 1.0 + 1e-5)
        self.assertGreaterEqual(float(cp.min(result)), -1.0 - 1e-5)


if __name__ == '__main__':
    unittest.main()
