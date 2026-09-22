import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestCentered(unittest.TestCase):

    def test_output_shape(self):
        from gpuphot.stats.subpixel_masked import _centered
        arr = cp.ones((10, 10), dtype=cp.float32)
        result = _centered(arr, newshape=(6, 6), axes=(0, 1))
        self.assertEqual(result.shape, (6, 6))

    def test_center_content(self):
        from gpuphot.stats.subpixel_masked import _centered
        arr = cp.arange(25, dtype=cp.float32).reshape(5, 5)
        result = _centered(arr, newshape=(3, 3), axes=(0, 1))
        # Center 3x3 of 5x5 starts at index 1
        expected = arr[1:4, 1:4]
        np.testing.assert_array_equal(result.get(), expected.get())

    def test_single_axis(self):
        from gpuphot.stats.subpixel_masked import _centered
        arr = cp.ones((10, 8), dtype=cp.float32)
        result = _centered(arr, newshape=(10, 4), axes=(1,))
        self.assertEqual(result.shape, (10, 4))

    def test_returns_cupy_array(self):
        from gpuphot.stats.subpixel_masked import _centered
        arr = cp.ones((8, 8), dtype=cp.float32)
        result = _centered(arr, newshape=(4, 4), axes=(0, 1))
        self.assertIsInstance(result, cp.ndarray)


if __name__ == '__main__':
    unittest.main()
