import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestFlip(unittest.TestCase):

    def test_flip_all_axes(self):
        from gpuphot.stats.subpixel_masked import _flip
        arr = cp.arange(12, dtype=cp.float32).reshape(3, 4)
        result = _flip(arr, axes=None)
        expected = arr[::-1, ::-1]
        np.testing.assert_array_equal(result.get(), expected.get())

    def test_flip_single_axis(self):
        from gpuphot.stats.subpixel_masked import _flip
        arr = cp.arange(12, dtype=cp.float32).reshape(3, 4)
        result = _flip(arr, axes=(0,))
        expected = arr[::-1, :]
        np.testing.assert_array_equal(result.get(), expected.get())

    def test_double_flip_is_identity(self):
        from gpuphot.stats.subpixel_masked import _flip
        arr = cp.random.rand(5, 7).astype(cp.float32)
        result = _flip(_flip(arr, axes=None), axes=None)
        np.testing.assert_array_equal(result.get(), arr.get())

    def test_flip_preserves_shape(self):
        from gpuphot.stats.subpixel_masked import _flip
        arr = cp.ones((4, 6), dtype=cp.float32)
        result = _flip(arr, axes=(1,))
        self.assertEqual(result.shape, (4, 6))

    def test_returns_cupy_array(self):
        from gpuphot.stats.subpixel_masked import _flip
        arr = cp.ones((3, 3), dtype=cp.float32)
        result = _flip(arr)
        self.assertIsInstance(result, cp.ndarray)


if __name__ == '__main__':
    unittest.main()
