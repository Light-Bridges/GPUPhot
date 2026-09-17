import unittest
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class TestCenter(unittest.TestCase):

    def test_crops_to_smaller_size(self):
        from gpuphot.stats.reduction import center
        im = np.ones((100, 100), dtype=np.float32)
        result = center(im, 50)
        self.assertEqual(result.shape, (50, 50))

    def test_larger_size_returns_original(self):
        from gpuphot.stats.reduction import center
        im = np.ones((30, 30), dtype=np.float32)
        result = center(im, 100)
        self.assertEqual(result.shape, (30, 30))

    def test_center_pixel_preserved(self):
        from gpuphot.stats.reduction import center
        im = np.zeros((100, 100), dtype=np.float32)
        im[50, 50] = 42.0
        result = center(im, 20)
        # Center pixel (50,50) -> offset (50-10=40, 40+20=60) -> result[10,10]
        self.assertAlmostEqual(float(result[10, 10]), 42.0)

    @unittest.skipUnless(HAS_CUPY, "CuPy required")
    def test_works_with_cupy(self):
        from gpuphot.stats.reduction import center
        im = cp.ones((100, 100), dtype=cp.float32)
        result = center(im, 50)
        self.assertEqual(result.shape, (50, 50))
        self.assertIsInstance(result, cp.ndarray)

    def test_returns_numpy_for_numpy_input(self):
        from gpuphot.stats.reduction import center
        im = np.ones((80, 80), dtype=np.float32)
        result = center(im, 40)
        self.assertIsInstance(result, np.ndarray)


if __name__ == '__main__':
    unittest.main()
