import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestGaussianKernel(unittest.TestCase):

    def test_output_shape(self):
        from .....gpuphot.phot.conv import gaussian_kernel
        lk, sigma = 5, 2
        result = gaussian_kernel(lk, sigma)
        expected_size = 2 * lk + 1
        self.assertEqual(result.shape, (expected_size, expected_size))

    def test_sums_to_one(self):
        from .....gpuphot.phot.conv import gaussian_kernel
        result = gaussian_kernel(5, 2)
        np.testing.assert_allclose(cp.sum(result).get(), 1.0, atol=1e-12)

    def test_symmetric(self):
        from .....gpuphot.phot.conv import gaussian_kernel
        result = gaussian_kernel(5, 2).get()
        np.testing.assert_allclose(result, result[::-1, :], atol=1e-12)
        np.testing.assert_allclose(result, result[:, ::-1], atol=1e-12)

    def test_peak_at_center(self):
        from .....gpuphot.phot.conv import gaussian_kernel
        lk = 5
        result = gaussian_kernel(lk, 2)
        peak_idx = cp.unravel_index(cp.argmax(result), result.shape)
        self.assertEqual(int(peak_idx[0]), lk)
        self.assertEqual(int(peak_idx[1]), lk)

    def test_larger_sigma_wider(self):
        from .....gpuphot.phot.conv import gaussian_kernel
        narrow = gaussian_kernel(10, 1)
        wide = gaussian_kernel(10, 5)
        # Wider kernel has smaller peak value (more spread)
        self.assertGreater(float(narrow[10, 10]), float(wide[10, 10]))

    def test_returns_cupy_array(self):
        from .....gpuphot.phot.conv import gaussian_kernel
        result = gaussian_kernel(3, 1)
        self.assertIsInstance(result, cp.ndarray)


if __name__ == '__main__':
    unittest.main()
