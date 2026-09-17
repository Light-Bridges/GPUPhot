import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestConvolveFft(unittest.TestCase):

    def test_identity_kernel(self):
        """Convolution with a delta kernel returns the original image."""
        from gpuphot.phot.conv import convolve_fft
        image = cp.random.rand(64, 64).astype(cp.float64)
        kernel = cp.zeros((5, 5), dtype=cp.float64)
        kernel[2, 2] = 1.0
        result = convolve_fft(image, kernel)
        np.testing.assert_allclose(result.get(), image.get(), atol=1e-10)

    def test_output_shape_matches_input(self):
        """Output shape equals input shape when do_pad=True."""
        from gpuphot.phot.conv import convolve_fft
        image = cp.random.rand(100, 80).astype(cp.float64)
        kernel = cp.ones((7, 7), dtype=cp.float64) / 49.0
        result = convolve_fft(image, kernel)
        self.assertEqual(result.shape, image.shape)

    def test_uniform_kernel_gives_mean(self):
        """Convolution with uniform kernel approximates local mean."""
        from gpuphot.phot.conv import convolve_fft
        image = cp.ones((64, 64), dtype=cp.float64) * 5.0
        kernel = cp.ones((3, 3), dtype=cp.float64) / 9.0
        result = convolve_fft(image, kernel)
        np.testing.assert_allclose(result.get(), 5.0, atol=1e-10)

    def test_returns_cupy_array(self):
        from gpuphot.phot.conv import convolve_fft
        image = cp.random.rand(32, 32).astype(cp.float64)
        kernel = cp.ones((3, 3), dtype=cp.float64) / 9.0
        result = convolve_fft(image, kernel)
        self.assertIsInstance(result, cp.ndarray)

    def test_no_pad(self):
        """do_pad=False still produces output (different shape possible)."""
        from gpuphot.phot.conv import convolve_fft
        image = cp.random.rand(64, 64).astype(cp.float64)
        kernel = cp.ones((5, 5), dtype=cp.float64) / 25.0
        result = convolve_fft(image, kernel, do_pad=False)
        self.assertIsInstance(result, cp.ndarray)
        self.assertEqual(result.shape, image.shape)


if __name__ == '__main__':
    unittest.main()
