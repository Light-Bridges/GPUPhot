import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestGetAperKernel(unittest.TestCase):

    def test_even_size_raises(self):
        from gpuphot.phot.conv import get_aper_kernel
        with self.assertRaises(ValueError):
            get_aper_kernel(radius=3.0, size=6)

    def test_output_shape(self):
        from gpuphot.phot.conv import get_aper_kernel
        kernel, area = get_aper_kernel(radius=3.0, size=7)
        self.assertEqual(kernel.shape, (7, 7))

    def test_kernel_is_binary(self):
        from gpuphot.phot.conv import get_aper_kernel
        kernel, area = get_aper_kernel(radius=3.0, size=7)
        unique_vals = cp.unique(kernel).get()
        self.assertTrue(set(unique_vals).issubset({0.0, 1.0}))

    def test_area_equals_sum(self):
        from gpuphot.phot.conv import get_aper_kernel
        kernel, area = get_aper_kernel(radius=3.0, size=7)
        np.testing.assert_allclose(float(area), float(cp.sum(kernel)))

    def test_center_pixel_is_one(self):
        from gpuphot.phot.conv import get_aper_kernel
        kernel, _ = get_aper_kernel(radius=2.0, size=5)
        self.assertEqual(float(kernel[2, 2]), 1.0)

    def test_returns_tuple(self):
        from gpuphot.phot.conv import get_aper_kernel
        result = get_aper_kernel(radius=2.0, size=5)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_small_radius(self):
        from gpuphot.phot.conv import get_aper_kernel
        kernel, area = get_aper_kernel(radius=0.5, size=3)
        # Only center pixel should be within radius 0.5
        self.assertEqual(float(area), 1.0)


if __name__ == '__main__':
    unittest.main()
