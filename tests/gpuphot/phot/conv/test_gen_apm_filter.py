import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestGenApmFilter(unittest.TestCase):

    def test_output_shape(self):
        from gpuphot.phot.conv import gen_apm_filter
        lk = 5
        result = gen_apm_filter(lk)
        expected = 2 * lk + 1
        self.assertEqual(result.shape, (expected, expected))

    def test_normalized_sums_to_one(self):
        from gpuphot.phot.conv import gen_apm_filter
        result = gen_apm_filter(5, norm=True)
        np.testing.assert_allclose(float(cp.sum(result)), 1.0, atol=1e-12)

    def test_unnormalized_sums_to_area(self):
        from gpuphot.phot.conv import gen_apm_filter
        result = gen_apm_filter(5, norm=False)
        total = float(cp.sum(result))
        self.assertGreater(total, 1.0)

    def test_inner_radius_creates_ring(self):
        from gpuphot.phot.conv import gen_apm_filter
        full = gen_apm_filter(5, li=0, norm=False)
        ring = gen_apm_filter(5, li=3, norm=False)
        self.assertGreater(float(cp.sum(full)), float(cp.sum(ring)))

    def test_center_pixel_with_no_inner(self):
        from gpuphot.phot.conv import gen_apm_filter
        result = gen_apm_filter(3, li=0, norm=False)
        # Center pixel at (3, 3) should be 1.0
        self.assertEqual(float(result[3, 3]), 1.0)

    def test_center_pixel_with_inner_radius(self):
        from gpuphot.phot.conv import gen_apm_filter
        result = gen_apm_filter(5, li=3, norm=False)
        # Center pixel at (5, 5) should be 0 (inside inner radius)
        self.assertEqual(float(result[5, 5]), 0.0)

    def test_returns_cupy_array(self):
        from gpuphot.phot.conv import gen_apm_filter
        result = gen_apm_filter(3)
        self.assertIsInstance(result, cp.ndarray)


if __name__ == '__main__':
    unittest.main()
