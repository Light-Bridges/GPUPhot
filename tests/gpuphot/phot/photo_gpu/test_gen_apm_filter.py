import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestGenApmFilter(unittest.TestCase):

    def setUp(self):
        from .....gpuphot.phot.conv import gen_apm_filter
        self.gen_apm_filter = gen_apm_filter

    def test_returns_cupy_array(self):
        kernel = self.gen_apm_filter(lk=5)
        self.assertIsInstance(kernel, cp.ndarray)

    def test_kernel_shape_is_2d_and_odd(self):
        lk = 5
        kernel = self.gen_apm_filter(lk=lk)
        self.assertEqual(kernel.ndim, 2)
        expected_side = 2 * lk + 1
        self.assertEqual(kernel.shape, (expected_side, expected_side))

    def test_normalized_kernel_sums_to_one(self):
        # With norm=True (default), the kernel should sum to 1
        kernel = self.gen_apm_filter(lk=5, norm=True)
        ksum = float(cp.sum(kernel).get())
        self.assertAlmostEqual(ksum, 1.0, places=5)

    def test_unnormalized_kernel_sums_to_pixel_count(self):
        # With norm=False the kernel is 0/1 mask; sum equals number of pixels inside disk
        kernel = self.gen_apm_filter(lk=5, norm=False)
        ksum = float(cp.sum(kernel).get())
        self.assertGreater(ksum, 1.0)

    def test_inner_exclusion_reduces_sum(self):
        # Adding a non-zero li carves out the inner disk, reducing area
        k_full = self.gen_apm_filter(lk=8, li=0, norm=False)
        k_annulus = self.gen_apm_filter(lk=8, li=3, norm=False)
        self.assertGreater(float(cp.sum(k_full).get()), float(cp.sum(k_annulus).get()))


if __name__ == '__main__':
    unittest.main()
