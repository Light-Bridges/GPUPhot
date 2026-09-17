import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestGenGaussFilter(unittest.TestCase):

    def setUp(self):
        from gpuphot.phot.photo_gpu import gen_gauss_filter
        self.gen_gauss_filter = gen_gauss_filter

    def test_returns_tuple_of_two(self):
        result = self.gen_gauss_filter(fw=4.0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_kernel_is_2d_cupy_array(self):
        k_app, lk = self.gen_gauss_filter(fw=4.0)
        self.assertIsInstance(k_app, cp.ndarray)
        self.assertEqual(k_app.ndim, 2)

    def test_kernel_is_square(self):
        k_app, lk = self.gen_gauss_filter(fw=4.0)
        self.assertEqual(k_app.shape[0], k_app.shape[1])

    def test_kernel_shape_is_odd(self):
        k_app, lk = self.gen_gauss_filter(fw=4.0)
        side = k_app.shape[0]
        self.assertEqual(side % 2, 1)

    def test_kernel_shape_consistent_with_lk(self):
        k_app, lk = self.gen_gauss_filter(fw=4.0)
        expected_side = int(2 * lk + 1)
        self.assertEqual(k_app.shape, (expected_side, expected_side))

    def test_kernel_sum_approximately_zero(self):
        # The kernel is mean-subtracted, so its sum should be ~0
        k_app, lk = self.gen_gauss_filter(fw=4.0)
        ksum = float(cp.sum(k_app).get())
        self.assertAlmostEqual(ksum, 0.0, places=4)

    def test_larger_fw_gives_larger_kernel(self):
        _, lk_small = self.gen_gauss_filter(fw=2.0)
        _, lk_large = self.gen_gauss_filter(fw=8.0)
        self.assertLess(int(lk_small), int(lk_large))

    def test_lk_is_positive(self):
        _, lk = self.gen_gauss_filter(fw=4.0)
        self.assertGreater(int(lk), 0)


if __name__ == '__main__':
    unittest.main()
