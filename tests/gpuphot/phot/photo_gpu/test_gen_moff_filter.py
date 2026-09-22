import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestGenMoffFilter(unittest.TestCase):

    def setUp(self):
        from gpuphot.phot.photo_gpu import gen_moff_filter
        self.gen_moff_filter = gen_moff_filter

    def test_returns_tuple_of_two(self):
        result = self.gen_moff_filter(alpha=3.0, beta=4.765)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_kernel_is_2d_cupy_array(self):
        k_app, lk = self.gen_moff_filter(alpha=3.0, beta=4.765)
        self.assertIsInstance(k_app, cp.ndarray)
        self.assertEqual(k_app.ndim, 2)

    def test_kernel_shape_is_odd(self):
        k_app, lk = self.gen_moff_filter(alpha=3.0, beta=4.765)
        h, w = k_app.shape
        self.assertEqual(h % 2, 1)
        self.assertEqual(w % 2, 1)

    def test_kernel_shape_consistent_with_lk(self):
        k_app, lk = self.gen_moff_filter(alpha=3.0, beta=4.765)
        expected_side = int(2 * lk + 1)
        self.assertEqual(k_app.shape, (expected_side, expected_side))

    def test_kernel_sum_approximately_zero(self):
        # The kernel is mean-subtracted, so its sum should be ~0
        k_app, lk = self.gen_moff_filter(alpha=3.0, beta=4.765)
        ksum = float(cp.sum(k_app).get())
        self.assertAlmostEqual(ksum, 0.0, places=4)

    def test_different_alpha_gives_different_kernel_size(self):
        _, lk_small = self.gen_moff_filter(alpha=2.0, beta=4.765)
        _, lk_large = self.gen_moff_filter(alpha=6.0, beta=4.765)
        self.assertLess(int(lk_small), int(lk_large))

    def test_different_beta_gives_different_kernel(self):
        k1, _ = self.gen_moff_filter(alpha=3.0, beta=4.0)
        k2, _ = self.gen_moff_filter(alpha=3.0, beta=7.0)
        if k1.shape == k2.shape:
            self.assertFalse(bool(cp.allclose(k1, k2).get()))
        else:
            self.assertNotEqual(k1.shape, k2.shape)

    def test_lk_is_positive_integer(self):
        _, lk = self.gen_moff_filter(alpha=3.0, beta=4.765)
        self.assertGreater(int(lk), 0)


if __name__ == '__main__':
    unittest.main()
