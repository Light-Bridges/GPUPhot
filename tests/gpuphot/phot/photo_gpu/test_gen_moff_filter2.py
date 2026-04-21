import unittest
import warnings

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestGenMoffFilter2(unittest.TestCase):

    def setUp(self):
        from .....gpuphot.phot.photo_gpu import gen_moff_filter2
        self.gen_moff_filter2 = gen_moff_filter2

    def test_returns_tuple_of_two(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = self.gen_moff_filter2(alpha=3.0, beta=4.765)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_emits_deprecation_warning(self):
        with self.assertWarns(DeprecationWarning):
            self.gen_moff_filter2(alpha=3.0, beta=4.765)

    def test_kernel_is_2d_cupy_array(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            k_app, lk = self.gen_moff_filter2(alpha=3.0, beta=4.765)
        self.assertIsInstance(k_app, cp.ndarray)
        self.assertEqual(k_app.ndim, 2)

    def test_kernel_shape_is_odd(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            k_app, lk = self.gen_moff_filter2(alpha=3.0, beta=4.765)
        h, w = k_app.shape
        self.assertEqual(h % 2, 1)
        self.assertEqual(w % 2, 1)

    def test_halved_alpha_produces_smaller_kernel_than_gen_moff_filter(self):
        # gen_moff_filter2 halves alpha internally, so for the same input alpha
        # its kernel should be smaller than gen_moff_filter with the same alpha
        from .....gpuphot.phot.photo_gpu import gen_moff_filter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            _, lk2 = self.gen_moff_filter2(alpha=6.0, beta=4.765)
        _, lk1 = gen_moff_filter(alpha=6.0, beta=4.765)
        self.assertLess(int(lk2), int(lk1))

    def test_lk_is_positive(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            _, lk = self.gen_moff_filter2(alpha=4.0, beta=4.765)
        self.assertGreater(int(lk), 0)


if __name__ == '__main__':
    unittest.main()
