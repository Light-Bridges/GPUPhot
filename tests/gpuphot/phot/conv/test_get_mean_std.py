import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestGetMeanStd(unittest.TestCase):

    def test_returns_tuple_of_two(self):
        from .....gpuphot.phot.conv import get_mean_std
        image = cp.random.rand(64, 64).astype(cp.float64)
        result = get_mean_std(image, lk=3)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_uniform_image_low_std(self):
        from .....gpuphot.phot.conv import get_mean_std
        image = cp.ones((64, 64), dtype=cp.float64) * 10.0
        mean_img, std_img = get_mean_std(image, lk=3)
        # Uniform image: std should be near zero
        np.testing.assert_allclose(std_img.get(), 0.0, atol=1e-8)

    def test_uniform_image_mean_value(self):
        from .....gpuphot.phot.conv import get_mean_std
        image = cp.ones((64, 64), dtype=cp.float64) * 7.0
        mean_img, _ = get_mean_std(image, lk=3)
        np.testing.assert_allclose(mean_img.get(), 7.0, atol=0.5)

    def test_std_false_returns_none(self):
        from .....gpuphot.phot.conv import get_mean_std
        image = cp.random.rand(64, 64).astype(cp.float64)
        mean_img, std_img = get_mean_std(image, lk=3, std=False)
        self.assertIsNone(std_img)

    def test_output_shapes_match_input(self):
        from .....gpuphot.phot.conv import get_mean_std
        image = cp.random.rand(50, 70).astype(cp.float64)
        mean_img, std_img = get_mean_std(image, lk=3)
        self.assertEqual(mean_img.shape, image.shape)
        self.assertEqual(std_img.shape, image.shape)


if __name__ == '__main__':
    unittest.main()
