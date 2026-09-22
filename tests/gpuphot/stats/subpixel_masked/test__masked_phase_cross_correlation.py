import unittest

import pytest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np

from ...conftest import NVRTC_WORKS


@unittest.skipUnless(HAS_CUPY, "CuPy required")
@pytest.mark.skipif(not NVRTC_WORKS, reason="CuPy NVRTC compilation broken")
class TestMaskedPhaseCrossCorrelation(unittest.TestCase):

    def test_identical_images_near_zero_shift(self):
        from gpuphot.stats.subpixel_masked import _masked_phase_cross_correlation
        image = cp.random.rand(32, 32).astype(cp.float32)
        mask = cp.ones((32, 32), dtype=bool)
        shifts = _masked_phase_cross_correlation(image, image, mask)
        np.testing.assert_allclose(shifts.get(), 0.0, atol=0.5)

    def test_raises_on_shape_mismatch_without_mask(self):
        from gpuphot.stats.subpixel_masked import _masked_phase_cross_correlation
        img1 = cp.ones((16, 16), dtype=cp.float32)
        img2 = cp.ones((32, 32), dtype=cp.float32)
        mask = cp.ones((16, 16), dtype=bool)
        with self.assertRaises(ValueError):
            _masked_phase_cross_correlation(img1, img2, mask)

    def test_raises_on_image_mask_shape_mismatch(self):
        from gpuphot.stats.subpixel_masked import _masked_phase_cross_correlation
        img = cp.ones((16, 16), dtype=cp.float32)
        mask = cp.ones((8, 8), dtype=bool)
        with self.assertRaises(ValueError):
            _masked_phase_cross_correlation(img, img, mask, mask)

    def test_returns_shift_vector(self):
        from gpuphot.stats.subpixel_masked import _masked_phase_cross_correlation
        image = cp.random.rand(32, 32).astype(cp.float32)
        mask = cp.ones((32, 32), dtype=bool)
        shifts = _masked_phase_cross_correlation(image, image, mask)
        self.assertEqual(shifts.shape, (2,))


if __name__ == '__main__':
    unittest.main()
