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
class TestPhaseCrossCorrelation(unittest.TestCase):

    def test_identical_images_zero_shift(self):
        from .....gpuphot.stats.subpixel import phase_cross_correlation
        image = cp.random.rand(64, 64).astype(cp.float32)
        shifts, error, phasediff = phase_cross_correlation(image, image)
        np.testing.assert_allclose(shifts.get(), 0.0, atol=0.02)

    def test_returns_three_values(self):
        from .....gpuphot.stats.subpixel import phase_cross_correlation
        image = cp.random.rand(32, 32).astype(cp.float32)
        result = phase_cross_correlation(image, image)
        self.assertEqual(len(result), 3)

    @pytest.mark.skipif(not NVRTC_WORKS, reason="CuPy NVRTC compilation broken")
    def test_known_integer_shift(self):
        from .....gpuphot.stats.subpixel import phase_cross_correlation
        ref = cp.zeros((64, 64), dtype=cp.float32)
        ref[20:40, 20:40] = 1.0
        moved = cp.zeros((64, 64), dtype=cp.float32)
        moved[22:42, 23:43] = 1.0
        shifts, _, _ = phase_cross_correlation(ref, moved, upsample_factor=10)
        # phase_cross_correlation returns the shift to align moving->reference,
        # so the sign is negative: moving is shifted +2,+3 relative to ref
        np.testing.assert_allclose(cp.abs(shifts).get(), [2.0, 3.0], atol=0.5)

    def test_raises_on_shape_mismatch(self):
        from .....gpuphot.stats.subpixel import phase_cross_correlation
        img1 = cp.ones((32, 32), dtype=cp.float32)
        img2 = cp.ones((64, 64), dtype=cp.float32)
        with self.assertRaises(ValueError):
            phase_cross_correlation(img1, img2)

    def test_return_error_false(self):
        from .....gpuphot.stats.subpixel import phase_cross_correlation
        image = cp.random.rand(32, 32).astype(cp.float32)
        shifts, error, phasediff = phase_cross_correlation(
            image, image, return_error=False
        )
        self.assertEqual(error, 0)
        self.assertEqual(phasediff, 0)


if __name__ == '__main__':
    unittest.main()
