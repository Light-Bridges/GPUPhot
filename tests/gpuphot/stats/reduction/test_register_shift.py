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
class TestRegisterShift(unittest.TestCase):

    def test_identical_frames_preserved(self):
        from gpuphot.stats.reduction import register_shift
        rng = np.random.default_rng(42)
        frame = rng.normal(1000, 50, (256, 256)).astype(np.float32)
        stack = cp.asarray(np.stack([frame, frame, frame]))
        result = register_shift(stack, uf=10, n=200)
        self.assertEqual(result.shape, stack.shape)

    def test_output_shape_matches_input(self):
        from gpuphot.stats.reduction import register_shift
        rng = np.random.default_rng(42)
        data = rng.normal(500, 30, (3, 128, 128)).astype(np.float32)
        stack = cp.asarray(data)
        result = register_shift(stack, uf=10, n=100)
        self.assertEqual(result.shape, (3, 128, 128))


if __name__ == '__main__':
    unittest.main()
