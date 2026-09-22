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
class TestCalculateTileNanmean(unittest.TestCase):

    def test_output_shape(self):
        from gpuphot.phot.utils import calculate_tile_nanmean
        tiles = cp.ones((6, 4, 4), dtype=cp.float32)
        result = calculate_tile_nanmean(tiles)
        self.assertEqual(result.shape, (6,))

    def test_known_mean(self):
        from gpuphot.phot.utils import calculate_tile_nanmean
        tiles = cp.ones((3, 4, 4), dtype=cp.float32) * 7.0
        result = calculate_tile_nanmean(tiles)
        np.testing.assert_allclose(result.get(), 7.0, atol=1e-6)

    def test_handles_nan(self):
        from gpuphot.phot.utils import calculate_tile_nanmean
        tiles = cp.ones((2, 4, 4), dtype=cp.float32) * 10.0
        tiles[0, 0, 0] = cp.nan
        result = calculate_tile_nanmean(tiles)
        # First tile mean should still be ~10.0 (nan ignored)
        self.assertAlmostEqual(float(result[0]), 10.0, places=3)

    def test_returns_cupy_array(self):
        from gpuphot.phot.utils import calculate_tile_nanmean
        tiles = cp.ones((3, 4, 4), dtype=cp.float32)
        result = calculate_tile_nanmean(tiles)
        self.assertIsInstance(result, cp.ndarray)


if __name__ == '__main__':
    unittest.main()
