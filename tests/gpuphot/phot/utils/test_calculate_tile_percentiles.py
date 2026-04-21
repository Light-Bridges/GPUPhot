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
class TestCalculateTilePercentiles(unittest.TestCase):

    def test_output_length(self):
        from .....gpuphot.phot.utils import calculate_tile_percentiles
        tiles = cp.ones((4, 8, 8), dtype=cp.float32)
        result = calculate_tile_percentiles(tiles, qt=50)
        self.assertEqual(len(result), 4)

    def test_known_percentile(self):
        from .....gpuphot.phot.utils import calculate_tile_percentiles
        tiles = cp.ones((3, 4, 4), dtype=cp.float32) * 10.0
        result = calculate_tile_percentiles(tiles, qt=50)
        np.testing.assert_allclose(result.get(), 10.0, atol=1e-5)

    def test_nan_tiles_filtered(self):
        from .....gpuphot.phot.utils import calculate_tile_percentiles
        tiles = cp.ones((5, 4, 4), dtype=cp.float32)
        tiles[2, 0, 0] = cp.nan  # tile 2 has NaN
        result = calculate_tile_percentiles(tiles, qt=50)
        self.assertEqual(len(result), 4)  # one tile filtered out

    def test_returns_cupy_array(self):
        from .....gpuphot.phot.utils import calculate_tile_percentiles
        tiles = cp.ones((3, 4, 4), dtype=cp.float32)
        result = calculate_tile_percentiles(tiles, qt=70)
        self.assertIsInstance(result, cp.ndarray)


if __name__ == '__main__':
    unittest.main()
