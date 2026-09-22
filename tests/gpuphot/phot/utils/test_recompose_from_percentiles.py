import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestRecomposeFromPercentiles(unittest.TestCase):

    def test_output_shape(self):
        from gpuphot.phot.utils import recompose_from_percentiles
        original_shape = (64, 64)
        block_size = 16
        n_tiles = (64 // 16) * (64 // 16)  # 16
        percentiles = cp.ones(n_tiles, dtype=cp.float32) * 5.0
        result = recompose_from_percentiles(percentiles, original_shape, block_size)
        self.assertEqual(result.shape, original_shape)

    def test_tile_regions_have_correct_values(self):
        from gpuphot.phot.utils import recompose_from_percentiles
        original_shape = (32, 32)
        block_size = 16
        # 4 tiles: values 1, 2, 3, 4
        percentiles = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
        result = recompose_from_percentiles(percentiles, original_shape, block_size)
        # Top-left tile should be 1.0
        np.testing.assert_allclose(result[:16, :16].get(), 1.0)
        # Top-right tile should be 2.0
        np.testing.assert_allclose(result[:16, 16:32].get(), 2.0)

    def test_returns_cupy_array(self):
        from gpuphot.phot.utils import recompose_from_percentiles
        percentiles = cp.ones(4, dtype=cp.float32)
        result = recompose_from_percentiles(percentiles, (32, 32), 16)
        self.assertIsInstance(result, cp.ndarray)


if __name__ == '__main__':
    unittest.main()
