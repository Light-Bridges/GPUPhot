import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestDecomposeIntoTiles(unittest.TestCase):

    def test_output_shape(self):
        from .....gpuphot.phot.utils import decompose_into_tiles
        image = cp.ones((64, 64), dtype=cp.float32)
        tiles = decompose_into_tiles(image, block_size=16)
        # 64/16 = 4 tiles per side, 4*4 = 16 tiles
        self.assertEqual(tiles.shape, (16, 16, 16))

    def test_number_of_tiles(self):
        from .....gpuphot.phot.utils import decompose_into_tiles
        image = cp.ones((100, 80), dtype=cp.float32)
        tiles = decompose_into_tiles(image, block_size=20)
        # 100/20=5, 80/20=4 -> 20 tiles
        self.assertEqual(tiles.shape[0], 20)

    def test_tile_content_matches(self):
        from .....gpuphot.phot.utils import decompose_into_tiles
        image = cp.arange(64 * 64, dtype=cp.float32).reshape(64, 64)
        tiles = decompose_into_tiles(image, block_size=32)
        # First tile should be top-left 32x32 block
        np.testing.assert_array_equal(tiles[0].get(), image[:32, :32].get())

    def test_returns_cupy_array(self):
        from .....gpuphot.phot.utils import decompose_into_tiles
        image = cp.ones((32, 32), dtype=cp.float32)
        tiles = decompose_into_tiles(image, block_size=16)
        self.assertIsInstance(tiles, cp.ndarray)

    def test_ignores_remainder(self):
        from .....gpuphot.phot.utils import decompose_into_tiles
        # 50 / 16 = 3 (remainder 2 ignored)
        image = cp.ones((50, 50), dtype=cp.float32)
        tiles = decompose_into_tiles(image, block_size=16)
        self.assertEqual(tiles.shape[0], 9)  # 3*3


if __name__ == '__main__':
    unittest.main()
