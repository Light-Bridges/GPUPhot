import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestUpsampledDft(unittest.TestCase):

    def test_output_shape(self):
        from gpuphot.stats.subpixel import _upsampled_dft
        data = cp.random.rand(16, 16).astype(cp.complex64)
        result = _upsampled_dft(data, upsampled_region_size=3, upsample_factor=10)
        self.assertEqual(result.shape, (3, 3))

    def test_wrong_axis_offsets_raises(self):
        from gpuphot.stats.subpixel import _upsampled_dft
        data = cp.random.rand(16, 16).astype(cp.complex64)
        with self.assertRaises(ValueError):
            _upsampled_dft(data, upsampled_region_size=3, axis_offsets=[0, 0, 0])

    def test_returns_complex_array(self):
        from gpuphot.stats.subpixel import _upsampled_dft
        data = cp.random.rand(8, 8).astype(cp.complex64)
        result = _upsampled_dft(data, upsampled_region_size=2)
        self.assertTrue(cp.iscomplexobj(result))

    def test_upsample_factor_one(self):
        from gpuphot.stats.subpixel import _upsampled_dft
        data = cp.random.rand(8, 8).astype(cp.complex64)
        result = _upsampled_dft(data, upsampled_region_size=4, upsample_factor=1)
        self.assertEqual(result.shape, (4, 4))


if __name__ == '__main__':
    unittest.main()
