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
class TestCreateCoeffMap(unittest.TestCase):

    def test_output_shape(self):
        from .....gpuphot.phot.psf import create_coeff_map
        img_shape = (256, 256)
        n_stars, n_coeffs = 5, 3
        positions = cp.array([[64, 64], [128, 128], [192, 192],
                               [64, 192], [192, 64]], dtype=cp.float32)
        coefficients = cp.random.rand(n_coeffs, n_stars).astype(cp.float32)
        result = create_coeff_map(img_shape, positions, coefficients,
                                  pxscale=1.0, tile_section=32)
        self.assertEqual(result.shape[0], n_coeffs)
        self.assertEqual(result.shape[1], img_shape[0])
        self.assertEqual(result.shape[2], img_shape[1])

    def test_returns_cupy_array(self):
        from .....gpuphot.phot.psf import create_coeff_map
        img_shape = (128, 128)
        positions = cp.array([[32, 32], [64, 64], [96, 96]], dtype=cp.float32)
        coefficients = cp.random.rand(2, 3).astype(cp.float32)
        result = create_coeff_map(img_shape, positions, coefficients,
                                  pxscale=1.0, tile_section=32)
        self.assertIsInstance(result, cp.ndarray)

    def test_no_nan_in_output(self):
        from .....gpuphot.phot.psf import create_coeff_map
        img_shape = (128, 128)
        positions = cp.array([[32, 32], [64, 64], [96, 96]], dtype=cp.float32)
        coefficients = cp.random.rand(2, 3).astype(cp.float32)
        result = create_coeff_map(img_shape, positions, coefficients,
                                  pxscale=1.0, tile_section=32)
        self.assertFalse(bool(cp.any(cp.isnan(result))))


if __name__ == '__main__':
    unittest.main()
