import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestRecreateNormedStar(unittest.TestCase):

    def test_output_shape(self):
        from gpuphot.phot.psf import recreate_normed_star
        n_comp, h, w, psf_size = 3, 100, 100, 11
        coeff_map = cp.random.rand(n_comp, h, w).astype(cp.float32)
        eigen_psfs = cp.random.rand(n_comp, psf_size, psf_size).astype(cp.float32)
        result = recreate_normed_star(coeff_map, eigen_psfs, (50, 50))
        self.assertEqual(result.shape, (psf_size, psf_size))

    def test_zero_coefficients_zero_star(self):
        from gpuphot.phot.psf import recreate_normed_star
        n_comp, h, w, psf_size = 3, 50, 50, 9
        coeff_map = cp.zeros((n_comp, h, w), dtype=cp.float32)
        eigen_psfs = cp.random.rand(n_comp, psf_size, psf_size).astype(cp.float32)
        result = recreate_normed_star(coeff_map, eigen_psfs, (25, 25))
        np.testing.assert_allclose(result.get(), 0.0, atol=1e-6)

    def test_single_component(self):
        from gpuphot.phot.psf import recreate_normed_star
        h, w, psf_size = 20, 20, 5
        coeff_map = cp.ones((1, h, w), dtype=cp.float32) * 2.0
        eigen_psfs = cp.ones((1, psf_size, psf_size), dtype=cp.float32) * 3.0
        result = recreate_normed_star(coeff_map, eigen_psfs, (10, 10))
        np.testing.assert_allclose(result.get(), 6.0, atol=1e-5)

    def test_returns_cupy_array(self):
        from gpuphot.phot.psf import recreate_normed_star
        coeff_map = cp.random.rand(2, 30, 30).astype(cp.float32)
        eigen_psfs = cp.random.rand(2, 7, 7).astype(cp.float32)
        result = recreate_normed_star(coeff_map, eigen_psfs, (15, 15))
        self.assertIsInstance(result, cp.ndarray)


if __name__ == '__main__':
    unittest.main()
