import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestProjectAllStarsOntoEigenpsfs(unittest.TestCase):

    def test_output_shape(self):
        from gpuphot.phot.psf import project_all_stars_onto_eigenpsfs
        n_stars, n_components, size = 10, 3, 11
        stars = cp.random.rand(n_stars, size, size).astype(cp.float32)
        eigen_psfs = cp.random.rand(n_components, size, size).astype(cp.float32)
        result = project_all_stars_onto_eigenpsfs(stars, eigen_psfs)
        self.assertEqual(result.shape, (n_stars, n_components))

    def test_zero_stars_zero_coefficients(self):
        from gpuphot.phot.psf import project_all_stars_onto_eigenpsfs
        n_stars, n_components, size = 5, 3, 11
        stars = cp.zeros((n_stars, size, size), dtype=cp.float32)
        eigen_psfs = cp.random.rand(n_components, size, size).astype(cp.float32)
        result = project_all_stars_onto_eigenpsfs(stars, eigen_psfs)
        np.testing.assert_allclose(result.get(), 0.0, atol=1e-6)

    def test_returns_cupy_array(self):
        from gpuphot.phot.psf import project_all_stars_onto_eigenpsfs
        stars = cp.random.rand(5, 7, 7).astype(cp.float32)
        eigen_psfs = cp.random.rand(2, 7, 7).astype(cp.float32)
        result = project_all_stars_onto_eigenpsfs(stars, eigen_psfs)
        self.assertIsInstance(result, cp.ndarray)

    def test_single_star(self):
        from gpuphot.phot.psf import project_all_stars_onto_eigenpsfs
        stars = cp.random.rand(1, 9, 9).astype(cp.float32)
        eigen_psfs = cp.random.rand(3, 9, 9).astype(cp.float32)
        result = project_all_stars_onto_eigenpsfs(stars, eigen_psfs)
        self.assertEqual(result.shape, (1, 3))


if __name__ == '__main__':
    unittest.main()
