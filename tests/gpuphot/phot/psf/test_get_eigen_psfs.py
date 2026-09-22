import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestGetEigenPsfs(unittest.TestCase):

    def test_output_shape(self):
        from gpuphot.phot.psf import get_eigen_psfs
        n_stars, size, n_components = 20, 11, 3
        stars = cp.random.rand(n_stars, size, size).astype(cp.float32)
        result = get_eigen_psfs(stars, n_components=n_components)
        self.assertEqual(result.shape, (n_components, size, size))

    def test_fewer_components_than_stars(self):
        from gpuphot.phot.psf import get_eigen_psfs
        n_stars, size = 15, 9
        stars = cp.random.rand(n_stars, size, size).astype(cp.float32)
        result = get_eigen_psfs(stars, n_components=5)
        self.assertEqual(result.shape[0], 5)

    def test_single_component(self):
        from gpuphot.phot.psf import get_eigen_psfs
        stars = cp.random.rand(10, 7, 7).astype(cp.float32)
        result = get_eigen_psfs(stars, n_components=1)
        self.assertEqual(result.shape[0], 1)

    def test_returns_array(self):
        from gpuphot.phot.psf import get_eigen_psfs
        stars = cp.random.rand(10, 7, 7).astype(cp.float32)
        result = get_eigen_psfs(stars, n_components=2)
        self.assertTrue(hasattr(result, 'shape'))


if __name__ == '__main__':
    unittest.main()
