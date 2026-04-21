import unittest
import numpy as np


class TestFilterCentroidsKdtree(unittest.TestCase):

    def test_removes_close_centroids(self):
        from .....gpuphot.phot.psf import filter_centroids_kdtree
        centroids = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [20.0, 20.0],
        ])
        result = filter_centroids_kdtree(centroids, min_distance=5.0)
        self.assertLess(len(result), len(centroids))

    def test_keeps_distant_centroids(self):
        from .....gpuphot.phot.psf import filter_centroids_kdtree
        centroids = np.array([
            [0.0, 0.0],
            [100.0, 0.0],
            [0.0, 100.0],
            [100.0, 100.0],
        ])
        result = filter_centroids_kdtree(centroids, min_distance=5.0)
        self.assertEqual(len(result), len(centroids))

    def test_returns_ndarray(self):
        from .....gpuphot.phot.psf import filter_centroids_kdtree
        centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
        result = filter_centroids_kdtree(centroids, min_distance=1.0)
        self.assertIsInstance(result, np.ndarray)

    def test_all_close_filtered(self):
        from .....gpuphot.phot.psf import filter_centroids_kdtree
        centroids = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ])
        result = filter_centroids_kdtree(centroids, min_distance=100.0)
        self.assertEqual(len(result), 0)

    def test_preserves_columns(self):
        from .....gpuphot.phot.psf import filter_centroids_kdtree
        centroids = np.array([[0.0, 0.0], [50.0, 50.0], [100.0, 100.0]])
        result = filter_centroids_kdtree(centroids, min_distance=1.0)
        self.assertEqual(result.shape[1], 2)


if __name__ == '__main__':
    unittest.main()
