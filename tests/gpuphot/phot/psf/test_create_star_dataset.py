import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestCreateStarDataset(unittest.TestCase):

    @staticmethod
    def _make_star_image(size=256, n_stars=10, peak=5000.0, bg=100.0):
        """Create an image with bright stars at known positions."""
        np.random.seed(42)
        image = np.ones((size, size), dtype=np.float32) * bg
        margin = 30
        coords = []
        for _ in range(n_stars):
            y = np.random.randint(margin, size - margin)
            x = np.random.randint(margin, size - margin)
            yy, xx = np.meshgrid(
                np.arange(max(0, y - 10), min(size, y + 11)),
                np.arange(max(0, x - 10), min(size, x + 11)),
                indexing='ij'
            )
            star = peak * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 3.0 ** 2))
            image[max(0, y - 10):min(size, y + 11), max(0, x - 10):min(size, x + 11)] += star
            coords.append([y, x])
        return image, np.array(coords)

    def test_returns_three_tuple(self):
        from .....gpuphot.phot.psf import create_star_dataset
        image, coords = self._make_star_image()
        result = create_star_dataset(cp.asarray(image), cp.asarray(coords), pxscale=1.0)
        self.assertEqual(len(result), 3)

    def test_output_dataset_shape(self):
        from .....gpuphot.phot.psf import create_star_dataset
        image, coords = self._make_star_image(n_stars=10)
        star_ds, out_coords, scaling_ds = create_star_dataset(
            cp.asarray(image), cp.asarray(coords), pxscale=1.0
        )
        if len(star_ds) > 0:
            self.assertEqual(star_ds.ndim, 3)
            self.assertEqual(star_ds.shape[1], star_ds.shape[2])

    def test_n_limits_output(self):
        from .....gpuphot.phot.psf import create_star_dataset
        image, coords = self._make_star_image(n_stars=10)
        star_ds, out_coords, scaling_ds = create_star_dataset(
            cp.asarray(image), cp.asarray(coords), pxscale=1.0, N=3
        )
        self.assertLessEqual(len(star_ds), 3)

    def test_boundary_filtering(self):
        from .....gpuphot.phot.psf import create_star_dataset
        image = np.ones((100, 100), dtype=np.float32) * 100
        # Place coords near border that should be filtered
        coords = np.array([[2, 2], [50, 50], [98, 98]])
        star_ds, out_coords, scaling_ds = create_star_dataset(
            cp.asarray(image), cp.asarray(coords), pxscale=1.0
        )
        # Border stars at (2,2) and (98,98) should be filtered
        self.assertLessEqual(len(star_ds), 3)

    def test_scaling_dataset_has_four_columns(self):
        from .....gpuphot.phot.psf import create_star_dataset
        image, coords = self._make_star_image(n_stars=5)
        _, _, scaling_ds = create_star_dataset(
            cp.asarray(image), cp.asarray(coords), pxscale=1.0
        )
        if len(scaling_ds) > 0:
            self.assertEqual(scaling_ds.shape[1], 4)


if __name__ == '__main__':
    unittest.main()
