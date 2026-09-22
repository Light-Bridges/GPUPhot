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
class TestDetectIsolatedStars(unittest.TestCase):

    @staticmethod
    def _make_star_field(size=512, n_stars=20, peak=5000.0, bg=100.0, fwhm=5.0):
        """Create synthetic image with isolated Gaussian stars."""
        np.random.seed(42)
        image = np.ones((size, size), dtype=np.float64) * bg
        sigma = fwhm / 2.355
        margin = 50
        coords = []
        for _ in range(n_stars):
            y = np.random.randint(margin, size - margin)
            x = np.random.randint(margin, size - margin)
            yy, xx = np.meshgrid(
                np.arange(max(0, y - 20), min(size, y + 21)),
                np.arange(max(0, x - 20), min(size, x + 21)),
                indexing='ij'
            )
            star = peak * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))
            image[max(0, y - 20):min(size, y + 21), max(0, x - 20):min(size, x + 21)] += star
            coords.append((y, x))
        return image, coords

    def test_returns_cupy_array(self):
        from gpuphot.phot.psf import detect_isolated_stars
        image, _ = self._make_star_field()
        img = cp.asarray(image)
        rms = cp.ones_like(img) * 10.0
        result = detect_isolated_stars(img, rms, pxscale=1.0, min_snr=3, dist_asec=30)
        self.assertIsInstance(result, cp.ndarray)

    def test_detects_some_stars(self):
        from gpuphot.phot.psf import detect_isolated_stars
        image, _ = self._make_star_field(n_stars=30, peak=10000.0)
        img = cp.asarray(image)
        rms = cp.ones_like(img) * 5.0
        result = detect_isolated_stars(img, rms, pxscale=1.0, min_snr=3, dist_asec=20)
        self.assertGreater(len(result), 0)

    def test_coords_within_bounds(self):
        from gpuphot.phot.psf import detect_isolated_stars
        size = 512
        image, _ = self._make_star_field(size=size, n_stars=30, peak=10000.0)
        img = cp.asarray(image)
        rms = cp.ones_like(img) * 5.0
        result = detect_isolated_stars(img, rms, pxscale=1.0, min_snr=3, dist_asec=20)
        coords = result.get()
        self.assertTrue(np.all(coords[:, 0] >= 0))
        self.assertTrue(np.all(coords[:, 0] < size))
        self.assertTrue(np.all(coords[:, 1] >= 0))
        self.assertTrue(np.all(coords[:, 1] < size))

    def test_raises_on_empty_image(self):
        from gpuphot.phot.psf import detect_isolated_stars
        from gpuphot.exceptions import InsufficientStarsError
        img = cp.zeros((256, 256), dtype=cp.float64)
        rms = cp.ones((256, 256), dtype=cp.float64)
        with self.assertRaises((InsufficientStarsError, Exception)):
            detect_isolated_stars(img, rms, pxscale=1.0, min_snr=100)


if __name__ == '__main__':
    unittest.main()
