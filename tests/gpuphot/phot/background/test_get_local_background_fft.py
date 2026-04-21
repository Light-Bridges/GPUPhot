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
class TestGetLocalBackgroundFFT(unittest.TestCase):

    def setUp(self):
        from .....gpuphot.phot.background import get_local_background_fft
        self.get_local_background_fft = get_local_background_fft

    def _make_image(self, size=512, bg=1000.0, noise_std=10.0, seed=42):
        """Create a synthetic sky image with Gaussian noise."""
        rng = np.random.default_rng(seed)
        return cp.asarray(rng.normal(bg, noise_std, (size, size)).astype(np.float32))

    def _make_image_with_stars(self, size=512, bg=1000.0, noise_std=10.0,
                                n_stars=10, peak=5000.0, seed=42):
        """Create a synthetic sky image with stars (Gaussian profiles)."""
        rng = np.random.default_rng(seed)
        data = rng.normal(bg, noise_std, (size, size)).astype(np.float64)
        margin = 50
        for _ in range(n_stars):
            y = rng.integers(margin, size - margin)
            x = rng.integers(margin, size - margin)
            yy, xx = np.meshgrid(
                np.arange(max(0, y - 15), min(size, y + 16)),
                np.arange(max(0, x - 15), min(size, x + 16)),
                indexing='ij'
            )
            star = peak * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 3.0 ** 2))
            data[max(0, y - 15):min(size, y + 16),
                 max(0, x - 15):min(size, x + 16)] += star
        return cp.asarray(data.astype(np.float32))

    def test_returns_tuple_of_two(self):
        image = self._make_image()
        result = self.get_local_background_fft(image, pxscale=1.0, tile_section=128)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_background_is_cupy_array(self):
        image = self._make_image()
        bg, std = self.get_local_background_fft(image, pxscale=1.0, tile_section=128)
        self.assertIsInstance(bg, cp.ndarray)

    def test_std_is_none_when_get_std_false(self):
        image = self._make_image()
        bg, std = self.get_local_background_fft(
            image, pxscale=1.0, tile_section=128, get_std=False
        )
        self.assertIsNone(std)

    def test_std_is_array_when_get_std_true(self):
        image = self._make_image()
        bg, std = self.get_local_background_fft(
            image, pxscale=1.0, tile_section=128, get_std=True
        )
        self.assertIsInstance(std, cp.ndarray)

    def test_output_shape_matches_input(self):
        image = self._make_image(size=256)
        bg, std = self.get_local_background_fft(
            image, pxscale=1.0, tile_section=64
        )
        self.assertEqual(bg.shape, (256, 256))

    def test_uniform_image_background_near_input(self):
        image = self._make_image(size=512, bg=1000.0, noise_std=5.0)
        bg, _ = self.get_local_background_fft(
            image, pxscale=1.0, tile_section=128, get_std=False
        )
        bg_mean = float(cp.mean(bg))
        self.assertAlmostEqual(bg_mean, 1000.0, delta=50.0)

    def test_star_field_background_excludes_stars(self):
        """Background of star field should be near the sky level, not pulled up by stars."""
        image = self._make_image_with_stars(
            size=512, bg=500.0, noise_std=5.0, n_stars=10, peak=10000.0
        )
        bg, _ = self.get_local_background_fft(
            image, pxscale=1.0, tile_section=128, get_std=False
        )
        bg_mean = float(cp.mean(bg))
        # Background should be near 500, not near 500 + star contribution
        self.assertLess(bg_mean, 800.0)

    def test_no_nan_in_output(self):
        image = self._make_image(size=256)
        bg, _ = self.get_local_background_fft(
            image, pxscale=1.0, tile_section=64, get_std=False
        )
        self.assertEqual(int(cp.sum(cp.isnan(bg))), 0)


if __name__ == '__main__':
    unittest.main()
