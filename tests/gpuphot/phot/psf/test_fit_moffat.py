import unittest
import numpy as np

import pytest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from ...conftest import NVRTC_WORKS


@unittest.skipUnless(HAS_CUPY, "CuPy required")
@pytest.mark.skipif(not NVRTC_WORKS, reason="CuPy NVRTC compilation broken")
class TestFitMoffat(unittest.TestCase):

    @staticmethod
    def _make_synthetic_star(size=31, A=1000.0, R=3.0, B=2.5):
        """Create a synthetic Moffat star image."""
        center = size // 2
        y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        star = A * (1 + (r / R) ** 2) ** (-B) + 100.0
        return star

    def test_returns_five_tuple(self):
        from gpuphot.phot.psf import fit_moffat
        star = self._make_synthetic_star()
        result = fit_moffat(cp.asarray(star))
        self.assertEqual(len(result), 5)

    def test_fwhm_positive(self):
        from gpuphot.phot.psf import fit_moffat
        star = self._make_synthetic_star()
        _, _, _, fwhm, _ = fit_moffat(cp.asarray(star))
        self.assertGreater(float(fwhm), 0.0)

    def test_works_with_numpy_input(self):
        from gpuphot.phot.psf import fit_moffat
        star = self._make_synthetic_star()
        r, Z, result, fwhm, fwhm_err = fit_moffat(star)
        self.assertIsInstance(r, np.ndarray)
        self.assertGreater(float(fwhm), 0.0)

    def test_fit_result_has_params(self):
        from gpuphot.phot.psf import fit_moffat
        star = self._make_synthetic_star()
        _, _, result, _, _ = fit_moffat(cp.asarray(star))
        self.assertIn('A', result.params)
        self.assertIn('R', result.params)
        self.assertIn('B', result.params)

    def test_reasonable_fwhm(self):
        from gpuphot.phot.psf import fit_moffat
        star = self._make_synthetic_star(size=31, R=3.0, B=2.5)
        _, _, _, fwhm, _ = fit_moffat(cp.asarray(star))
        self.assertGreater(float(fwhm), 1.0)
        self.assertLess(float(fwhm), 30.0)


if __name__ == '__main__':
    unittest.main()
