import unittest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@unittest.skipUnless(HAS_CUPY, "CuPy required")
class TestMoffatFwhm(unittest.TestCase):

    def test_known_value(self):
        """FWHM = 2*R*sqrt(2^(1/B)-1). For R=1, B=1: FWHM=2."""
        from .....gpuphot.phot.psf import moffat_fwhm
        fwhm, fwhm_err = moffat_fwhm(R=1.0, B=1.0, R_err=0.0, B_err=0.0)
        self.assertAlmostEqual(float(fwhm), 2.0, places=5)

    def test_positive_output(self):
        from .....gpuphot.phot.psf import moffat_fwhm
        fwhm, fwhm_err = moffat_fwhm(R=2.0, B=3.0, R_err=0.1, B_err=0.1)
        self.assertGreater(float(fwhm), 0.0)

    def test_returns_tuple(self):
        from .....gpuphot.phot.psf import moffat_fwhm
        result = moffat_fwhm(R=1.0, B=2.0, R_err=0.1, B_err=0.05)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_zero_errors(self):
        from .....gpuphot.phot.psf import moffat_fwhm
        fwhm, fwhm_err = moffat_fwhm(R=1.5, B=2.5, R_err=0.0, B_err=0.0)
        self.assertAlmostEqual(float(fwhm_err), 0.0, places=10)

    def test_none_r_err(self):
        from .....gpuphot.phot.psf import moffat_fwhm
        fwhm, fwhm_err = moffat_fwhm(R=1.0, B=2.0, R_err=None, B_err=0.1)
        self.assertGreater(float(fwhm), 0.0)

    def test_larger_R_larger_fwhm(self):
        from .....gpuphot.phot.psf import moffat_fwhm
        fwhm1, _ = moffat_fwhm(R=1.0, B=2.0, R_err=0.0, B_err=0.0)
        fwhm2, _ = moffat_fwhm(R=3.0, B=2.0, R_err=0.0, B_err=0.0)
        self.assertGreater(float(fwhm2), float(fwhm1))


if __name__ == '__main__':
    unittest.main()
