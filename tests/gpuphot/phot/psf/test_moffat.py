import unittest
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class TestMoffat(unittest.TestCase):

    def test_peak_at_center(self):
        """Moffat(r=r0) = A."""
        from .....gpuphot.phot.psf import moffat
        r = np.array([0.0, 1.0, 2.0, 3.0])
        result = moffat(r, A=10.0, r0=0.0, B=2.0, R=1.0)
        self.assertAlmostEqual(float(result[0]), 10.0)

    def test_decays_with_distance(self):
        from .....gpuphot.phot.psf import moffat
        r = np.array([0.0, 1.0, 5.0, 10.0])
        result = moffat(r, A=1.0, r0=0.0, B=2.0, R=1.0)
        for i in range(len(result) - 1):
            self.assertGreater(float(result[i]), float(result[i + 1]))

    def test_known_value(self):
        """moffat(r=1, A=1, r0=0, B=1, R=1) = 1 * (1 + 1)**(-1) = 0.5."""
        from .....gpuphot.phot.psf import moffat
        r = np.array([1.0])
        result = moffat(r, A=1.0, r0=0.0, B=1.0, R=1.0)
        self.assertAlmostEqual(float(result[0]), 0.5)

    @unittest.skipUnless(HAS_CUPY, "CuPy required")
    def test_works_with_cupy(self):
        from .....gpuphot.phot.psf import moffat
        r = cp.array([0.0, 1.0, 2.0])
        result = moffat(r, A=5.0, r0=0.0, B=2.0, R=1.0)
        self.assertIsInstance(result, cp.ndarray)
        self.assertAlmostEqual(float(result[0]), 5.0)

    def test_shifted_center(self):
        from .....gpuphot.phot.psf import moffat
        r = np.array([3.0])
        result = moffat(r, A=1.0, r0=3.0, B=2.0, R=1.0)
        self.assertAlmostEqual(float(result[0]), 1.0)


if __name__ == '__main__':
    unittest.main()
