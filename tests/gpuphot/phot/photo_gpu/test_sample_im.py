import unittest
import numpy as np


class TestSampleIm(unittest.TestCase):
    """
    Tests for sample_im. Uses numpy only (no CuPy required).
    Requires a large image (>= 512 + nc*delta pixels per side).
    Default nc=50 needs roughly 3000x3000 or larger for safety.
    """

    def setUp(self):
        from gpuphot.phot.photo_gpu import sample_im
        self.sample_im = sample_im

    def _large_image(self, size=3500, value=1000.0):
        rng = np.random.default_rng(42)
        return rng.normal(loc=value, scale=50.0, size=(size, size)).astype(np.float32)

    def test_output_is_two_tuple(self):
        img = self._large_image()
        result = self.sample_im(img, nc=50, ns=10)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_sampled_chunks_shape(self):
        ns = 10
        img = self._large_image()
        chunks, maxvals = self.sample_im(img, nc=50, ns=ns)
        self.assertEqual(chunks.shape, (ns, 512, 512, 1))

    def test_max_values_shape(self):
        ns = 8
        img = self._large_image()
        chunks, maxvals = self.sample_im(img, nc=50, ns=ns)
        self.assertEqual(maxvals.shape, (ns,))

    def test_no_nan_in_chunks(self):
        img = self._large_image()
        chunks, maxvals = self.sample_im(img, nc=50, ns=10)
        self.assertFalse(np.any(np.isnan(chunks)))

    def test_no_inf_in_chunks(self):
        img = self._large_image()
        chunks, maxvals = self.sample_im(img, nc=50, ns=10)
        self.assertFalse(np.any(np.isinf(chunks)))

    def test_max_values_are_positive(self):
        img = self._large_image(value=1000.0)
        chunks, maxvals = self.sample_im(img, nc=50, ns=10)
        self.assertTrue(np.all(maxvals > 0))

    def test_output_is_numpy_arrays(self):
        img = self._large_image()
        chunks, maxvals = self.sample_im(img, nc=50, ns=5)
        self.assertIsInstance(chunks, np.ndarray)
        self.assertIsInstance(maxvals, np.ndarray)


if __name__ == '__main__':
    unittest.main()
