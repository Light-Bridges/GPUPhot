import unittest
import numpy as np


class TestPredMof(unittest.TestCase):
    """
    Tests for pred_mof. Pure numpy, no CuPy required.

    pred_mof formula:
        alpha = pred[:, 1]
        beta  = pred[:, 0] * 0.4 + 4.565
        nstar = pred[:, 2] * 200
        fwhm  = 2 * alpha * sqrt(2^(1/beta) - 1)
    """

    def setUp(self):
        from .....gpuphot.phot.photo_gpu import pred_mof
        self.pred_mof = pred_mof

    def _make_pred(self, n=5):
        rng = np.random.default_rng(0)
        # col 0: raw beta param, col 1: alpha, col 2: raw nstar param
        return rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32)

    def test_output_is_four_tuple(self):
        pred = self._make_pred(5)
        result = self.pred_mof(pred)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_all_outputs_are_arrays(self):
        pred = self._make_pred(5)
        alpha, beta, nstar, fwhm = self.pred_mof(pred)
        for arr in (alpha, beta, nstar, fwhm):
            self.assertIsInstance(arr, np.ndarray)

    def test_output_shapes_match_input_rows(self):
        n = 7
        pred = self._make_pred(n)
        alpha, beta, nstar, fwhm = self.pred_mof(pred)
        for arr in (alpha, beta, nstar, fwhm):
            self.assertEqual(arr.shape, (n,))

    def test_alpha_equals_pred_col1(self):
        pred = self._make_pred(5)
        alpha, beta, nstar, fwhm = self.pred_mof(pred)
        np.testing.assert_array_almost_equal(alpha, pred[:, 1])

    def test_beta_formula(self):
        pred = self._make_pred(5)
        alpha, beta, nstar, fwhm = self.pred_mof(pred)
        expected_beta = pred[:, 0] * 0.4 + 4.565
        np.testing.assert_array_almost_equal(beta, expected_beta)

    def test_nstar_formula(self):
        pred = self._make_pred(5)
        alpha, beta, nstar, fwhm = self.pred_mof(pred)
        expected_nstar = pred[:, 2] * 200
        np.testing.assert_array_almost_equal(nstar, expected_nstar)

    def test_fwhm_formula_known_values(self):
        # Build a pred row with known alpha and beta for a deterministic check
        alpha_val = 2.0
        beta_raw = 0.0   # -> beta = 0.0*0.4 + 4.565 = 4.565
        beta_val = beta_raw * 0.4 + 4.565
        pred = np.array([[beta_raw, alpha_val, 0.5]], dtype=np.float64)
        alpha, beta, nstar, fwhm = self.pred_mof(pred)
        expected_fwhm = 2.0 * alpha_val * np.sqrt(2 ** (1.0 / beta_val) - 1.0)
        self.assertAlmostEqual(float(fwhm[0]), expected_fwhm, places=6)

    def test_fwhm_positive_for_positive_alpha_beta(self):
        pred = self._make_pred(10)
        # Ensure beta raw values > 0 so beta > 4.565 > 0
        pred[:, 0] = np.abs(pred[:, 0])
        pred[:, 1] = np.abs(pred[:, 1]) + 0.1  # alpha > 0
        alpha, beta, nstar, fwhm = self.pred_mof(pred)
        self.assertTrue(np.all(fwhm > 0))


if __name__ == '__main__':
    unittest.main()
