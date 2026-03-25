import unittest
from astropy.io.fits import Header


class TestDeleteHeaderFrom(unittest.TestCase):
    """
    Tests for delete_header_from. Pure Python/astropy, no CuPy required.
    """

    def setUp(self):
        from .....gpuphot.utils.headers import delete_header_from
        self.delete_header_from = delete_header_from

    def _make_header(self):
        h = Header()
        h['SIMPLE'] = True
        h['NAXIS'] = 2
        h['NAXIS1'] = 100
        h['NAXIS2'] = 100
        h.add_comment('BEGIN SECTION_A')
        h.add_comment('Some comment line')
        h.add_comment('END SECTION_A')
        return h

    def test_returns_header(self):
        h = self._make_header()
        result = self.delete_header_from(h, 'SECTION_A')
        self.assertIsInstance(result, Header)

    def test_matching_block_is_removed(self):
        h = self._make_header()
        result = self.delete_header_from(h, 'END SECTION_A')
        # After deletion the val should no longer appear in values
        found = any('END SECTION_A' in str(v) for v in result.values())
        self.assertFalse(found)

    def test_returns_unchanged_when_val_not_found(self):
        h = self._make_header()
        original_len = len(h)
        result = self.delete_header_from(h, 'NONEXISTENT_VALUE')
        self.assertEqual(len(result), original_len)

    def test_works_with_real_astropy_header(self):
        h = Header()
        h['BITPIX'] = -32
        h.add_comment('GPUPHOT ASTROMETRY START')
        h.add_comment('RA = 10.0')
        h.add_comment('DEC = 20.0')
        h.add_comment('GPUPHOT ASTROMETRY END')
        result = self.delete_header_from(h, 'GPUPHOT ASTROMETRY END')
        # The END marker and everything after it should be removed
        found = any('GPUPHOT ASTROMETRY END' in str(v) for v in result.values())
        self.assertFalse(found)

    def test_mandatory_keywords_preserved(self):
        h = self._make_header()
        result = self.delete_header_from(h, 'NONEXISTENT')
        self.assertIn('NAXIS', result)


if __name__ == '__main__':
    unittest.main()
