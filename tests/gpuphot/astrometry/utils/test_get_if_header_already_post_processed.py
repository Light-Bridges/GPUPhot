import os
import unittest

from astropy.io import fits

from .....gpuphot.astrometry.utils import get_if_header_already_post_processed


class TestGetIfHeaderAlreadyPostProcessed(unittest.TestCase):

    def _test_fits_file(self, file_path, postprocess, expected_result):
        with fits.open(file_path) as hdul:
            header = hdul[0].header
            result = get_if_header_already_post_processed(header, postprocess)
            self.assertEqual(expected_result, result, f'El resultado para {file_path} no es el esperado.')

    def test_image_calib_1(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT1_iKon936-1_MasterFlat_Ha_Bin11.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_calib_2(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT1_iKon936-1_MasterFlat_Lum_Bin11.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_calib_3(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT1_iKon936-1_MasterFlat_SDSSg_Bin11.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_calib_4(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT1_iKon936-1_MasterFlat_SDSSi_Bin11.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_calib_5(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT1_iKon936-1_MasterFlat_SDSSr_Bin11.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_calib_6(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT1_iKon936-1_MasterFlat_SDSSu_Bin11.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_calib_7(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT1_iKon936-1_MasterFlat_SDSSzs_Bin11.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_calib_8(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT2_QHY411-2_MasterFlat_Ha_Bin11.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_calib_9(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT2_QHY411-2_MasterFlat_Lum_Bin11.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_calib_10(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT2_QHY411-2_MasterFlat_SDSSg_Bin11.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_calib_11(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT2_QHY411-2_MasterFlat_SDSSi_Bin11.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_calib_12(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT2_QHY411-2_MasterFlat_SDSSr_Bin11.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_prered_1(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'prered',
                                 'TTT1_iKon936-1_2024-07-11-02-43-55-383463_Chariklo.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_prered_2(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'prered',
                                 'TTT1_iKon936-1_2024-07-11-02-46-43-564176_chiron.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_prered_3(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'prered',
                                 'TTT2_QHY411-2_2024-07-11-02-57-08-468122_chiron.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_prered_4(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'prered',
                                 'TTT2_QHY411-2_2024-07-11-02-57-59-361847_chiron.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_raw_1(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw',
                                 'TTT1_iKon936-1_2024-07-11-02-41-45-707922_Chariklo.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_raw_2(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw',
                                 'TTT1_iKon936-1_2024-07-11-02-43-55-383463_Chariklo.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_raw_3(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw',
                                 'TTT2_QHY411-2_2024-07-11-02-57-08-468122_chiron.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_raw_4(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw',
                                 'TTT2_QHY411-2_2024-07-11-02-57-59-361847_chiron.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_red_1(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'red',
                                 'TTT1_iKon936-1_2024-07-11-02-41-45-707922_Chariklo.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = True
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_red_2(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'red',
                                 'TTT1_iKon936-1_2024-07-11-02-43-55-383463_Chariklo.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = True
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_red_3(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'red',
                                 'TTT2_QHY411-2_2024-07-11-02-57-08-468122_chiron.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = True
        self._test_fits_file(file_path, postprocess, expected_result)

    def test_image_red_4(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'red',
                                 'TTT2_QHY411-2_2024-07-11-02-57-59-361847_chiron.fits')
        postprocess = 'PHOTOMETRY'
        expected_result = False
        self._test_fits_file(file_path, postprocess, expected_result)


if __name__ == '__main__':
    unittest.main()
