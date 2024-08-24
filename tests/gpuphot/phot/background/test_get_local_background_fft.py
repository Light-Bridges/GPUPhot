import os
import unittest

import cupy as cp
from astropy.io import fits

from gpuphot.phot.background import get_local_background_fft


class TestGetLocalBackgroundFFT(unittest.TestCase):

    @staticmethod
    def plate_scale_mm(focal):
        return 206265 / focal

    @staticmethod
    def plate_scale_px(microns, focal):
        return TestGetLocalBackgroundFFT.plate_scale_mm(focal) * microns / 1000

    def _load_fits_image(self, file_path):
        """
        Load an image from a FITS file and return as a cupy array.

        Parameters
        ----------
        file_path : str
            Path to the FITS file.

        Returns
        -------
        cupy.ndarray
            Image data as a cupy array.
        """
        return cp.array(fits.getdata(file_path))

    def _load_fits_header(self, file_path):
        return fits.getheader(file_path)

    def _test_fits_file(self, file_path, expected_result):
        """
        Test get_local_background_fft with a given FITS file.

        Parameters
        ----------
        file_path : str
            Path to the FITS file.
        pxscale : float
            Pixel scale in arcsec/pixel.
        expected_result : float
            Expected result for assertion (dummy in this case).
        """
        image = self._load_fits_image(file_path)
        imheader = self._load_fits_header(file_path)
        print('PXSIZE:', imheader['PXSIZE'] if 'PXSIZE' in imheader else 'N/A')
        print('FOCALEN:', imheader['FOCALEN'] if 'FOCALEN' in imheader else 'N/A')
        print('XBINNING:', imheader['XBINNING'] if 'XBINNING' in imheader else 'N/A')
        print('KS:', imheader['KS'] if 'KS' in imheader else 'N/A')
        get_std = False
        scale = TestGetLocalBackgroundFFT.plate_scale_px(imheader['PXSIZE'], imheader['FOCALEN']) * imheader['XBINNING']
        ks = int(imheader['KS']) if 'KS' in imheader else 2
        (img_filled_m, img_filled_2) = get_local_background_fft(image=image, pxscale=scale, ks=ks, get_std=get_std)
        self.assertIsInstance(img_filled_m, cp.ndarray, 'The returned background should be a cupy array.')
        if get_std:
            self.assertIsInstance(img_filled_2, cp.ndarray,
                                  'The returned standard deviation image should be a cupy array when get_std is True.')
        else:
            self.assertIsNone(img_filled_2,
                              'The returned standard deviation image should be None when get_std is False.')

    def test_image_calib_1(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT1_iKon936-1_MasterFlat_Ha_Bin11.fits')
        expected_result = 0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_calib_2(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT1_iKon936-1_MasterFlat_Lum_Bin11.fits')
        expected_result = 0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_calib_3(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT1_iKon936-1_MasterFlat_SDSSg_Bin11.fits')
        expected_result = 0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_calib_4(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT1_iKon936-1_MasterFlat_SDSSi_Bin11.fits')
        expected_result = 0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_calib_5(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT1_iKon936-1_MasterFlat_SDSSr_Bin11.fits')
        expected_result = 0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_calib_6(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT1_iKon936-1_MasterFlat_SDSSu_Bin11.fits')
        expected_result = 0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_calib_7(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT1_iKon936-1_MasterFlat_SDSSzs_Bin11.fits')
        expected_result = 0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_calib_8(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT2_QHY411-2_MasterFlat_Ha_Bin11.fits')
        expected_result = 0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_calib_9(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT2_QHY411-2_MasterFlat_Lum_Bin11.fits')
        expected_result = 0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_calib_10(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT2_QHY411-2_MasterFlat_SDSSg_Bin11.fits')
        expected_result = 0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_calib_11(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT2_QHY411-2_MasterFlat_SDSSi_Bin11.fits')
        expected_result = 0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_calib_12(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
                                 'TTT2_QHY411-2_MasterFlat_SDSSr_Bin11.fits')
        expected_result = 0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_prered_1(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'prered',
                                 'TTT1_iKon936-1_2024-07-11-02-43-55-383463_Chariklo.fits')
        expected_result = -0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_prered_2(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'prered',
                                 'TTT1_iKon936-1_2024-07-11-02-46-43-564176_chiron.fits')
        expected_result = -0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_prered_3(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'prered',
                                 'TTT2_QHY411-2_2024-07-11-02-57-08-468122_chiron.fits')
        expected_result = -0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_prered_4(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'prered',
                                 'TTT2_QHY411-2_2024-07-11-02-57-59-361847_chiron.fits')
        expected_result = -0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_raw_1(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw',
                                 'TTT1_iKon936-1_2024-07-11-02-41-45-707922_Chariklo.fits')
        expected_result = -0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_raw_2(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw',
                                 'TTT1_iKon936-1_2024-07-11-02-43-55-383463_Chariklo.fits')
        expected_result = -0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_raw_3(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw',
                                 'TTT2_QHY411-2_2024-07-11-02-57-08-468122_chiron.fits')
        expected_result = -0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_raw_4(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw',
                                 'TTT2_QHY411-2_2024-07-11-02-57-59-361847_chiron.fits')
        expected_result = 0.0
        self._test_fits_file(file_path, expected_result)

    def test_image_red_1(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'red',
                                 'TTT1_iKon936-1_2024-07-11-02-41-45-707922_Chariklo.fits')
        expected_result = -1.6186416591554544
        self._test_fits_file(file_path, expected_result)

    def test_image_red_2(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'red',
                                 'TTT1_iKon936-1_2024-07-11-02-43-55-383463_Chariklo.fits')
        expected_result = -1.6227922267716928
        self._test_fits_file(file_path, expected_result)

    def test_image_red_3(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'red',
                                 'TTT2_QHY411-2_2024-07-11-02-57-08-468122_chiron.fits')
        expected_result = -174.29739930099691
        self._test_fits_file(file_path, expected_result)

    def test_image_red_4(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'red',
                                 'TTT2_QHY411-2_2024-07-11-02-57-59-361847_chiron.fits')
        expected_result = -0.0
        self._test_fits_file(file_path, expected_result)


if __name__ == '__main__':
    test_loader = unittest.TestLoader()
    test_names = test_loader.getTestCaseNames(TestGetLocalBackgroundFFT)
    for test_name in test_names:
        print(f'Running {test_name}...')
        suite = unittest.TestSuite()
        suite.addTest(TestGetLocalBackgroundFFT(test_name))
        runner = unittest.TextTestRunner()
        runner.run(suite)
        suite._tests.clear()
