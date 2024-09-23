# import os
# import unittest
#
# from astropy.io import fits
#
# from gpuphot.gpuphot.astrometry.utils import get_scale
#
#
# class TestGetScale(unittest.TestCase):
#
#     def _test_fits_file(self, file_path, expected_result):
#         with fits.open(file_path) as hdul:
#             header = hdul[0].header
#             wcs_params = {key: header.get(key, 0.0) for key in ['CD1_1', 'CD1_2']}
#             result = get_scale(hwcs=wcs_params)
#             self.assertAlmostEqual(expected_result, result, places=6,
#                                    msg=f'El resultado para {file_path} no es el esperado.')
#
#     def test_image_calib_1(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
#                                  'TTT1_iKon936-1_MasterFlat_Ha_Bin11.fits')
#         expected_result = 0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_calib_2(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
#                                  'TTT1_iKon936-1_MasterFlat_Lum_Bin11.fits')
#         expected_result = 0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_calib_3(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
#                                  'TTT1_iKon936-1_MasterFlat_SDSSg_Bin11.fits')
#         expected_result = 0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_calib_4(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
#                                  'TTT1_iKon936-1_MasterFlat_SDSSi_Bin11.fits')
#         expected_result = 0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_calib_5(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
#                                  'TTT1_iKon936-1_MasterFlat_SDSSr_Bin11.fits')
#         expected_result = 0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_calib_6(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
#                                  'TTT1_iKon936-1_MasterFlat_SDSSu_Bin11.fits')
#         expected_result = 0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_calib_7(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
#                                  'TTT1_iKon936-1_MasterFlat_SDSSzs_Bin11.fits')
#         expected_result = 0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_calib_8(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
#                                  'TTT2_QHY411-2_MasterFlat_Ha_Bin11.fits')
#         expected_result = 0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_calib_9(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
#                                  'TTT2_QHY411-2_MasterFlat_Lum_Bin11.fits')
#         expected_result = 0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_calib_10(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
#                                  'TTT2_QHY411-2_MasterFlat_SDSSg_Bin11.fits')
#         expected_result = 0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_calib_11(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
#                                  'TTT2_QHY411-2_MasterFlat_SDSSi_Bin11.fits')
#         expected_result = 0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_calib_12(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'calib',
#                                  'TTT2_QHY411-2_MasterFlat_SDSSr_Bin11.fits')
#         expected_result = 0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_prered_1(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'prered',
#                                  'TTT1_iKon936-1_2024-07-11-02-43-55-383463_Chariklo.fits')
#         expected_result = -0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_prered_2(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'prered',
#                                  'TTT1_iKon936-1_2024-07-11-02-46-43-564176_chiron.fits')
#         expected_result = -0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_prered_3(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'prered',
#                                  'TTT2_QHY411-2_2024-07-11-02-57-08-468122_chiron.fits')
#         expected_result = -0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_prered_4(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'prered',
#                                  'TTT2_QHY411-2_2024-07-11-02-57-59-361847_chiron.fits')
#         expected_result = -0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_raw_1(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw',
#                                  'TTT1_iKon936-1_2024-07-11-02-41-45-707922_Chariklo.fits')
#         expected_result = -0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_raw_2(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw',
#                                  'TTT1_iKon936-1_2024-07-11-02-43-55-383463_Chariklo.fits')
#         expected_result = -0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_raw_3(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw',
#                                  'TTT2_QHY411-2_2024-07-11-02-57-08-468122_chiron.fits')
#         expected_result = -0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_raw_4(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw',
#                                  'TTT2_QHY411-2_2024-07-11-02-57-59-361847_chiron.fits')
#         expected_result = 0.0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_red_1(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'red',
#                                  'TTT1_iKon936-1_2024-07-11-02-41-45-707922_Chariklo.fits')
#         expected_result = 0.5069605059868271
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_red_2(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'red',
#                                  'TTT1_iKon936-1_2024-07-11-02-43-55-383463_Chariklo.fits')
#         expected_result = 0.5069364981087163
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_red_3(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'red',
#                                  'TTT2_QHY411-2_2024-07-11-02-57-08-468122_chiron.fits')
#         expected_result = 0.14153757996757674
#         self._test_fits_file(file_path, expected_result)
#
#     def test_image_red_4(self):
#         file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'red',
#                                  'TTT2_QHY411-2_2024-07-11-02-57-59-361847_chiron.fits')
#         expected_result = 0
#         self._test_fits_file(file_path, expected_result)
#
#     def test_ccw_combination_1(self):
#         hwcs = {'CD1_1': 1.0, 'CD1_2': 0.0}
#         expected_result = 3600.0
#         result = get_scale(hwcs)
#         self.assertAlmostEqual(result, expected_result, places=6)
#
#     def test_ccw_combination_2(self):
#         hwcs = {'CD1_1': 0.0, 'CD1_2': -1.0}
#         expected_result = 3600.0
#         result = get_scale(hwcs)
#         self.assertAlmostEqual(result, expected_result, places=6)
#
#     def test_ccw_combination_3(self):
#         hwcs = {'CD1_1': -1.0, 'CD1_2': 0.0}
#         expected_result = 3600.0
#         result = get_scale(hwcs)
#         self.assertAlmostEqual(result, expected_result, places=6)
#
#     def test_ccw_combination_4(self):
#         hwcs = {'CD1_1': 0.0, 'CD1_2': 1.0}
#         expected_result = 3600.0
#         result = get_scale(hwcs)
#         self.assertAlmostEqual(result, expected_result, places=6)
#
#
# if __name__ == '__main__':
#     unittest.main()
