from unittest import TestCase

import numpy as np

from gpuphot.astrometry.utils import (
    get_target_ephemeris,
    px_to_wcs,
    wcs_to_px,
    get_target_ra_dec,
)


class TestAstrometryUtils(TestCase):
    _header1 = {
        "SIMPLE": True,
        "BITPIX": -32,
        "NAXIS": 2,
        "NAXIS1": 2048,
        "NAXIS2": 2048,
        "TELESCOP": "TTT1",
        "SITELAT": 28.29871121,
        "SITELONG": -16.50956893,
        "SITEALT": 2359.12,
        "DIAMETER": 800.0,
        "FOCAL": 6.85,
        "FOCALEN": 5480.0,
        "TRACK": 1,
        "SHMODE": "Mechanical shutter",
        "DRMODE": None,
        "COMODE": None,
        "RDMODE": "Multi track",
        "RDNOISE": 7.0,
        "INTEMP": -55.883,
        "SATLEVEL": 65536,
        "GAIN": 1.0,
        "DC": 0.02,
        "SCALEORI": 0.508,
        "PARITY": "pos",
        "XBINNING": 1,
        "YBINNING": 1,
        "EFFECTSX": 0,
        "EFFECTNX": 2048,
        "EFFECTSY": 0,
        "EFFECTNY": 2048,
        "STATSX": 0,
        "STATNX": 2048,
        "STATSY": 0,
        "STATNY": 2048,
        "INSTRUME": "iKon936-TTT1",
        "INMODEL": "iKon936",
        "INSERIAL": 27305,
        "INSDK": "2.104",
        "OFFSET": None,
        "PXSIZE": 13.5,
        "ORISIZEX": 2048,
        "ORISIZEY": 2048,
        "ALISIZEX": 2048,
        "ALISIZEY": 2048,
        "FINSIZEX": 2048,
        "FINSIZEY": 2048,
        "CAMERA": "iKon936-1",
        "INFIRMW": "20.12",
        "HSSPEED": 1.0,
        "VSSPEED": 38.549999,
        "PREAMPGA": 4.0,
        "DATE-OBS": "2023-02-08T20:53:25.452554",
        "JD-OBS": 2459984.370433479,
        "MJD-OBS": 59983.87043347863,
        "PCDATE": "2023-02-08T20:53:25.452554",
        "COOLERST": "DRV_TEMP_NOT_REACHED",
        "EXPTIME": 20.022,
        "WINDOWSX": 0,
        "WINDOWNX": 2048,
        "WINDOWSY": 0,
        "WINDOWNY": 2048,
        "OBJECT": "Kellyoconnor",
        "FILTER": "SDSSi",
        "POINTRA": 3.652383384165774,
        "POINTDEC": 11.85168222874598,
        "UT1": "2023-02-08T20:53:25.452554",
        "PCDAT1": "2023-02-08T20:53:25.452554",
        "EXPT1": 20.022,
        "TEMP1": -55.883,
        "FOCUS": 12190,
        "UTOBS": "2023-02-08T20:53:25.452554",
        "INTEGT": 20.0,
        "TOTIMA": 1,
        "OBLINEID": 81158,
        "HUMIDITY": 4.2,
        "MIRRHUM": 15.612,
        "PRESSURE": 766.0,
        "AMBTEMP": 3.41,
        "MIRRTEMP": 0.96,
        "CLOUD": 0.0,
        "ILLUMINA": 1.6,
        "WINDDIR": 106.0,
        "WINDVEL": 2.33,
        "DUSTPLA": 0.003,
        "DUSTPM1": 0.04,
        "DUSTPM10": 0.04,
        "DUSTPM25": 0.04,
        "PWV": 2.04,
        "TESSMAG": 20.84,
        "SKYIRTEM": -49.75,
        "BIASCORR": "bias+dark",
        "BIASMEAN": 268.871,
        "BIASSTD": 6.412,
        "BIASN": 21,
        "BIASTEMP": -53.287,
        "BIASDATE": "2023-03-01T07:39:43.292806",
        "DARKMEAN": 0.662,
        "DARKSTD": 0.287,
        "DARKN": 21,
        "DARKTEMP": -53.287,
        "DARKDATE": "2023-03-01T07:44:13.077313",
        "FLATMEAN": 14836.159,
        "FLATSTD": 539.274,
        "FLATN": 11,
        "FLATTEMP": -55.883,
        "FLATDATE": "2023-02-09T08:48:35.819472",
        "DATE": "2023-04-15T03:26:03",
        "RA": 54.78902141028366,
        "DEC": 11.85320287613557,
        "RAHMS": "03:39:9.365138",
        "DECDMS": "11:51:11.530354",
        "AZ": 233.523691,
        "ALT": 64.850553,
        "ZD": 25.149447,
        "AIRMASS": 1.104725,
        "LONGAL": 174.721683,
        "LATGAL": -33.665416,
        "LONECL": 55.312877,
        "LATECL": -7.447191,
        "WCSAXES": 2,
        "EQUINOX": 2000.0,
        "LONPOLE": 180.0,
        "LATPOLE": 0.0,
        "CRVAL1": 54.650272494715,
        "CRVAL2": 11.85584513410472,
        "CRPIX1": 212.0882167819279,
        "CRPIX2": 1545.511680545574,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CD1_1": 0.000116910747794951,
        "CD1_2": -7.836721585484e-05,
        "CD2_1": -7.8326675648825e-05,
        "CD2_2": -0.00011694055432154,
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "A_ORDER": 3,
        "A_0_0": 0.0,
        "A_0_1": 0.0,
        "A_0_2": 7.85130602880151e-07,
        "A_0_3": 4.87714430139195e-10,
        "A_1_0": 0.0,
        "A_1_1": -5.2398680309053e-07,
        "A_1_2": 1.06085839651212e-10,
        "A_2_0": -7.7585895073241e-07,
        "A_2_1": 3.74421642914737e-10,
        "A_3_0": 6.63685497970571e-10,
        "B_ORDER": 3,
        "B_0_0": 0.0,
        "B_0_1": 0.0,
        "B_0_2": 2.33951184393364e-07,
        "B_0_3": 2.58285053217136e-10,
        "B_1_0": 0.0,
        "B_1_1": 2.09009183645733e-07,
        "B_1_2": 1.53541176428838e-10,
        "B_2_0": 5.12947344849658e-07,
        "B_2_1": 2.46888610282133e-11,
        "B_3_0": -2.1443451661585e-10,
        "AP_ORDER": 3,
        "AP_0_0": 1.5219546870604e-05,
        "AP_0_1": -2.6448675612831e-08,
        "AP_0_2": -7.8428516167927e-07,
        "AP_0_3": -4.870677038752e-10,
        "AP_1_0": 5.00186329487163e-07,
        "AP_1_1": 5.24466855304315e-07,
        "AP_1_2": -1.0535107700489e-10,
        "AP_2_0": 7.73085658503646e-07,
        "AP_2_1": -3.7411460141753e-10,
        "AP_3_0": -6.6162747930699e-10,
        "BP_ORDER": 3,
        "BP_0_0": -4.8112570978542e-05,
        "BP_0_1": -1.2852388760091e-08,
        "BP_0_2": -2.3355636615901e-07,
        "BP_0_3": -2.57929193847e-10,
        "BP_1_0": -2.3905053194158e-07,
        "BP_1_1": -2.0901938882307e-07,
        "BP_1_2": -1.5314954797012e-10,
        "BP_2_0": -5.1235862137302e-07,
        "BP_2_1": -2.4328225981164e-11,
        "BP_3_0": 2.14212738169679e-10,
        "FOVX": 0.2889955555555556,
        "FOVY": 0.2889955555555556,
        "ZP": 0.0,
        "EZP": 0.0,
        "FWHM": 2.96001935005188,
        "EFWHM": 0.3019448220729828,
        "M_LIM": 0.0,
        "M_SKY": -5.202066868645168,
        "SKY": 160.2217864990234,
        "ESKY": 16.04984092712402,
        "SCALE": 0.508,
        "CCW": -146.1756287619255,
        "AP_SNR": 49,
        "AP_FLUX": 49,
    }
    _header2 = {
        "SIMPLE": True,
        "BITPIX": -32,
        "NAXIS": 2,
        "NAXIS1": 14200,
        "NAXIS2": 10650,
        "TELESCOP": "TTT2",
        "SITELAT": 28.29871203,
        "SITELONG": -16.50948217,
        "SITEALT": 2359.11,
        "DIAMETER": 800.0,
        "FOCAL": 6.85,
        "FOCALEN": 5480.0,
        "TRACK": 1,
        "SHMODE": "Rolling shutter",
        "DRMODE": "HDR",
        "COMODE": "Mono 16",
        "RDMODE": "Full Frame Read Mode #4",
        "RDNOISE": 2.8,
        "INTEMP": -15.0,
        "SATLEVEL": 65536,
        "GAIN": 1.024,
        "DC": 0.007,
        "SCALEORI": 0.142,
        "PARITY": "pos",
        "XBINNING": 1,
        "YBINNING": 1,
        "OVERSCSX": 51,
        "OVERSCNX": 14253,
        "OVERSCSY": 10659,
        "OVERSCNY": 88,
        "EFFECTSX": 51,
        "EFFECTNX": 14253,
        "EFFECTSY": 0,
        "EFFECTNY": 10650,
        "STATSX": 6640,
        "STATNX": 1024,
        "STATSY": 4862,
        "STATNY": 1024,
        "INSTRUME": "QHY411-TTT2",
        "INMODEL": "QHY411MERIS",
        "INSERIAL": "2566a14f9ee62fe1a",
        "INSDK": "22-7-25.16",
        "OFFSET": 10.0,
        "PXSIZE": 3.76,
        "ORISIZEX": 14304,
        "ORISIZEY": 10748,
        "ALISIZEX": 14304,
        "ALISIZEY": 10748,
        "FINSIZEX": 14304,
        "FINSIZEY": 10748,
        "CAMERA": "QHY411-2",
        "INFIRMW": "2017_5_6",
        "DATE-OBS": "2023-02-19T20:56:11.676285",
        "GPSDAT": "2023-02-19T20:56:11.676285",
        "GPSLAT": 28.298815,
        "GPSLON": -16.50944833333334,
        "SEQNUM": 41,
        "JD-OBS": 2459995.372357364,
        "MJD-OBS": 59994.87235736441,
        "PCDATE": "2023-02-19T20:56:11.683098",
        "EXPTIME": 5.326963,
        "GAMODE": 0.0,
        "WINDOWSX": 0,
        "WINDOWNX": 14304,
        "WINDOWSY": 0,
        "WINDOWNY": 10748,
        "OBJECT": "1993VB",
        "FILTER": "Lum",
        "POINTRA": 4.423111699254016,
        "POINTDEC": 26.56427818078933,
        "UT1": "2023-02-19T20:56:11.676285",
        "GPSDAT1": "2023-02-19T20:56:11.676285",
        "PCDAT1": "2023-02-19T20:56:11.683098",
        "EXPT1": 5.326963,
        "TEMP1": -15.0,
        "SEQNUM1": 41,
        "FOCUS": 13898,
        "UTOBS": "2023-02-19T20:56:11.676285",
        "INTEGT": 5.3,
        "TOTIMA": 1,
        "OBLINEID": 111545,
        "HUMIDITY": 55.44,
        "BIASCORR": "bias",
        "BIASMEAN": 171.718,
        "BIASSTD": 3.32,
        "BIASOVER": 171.734,
        "BIASN": 21,
        "BIASTEMP": -12.3,
        "BIASDATE": "2023-02-19T18:36:44.834639",
        "FLATMEAN": 14175.844,
        "FLATSTD": 129.843,
        "FLATN": 11,
        "FLATTEMP": -13.0,
        "FLATDATE": "2023-02-19T07:18:19.075587",
        "DATE": "2023-04-15T01:38:21",
        "RA": 66.42826273907058,
        "DEC": 26.62161140115776,
        "RAHMS": "04:25:42.783057",
        "DECDMS": "26:37:17.801044",
        "AZ": 269.626985,
        "ALT": 71.937176,
        "ZD": 18.062824,
        "AIRMASS": 1.051838,
        "LONGAL": 171.230494,
        "LATGAL": -15.555384,
        "LONECL": 68.973102,
        "LATECL": 4.886683,
        "WCSAXES": 2,
        "EQUINOX": 2000.0,
        "LONPOLE": 180.0,
        "LATPOLE": 0.0,
        "CRVAL1": 66.5731118308627,
        "CRVAL2": 26.70513733216426,
        "CRPIX1": 9764.406291715912,
        "CRPIX2": 8199.798698904298,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CD1_1": 6.84340927896256e-06,
        "CD1_2": 3.87017232734834e-05,
        "CD2_1": 3.86905151533621e-05,
        "CD2_2": -6.8299073795454e-06,
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "A_ORDER": 3,
        "A_0_0": 0.0,
        "A_0_1": 0.0,
        "A_0_2": 4.73780990402977e-08,
        "A_0_3": 4.22005169691596e-12,
        "A_1_0": 0.0,
        "A_1_1": -5.7176900597633e-08,
        "A_1_2": -3.5659289348658e-12,
        "A_2_0": -1.3129239874033e-07,
        "A_2_1": -1.0242455126531e-11,
        "A_3_0": -1.7383258240407e-11,
        "B_ORDER": 3,
        "B_0_0": 0.0,
        "B_0_1": 0.0,
        "B_0_2": 6.69007879531292e-09,
        "B_0_3": -2.573733153067e-14,
        "B_1_0": 0.0,
        "B_1_1": -4.036529341206e-08,
        "B_1_2": -9.3644790143299e-12,
        "B_2_0": -7.032292721734e-08,
        "B_2_1": -3.333143641493e-12,
        "B_3_0": -1.0658621983011e-11,
        "AP_ORDER": 3,
        "AP_0_0": -0.00161476092667527,
        "AP_0_1": -1.7220915038522e-07,
        "AP_0_2": -4.7363792125673e-08,
        "AP_0_3": -4.2195087651648e-12,
        "AP_1_0": 9.56583422983116e-08,
        "AP_1_1": 5.74198698131822e-08,
        "AP_1_2": 3.58426306939799e-12,
        "AP_2_0": 1.31683014580356e-07,
        "AP_2_1": 1.02894027357783e-11,
        "AP_3_0": 1.74308119033378e-11,
        "BP_ORDER": 3,
        "BP_0_0": -0.00128863202567437,
        "BP_0_1": -1.172848357772e-07,
        "BP_0_2": -6.6351435582233e-09,
        "BP_0_3": 3.21998549459442e-14,
        "BP_1_0": 4.04272314790572e-08,
        "BP_1_1": 4.05799078044546e-08,
        "BP_1_2": 9.38479325507474e-12,
        "BP_2_0": 7.05901127290546e-08,
        "BP_2_1": 3.37127115036146e-12,
        "BP_3_0": 1.06902353633817e-11,
        "FOVX": 0.4200833333333333,
        "FOVY": 0.5601111111111111,
        "ZP": 23.15573159902409,
        "EZP": 0.06360210866361185,
        "FWHM": 5.001328468322754,
        "EFWHM": 0.56470787525177,
        "M_LIM": 18.720908109705,
        "M_SKY": 24.4681875057086,
        "SKY": 3.135298728942871,
        "ESKY": 4.061130523681641,
        "SCALE": 0.142,
        "CCW": 100.0193684428295,
        "AP_SNR": 49,
        "AP_FLUX": 49,
    }

    def _split_target_name(self, s):
        result = ""
        prev_type = None

        for char in s:
            current_type = "letter" if char.isalpha() else "number"

            # Check if there's a transition from letter to number or vice versa
            if prev_type is not None and prev_type != current_type:
                result += " "

            result += char
            prev_type = current_type

        return result

    def test_get_target_ephemeris(self):
        _site_lat = +28.300332
        _site_lon = -16.512206
        _site_elev = 2390
        _target_name = "Ceres"
        _date_obs = 2458133.33546
        _ephems = (9.434766666666667, 27.97686)

        ephems = get_target_ephemeris(
            _target_name, _date_obs, _site_lat, _site_lon, _site_elev
        )

        self.assertEquals(ephems, _ephems)

    def test_get_target_ephemeris_location(self):
        _site_lat = -34.6037
        _site_lon = -58.3816
        _site_elev = 50
        _target_name = "Ceres"
        _date_obs = 2458133.33546
        _ephems = (9.434723333333332, 27.97789)

        ephems = get_target_ephemeris(
            _target_name, _date_obs, _site_lat, _site_lon, _site_elev
        )

        self.assertEquals(ephems, _ephems)

    def test_get_target_ephemeris_target(self):
        _site_lat = +28.300332
        _site_lon = -16.512206
        _site_elev = 2390
        _target_name = "Mars Barycenter"
        _date_obs = 2458133.33546
        _ephems = (15.355653333333334, -17.6265)

        ephems = get_target_ephemeris(
            _target_name, _date_obs, _site_lat, _site_lon, _site_elev
        )

        self.assertEquals(ephems, _ephems)

    def test_get_target_ephemeris_date(self):
        _site_lat = +28.300332
        _site_lon = -16.512206
        _site_elev = 2390
        _target_name = "Ceres"
        _date_obs = 2459000.0  # Cambia la fecha de observación
        _ephems = (22.944357333333336, -17.19894)

        ephems = get_target_ephemeris(
            _target_name, _date_obs, _site_lat, _site_lon, _site_elev
        )

        self.assertEquals(ephems, _ephems)

    def test_px_to_wcs_h1(self):
        x = 100
        y = 150
        wcs_coords = px_to_wcs(x, y, self._header1)

        expected_coords = [np.array(54.74875915682997), np.array(12.027625065441386)]

        self.assertListEqual(
            [float(wcs_coords[0]), float(wcs_coords[1])], expected_coords
        )

    def test_px_to_wcs_h2(self):
        x = 100
        y = 150
        wcs_coords = px_to_wcs(x, y, self._header2)

        expected_coords = [np.array(66.15202567431312), np.array(26.385933695156663)]

        self.assertListEqual(
            [float(wcs_coords[0]), float(wcs_coords[1])], expected_coords
        )

    def test_wcs_to_px_h1(self):
        cat = {"RA": 54.74875915682997, "DEC": 12.027625065441386}
        wcs_coords = wcs_to_px(cat, self._header1)

        expected_pix = [np.array(100), np.array(150)]

        self.assertListEqual([int(wcs_coords[0]), int(wcs_coords[1])], expected_pix)

    def test_wcs_to_px_h2(self):
        cat = {"RA": 66.15202567431312, "DEC": 26.385933695156663}
        wcs_coords = wcs_to_px(cat, self._header2)

        expected_pix = [np.array(100), np.array(150)]

        self.assertListEqual([int(wcs_coords[0]), int(wcs_coords[1])], expected_pix)

    def test_get_target_ra_dec_without_moving_h1(self):
        target_name = self._split_target_name(self._header1["OBJECT"])
        moving_target = False
        ra_dec_dict = None
        result = get_target_ra_dec(
            target_name, moving_target, self._header1, ra_dec_dict
        )
        expected_result = tuple([self._header1["POINTRA"], self._header1["POINTDEC"]])

        self.assertTupleEqual(result, expected_result)

    def test_get_target_ra_dec_without_moving_h2(self):
        target_name = self._split_target_name(self._header2["OBJECT"])
        moving_target = False
        ra_dec_dict = None
        result = get_target_ra_dec(
            target_name, moving_target, self._header2, ra_dec_dict
        )
        expected_result = tuple([self._header2["POINTRA"], self._header2["POINTDEC"]])

        self.assertTupleEqual(result, expected_result)

    def test_get_target_ra_dec_moving_h1(self):
        target_name = self._split_target_name(self._header1["OBJECT"])
        moving_target = True
        ra_dec_dict = None  # This parameter is not needed for ephemeris lookup
        result = get_target_ra_dec(
            target_name, moving_target, self._header1, ra_dec_dict
        )
        expected_result = tuple([self._header1["POINTRA"], self._header1["POINTDEC"]])

        tolerance = 0.001

        self.assertTrue(abs(result[0] - expected_result[0]) < tolerance)
        self.assertTrue(abs(result[1] - expected_result[1]) < tolerance)

    def test_get_target_ra_dec_moving_h2(self):
        target_name = self._split_target_name(self._header2["OBJECT"])
        moving_target = True
        ra_dec_dict = None  # This parameter is not needed for ephemeris lookup
        result = get_target_ra_dec(
            target_name, moving_target, self._header2, ra_dec_dict
        )

        expected_result = tuple([self._header2["POINTRA"], self._header2["POINTDEC"]])

        tolerance = 0.1

        self.assertTrue(abs(result[0] - expected_result[0]) < tolerance)
        self.assertTrue(abs(result[1] - expected_result[1]) < tolerance)

    def test_get_target_ra_dec_moving_precalculated_ra_dec_h1(self):
        target_name = self._split_target_name(self._header1["OBJECT"])
        moving_target = True
        # Assuming ra_dec_dict is pre-calculated and available
        ra_dec_dict = {
            "time_index": [],  # Replace with the actual list of time objects
            "ra": [],  # Replace with the actual list of RA values
            "dec": [],  # Replace with the actual list of Dec values
        }
        result = get_target_ra_dec(
            target_name, moving_target, self._header1, ra_dec_dict
        )

        self.fail()

    def test_get_target_ra_dec_moving_precalculated_ra_dec_h2(self):
        target_name = self._split_target_name(self._header2["OBJECT"])
        moving_target = True
        # Assuming ra_dec_dict is pre-calculated and available
        ra_dec_dict = {
            "time_index": [],  # Replace with the actual list of time objects
            "ra": [],  # Replace with the actual list of RA values
            "dec": [],  # Replace with the actual list of Dec values
        }
        result = get_target_ra_dec(
            target_name, moving_target, self._header2, ra_dec_dict
        )

        self.fail()

    def test_get_target_pix(self):
        self.fail()

    def test_get_ccw(self):
        self.fail()

    def test_get_if_header_already_post_processed(self):
        self.fail()

    def test_cat_input_from_header(self):
        self.fail()
