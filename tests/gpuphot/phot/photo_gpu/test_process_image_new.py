import os
import signal
# from ...utils_tests import get_tests_data_path
# from gpuphot.logger.hierarchical_logging import setup_logger, hierarchical_debug
# from gpuphot.phot.photo_gpu import process_image_new
# from gpuphot.stats.s_util import free_gpu_mem
import unittest

import numpy as np
from astropy.io import fits

from gpuphot.logger.hierarchical_logging import hierarchical_debug, setup_logger
from gpuphot.phot.photo_gpu import process_image_new
from gpuphot.stats.s_util import free_gpu_mem
from tests.gpuphot.utils_tests import get_tests_data_path

logger = setup_logger(__name__)


class TimeoutError(Exception):
    pass


@hierarchical_debug(logger)
def timeout_handler(signum, frame):
    raise TimeoutError("El tiempo límite del test ha sido excedido.")


class TestProcess_image_new(unittest.TestCase):

    def _process_image_v2(self, imdata, imheader, astrom=True, center_factor=0.3, ks=2, tile_section=500, border=50,
                          SP_filt=True,
                          pca_method=True, CR_filt=False):
        signal.signal(signal.SIGALRM, timeout_handler)
        timeout_duration = 10
        signal.alarm(timeout_duration)

        try:
            df_phot, imheader = process_image_new(imdata, imheader, center_factor=center_factor, ks=ks, astrom=astrom,
                                                  border=border, tile_section=tile_section, SP_filt=SP_filt,
                                                  pca_method=pca_method, CR_filt=CR_filt)

        except TimeoutError as e:
            logger.error(e)
            df_phot = None
        finally:
            signal.alarm(0)

        return df_phot, imheader

    def _task_caller(self, reduced_path):
        logger.info(f"Processing {reduced_path}")

        im = fits.getdata(reduced_path).astype(np.float32)
        headers = fits.getheader(reduced_path)

        # with fits.open(reduced_path) as ima:
        #     im = ima[0].data.astype(np.float32)
        #     headers = ima[0].header

        # Apply photometry
        if headers['INMODEL'] == 'iKon936':
            tile_section = 500
            center_factor = .9
            SP_filt = False
            pca_method = False
            CR_filt = False
            border = 50
        elif headers['INMODEL'] == 'QHY411MERIS':
            tile_section = 3000
            center_factor = .5
            SP_filt = True
            pca_method = True
            CR_filt = False
            border = 100

        phot_df, hwcs = self._process_image_v2(im, headers, astrom=True, tile_section=tile_section, border=border,
                                               center_factor=center_factor, SP_filt=SP_filt, pca_method=pca_method,
                                               CR_filt=CR_filt)

    def test_raw_1(self):

        directory_path = os.path.join(get_tests_data_path())

        logger.info(f'Root folder: {directory_path}')
        for (root, dirs, files) in os.walk(directory_path):
            for file in files:
                if file.endswith('.fits'):
                    try:
                        self._task_caller(os.path.join(root, file))
                    except Exception as e:
                        logger.error(f"Error processing {file}: {e}")
                        logger.exception(e)
                        free_gpu_mem()


if __name__ == '__main__':
    unittest.main()
