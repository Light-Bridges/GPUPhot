import os
import time
import traceback
import unittest
from pathlib import Path
from unittest import TestCase

import numpy as np
from astropy.io import fits

from gpuphot.logger.hierarchical_logging import setup_logger
from gpuphot.phot.photo_gpu import get_detections, init_gpu
from gpuphot.stats.s_util import free_gpu_mem
from gpuphot.utils.astro import plate_scale_px

# from tensorflow.python.keras.models import load_model

logger = setup_logger(__name__)


class Test(TestCase):
    def get_fwhm_model(model_path=Path(__file__).parent.parent):
        import tensorflow as tf
        from tensorflow.python.keras.models import load_model
        # Obtener la ruta del directorio actual del script
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construir la ruta al directorio deseado
        name = os.path.join(current_dir, 'gpuphot', 'fwhm', 'fwhm_3_2_mofatt_ns_mix_100_model')

        if tf.__version__ == '2.4.1':
            name = name + '_old'

        logger.debug(f'Loading model {name}')

        model = load_model(name)
        return model

    def test_get_detections(self):

        directory_path = os.path.join(os.path.dirname(__file__), 'data')
        print(f'Processing {directory_path}')
        for (root, dirs, files) in os.walk(directory_path):
            for file in files:
                if file.endswith('.fits'):
                    if 'TTT1' in file:
                        image_path = os.path.join(root, file)
                        print(f'Processing {image_path}')
                        try:
                            print(f'Processing {image_path}')


                            model = self.get_fwhm_model()

                            init_gpu()

                            # Read reduced image
                            with fits.open(image_path) as ima:
                                im = ima[0].data.astype(np.float32)
                                headers = ima[0].header

                            # Apply photometry
                            gain = headers['GAIN']
                            rnois = headers['RDNOISE']
                            scale = np.round(plate_scale_px(headers['PXSIZE'], headers['FOCALEN']), 3)
                            # scale = head['SCALEORI']

                            dfm, sky, rms, fw, efw, ap = get_detections(model, im, det=3, gain=gain, rdnoise=rnois,
                                                                        scale=scale)
                            sm = np.mean(sky).tolist()
                            rm = np.mean(rms).tolist()

                            del sky, rms
                            free_gpu_mem()

                            logger.info(f'Detection: fw:{fw},sky:{sm},rms:{rm}', extra={'fw': fw, 'sky': sm, 'rms': rm})

                            # image_cp = cp.asarray(fits.getdata(image_path))
                            # resul = SP_filter_cupy(image_cp)
                            end_time = time.time()
                        except Exception as e:
                            print(f'Error processing {image_path}: {e}')
                            print('Traceback:')
                            traceback.print_exc()
if __name__ == '__main__':
    unittest.main()
