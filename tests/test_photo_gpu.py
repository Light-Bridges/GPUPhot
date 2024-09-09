import logging
import os
import time
import traceback
from unittest import TestCase

import numpy as np
from astropy.io import fits

from gpuphot.phot.photo_gpu import get_fwhm_model, get_detections, init_gpu
from gpuphot.stats.s_util import free_gpu_mem
from gpuphot.utils.astro import plate_scale_px

logger = logging.getLogger(__name__)

class Test(TestCase):
    def test_get_detections(self):

        directory_path = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')
        for (root, dirs, files) in os.walk(directory_path):
            for file in files:
                if file.endswith('.fits'):
                    if 'TTT1' in file:
                        image_path = os.path.join(root, file)
                        print(f'Processing {image_path}')
                        try:
                            print(f'Processing {image_path}')
                            start_time = time.time()

                            init_gpu()

                            model = get_fwhm_model()


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
                            print(f'Elapsed time for {file}: {end_time - start_time:.2f} seconds')
                        except Exception as e:
                            print(f'Error processing {image_path}: {e}')
                            print('Traceback:')
                            traceback.print_exc()
