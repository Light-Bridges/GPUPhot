import gc
import logging

import cupy as cp
import numpy as np
from astroalign import find_transform
from astropy.io import fits
from cupyx.scipy.ndimage import shift
from skimage.transform._warps_cy import _warp_fast

from cv.gpuphot.gpuphot.stats.reduction import center
from cv.gpuphot.gpuphot.stats.subpixel import phase_cross_correlation as phase_cross_correlation_gpu
from ttt.equipment.models import Header

logger = logging.getLogger(__name__)


def reduction_atlas(cubes_path, size=2048, internal_shift=False):
    # Read cubes
    heads = [];
    idx = []
    for c, cube in enumerate(cubes_path):

        logger.info('Reading cube %s' % cube)
        with fits.open(cube) as ima:
            data = ima[1].data.astype(np.float32)
            ima_header = ima[1].header

        if c == 0:
            im0cp = cp.asarray(data[0, :])
            nnew, avgnew = dyn_avg(im0cp, cp.zeros_like(im0cp), cp.zeros_like(im0cp))
            logger.info('Shifting and combining')
            for j in range(1, data.shape[0]):
                im = cp.asarray(data[j, :])
                if internal_shift:
                    shifted, _, _ = phase_cross_correlation_gpu(center(im0cp, size), center(im, size))
                    im = shift(im, shift=(shifted[0], shifted[1]), mode='constant')

                nnew, avgnew = dyn_avg(im, nnew, avgnew)
                del im

            f0 = cp.nansum(center(avgnew, size)).get()
            heads = [ima_header]

            del im0cp, data
            gc.collect()

        else:
            im1cp = cp.asarray(data[0, :])
            nnew1, avgnew1 = dyn_avg(im1cp, cp.zeros_like(im1cp), cp.zeros_like(im1cp))
            logger.info('Shifting and combining')
            for j in range(1, data.shape[0]):
                im = cp.asarray(data[j, :])
                if internal_shift:
                    shifted, _, _ = phase_cross_correlation_gpu(center(im1cp, size), center(im, size))
                    im = shift(im, shift=(shifted[0], shifted[1]), mode='constant')
                nnew1, avgnew1 = dyn_avg(im, nnew1, avgnew1)
                del im
            del im1cp, data
            gc.collect()

            logger.info('Aligning')
            avgnew1_cpu = avgnew1.get().astype(np.float32)
            transf, _ = find_transform(source=avgnew1_cpu, target=avgnew.get().astype(np.float32), min_area=100)
            matrix = np.linalg.inv(transf.params).astype(np.float32)
            alig = _warp_fast(avgnew1_cpu, matrix, output_shape=avgnew.shape, order=3)
            nnew1 = _warp_fast(nnew1.get().astype(np.float32), matrix, output_shape=avgnew.shape, order=3)
            nnew1 = cp.asarray(nnew1, dtype=cp.int16)

            im = cp.asarray(alig)
            fc = f0 / cp.nansum(center(im, size)).get()
            im *= fc
            nnew, avgnew = sum_dyn_avg(nnew, avgnew, nnew1, im)

            ima_header['ROT'] = (np.round(transf.rotation * 180 / np.pi, 4), Header.objects.get(name='ROT').description)
            ima_header['SHX'] = (np.round(transf.translation[1], 2), Header.objects.get(name='SHX').description)
            ima_header['SHY'] = (np.round(transf.translation[0], 2), Header.objects.get(name='SHY').description)
            ima_header['ZOO'] = (np.round(transf.scale, 4), Header.objects.get(name='ZOO').description)
            ima_header['FC'] = (np.round(fc, 4), Header.objects.get(name='FC').description)

            heads.append(ima_header)

            del avgnew1, alig
            gc.collect()

    logger.info('Reduction finished')

    avgnew[avgnew < 0] = 0
    avgnew[avgnew > 2 ** 16] = 2 ** 16 - 1

    return avgnew, nnew, heads


def dyn_avg(valuenew, nold, avgold):
    nnew = nold + (valuenew != 0).astype(cp.int16)
    if cp.sum(nold) == 0:
        return nnew, valuenew
    else:
        avgnew = avgold + (valuenew - avgold) / nnew
        return nnew, avgnew


def sum_dyn_avg(n1, avg1, n2, avg2):
    n_combined = n1 + n2
    avg_combined = (n1 * avg1 + n2 * avg2) / n_combined
    return n_combined, avg_combined
