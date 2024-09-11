import logging
import time

logging.basicConfig(level=logging.DEBUG, format=
'%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import gc
import logging
import cupy as cp
import numpy as np
from astroalign import find_transform
from astropy.io import fits
from cupyx.scipy.ndimage import shift
from skimage.transform._warps_cy import _warp_fast
from ..stats.reduction import center
from ..stats.subpixel import phase_cross_correlation as phase_cross_correlation_gpu

logger = logging.getLogger(__name__)


def dyn_avgstd(valuenew, nold, avgold, stdold):
    logger.debug(
        f'Iniciando función dyn_avgstd(valuenew={valuenew}, nold={nold}, avgold={avgold}, stdold={stdold})'
    )
    start_time = time.time()
    valuenew = cp.asarray(valuenew, dtype=np.double)
    nnew = nold + (valuenew != 0).astype(cp.int32)
    if cp.sum(nold) == 0:
        avgnew = cp.asarray(valuenew, dtype=np.double)
        stdnew = cp.zeros_like(valuenew, dtype=np.double)
    else:
        avgnew = avgold + (valuenew - avgold) / nnew
        stdnew = np.sqrt(nold / nnew * stdold ** 2 + (valuenew - avgnew) *
                         (valuenew - avgold) / nnew)
    logger.debug(
        f'Función dyn_avgstd completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return nnew, avgnew, stdnew


def reduction_atlas(cubes_path, size=2048, internal_shift=False):
    logger.debug(
        f'Iniciando función reduction_atlas(cubes_path={cubes_path}, size={size}, internal_shift={internal_shift})'
    )
    start_time = time.time()
    """
    Perform reduction on a list of FITS image cubes to produce a combined image.

    This function reads a series of FITS cubes, applies internal shifting if required, and combines them into a single averaged image. It uses GPU acceleration for computational efficiency.

    Parameters
    ----------
    cubes_path : list of str
        List of file paths to the FITS image cubes.
    size : int, optional
        Size of the subimage to be centered and processed, by default 2048.
    internal_shift : bool, optional
        Whether to apply internal shifting to align images within each cube, by default False.

    Returns
    -------
    tuple
        (avgnew, nnew, heads, temp_header) where:
        - avgnew is the final combined image.
        - nnew is the count of non-zero pixel contributions for each pixel.
        - heads is a list of headers from the FITS files.
        - temp_header is a dictionary with transformation parameters for the alignment.
    """
    heads = []
    for c, cube in enumerate(cubes_path):
        logger.info('Reading cube %s' % cube)
        with fits.open(cube) as ima:
            data = ima[1].data.astype(np.float32)
            ima_header = ima[1].header
        # if c == 0:
        im0cp = cp.asarray(data[0, :])
        nnew, avgnew = dyn_avg(im0cp, cp.zeros_like(im0cp), cp.
                               zeros_like(im0cp))
        logger.info('Shifting and combining')
        for j in range(1, data.shape[0]):
            im = cp.asarray(data[j, :])
            if internal_shift:
                shifted, _, _ = phase_cross_correlation_gpu(center(
                    im0cp, size), center(im, size))
                im = shift(im, shift=(shifted[0], shifted[1]), mode=
                'constant')
            nnew, avgnew = dyn_avg(im, nnew, avgnew)
            del im
        if c == 0:
            f0 = cp.nansum(center(avgnew, size)).get()
            heads = [ima_header]
            del im0cp, data
            gc.collect()
        else:
            # im1cp = cp.asarray(data[0, :])
            # nnew1, avgnew1 = dyn_avg(im1cp, cp.zeros_like(im1cp), cp.
            #                          zeros_like(im1cp))
            # logger.info('Shifting and combining')
            # for j in range(1, data.shape[0]):
            #     im = cp.asarray(data[j, :])
            #     if internal_shift:
            #         shifted, _, _ = phase_cross_correlation_gpu(center(
            #             im1cp, size), center(im, size))
            #         im = shift(im, shift=(shifted[0], shifted[1]), mode=
            #         'constant')
            #     nnew1, avgnew1 = dyn_avg(im, nnew1, avgnew1)
            #     del im
            del im0cp, data
            gc.collect()
            logger.info('Aligning')
            avgnew1_cpu = avgnew.get().astype(np.float32)
            transf, _ = find_transform(source=avgnew1_cpu, target=avgnew.
                                       get().astype(np.float32), min_area=100)
            matrix = np.linalg.inv(transf.params).astype(np.float32)
            alig = _warp_fast(avgnew1_cpu, matrix, output_shape=avgnew.
                              shape, order=3)
            nnew1 = _warp_fast(nnew1.get().astype(np.float32), matrix,
                               output_shape=avgnew.shape, order=3)
            nnew1 = cp.asarray(nnew1, dtype=cp.int16)
            im = cp.asarray(alig)
            fc = f0 / cp.nansum(center(im, size)).get()
            im *= fc
            nnew, avgnew = sum_dyn_avg(nnew, avgnew, nnew1, im)
            temp_header = {'ROT': np.round(transf.rotation * 180 / np.pi, 4
                                           ), 'SHX': np.round(transf.translation[1], 2), 'SHY': np.
            round(transf.translation[0], 2), 'ZOO': np.round(transf.
                                                             scale, 4), 'FC': np.round(fc, 4)}
            heads.append(ima_header)
            del avgnew, alig
            gc.collect()
    logger.info('Reduction finished')
    avgnew[avgnew < 0] = 0
    avgnew[avgnew > 2 ** 16] = 2 ** 16 - 1
    logger.debug(
        f'Función reduction_atlas completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return avgnew, nnew, heads, temp_header


def dyn_avg(valuenew, nold, avgold):
    logger.debug(
        f'Iniciando función dyn_avg(valuenew={valuenew}, nold={nold}, avgold={avgold})'
    )
    start_time = time.time()
    """
    Compute the dynamic average for image combination.

    Parameters
    ----------
    valuenew : cupy.ndarray
        New image values to be averaged.
    nold : cupy.ndarray
        Previous count of non-zero pixel contributions.
    avgold : cupy.ndarray
        Previous average image values.

    Returns
    -------
    tuple
        (nnew, avgnew) where nnew is the updated count of non-zero pixel contributions
        and avgnew is the updated average image values.
    """
    nnew = nold + (valuenew != 0).astype(cp.int16)
    if cp.sum(nold) == 0:
        logger.debug(
            f'Función dyn_avg completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
        )
        return nnew, valuenew
    else:
        avgnew = avgold + (valuenew - avgold) / nnew
        logger.debug(
            f'Función dyn_avg completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
        )
        return nnew, avgnew


def sum_dyn_avg(n1, avg1, n2, avg2):
    logger.debug(
        f'Iniciando función sum_dyn_avg(n1={n1}, avg1={avg1}, n2={n2}, avg2={avg2})'
    )
    start_time = time.time()
    """
    Combine two sets of dynamic averages.

    Parameters
    ----------
    n1 : cupy.ndarray
        First set of non-zero pixel contributions.
    avg1 : cupy.ndarray
        First set of average image values.
    n2 : cupy.ndarray
        Second set of non-zero pixel contributions.
    avg2 : cupy.ndarray
        Second set of average image values.

    Returns
    -------
    tuple
        (n_combined, avg_combined) where n_combined is the combined count of non-zero pixel contributions
        and avg_combined is the combined average image values.
    """
    n_combined = n1 + n2
    avg_combined = (n1 * avg1 + n2 * avg2) / n_combined
    logger.debug(
        f'Función sum_dyn_avg completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
    return n_combined, avg_combined
