import gc
import logging

import cupy as cp
import numpy as np
from astroalign import find_transform

# from mcs.utils.meteo import get_meteo
# from ttt.models import Header
from astropy.io import fits
from cupyx.scipy.ndimage import shift
from skimage.transform._warps_cy import _warp_fast

# from ttt.models import ObservingBlock, ObservingBlockLine
# from django.utils import timezone
from gpuphot.stats.reduction import center
from gpuphot.stats.subpixel import (
    phase_cross_correlation as phase_cross_correlation_gpu,
)

logger = logging.getLogger("cv")


def dyn_avgstd(valuenew, nold, avgold, stdold):
    valuenew = cp.asarray(valuenew, dtype=np.double)
    nnew = nold + (valuenew != 0).astype(cp.int32)
    if cp.sum(nold) == 0:
        avgnew = cp.asarray(valuenew, dtype=np.double)
        stdnew = cp.zeros_like(valuenew, dtype=np.double)
    else:
        avgnew = avgold + (valuenew - avgold) / nnew
        stdnew = np.sqrt(
            nold / nnew * stdold**2 + (valuenew - avgnew) * (valuenew - avgold) / nnew
        )
    return nnew, avgnew, stdnew


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


def reduction_atlas(cubes_path, size=2048, internal_shift=False):
    # Read cubes
    heads = []
    # idx = []
    for c, cube in enumerate(cubes_path):
        logger.info("Reading cube %s" % cube)
        with fits.open(cube) as ima:
            data = ima[1].data.astype(np.float32)
            ima_header = ima[1].header

        if c == 0:
            im0cp = cp.asarray(data[0, :])
            nnew, avgnew = dyn_avg(im0cp, cp.zeros_like(im0cp), cp.zeros_like(im0cp))
            logger.info("Shifting and combining")
            for j in range(1, data.shape[0]):
                im = cp.asarray(data[j, :])
                if internal_shift:
                    shifted, _, _ = phase_cross_correlation_gpu(
                        center(im0cp, size), center(im, size)
                    )
                    im = shift(im, shift=(shifted[0], shifted[1]), mode="constant")

                nnew, avgnew = dyn_avg(im, nnew, avgnew)
                del im

            f0 = cp.nansum(center(avgnew, size)).get()
            heads = [ima_header]

            del im0cp, data
            gc.collect()

        else:
            im1cp = cp.asarray(data[0, :])
            nnew1, avgnew1 = dyn_avg(im1cp, cp.zeros_like(im1cp), cp.zeros_like(im1cp))
            logger.info("Shifting and combining")
            for j in range(1, data.shape[0]):
                im = cp.asarray(data[j, :])
                if internal_shift:
                    shifted, _, _ = phase_cross_correlation_gpu(
                        center(im1cp, size), center(im, size)
                    )
                    im = shift(im, shift=(shifted[0], shifted[1]), mode="constant")
                nnew1, avgnew1 = dyn_avg(im, nnew1, avgnew1)
                del im
            del im1cp, data
            gc.collect()

            logger.info("Aligning")
            avgnew1_cpu = avgnew1.get().astype(np.float32)
            transf, _ = find_transform(
                source=avgnew1_cpu, target=avgnew.get().astype(np.float32), min_area=100
            )  # FIXMEavgnew is not defined
            matrix = np.linalg.inv(transf.params).astype(np.float32)
            alig = _warp_fast(avgnew1_cpu, matrix, output_shape=avgnew.shape, order=3)
            nnew1 = _warp_fast(
                nnew1.get().astype(np.float32),
                matrix,
                output_shape=avgnew.shape,
                order=3,
            )
            nnew1 = cp.asarray(nnew1, dtype=cp.int16)

            im = cp.asarray(alig)
            fc = f0 / cp.nansum(center(im, size)).get()  # FIXME f0 is not defined
            im *= fc
            nnew, avgnew = sum_dyn_avg(
                nnew, avgnew, nnew1, im
            )  # FIXME nnew is not defined

            ima_header["ROT"] = np.round(
                transf.rotation * 180 / np.pi, 4
            )  # , Header.objects.get(name='ROT').description)
            ima_header["SHX"] = np.round(
                transf.translation[1], 2
            )  # , Header.objects.get(name='SHX').description)
            ima_header["SHY"] = np.round(
                transf.translation[0], 2
            )  # , Header.objects.get(name='SHY').description)
            ima_header["ZOO"] = np.round(
                transf.scale, 4
            )  # , Header.objects.get(name='ZOO').description)
            ima_header["FC"] = np.round(
                fc, 4
            )  # , Header.objects.get(name='FC').description)

            heads.append(ima_header)

            del avgnew1, alig
            gc.collect()

    logger.info("Reduction finished")

    avgnew[avgnew < 0] = 0  # FIXME avgnew is not defined
    avgnew[avgnew > 2**16] = 2**16 - 1

    return avgnew, nnew, heads  # FIXME nnew is not defined


# def complete_header_atlas(header_list):
#     logger.info(f'creating atlas header')
#
#     deletions = ['PCOUNT',
#                  'GCOUNT',
#                  'SEQNUM',
#                  'NAXIS3',
#                  # 'CAMERA',
#                  'ORISIZEX',
#                  'ORISIZEY',
#                  'ALISIZEX',
#                  'ALISIZEY',
#                  'FINSIZEX',
#                  'FINSIZEY']
#
#     recursive_deletions = ['SEQNUM', 'TEMP', 'EXPT', 'PCDAT']
#
#     for head in header_list:
#
#         for d in deletions: del head[d]
#         for d in recursive_deletions:
#             for i in range(1, int(head['TOTIMA'] + 1)): del head['%s%i' % (d, i)]
#
#     header = header_list[0]
#     header.insert('EXPTIME', ('SUBEXPT', header['EXPTIME'], 'Sub-exposure time, in seconds'))
#     header['EXPTIME'] = header['EXPTIME'] * header['TOTIMA']
#
#     header.insert('UT1', ('COMMENT', '  *******  %s ******* ' % header['CAMERA'].upper()))
#     for i in range(1, int(header['TOTIMA'] + 1)):
#         header.rename_keyword('UT%i' % i, '%sUT%i' % (header['CAMERA'].upper(), i))
#
#     for head in header_list[1:]:
#         header.insert('UTOBS', ('COMMENT', '  *******  %s ******* ' % head['CAMERA'].upper()))
#
#         for p in ['ROT', 'SHX', 'SHY', 'ZOO', 'FC']:
#             header.insert('UTOBS', ('%s%s' % (head['CAMERA'].upper(), p),
#                                     head[p], head.comments[p]))
#
#         for i in range(1, int(head['TOTIMA'] + 1)):
#
#             if i == 1: header['CAMERA'] = header['CAMERA'] + ', ' + head['CAMERA']
#
#             header.insert('UTOBS', ('%sUT%i' % (head['CAMERA'].upper(), i),
#                                     head['UT%i' % i], head.comments['UT%i' % i]
#                                     ))
#
#     header.insert('UTOBS', ('COMMENT', ' *************************** '))
#     header.insert('UTOBS', ('STACKIN', 'Average', 'Stacking method'))
#     header['TOTIMA'] = len(header_list) * header['TOTIMA']
#     header['RDNOISE'] = np.round(header['RDNOISE'] * np.sqrt(header['TOTIMA']), 2)
#     header['GAIN'] = np.round(header['RDNOISE'] * np.sqrt(header['TOTIMA']), 4)
#     header['FILTER'] = 'w'
#     header.insert('POINTRA', ('RA-MNT', header['POINTRA'] * 15, '[deg] Right ascension (J2000) sent to telescope'))
#     header.insert('POINTRA', ('DEC-MNT', header['POINTDEC'], header.comments['POINTDEC']))
#     header['TELESCOP'] = 'ATLAS-TDO'
#     header.insert('RDNOISE', ('HSYNC', 46, Header.objects.get(name='HSYNC').description))
#
#     # Add weather data to the header
#     header['COMINIT'] = 'e'
#     header.insert('COMINIT', ('COMMENT', '***************************'))
#     header.insert('COMINIT', ('COMMENT', '        WEATHER DATA       '))
#     header.insert('COMINIT', ('COMMENT', '***************************'))
#
#     try:
#         for (i, k) in get_meteo(header['DATE-OBS'], header['TELESCOP']).items():
#             header[i] = (k, Header.objects.get(name=i).description)
#     except Exception as e:
#         logger.warning(f'exception with weather header')
#     del header['COMINIT']
#     logger.info(f'added weather data to header')
#
#     header['DATE'] = (datetime.datetime.utcnow().isoformat(), '[UTC] Date of file creation')
#
#     return header


# def get_block_number(proposal, start_date, end_date):
#     blocks = ObservingBlock.objects.filter(run__proposal=proposal, scheduled_time__gte=start_date,
#                                            scheduled_time__lte=end_date)
#     blocks = blocks.exclude(lines__status__in=[ObservingBlockLine.StatusChoices.failed,
#                                                ObservingBlockLine.StatusChoices.reduction_failed,
#                                                ObservingBlockLine.StatusChoices.prereduction_failed])
#     blocks = blocks.order_by('scheduled_time')
#     blocks = {block.id: i + 1 for i, block in enumerate(blocks)}
#     return blocks


# def get_date_info(block_id):
#     block = ObservingBlock.objects.get(id=block_id)
#     proposal_id = block.run.proposal.id
#     scheduled_time = block.scheduled_time.astimezone(timezone.utc)  # Convert to UTC timezone
#     if scheduled_time.time() > datetime.time(17, 0, 0):
#         start_date = timezone.make_aware(datetime.datetime.combine(scheduled_time.date(), datetime.time(17, 0, 0)),
#                                          timezone.utc)
#         end_date = timezone.make_aware(
#             datetime.datetime.combine(scheduled_time.date() + datetime.timedelta(days=1), datetime.time(12, 0, 0)),
#             timezone.utc)
#     else:
#         start_date = timezone.make_aware(
#             datetime.datetime.combine(scheduled_time.date() - datetime.timedelta(days=1), datetime.time(17, 0, 0)),
#             timezone.utc)
#         end_date = timezone.make_aware(datetime.datetime.combine(scheduled_time.date(), datetime.time(12, 0, 0)),
#                                        timezone.utc)
#     return proposal_id, start_date, end_date


# def get_block_order_position(block_id):
#     proposal_id, start_date, end_date = get_date_info(block_id)
#     seq_num = get_block_number(proposal_id, start_date, end_date)[block_id]
#     return seq_num


def create_inst_file(head):
    inst = fits.Header()
    for j in [
        "TELESCOP",
        "SITENAME",
        "SITECODE",
        "SITELAT",
        "SITELONG",
        "SITEALT",
        "DIAMETER",
        "FOCAL",
        "FOCALEN",
        "MOUNT",
        "TRACK",
        "RA-MNT",
        "DEC-MNT",
        "POINTRA",
        "POINTDEC",
        "HUMIDITY",
        "PRESSURE",
        "AMBTEMP",
        "CLOUD",
        "ILLUMINA",
        "WINDDIR",
        "WINDVEL",
        "DUSTPLA",
        "DEWPOINT",
        "DUSTPM1",
        "DUSTPM10",
        "DUSTPM25",
        "PWV",
        "MIRRTEMP",
        "MIRRHUM",
    ]:
        try:
            inst[j] = (head[j], head.comments[j])
        except:
            pass
    return inst


# def send_atlas_files(im, n, head):
#     line_id = head['OBLINEID']
#     obl = ObservingBlockLine.objects.get(id=line_id)
#     block_id = ObservingBlockLine.objects.get(id=line_id).block.id
#     try:
#         seq_num = get_block_order_position(block_id)
#     except:
#         logger.error(f'seq_number for block {block_id} not found')
#         raise
#     reduced_path = obl.reduced_image.path
#
#     data = np.array([im.get(), n.get()])
#
#     head['BITPIX'] = 16
#     data[data == 2 ** 16] = 2 ** 16 - 1
#     data = data.astype(np.uint16)
#     inst = create_inst_file(head)
#
#     try:
#         mjd = int(head['MJD-OBS'] + 0.501 + -1 / 24)
#     except:
#         jd, mjd = date_to_jd(head['DATE-OBS'])
#         head.insert('PCDATE', ('JD-OBS', jd, Header.objects.get(name='JD-OBS').description))
#         head.insert('PCDATE', ('MJD-OBS', mjd, Header.objects.get(name='MJD-OBS').description))
#         mjd = int(head['MJD-OBS'] + 0.501 + -1 / 24)
#
#     name = '05r%io%04dw0.stk' % (mjd, seq_num)
#
#     atlas_local_path = reduced_path.split('/')[:-3] + ['atlas-tdo', str(mjd)]
#     os.makedirs(('/').join(atlas_local_path), exist_ok=True)
#
#     atlas_local_file = ('/').join(atlas_local_path + [name])
#
#     HDU = fits.CompImageHDU(data, head)
#     HDU.writeto(atlas_local_file, overwrite=True)
#     inst_file_name = '05r%io%04dw.inst' % (mjd, seq_num)
#     with open(('/').join(atlas_local_path + [inst_file_name]), 'w') as f:
#         for line in inst.cards:
#             f.write(f"{line}\n")
#     logger.info(f'atlas file sent')
