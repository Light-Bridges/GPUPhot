import gc
import logging
import os
import sys
from glob import glob

import astroalign as aa
import numpy as np
from astropy.io import fits

try:
    import cupy as cp

    gpu_enable = True
except Exception as e:
    print(f'Exception with cp: {e}')
    gpu_enable = False
try:
    from cupyx.scipy.ndimage import affine_transform, convolve, shift, gaussian_filter, binary_erosion
except Exception as e:
    pass
try:
    from cupyx.scipy.ndimage import fourier_shift
except Exception as e:
    pass
try:
    from src.phot.photo_gpu import get_sky, get_fwhm_mof, gen_moff_filter2, detect_gpu
except Exception as e:
    pass
try:
    from src.stats.subpixel import phase_cross_correlation as phase_cross_correlation_gpu
except Exception as e:
    pass
try:
    from src.stats.io import fitsloader, mkhdu
except Exception as e:
    pass
try:
    from src.stats.utils import free_gpu_mem
except Exception as e:
    pass

logger = logging.getLogger("pre_reduction")

############
# MRA
############

aa.NUM_NEAREST_NEIGHBORS = 5
aa.PIXEL_TOL = 3


class prered():

    def __init__(self, telescope, instrument, camera, calib_path):

        self.telescope = telescope
        # self.camera_db = db.Camera.objects.get(name=camera)
        self.instrument = instrument
        self.camera = camera

        self.calib_path = os.path.expanduser(calib_path)
        if not os.path.exists(self.calib_path):
            os.makedirs(self.calib_path)

        # Check if GPU is enable. If so, define cupy as working package
        self.gpu_enable = gpu_enable
        if self.gpu_enable:
            self.xp = cp
            free_gpu_mem()
        else:
            self.xp = np
        gc.collect()

        # Initialice load function
        self.load_masters(None, None, None)
        logger.info(f'prered class compiled...',
                    extra={'telescope': telescope, 'instrument': instrument, 'camera': camera})

    def load_masters(self, masterbias_path, masterdark_path, masterflat_path):
        """
        Load MasterBias (and calculate overscan median, in case), MasterDark and MasterFlats.
        Input:
            masterbias_path   :  Absolute path to the MasterBias (str)
            masterdark_path   :  Absolute path to the MasterDark (str)
            masterflat_path   :  Absolute path to the MasterFlat (str)
        """
        # Load MasterBias
        if masterbias_path is None:
            self.masterbias, self.header_masterbias = None, None
        else:
            try:
                self.masterbias, self.header_masterbias = fitsloader(masterbias_path, hdu_index=0,
                                                                     dtype=self.xp.float32, xp=self.xp)
                self.masterbias[self.xp.isnan(self.masterbias)] = 0
                # Calculate MasterBias overscan median
                try:
                    if self.header_masterbias['OVERSCNX'] != 0 and self.header_masterbias['OVERSCNY'] != 0:
                        ix, nx, iy, ny = (self.header_masterbias['OVERSCSX'], self.header_masterbias['OVERSCNX'],
                                          self.header_masterbias['OVERSCSY'], self.header_masterbias['OVERSCNY'])
                        self.bias_overscan_med = self.xp.median(self.masterbias[iy:iy + ny, ix:ix + nx]).item()
                    else:
                        self.bias_overscan_med = None
                except:
                    self.bias_overscan_med = None
            except Exception as e:
                logger.error(f'MasterBias exception: {e}',
                             extra={'telescope': self.telescope, 'instrument': self.instrument, 'camera': self.camera})
                self.masterbias, self.header_masterbias = None, None

        # Load MasterDark
        if masterdark_path is None:
            self.masterdark, self.header_masterdark = None, None
        else:
            try:
                self.masterdark, self.header_masterdark = fitsloader(masterdark_path, hdu_index=0,
                                                                     dtype=self.xp.float32, xp=self.xp)
                self.masterdark[self.xp.isnan(self.masterdark)] = 0
            except:
                logger.warning(f'MasterDark not found in %s' % masterdark_path,
                               extra={'telescope': self.telescope, 'instrument': self.instrument,
                                      'camera': self.camera})
                self.masterdark, self.header_masterdark = None, None

        # Load MasterFlat
        if masterflat_path is None:
            self.masterflat, self.header_masterflat = None, None
        else:
            try:
                self.masterflat, self.header_masterflat = fitsloader(masterflat_path, hdu_index=0,
                                                                     dtype=self.xp.float32, xp=self.xp)
                self.masterflat[self.xp.isnan(self.masterflat)] = 1
            except:
                logger.warning(f'MasterFlat not found in %s' % masterflat_path,
                               extra={'telescope': self.telescope, 'instrument': self.instrument,
                                      'camera': self.camera})
                self.masterflat, self.header_masterflat = None, None

    def imcombine(self, data, overscan, texp, norm=False, it=5, n=3):
        """
        Stack data cube using a sigma-clipped average adding frames one by one.

        Input:
            data         :  Data cube with dimension [N, nx, ny] (numpy.ndarray)
            overscan     :  Overscan region (ix, nx, iy, ny) (tuple of ints)
            texp         :  Exposure time (float)
            norm         :  True if corrected image is normed, False if not (bool)
            it           :  Number of iterations in sigma-clipping (int)
            n            :  Low and high sigma clipping factor (int)
        Output:
            center       :  Stacked average image with dimension [nx, ny] (numpy.ndarray or cupy.ndarray)
            sigma        :  Stacked std. image with dimension [nx, ny] (numpy.ndarray or cupy.ndarray)
            overm        :  Mean of median on the overscan region of the data cube (float)
            overs        :  Mean of standard deviation on the overscan region of the data cube (float)
        """

        im0 = data[0, :]
        delta0 = -1
        iplus = self.xp.zeros_like(im0, dtype=self.xp.float32) + 1.e10
        iminu = self.xp.zeros_like(im0, dtype=self.xp.float32) - 1.e10

        for j in range(it):
            center = self.xp.zeros_like(im0, dtype=self.xp.double)
            sigma = self.xp.zeros_like(im0, dtype=self.xp.double)
            mask = self.xp.zeros_like(im0, dtype=self.xp.float32)
            delta = 0
            overml = []
            oversl = []
            for i in range(data.shape[0]):

                im, overm, overs = self.correct_biasflat(data[i, :], overscan, texp, norm)

                mk = (im >= iminu)
                mk = mk * (im <= iplus)
                im = im * mk
                center = center + im
                sigma = sigma + im * im
                delta = delta + self.xp.sum(mk == 0)
                mask = mask + mk
                overml.append(overm)
                oversl.append(overs)

                del im, mk
                if self.gpu_enable:
                    free_gpu_mem()
                gc.collect()

            if None not in overml:
                overm = np.mean(overml)
            else:
                overm = None
            if None not in oversl:
                overs = np.mean(oversl)
            else:
                overs = None

            center = center / mask
            sigma = sigma / mask
            sigma = self.xp.sqrt(sigma - center * center)
            if delta == delta0:
                break
            delta0 = delta
            iplus = center + n * sigma
            iminu = center - n * sigma

        del iplus, iminu, im0, delta0, mask, delta
        if self.gpu_enable:
            free_gpu_mem()
        gc.collect()

        return center, sigma, overm, overs  # FIXME center, sigma, overm, overs reference before assignment

    def correct_biasflat(self, im, overscan, texp, norm):
        """
        Correct raw image from pedestal (bias) and flat field:
        im_corr = (im - bias - dark) / flat

        Input:
            im           :  Raw image (numpy.ndarray or cupy.ndarray)
            overscan     :  Overscan region (ix, nx, iy, ny) (tuple of ints)
            texp         :  Exposure time of raw image (float)
            norm         :  True if corrected image is normed, False if not (bool)
        Output:
            im           :  Corrected image (numpy.ndarray or cupy.ndarray)
            overm        :  Median on the overscan region (float)
            overs        :  Standard deviation on the overscan region (float)
        """
        im = self.xp.asarray(im, dtype=self.xp.float32)

        if self.masterbias is not None or overscan is not None:
            im, overm, overs = self.sustract_pedestal(im, overscan)
        else:
            overm = None
            overs = None
        if self.masterdark is not None:
            im = im - self.masterdark * texp
        else:
            overm = None
            overs = None
        if self.masterflat is not None:
            im = im / self.masterflat
        if norm:
            cx, cy = int(im.shape[0] / 2), int(im.shape[1] / 2)
            lx, ly = int(cx * 0.5), int(cy * 0.5)
            im = im / self.xp.mean(im[cx - lx:cx + lx, cy - ly:cy + ly]).item()

        return im, overm, overs

    def sustract_pedestal(self, data, overscan):
        """
        Sustract pedestal (bias or bias+dark) from data array using a MasterBias frame, the overscan region
        or both of them combined.

        Input:
            data           :  Data array to be corrected (numpy.ndarray or cupy.ndarray)
            overscan       :  Overscan region (ix, nx, iy, ny) (tuple of ints)
            dark_factor    :  Ratio between raw image and dark exposure time
        Output:
            data           :  Corrected data array (numpy.ndarray or cupy.ndarray)
            overscan_med   :  Mean on the overscan region (float)
            overscan_std   :  Standard deviation on the overscan region (float)
        """
        if self.masterbias is None and overscan is None:
            logger.warning(f'MasterBias frame or overscan region must be included',
                           extra={'telescope': self.telescope, 'instrument': self.instrument, 'camera': self.camera})
            overscan_med = None
            overscan_std = None

        elif self.masterbias is None and overscan is not None:
            ix, nx, iy, ny = overscan
            overscan_med = self.xp.mean(data[iy:iy + ny, ix:ix + nx])
            data = data - overscan_med
            overscan_med = overscan_med.item()
            overscan_std = self.xp.std(data[iy:iy + ny, ix:ix + nx]).item()

        elif self.masterbias is not None:
            if data.shape != self.masterbias.shape:
                logger.warning(f'Data and bias frame have different shapes!',
                               extra={'telescope': self.telescope, 'instrument': self.instrument,
                                      'camera': self.camera})
                overscan_med = None
                overscan_std = None
            else:
                try:
                    ix, nx, iy, ny = overscan
                    overscan_med = self.xp.mean(data[iy:iy + ny, ix:ix + nx])
                    factor = overscan_med - self.bias_overscan_med
                    overscan_med = overscan_med.item()
                    overscan_std = self.xp.std(data[iy:iy + ny, ix:ix + nx]).item()
                except:
                    factor = 0.
                    overscan_med = None
                    overscan_std = None
                data = data - self.masterbias + factor

        else:
            logger.warning(f'Pedestal not subtracted',
                           extra={'telescope': self.telescope, 'instrument': self.instrument, 'camera': self.camera})
            overscan_med = None
            overscan_std = None
        return data, overscan_med, overscan_std


#################


def build_master(dir_name, crop):
    files = glob(dir_name + '/Bias/*.fits')
    fc, hb = read_images(files, crop)
    siz = np.prod(fc.shape) * sys.getsizeof(fc[0, 0, 0]) / 1024 / 1024 / 1024 / 8
    logger.info("Master bias " + str(fc.shape[0]) + 'files Size = ' + str(siz) + 'GB',
                extra={'nbias': fc.shape, 'siz': siz})

    # print( str(siz)+ ' GB')
    mbiasg, _ = stack_sigmaclip(fc, it=5, n=3)
    mbias = mbiasg.get()
    del (fc, mbiasg)
    free_gpu_mem()
    hdu = fits.PrimaryHDU(mbias)
    # hdu.header = hb[
    hdu.writeto(dir_name + '/master_bias.fits', overwrite=True)

    files = glob(dir_name + '/Flat/*.fits')
    fc, hf = read_images(files, crop)
    siz = np.prod(fc.shape) * sys.getsizeof(fc[0, 0, 0]) / 1024 / 1024 / 1024 / 8
    logger.info("Master flat " + str(fc.shape[0]) + 'files Size = ' + str(siz) + 'GB',
                extra={'nflat': fc.shape, 'siz': siz})

    mflatg, hs = stack_sigmaclip(fc, it=5, n=3, master=True, mbias=mbias)
    mflat = mflatg.get()
    del (fc, mflatg)
    free_gpu_mem()
    hdu = fits.PrimaryHDU(mflat)
    # hdu.header = hs[0]
    hdu.writeto(dir_name + '/master_flat.fits', overwrite=True)


def reduce_cube(dir_name, crop, register=True, fcoadd=False, model=None):
    file = dir_name + '/master_bias.fits'
    with fits.open(file) as ima:
        mbias_c = ima[0].data.astype(np.float32)

    file = dir_name + '/master_flat.fits'
    with fits.open(file) as ima:
        mflat_c = ima[0].data.astype(np.float32)

    files = np.sort(glob(dir_name + '/Science/*.fits'))
    fc, hs = read_images(files, crop)
    siz = np.prod(fc.shape) * sys.getsizeof(fc[0, 0, 0]) / 1024 / 1024 / 1024 / 8
    logger.info("Cube " + str(fc.shape[0]) + 'files Size = ' + str(siz) + 'GB', extra={'nbias': fc.shape, 'siz': siz})

    mbias = cp.asarray(mbias_c, dtype=cp.float32)
    mflat = cp.asarray(mflat_c, dtype=cp.float32)

    for i in range(fc.shape[0]):
        im = cp.asarray(fc[i, :, :], dtype=cp.float32)
        fc[i, :, :] = ((im - mbias) / mflat).get()
    del (im, mbias, mflat)  # FIXME im reference before assignment
    free_gpu_mem()

    if register:
        fc = register_shift(fc)
    if fcoadd:
        nmax = 5000
        im = coadd(fc[:, nmax:, nmax:], model, minpix=5)
    else:
        im, _ = stack_sigmaclip(fc, it=5, n=3)

    free_gpu_mem()
    return im, hs[0]


def read_images(files, crop, ):
    x0, x1, y0, y1 = crop
    with fits.open(files[0]) as ima:
        imc = ima[0].data[x0:-x1:, y0:-y1:].astype(np.float32)
    cube = np.zeros([len(files)] + list(imc.shape))
    cube[0] = imc
    heads = [ima[0].header]
    for i, file in enumerate(files):
        print(file)
        if i > 0:
            try:
                with fits.open(file) as ima:
                    imc = ima[0].data[x0:-x1:, y0:-y1:].astype(np.float32)
                    cube[i] = imc
                    heads.append(ima[0].header)
            except:
                next()
    return cube, heads


def stack_sigmaclip(data, it=5, n=3, master=False, mbias=None, alpha=False, beta=False, tim=None):
    nim = data.shape[0]
    if tim == None:
        w = cp.ones(nim, dtype=cp.float32)
        f = 1.0
    else:
        w = cp.asarray(tim) / np.sum(tim)
        f = nim

    delta0 = -1
    iplus = cp.zeros_like(data[0], dtype=cp.float32) + 1.e10
    iminu = cp.zeros_like(data[0], dtype=cp.float32) - 1.e10
    if alpha:
        gf, lk = gen_moff_filter2(alpha, beta)

    for iit in range(it):
        center = cp.zeros_like(data[0], dtype=cp.float32)
        sigma = cp.zeros_like(data[0], dtype=cp.float32)
        mask = cp.zeros_like(data[0], dtype=cp.float32)
        delta = 0
        for i in range(data.shape[0]):
            im = cp.asarray(data[i, :], dtype=cp.float32)
            if master:
                im = im - cp.asarray(mbias, dtype=cp.float32)
                im = im / cp.mean(im)
            if alpha:
                im = convolve(im, gf, origin=(0, 0))  # FIXME gf reference before assignment

            mk = (im >= iminu)
            mk = mk * (im <= iplus)
            im = im * mk
            center = center + im * w[i]
            sigma = sigma + im * im * w[i]
            delta = delta + cp.sum(mk == 0)
            mask = mask + mk * w[i]
        center = center / mask
        sigma = sigma / mask
        sigma = sigma - center * center
        sigma[sigma < 1.0] = 1.0
        sigma = cp.sqrt(sigma)
        if delta == delta0:
            break
        delta0 = delta
        iplus = center + n * sigma
        iminu = center - n * sigma
        print('it=', iit, 'masked ', delta)
    del (iplus, iminu, im, mask)  # FIXME im and mask reference before assignment
    free_gpu_mem()
    center = center * f  # FIXME center reference before assignment
    sigma = sigma * f  # FIXME sigma reference before assignment
    return center, sigma


def coadd(data, model, minpix=5):
    im0 = data[0]
    p_hat = cp.zeros_like(im0, dtype=cp.complex128)
    s_hat = cp.zeros_like(im0, dtype=cp.complex128)
    ws = 0
    wp = 0
    ss = 0
    for i in range(data.shape[0]):
        # print(i)
        fw, efw, alpha, beta = get_fwhm_mof(model, data[i], step=50, ns=50)
        gf, lk = gen_moff_filter2(alpha, beta)
        im2 = cp.asarray(data[i])
        sky, rms, _ = get_sky(im2, fw, qt=99)

        if minpix > 0:
            it0, mask = get_coord_obj(im2, sky, rms, alpha=alpha, beta=beta, n=40, pix_cut=20, minpix=minpix)
            im2[mask] = sky[mask]

        Fj = cp.median(im2)

        sigj = cp.median(rms)
        ss += sigj
        p_hatj = cp.fft.fft2(gf, s=im2.shape)
        p_hatj = fourier_shift(p_hatj, (-lk, -lk))
        wsj = Fj / sigj ** 2
        wpj = (Fj / sigj) ** 2
        ws += wsj
        wp += wpj
        p_hat += p_hatj * p_hatj.conj() * wpj
        fim = cp.fft.fft2(im2)
        s_hat += fim * p_hatj.conj() * wsj

    free_gpu_mem()

    P_r_hat = cp.sqrt(p_hat)
    R = cp.real(cp.fft.ifft2(s_hat / P_r_hat)) * ss / cp.sqrt(data.shape[0]) / data.shape[0]
    return R


def get_coord_obj(im, sky, rms, alpha, beta, n=20, pix_cut=10, minpix=4):
    dfm, mask, max_mem = detect_gpu(im, sky, rms, 5, mode='m', alpha=alpha, beta=beta, mincut=pix_cut, minpix=minpix)
    dfm = dfm[dfm.npix > pix_cut].sort_values('flux', ascending=False)[:n]
    co1 = np.asarray(dfm[['xcentroid', 'ycentroid']])
    return co1, mask


def center(im, size):
    if im.shape[0] > size:
        c0 = int((im.shape[0] - size) / 2)
    if im.shape[1] > size:
        c1 = int((im.shape[1] - size) / 2)
    return cp.asarray(im[c0:-c0:, c1:-c1:])  # FIXME c0 and c1 referenced before assignment


def register_shift(fc, uf=100, n=1000):
    fc1 = fc.copy()
    im0 = center(fc[0], n)
    im0 = binary_erosion(im0 > (im0.mean() + im0.std()))
    for i in np.arange(1, fc.shape[0]):
        im1 = center(fc[i], n)
        im1 = binary_erosion(im1 > (im1.mean() + im1.std()))
        shifted, _, _ = phase_cross_correlation_gpu(im0.get(), im1.get(), upsample_factor=uf)
        if (np.abs(shifted[0]) > 300) | np.abs((shifted[1]) > 300):
            # shifted, _, _ = phase_cross_correlation_gpu(fc[0],fc[i],upsample_factor=uf,reference_mask = ref_mas, overlap_ratio = .9)
            shifted = (0, 0)
        fc1[i] = shift(cp.asarray(fc[i]), shift=(shifted[0], shifted[1]), order=1, mode='constant').get()

        print(f"Detected subpixel offset (y, x): {shifted}")
    return fc1


def register_shift_2(fc0, fc1, uf=100):
    shifted, error, diffphase = phase_cross_correlation_gpu(fc0, fc1, upsample_factor=uf)
    fcs = shift(cp.asarray(fc1), shift=(shifted[0], shifted[1]), mode='constant').get()
    return fcs, shifted


def register(fc, model, n=30, cut=10, minpix=20):
    fc2 = fc.copy()
    im = cp.asarray(fc[0, :, :])
    fw, efw, alpha, beta = get_fwhm_mof(model, im.get(), step=50, ns=50)
    sky, rms, _ = get_sky(im, fw, qt=85)

    it0, mask = get_coord_obj(im, sky, rms, alpha=alpha, beta=beta, n=n, pix_cut=cut, minpix=minpix)
    if minpix > 0:
        fc2[0, :, :][mask.get()] = sky[mask].get()
    for i in np.arange(1, fc.shape[0]):
        print(i)
        im1 = cp.asarray(fc[i, :, :])
        it1, mask = get_coord_obj(im1, sky, rms, alpha=alpha, beta=beta, n=n, pix_cut=cut, minpix=minpix)
        if minpix > 0:
            im1[mask] = sky[mask]
        # print(it0,it1)
        transf, a = aa.find_transform(it1, it0, max_control_points=n)
        # print(transf.params)
        fc2[i, :, :] = affine_transform(im1, cp.asarray(transf.params)).get()

    return fc2


def register_atlas(tgt, sc, model, cut=10, minpix=4):
    im = cp.asarray(tgt)
    fw, efw, alpha, beta = get_fwhm_mof(model, tgt, step=50, ns=50)
    sky, rms, _ = get_sky(im, fw, qt=85)

    f0 = cp.sum((im - sky.get())[int(im.shape[0] / 2 - 1024):int(im.shape[0] / 2 + 1024),
                int(im.shape[1] / 2 - 1024):int(im.shape[1] / 2 + 1024)])

    it0, mask = get_coord_obj(im, sky, rms, alpha=alpha, beta=beta, n=40, pix_cut=cut, minpix=minpix)
    if minpix > 0: tgt[mask.get()] = sky[mask].get()

    fwtgt = fw

    im = cp.asarray(sc)
    fw, efw, alpha, beta = get_fwhm_mof(model, sc, step=50, ns=50)
    sky, rms, _ = get_sky(im, fw, qt=85)

    f = cp.sum((im - sky.get())[int(im.shape[0] / 2 - 1024):int(im.shape[0] / 2 + 1024),
               int(im.shape[1] / 2 - 1024):int(im.shape[1] / 2 + 1024)])

    it1, mask = get_coord_obj(im, sky, rms, alpha=alpha, beta=beta, n=40, pix_cut=cut, minpix=minpix)
    if minpix > 0: im[mask] = sky[mask]

    transf, a = aa.find_transform(it1, it0, max_control_points=30)
    sc = affine_transform(im, cp.asarray(transf.params)).get() * f0 / f

    return tgt, fwtgt, sc, fw, f0 / f


def register2(fc):
    # im = cp.asarray(fc[0, :, :])
    for i in np.arange(1, fc.shape[0]):
        print(i)
        fc[i, :, :], _ = aa.register(fc[0, :, :], fc[i, :, :])

    return fc
