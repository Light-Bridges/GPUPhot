import gc
import time
import traceback

import cupy as cp
import numpy as np
import pandas as pd
import tensorflow as tf
from cupyx.scipy.ndimage import gaussian_filter, convolve, label, sum as nd_sum, mean as nd_mean, maximum_filter, \
    median_filter
from matplotlib import pyplot as plt
from sklearn.linear_model import RANSACRegressor

from .catalog import crossmatch_sources
from .convo import fill_image, get_aper_kernel, convolve_fft, gen_apm_filter
from ..logger.hierarchical_logging import setup_logger, hierarchical_debug
from ..stats.s_util import free_gpu_mem

logger = setup_logger(__name__)


# @hierarchical_debug(logger)
# def get_solver():
#     """
#     Get the astrometry solver with index files.
#
#     Returns
#     -------
#     astrometry.Solver
#         Configured astrometry solver instance.
#     """
#     if os.path.exists('/data'):
#         cache = '/data/astrometry_cache'
#     else:
#         cache = '/mnt/data/astrometry_cache'
#     solver = astrometry.Solver(astrometry.series_5200.index_files(
#         cache_directory=cache, scales={0, 1, 2, 3, 4, 5, 6}) + astrometry.
#                                series_4100.index_files(cache_directory=cache, scales={7, 8, 9, 10,
#                                                                                       11}))
#
#     return solver


@hierarchical_debug(logger)
def init_gpu():
    """ """
    logger.debug('Tensorflow version ' + tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    logger.debug('GPUs:', gpus)
    tf.config.set_logical_device_configuration(gpus[0], [tf.config.
                                               LogicalDeviceConfiguration(memory_limit=1024)])


@hierarchical_debug(logger)
def get_detections(model_mo, im, det=2, gain=1.024, rdnoise=2.3, scale=0.21):
    """Detect sources in an image using a model.

    :param model_mo: Model to use for detection.
    :type model_mo: Model
    :param im: Input image.
    :type im: ndarray
    :param det: Detection threshold, by default 2.
    :type det: float, optional
    :param gain: Gain value, by default 1.024.
    :type gain: float, optional
    :param rdnoise: Read noise value, by default 2.3.
    :type rdnoise: float, optional
    :param scale: Pixel scale in arcsec/pixel, by default 0.21.
    :type scale: float, optional

    
    """
    fw, efw, alpha, beta = get_fwhm_mof(model_mo, im, step=50, ns=50)
    logger.debug('FWHM = ' + str(fw))
    im_g = cp.asarray(im)
    sky, rms, _ = get_sky(im_g, fw, qt=80)
    logger.debug('Sky:', sky.mean(), ' RMS: ', rms.mean())
    dfm, lk = daofind_gpu_fast(im_g, sky, rms, det, mode='m', alpha=alpha,
                               beta=beta)
    dfm['snr'] = dfm.flux_a * gain / np.sqrt((dfm.flux_a + dfm.sks) * gain +
                                             rdnoise ** 2 * np.pi * lk ** 2)
    dfm = dfm[dfm.xcentroid == dfm.xcentroid]
    dfm = dfm[dfm.flux_a == dfm.flux_a]
    a = dfm.peak / dfm.flux_a
    dfm = dfm[a < scale / 2]
    dfm['idx'] = np.round(dfm.xcentroid) * im.shape[0] + np.round(dfm.ycentroid
                                                                  )
    del im_g
    free_gpu_mem()

    return dfm, sky, rms, fw, efw, 2 * lk + 1


@hierarchical_debug(logger)
def daofind_gpu_fast(img, sky, rms, sdet, mode='g', fw=0, alpha=0, beta=0,
                     mem=cp.get_default_pinned_memory_pool()):
    """Detect sources in an image using DAOFind algorithm on GPU.

    :param img: Input image.
    :type img: ndarray
    :param sky: Sky background.
    :type sky: ndarray
    :param rms: RMS noise.
    :type rms: ndarray
    :param sdet: Detection threshold.
    :type sdet: float
    :param mode: Mode for detection ('g' for Gaussian, 'm' for Moffat), by default 'g'.
    :type mode: str, optional
    :param fw: Full width at half maximum, by default 0.
    :type fw: float, optional
    :param alpha: Alpha parameter for Moffat filter, by default 0.
    :type alpha: float, optional
    :param beta: Beta parameter for Moffat filter, by default 0.
    :type beta: float, optional
    :param mem: Memory pool, by default cp.get_default_pinned_memory_pool().
    :type mem: cupy.cuda.memory.PinnedMemoryPool, optional

    
    """
    thres = sdet * cp.mean(rms)
    sigma_r = fw / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if mode == 'g':
        gf, lk = gen_gauss_filter(sigma_r)
    else:
        gf, lk = gen_moff_filter(alpha, beta)
    sky = gaussian_filter(sky, 3 * lk)
    rms = gaussian_filter(rms, 3 * lk)
    g = convolve(img - sky, gf, origin=(0, 0))
    g1 = (g / rms > sdet).astype(cp.int32)
    lbs = label(g1)
    ids = cp.asarray([range(lbs[1])])
    npix = nd_sum(g1, lbs[0], ids)
    del g1
    mem.free_all_blocks()
    idx = cp.indices(img.shape, dtype=cp.int16)
    im1 = g * idx
    x = nd_mean(im1[0, :, :], lbs[0], ids) / nd_mean(g, lbs[0], ids) + 1
    y = nd_mean(im1[1, :, :], lbs[0], ids) / nd_mean(g, lbs[0], ids) + 1
    coor = cp.round(x).astype(cp.int), cp.round(y).astype(cp.int)
    lk = np.round(lk * 3).astype(np.int)
    sk_s = (sky[coor] - cp.mean(sky)) / rms[coor]
    del rms
    mem.free_all_blocks()
    k_app = gen_ap_filter(lk)
    fot_p = maximum_filter(g, footprint=k_app, origin=(0, 0))
    flux_p = fot_p[coor] / thres
    del g, fot_p
    mem.free_all_blocks()
    mpea = maximum_filter(img - sky, footprint=k_app, origin=(0, 0))
    peak = mpea[coor]
    del mpea
    fot_a = convolve(img - sky, k_app, origin=(0, 0))
    flux_a = fot_a[coor]
    del fot_a, k_app
    mem.free_all_blocks()
    res = np.asarray([y.get(), x.get(), peak.get(), flux_a.get(), flux_p.
                     get(), npix.get(), sk_s.get()]).transpose().reshape((-1, 7))
    df = pd.DataFrame(res, columns=['xcentroid', 'ycentroid', 'peak',
                                    'flux_a', 'flux_p', 'npix', 'sks'])
    df = df[df.flux_a > 0]

    return df, lk


@hierarchical_debug(logger)
def gen_ap_filter(lk):
    """Generate an aperture filter.

    :param lk: Aperture radius.
    :type lk: int

    
    """
    k_dim = 2 * lk + 1, 2 * lk + 1
    indi = cp.indices(k_dim)
    fw2 = lk ** 2
    k_app = cp.zeros(k_dim)
    struc = cp.where((lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2 <
                     fw2)
    k_app[struc] = 1

    return k_app


@hierarchical_debug(logger)
def gen_moff_filter(alpha, beta):
    """Generate a Moffat filter.

    :param alpha: Alpha parameter for Moffat filter.
    :type alpha: float
    :param beta: Beta parameter for Moffat filter.
    :type beta: float

    
    """
    fw = alpha * (2 * np.sqrt(2 ** (1 / beta) - 1))
    sigma_r = fw / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    lk = np.ceil(sigma_r).astype(np.int16) * 4
    k_dim = 2 * lk + 1, 2 * lk + 1
    indi = cp.indices(k_dim)
    r2 = (lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2
    ker = (1 + r2 / alpha ** 2) ** -beta
    ksum = cp.sum(ker)
    ksum2 = cp.sum(ker * ker)
    n = k_dim[0] ** 2
    k_app = (ker - ksum / n) / (ksum2 - ksum * ksum / n)

    return k_app, lk


@hierarchical_debug(logger)
def get_sky(im_g, fw, qt=90, mem=cp.get_default_memory_pool()):
    """Estimate the sky background and RMS noise.

    :param im_g: Input image.
    :type im_g: ndarray
    :param fw: Full width at half maximum.
    :type fw: float
    :param qt: Quantile for sky estimation, by default 90.
    :type qt: float, optional
    :param mem: Memory pool, by default cp.get_default_memory_pool().
    :type mem: cupy.cuda.memory.PinnedMemoryPool, optional

    
    """
    sigma_r = fw / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    lk = np.ceil(sigma_r).astype(np.int16) * 4
    k_app = gen_apm_filter(2 * lk)
    fot_m = convolve(im_g, k_app, origin=(0, 0))
    fot_m2 = convolve(im_g * im_g, k_app, origin=(0, 0))
    del k_app
    fot_m2 = cp.sqrt(fot_m2 - fot_m * fot_m)
    cut1 = cp.percentile((fot_m * fot_m2).flatten(), qt)
    mask = fot_m * fot_m2 > cut1
    mm = mem.used_bytes()
    fot_m[mask] = None
    fot_m2[mask] = None
    del (cut1, mask)
    fot_m = cov_nan(fot_m, 20)
    fot_m2 = cov_nan(fot_m2, 20)

    return fot_m, fot_m2, mm


@hierarchical_debug(logger)
def SP_filter_cupy(img, filter_size=3, high_threshold_factor=10,
                   low_threshold_factor=5, scaling_factor=1.4826):
    """Apply a median filter to remove salt-and-pepper noise.

    :param img: Input image.
    :type img: cupy.ndarray
    :param filter_size: Size of the median filter (default is 3).
    :type filter_size: int, optional
    :param high_threshold_factor: Factor to determine the high threshold for noise detection (default is 10).
    :type high_threshold_factor: float, optional
    :param low_threshold_factor: Factor to determine the low threshold for noise detection (default is 5).
    :type low_threshold_factor: float, optional
    :param scaling_factor: Scaling factor for estimating the standard deviation from MAD (default is 1.4826).
    :type scaling_factor: float, optional

    
    """
    med_filter = median_filter(img, size=filter_size)
    dif = img - med_filter
    med = cp.nanmedian(dif)
    ms = scaling_factor * cp.nanmedian(cp.abs(dif - med))
    mask = (dif > med + high_threshold_factor * ms) | (dif < med -
                                                       low_threshold_factor * ms)
    img[mask] = med_filter[mask]
    del med_filter, dif, med, ms, mask

    return img


@hierarchical_debug(logger)
def gen_moff_filter2(alpha, beta):
    """Generate a Moffat filter with adjusted alpha.

    :param alpha: Alpha parameter for Moffat filter.
    :type alpha: float
    :param beta: Beta parameter for Moffat filter.
    :type beta: float

    
    """
    alpha = alpha / 2
    fw = alpha * (2 * np.sqrt(2 ** (1 / beta) - 1))
    sigma_r = fw / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    lk = np.ceil(sigma_r).astype(np.int16) * 4
    k_dim = 2 * lk + 1, 2 * lk + 1
    indi = cp.indices(k_dim)
    r2 = (lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2
    ker = (1 + r2 / alpha ** 2) ** -beta
    ksum = cp.sum(ker)
    k_app = ker / ksum

    return k_app, lk


#
# @hierarchical_debug(logger)
# def process_image_new(imdata, imheader, center_factor=0.5, ks=2, astrom=False, tile_section=3000, color_range=0.3,
#                       SP_filt=True, pca_method=True, border=50, CR_filt=False):
#     """
#     Process an astronomical image.
#
#     Parameters
#     ----------
#     imdata : ndarray
#         Image data.
#     imheader : dict
#         FITS header.
#     center_factor : float, optional
#         Factor for center region, by default 0.5.
#     ks : int, optional
#         Aperture size for dilation, by default 2.
#     astrom : bool, optional
#         Whether to perform astrometry, by default False.
#     tile_section : int, optional
#         Tile section size, by default 3000.
#     color_range : float, optional
#         Color range for solar filter, by default 0.3.
#     SP_filt : bool, optional
#         Whether to apply salt-and-pepper filter, by default True.
#     pca_method : bool, optional
#         Whether to use PCA method, by default True.
#     border : int, optional
#         Border size to exclude, by default 50.
#     CR_filt : bool, optional
#         Whether to apply cosmic ray filter, by default False.
#
#     Returns
#     -------
#     tuple
#         Dataframe of photometry, updated header, and calibration dictionary.
#     """
#     mempool = cp.get_default_memory_pool()
#     img_cp = cp.asarray(imdata)
#     scale = plate_scale_px(imheader['PXSIZE'], imheader['FOCALEN']) * imheader[
#         'XBINNING']
#     back, _ = get_local_background_fft(img_cp, scale, ks=ks)
#     gc.collect()
#     if SP_filt:
#         img = SP_filter_cupy(img_cp - back)
#     else:
#         img = img_cp - back
#     try:
#         rdnoise = imheader['GAIN'] * imheader['BIASSTD']
#     except:
#         rdnoise = imheader['RDNOISE']
#     rms = cp.sqrt(cp.abs(back) * imheader['GAIN'] + rdnoise ** 2) / imheader[
#         'GAIN'] / cp.sqrt(imheader['TOTIMA'])
#     sources = detect_isolated_stars(img[border:-border, border:-border],
#                                     rms[border:-border, border:-border], scale, sat_lim=imheader[
#                                                                                             'SATLEVEL'] * 0.8,
#                                     min_snr=10, dist_asec=10)
#     sources = sources + border
#     if len(sources) == 0:
#
#         return None, imheader
#     else:
#         star_dataset, coord, scaling = create_star_dataset(img, sources,
#                                                            scale, CR_filter=CR_filt)
#         normed_star_dataset = (star_dataset.astype(cp.double) - scaling[:,
#                                                                 1][:, None, None]) / cp.sqrt(
#             scaling[:, 2][:, None, None])
#         if pca_method:
#             eigen_psfs = get_eigen_psfs(normed_star_dataset, n_components=5)
#             eigen_psfs = cp.asarray(eigen_psfs)
#             coefficients = project_all_stars_onto_eigenpsfs(normed_star_dataset
#                                                             , eigen_psfs)
#             coeff_map = create_coeff_map(imdata.shape, coord, coefficients.
#                                          T, scale, tile_section=tile_section)
#             mempool.free_all_blocks()
#             gc.collect()
#             sources, conv_ima_sigma = detect_sources_pca(img, rms, scale,
#                                                          eigen_psfs, coeff_map, min_snr=3)
#             del img
#             mempool.free_all_blocks()
#             gc.collect()
#             sources = sources[sources[:, 0] > border]
#             sources = sources[sources[:, 0] < imdata.shape[0] - border]
#             sources = sources[sources[:, 1] > border]
#             sources = sources[sources[:, 1] < imdata.shape[1] - border]
#             x1 = int(imdata.shape[1] * 0.25)
#             x2 = int(imdata.shape[1] * 0.75)
#             y1 = int(imdata.shape[0] * 0.25)
#             y2 = int(imdata.shape[0] * 0.75)
#             x = np.array([imdata.shape[1] // 2, x1, x2, x1, x2])
#             y = np.array([imdata.shape[0] // 2, y1, y2, y2, y1])
#             fwhm_label = ['FWHM', 'FWHMLL', 'FWHMLR', 'FWHMUL', 'FWHMUR']
#             fwhms = np.zeros(5)
#             for point in range(5):
#                 psf = recreate_normed_star(coeff_map, eigen_psfs, (x[point],
#                                                                    y[point]))
#                 try:
#                     _, _, _, fwhm, _ = fit_moffat(psf.get())
#                     fwhms[point] = fwhm
#                 except:
#                     fwhms[point] = 0
#             del (scaling, normed_star_dataset, eigen_psfs, coefficients,
#                  coeff_map, rms)
#         else:
#             sst = np.argsort(scaling[:, -1])[-3:]
#             psf = cp.mean(normed_star_dataset[sst, :, :], axis=0)
#             sources, conv_ima_sigma = detect_sources_kernel(img, rms, psf,
#                                                             scale, min_snr=3)
#             del img, rms
#             mempool.free_all_blocks()
#             gc.collect()
#             sources = sources[sources[:, 0] > border]
#             sources = sources[sources[:, 0] < imdata.shape[0] - border]
#             sources = sources[sources[:, 1] > border]
#             sources = sources[sources[:, 1] < imdata.shape[1] - border]
#             fwhm_label = ['FWHM']
#             _, _, _, fwhm, _ = fit_moffat(psf.get())
#             fwhms = np.array([fwhm])
#         imheader['FWHM'] = fwhms[0]
#         source_flux, source_noise, source_coord, pov = perform_opt_photometry(
#             img_cp - back, back, conv_ima_sigma, sources, coord, imheader,
#             center_factor=center_factor)
#         del img_cp, sources, star_dataset, coord, conv_ima_sigma
#         mempool.free_all_blocks()
#         gc.collect()
#         center_factor = np.min((center_factor, 1))
#         cf = np.min((0.3, 1))
#         xmin = int(imdata.shape[1] * 0.5 * (1 - cf))
#         xmax = int(imdata.shape[1] * 0.5 * (1 + cf))
#         ymin = int(imdata.shape[0] * 0.5 * (1 - cf))
#         ymax = int(imdata.shape[0] * 0.5 * (1 + cf))
#         m = cp.median(back[ymin:ymax, xmin:xmax])
#         s = cp.std(back[ymin:ymax, xmin:xmax])
#         mask = cp.abs(back[ymin:ymax, xmin:xmax] - m) < 3 * s
#         m = cp.median(back[ymin:ymax, xmin:xmax][mask])
#         fluxsky = np.round(m.get(), 6)
#         del back
#         mempool.free_all_blocks()
#         gc.collect()
#         if astrom:
#             dfm = pd.DataFrame({'xcentroid': source_coord[:, 1] + 1,
#                                 'ycentroid': source_coord[:, 0] + 1, 'flux': source_flux})
#             dfm = dfm.sort_values('flux', ascending=False)
#             imheader = astrometrice2(dfm, imheader, imdata.shape)
#         coocenter, FOV, filter, _, inmodel = cat_input_from_header(imheader)
#         result, catalog, ref_filter = catalog_results(coocenter, FOV / 2,
#                                                       filter, inmodel, maglimit=20)
#         wcs = WCS(imheader)
#         coords = SkyCoord(result['RA'], result['DEC'], unit=(u.deg, u.deg))
#         ra = coords.ra.deg
#         dec = coords.dec.deg
#         cat_x, cat_y = wcs.all_world2pix(ra, dec, 0, quiet=True)
#         result['X'] = cat_x
#         result['Y'] = cat_y
#         result.loc[(result['X'] < 0) | (result['X'] > imdata.shape[1]), 'X'
#         ] = np.nan
#         result.loc[(result['Y'] < 0) | (result['Y'] > imdata.shape[0]), 'Y'
#         ] = np.nan
#         result = result.dropna().reset_index(drop=True)
#         cf = np.min((center_factor, 1))
#         xmin = int(imdata.shape[1] * 0.5 * (1 - cf))
#         xmax = int(imdata.shape[1] * 0.5 * (1 + cf))
#         ymin = int(imdata.shape[0] * 0.5 * (1 - cf))
#         ymax = int(imdata.shape[0] * 0.5 * (1 + cf))
#         center_mask = (source_coord[:, 0] > ymin) & (source_coord[:, 0] < ymax
#                                                      ) & (source_coord[:, 1] > xmin) & (source_coord[:, 1] < xmax)
#         zp, ezp, catnstar, min_mag, max_mag = get_zeropoint(result,
#                                                             source_flux[center_mask], source_noise[center_mask],
#                                                             source_coord[center_mask, :], imheader['EXPT1'],
#                                                             dist_thres_px=
#                                                             int(fwhms[0]), solar_filter=color_range)
#         dic_calib = {'ZP': np.round(zp, 4), 'EZP': np.round(ezp, 4),
#                      'CATALOG': catalog, 'CATBAND': ref_filter, 'CATNSTAR': catnstar,
#                      'ZPMINMAG': np.round(min_mag, 2), 'ZPMAXMAG': np.round(max_mag,
#                                                                             2),
#                      'BVMIN': np.round(0.65 - color_range, 2), 'BVMAX': np.round
#             (0.65 + color_range, 2), 'APINTER': np.round(pov[1], 3),
#                      'APSLOPE': np.round(pov[0], 3), 'FLUXSKY': np.round(fluxsky, 6)}
#         for i, fwhm in enumerate(fwhms):
#             dic_calib[fwhm_label[i]] = np.round(fwhm, 2)
#         Y = source_coord[:, 0]
#         X = source_coord[:, 1]
#         RA, DEC = wcs.all_pix2world(X, Y, 0)
#         FLUX = source_flux
#         FLUXERR = source_noise
#         df_phot = pd.DataFrame({'X': np.round(X, 2), 'Y': np.round(Y, 2),
#                                 'RA': np.round(RA, 6), 'DEC': np.round(DEC, 6), 'FLUX': np.
#                                round(FLUX, 2), 'FLUXERR': np.round(FLUXERR, 2)})
#         df_phot_center = df_phot.loc[(df_phot['X'] > xmin) & (df_phot['X'] <
#                                                               xmax) & (df_phot['Y'] > ymin) & (
#                                              df_phot['Y'] < ymax)].reset_index(
#             drop=True)
#         df_phot_center['snr'] = df_phot_center['FLUX'] / df_phot_center[
#             'FLUXERR']
#         df_phot_center['mag'] = -2.5 * np.log10(df_phot_center['FLUX'] /
#                                                 imheader['EXPT1']) + dic_calib['ZP']
#         df_phot_center = df_phot_center.loc[df_phot_center['snr'] < 10
#                                             ].reset_index(drop=True)
#         try:
#             p, cov = np.polyfit(df_phot_center.mag, np.log10(df_phot_center
#                                                              .snr), 1, cov=True)
#             mag = np.linspace(14, 24, 1000)
#             snr = np.polyval(p, mag)
#             dic_calib['MAGLIM'] = np.round(mag[np.argmin(np.abs(snr - np.
#                                                                 log10(3)))], 2)
#             del p, cov, mag, snr
#         except:
#             dic_calib['MAGLIM'] = 0
#         del df_phot_center
#         ref_coords = df_phot[['RA', 'DEC']].to_numpy()
#         target_coord = np.array([float(imheader['POINTRA']) * 15, float(
#             imheader['POINTDEC'])]).reshape(1, 2)
#         _, ref_coords_matched_idx = crossmatch_sources(target_coord,
#                                                        ref_coords, thres_px=10 * fwhms[0] * scale / 3600)
#         if len(ref_coords_matched_idx) == 0:
#             target_snr = 0
#         else:
#             target_snr = (df_phot.iloc[ref_coords_matched_idx]['FLUX'] /
#                           df_phot.iloc[ref_coords_matched_idx]['FLUXERR']).values[0]
#         dic_calib['OBJECSNR'] = np.round(target_snr, 2)
#         phot_exists = get_if_header_already_post_processed(imheader,
#                                                            'PHOTOMETRY')
#         if phot_exists:
#             imheader = delete_header_from(imheader, 'PHOTOMETRY')
#         del (source_flux, source_coord, source_noise, result, coords, ra,
#              dec, cat_x, cat_y, wcs, Y, X, RA, DEC, FLUX, FLUXERR)
#         mempool.free_all_blocks()
#         gc.collect()
#
#         return df_phot, imheader, dic_calib

#
# @hierarchical_debug(logger)
# def perform_opt_photometry(img, back, conv_ima_sigma, source_coord,
#                            isolated_coord, imheader, labels=None, center_factor=1):
#     """
#     Perform optimal photometry on detected sources.
#
#     Parameters
#     ----------
#     img : ndarray
#         Image data.
#     back : ndarray
#         Background image.
#     conv_ima_sigma : ndarray
#         Convolved image sigma.
#     source_coord : ndarray
#         Coordinates of detected sources.
#     isolated_coord : ndarray
#         Coordinates of isolated sources.
#     imheader : dict
#         FITS header.
#     labels : ndarray, optional
#         Labels for sources, by default None.
#     center_factor : float, optional
#         Factor for center region, by default 1.
#
#     Returns
#     -------
#     tuple
#         Optimal flux, noise, coordinates, and photometry parameters.
#     """
#     mempool = cp.get_default_memory_pool()
#     gain = imheader['GAIN']
#     n = imheader['TOTIMA']
#     try:
#         rdnoise = imheader['GAIN'] * imheader['BIASSTD']
#     except:
#         rdnoise = imheader['RDNOISE']
#     try:
#         fwhm = imheader['FWHM']
#         max_radii = int(np.ceil(3.5 * fwhm))
#         min_radii = int(np.ceil(0.5 * fwhm))
#         radii = np.arange(min_radii, max_radii, 1)
#         M = int(np.argmin(np.abs(radii - 2.5 * fwhm)))
#     except:
#         pxscale = plate_scale_px(imheader['PXSIZE'], imheader['FOCALEN']
#                                  ) * imheader['XBINNING']
#         min_radii = int(np.ceil(0.5 / pxscale))
#         max_radii = int(np.ceil(5 / pxscale))
#         radii = np.arange(min_radii, max_radii, 1)
#         M = int(np.argmin(np.abs(radii - 2.5 / pxscale)))
#     source_flux, back_flux, area = batch_aperture_photometry(img, back, cp.
#                                                              round(source_coord).astype(cp.int32), radii)
#     conv_snr = conv_ima_sigma[cp.round(source_coord[:, 0]).astype(cp.int32),
#     cp.round(source_coord[:, 1]).astype(cp.int32)]
#     mempool.free_all_blocks()
#     if labels is None:
#         center_factor = np.min((center_factor, 1))
#         xmin = int(img.shape[1] * 0.5 * (1 - center_factor))
#         xmax = int(img.shape[1] * 0.5 * (1 + center_factor))
#         ymin = int(img.shape[0] * 0.5 * (1 - center_factor))
#         ymax = int(img.shape[0] * 0.5 * (1 + center_factor))
#         center_mask = (isolated_coord[:, 0] > ymin) & (isolated_coord[:, 0] <
#                                                        ymax) & (isolated_coord[:, 1] > xmin) & (isolated_coord[:, 1] <
#                                                                                                 xmax)
#         _, source_coords_matched_idx = crossmatch_sources(isolated_coord[
#                                                               center_mask].get(), source_coord.get(), thres_px=3)
#         center_isolated_flux = source_flux[:, source_coords_matched_idx]
#         m = cp.max(center_isolated_flux, axis=0)
#         brightest = np.argsort(m)[-20:]
#         fm = center_isolated_flux[:, brightest] / m[brightest]
#         corr_fa = cp.nanmedian(fm, axis=1)
#         corr_e = cp.nanstd(fm, axis=1)
#         corr_e[corr_e == 0] = 0
#         cmask = cp.abs(fm - corr_fa[:, None]) > 0.5 * corr_e[:, None]
#         fm[cmask] = cp.nan
#         corr_fact = cp.nanmedian(fm, axis=1)
#         corr_err = cp.nanstd(fm, axis=1)
#         fluxes = source_flux / corr_fact[:, None]
#         noise_phot = cp.sqrt(source_flux / corr_fact[:, None] * gain + area
#                              .reshape(-1, 1) * rdnoise ** 2 + back_flux * gain
#                              ) / gain / cp.sqrt(n)
#         noise_corr = source_flux * corr_err[:, None] / corr_fact[:, None] ** 2
#         noises = cp.sqrt(noise_phot ** 2 + noise_corr ** 2)
#         snr_corr = (fluxes / noises)[:, source_coords_matched_idx]
#         max_snrs_id_corr = np.argmax(snr_corr, axis=0)
#         opt_rad = max_snrs_id_corr.get()
#         pov = np.polyfit(np.log10(conv_snr[source_coords_matched_idx].get()
#                                   ), opt_rad, 1, cov=False)
#         opt_rad = np.ceil(np.fmax(np.fmin(pov[0] * np.log10(conv_snr.get()) +
#                                           pov[1], M), 0)).astype(int)
#         pov0 = pov.copy()
#         pov0[1] += min_radii
#         del (source_flux, back_flux, conv_snr, area, corr_fact, corr_err,
#              noise_phot, noise_corr)
#         opt_flux = fluxes[opt_rad, np.arange(source_coord.shape[0])]
#         opt_noise = noises[opt_rad, np.arange(source_coord.shape[0])]
#     else:
#         _, source_coords_matched_idx = crossmatch_sources(isolated_coord.
#                                                           get(), source_coord.get(), thres_px=3)
#         opt_flux = cp.zeros(source_coord.shape[0])
#         opt_noise = cp.zeros(source_coord.shape[0])
#         opt_rads = cp.zeros(source_coord.shape[0])
#         for lab in np.unique(labels):
#             lab_source_mask = cp.array(labels == lab)
#             lab_source_match = lab_source_mask[source_coords_matched_idx]
#             isolated_flux_lab = source_flux[:, source_coords_matched_idx][:,
#                                 lab_source_match]
#             isolated_back_flux_lab = back_flux[:, source_coords_matched_idx][
#                                      :, lab_source_match]
#             m = cp.max(isolated_flux_lab, axis=0)
#             brightest = np.argsort(m)[-10:]
#             fm = isolated_flux_lab[:, brightest] / m[brightest]
#             corr_fa = cp.nanmedian(fm, axis=1)
#             corr_e = cp.nanstd(fm, axis=1)
#             corr_e[corr_e == 0] = 0
#             cmask = cp.abs(fm - corr_fa[:, None]) > corr_e[:, None]
#             fm[cmask] = cp.nan
#             corr_fact = cp.nanmedian(fm, axis=1)
#             corr_err = cp.nanstd(fm, axis=1)
#             fluxes_lab = isolated_flux_lab / corr_fact[:, None]
#             noise_phot = cp.sqrt(isolated_flux_lab / corr_fact[:, None] *
#                                  gain + area.reshape(-1, 1) * rdnoise ** 2 +
#                                  isolated_back_flux_lab * gain) / gain / cp.sqrt(n)
#             noise_corr = isolated_flux_lab * corr_err[:, None] / corr_fact[
#                                                                  :, None] ** 2
#             noises_lab = cp.sqrt(noise_phot ** 2 + noise_corr ** 2)
#             del (isolated_flux_lab, isolated_back_flux_lab, noise_corr, m,
#                  brightest, fm, corr_fa, corr_e, cmask)
#             snr_corr = fluxes_lab / noises_lab
#             max_snrs_id_corr = np.argmax(snr_corr, axis=0)
#             opt_rad_lab = max_snrs_id_corr.get()
#             pov = np.polyfit(np.log10(conv_snr[source_coords_matched_idx][
#                                           lab_source_match].get()), opt_rad_lab, 1, cov=False)
#             if lab == -1:
#                 pov0 = pov.copy()
#                 pov0[1] += min_radii
#             fluxes_lab = source_flux[:, lab_source_mask] / corr_fact[:, None]
#             opt_rad = np.round(np.fmax(np.fmin(pov[0] * np.log10(conv_snr[
#                                                                      lab_source_mask].get()) + pov[1], M), 0)).astype(
#                 int)
#             opt_flux[lab_source_mask] = fluxes_lab[opt_rad, np.arange(
#                 fluxes_lab.shape[1])]
#             opt_noise[lab_source_mask] = noises_lab[opt_rad, np.arange(
#                 fluxes_lab.shape[1])]
#             opt_rads[lab_source_mask] = opt_rad
#             del (fluxes_lab, noise_phot, opt_rad, snr_corr,
#                  max_snrs_id_corr, pov, noises_lab)
#     mask = opt_flux > 0
#     opt_flux = opt_flux[mask]
#     opt_noise = opt_noise[mask]
#     source_coord = source_coord[mask, :]
#     mempool.free_all_blocks()
#     gc.collect()
#
#     return opt_flux.get(), opt_noise.get(), source_coord.get(), pov0


@hierarchical_debug(logger)
def batch_aperture_photometry(img, back, positions, radii):
    """Perform aperture photometry in batch mode.

    :param img: Image data.
    :type img: ndarray
    :param back: Background image.
    :type back: ndarray
    :param positions: Positions of sources.
    :type positions: ndarray
    :param radii: Aperture radii.
    :type radii: ndarray

    
    """
    mempool = cp.get_default_memory_pool()
    flux = cp.zeros((len(radii), len(positions)))
    back_flux = cp.zeros((len(radii), len(positions)))
    area = cp.zeros(len(radii))
    image_shape = img.shape
    kernel_shape = 2 * radii[-1] + 1, 2 * radii[-1] + 1
    padding = int((kernel_shape[0] - 1) / 2)
    new_image_shape = fill_image((image_shape[0] + 2 * padding, image_shape
    [1] + 2 * padding))
    img_c = cp.fft.rfft2(img, s=new_image_shape)
    back_c = cp.fft.rfft2(back, s=new_image_shape)
    for i, r in enumerate(radii):
        kernel, area[i] = get_aper_kernel(r, size=kernel_shape[0])
        kernel = cp.conj(cp.fft.rfft2(kernel, s=new_image_shape))
        convolved = cp.fft.irfft2(img_c * kernel, s=new_image_shape)
        convolved = cp.roll(convolved, shift=[padding, padding], axis=[0, 1])
        convolved = convolved[:image_shape[0], :image_shape[1]]
        flux[i, :] = convolved[positions[:, 0], positions[:, 1]]
        convolved = cp.fft.irfft2(back_c * kernel, s=new_image_shape)
        convolved = cp.roll(convolved, shift=[padding, padding], axis=[0, 1])
        convolved = convolved[:image_shape[0], :image_shape[1]]
        back_flux[i, :] = convolved[positions[:, 0], positions[:, 1]]
    del convolved, kernel, back_c, img_c
    mempool.free_all_blocks()
    gc.collect()

    return flux, back_flux, area


# @hierarchical_debug(logger)
# def get_raw_photometry(imdata, imheader, aperture_rad_asec=None, astrom=
# True, SP_filt=True, ks=2):
#     """
#     Get raw photometry for an astronomical image.
#
#     Parameters
#     ----------
#     imdata : ndarray
#         Image data.
#     imheader : dict
#         FITS header.
#     aperture_rad_asec : float, optional
#         Aperture radius in arcseconds, by default None.
#     astrom : bool, optional
#         Whether to perform astrometry, by default True.
#     SP_filt : bool, optional
#         Whether to apply salt-and-pepper filter, by default True.
#     ks : int, optional
#         Aperture size for dilation, by default 2.
#
#     Returns
#     -------
#     pd.DataFrame
#         Dataframe containing photometry results.
#     """
#
#     logger.debug('Start photometry extraction')
#     mempool = cp.get_default_memory_pool()
#     img_cp = cp.asarray(imdata)
#     scale = plate_scale_px(imheader['PXSIZE'], imheader['FOCALEN']) * imheader[
#         'XBINNING']
#     back, _ = get_local_background_fft(img_cp, scale, ks=ks)
#     gc.collect()
#
#     if SP_filt:
#         img = SP_filter_cupy(img_cp - back)
#     else:
#         img = img_cp - back
#
#     try:
#         rdnoise = imheader['GAIN'] * imheader['BIASSTD'] * np.sqrt(imheader
#                                                                    ['TOTIMA'])
#     except:
#         rdnoise = imheader['RDNOISE']
#     rms = cp.sqrt(cp.abs(back) * imheader['GAIN'] * np.sqrt(imheader[
#                                                                 'TOTIMA']) + rdnoise ** 2) / imheader['GAIN']
#
#     sources = detect_isolated_stars(img, rms, scale, sat_lim=imheader[
#                                                                  'SATLEVEL'] * 0.8, min_snr=10, dist_asec=10)
#     star_dataset, coord, scaling = create_star_dataset(img, sources, scale)
#     normed_star_dataset = (star_dataset.astype(cp.double) - scaling[:, 1][:,
#                                                             None, None]) / cp.sqrt(scaling[:, 2][:, None, None])
#     sst = np.argsort(scaling[:, -1])[-3:]
#     psf = cp.mean(normed_star_dataset[sst, :, :], axis=0)
#     sources = detect_sources_kernel(img, rms, psf, scale, min_snr=5)
#     del star_dataset, coord, scaling, sst, psf
#
#     if aperture_rad_asec is None:
#         aperture_rad_asec = 1
#     aperture_rad = int(max(aperture_rad_asec / scale, 1))
#     source_flux, _ = aperture_photometry(img, sources, aperture_rad)
#     source_neg_mask = source_flux > 0
#     source_flux = source_flux[source_neg_mask]
#     source_coord = sources[source_neg_mask]
#     del source_neg_mask
#     back_n, area = aperture_photometry(rms, source_coord, aperture_rad)
#     source_noise = cp.sqrt(source_flux * imheader['GAIN'] + area * rdnoise **
#                            2 + back_n * imheader['GAIN']) / imheader['GAIN']
#     snr_mask = (source_flux / source_noise >= 2).get()
#     del back_n, area
#     source_flux = source_flux[snr_mask].get()
#     source_coord = source_coord[snr_mask].get()
#     source_noise = source_noise[snr_mask].get()
#     del img, img_cp, rms, sources, back
#     mempool.free_all_blocks()
#     gc.collect()
#
#     if astrom:
#         dfm = pd.DataFrame({'xcentroid': source_coord[:, 1], 'ycentroid':
#             source_coord[:, 0], 'flux': source_flux})
#         dfm = dfm.sort_values('flux', ascending=False)
#         imheader = astrometrice2(dfm, imheader, imdata.shape)
#     wcs = WCS(imheader)
#     Y = source_coord[:, 0]
#     X = source_coord[:, 1]
#     RA, DEC = wcs.all_pix2world(X, Y, 0)
#     FLUX = source_flux
#     FLUXERR = source_noise
#     df_phot = pd.DataFrame({'DATE': imheader['DATE-OBS'], 'FILTER':
#         imheader['FILTER'], 'EXPTIME': imheader['EXPT1'], 'RA': np.round(RA,
#                                                                          6), 'DEC': np.round(DEC, 6),
#                             'FLUX': np.round(FLUX, 2), 'FLUXERR':
#                                 np.round(FLUXERR, 2)})
#     del X, Y, RA, DEC, FLUX, FLUXERR
#     mempool.free_all_blocks()
#     gc.collect()
#
#     return df_phot


@hierarchical_debug(logger)
def aperture_photometry(img, positions, aper_rad):
    """Perform aperture photometry.

    :param img: Image data.
    :type img: ndarray
    :param positions: Positions of sources.
    :type positions: ndarray
    :param aper_rad: Aperture radius.
    :type aper_rad: int

    
    """
    kernel, area = get_aper_kernel(aper_rad)
    conv_ima = convolve_fft(img, kernel)
    positions = cp.array(cp.round(positions)).astype(cp.int32)
    flux = conv_ima[positions[:, 0], positions[:, 1]]
    del conv_ima, kernel, positions

    return flux, area


@hierarchical_debug(logger)
def get_fwhm_mof(model, img, step=50, ns=25, mins=3):
    """Get the full width at half maximum using Moffat model.

    :param model: Model to use for FWHM estimation.
    :type model: Model
    :param img: Image data.
    :type img: ndarray
    :param step: Step size for sampling, by default 50.
    :type step: int, optional
    :param ns: Number of samples, by default 25.
    :type ns: int, optional
    :param mins: Minimum number of stars, by default 3.
    :type mins: int, optional

    
    """
    ims, cs = sample_im(img, step, ns)
    pred2 = model.predict(ims)
    alpha, beta, nstar, fwhm = pred_mof(pred2)
    fws = fwhm[nstar >= np.mean(nstar)]

    return np.mean(fws), np.std(fws), np.mean(alpha), np.mean(beta)


@hierarchical_debug(logger)
def cov_nan(img, nc=10):
    """Fill NaN values in an image using convolution.

    :param img: Input image.
    :type img: ndarray
    :param nc: Number of chunks, by default 10.
    :type nc: int, optional

    
    """
    delta = np.round(img.shape[0] / nc).astype(int)
    for i in range(nc):
        for j in range(nc):
            ii = img[delta * i:delta * (i + 1), delta * j:delta * (j + 1)]
            ii[cp.isnan(ii)] = cp.nanmean(ii)
            img[delta * i:delta * (i + 1), delta * j:delta * (j + 1)] = ii
    img[cp.isnan(img)] = cp.nanmean(img)
    img[cp.isinf(img)] = cp.nanmean(img)

    return img


#
# @hierarchical_debug(logger)
# def astrometrice2(dfm, head0, im_shape):
#     """
#     Perform astrometry on an image.
#
#     Parameters
#     ----------
#     dfm : pd.DataFrame
#         Dataframe containing detected sources.
#     head0 : dict
#         FITS header.
#     im_shape : tuple
#         Shape of the image.
#
#     Returns
#     -------
#     dict
#         Updated FITS header.
#     """
#     head = head0.copy()
#     arcsec_per_pixel = plate_scale_px(head['PXSIZE'], head['FOCALEN']) * head[
#         'XBINNING']
#     signal.signal(signal.SIGALRM, handler)
#     signal.alarm(120)
#     try:
#         solution = get_solver().solve(stars_xs=dfm['xcentroid'], stars_ys=
#         dfm['ycentroid'], size_hint=astrometry.SizeHint(
#             lower_arcsec_per_pixel=arcsec_per_pixel * 0.8,
#             upper_arcsec_per_pixel=arcsec_per_pixel * 1.2), position_hint=
#                                       astrometry.PositionHint(ra_deg=head['POINTRA'] * 360 / 24,
#                                                               dec_deg=head['POINTDEC'], radius_deg=0.5),
#                                       solution_parameters=
#                                       astrometry.SolutionParameters(logodds_callback=
#                                                                     logodds_callback_100, sip_order=3))
#         nmatches = len(solution.matches)
#         logger.debug(nmatches)
#         return solution
#     except:
#         pass
#     signal.alarm(0)
#
#     return None


@hierarchical_debug(logger)
def get_zeropoint(df_catalog, flux, noise, coord, exptime, solar_filter=0.3,
                  dist_thres_px=3, N=50, plot=False):
    """Calculate the zeropoint for photometry.

    :param df_catalog: Catalog dataframe.
    :type df_catalog: pd.DataFrame
    :param flux: Flux values.
    :type flux: ndarray
    :param noise: Noise values.
    :type noise: ndarray
    :param coord: Coordinates of sources.
    :type coord: ndarray
    :param exptime: Exposure time.
    :type exptime: float
    :param solar_filter: Solar filter, by default 0.3.
    :type solar_filter: float, optional
    :param dist_thres_px: Distance threshold in pixels, by default 3.
    :type dist_thres_px: int, optional
    :param N: Number of brightest stars to use, by default 50.
    :type N: int, optional
    :param plot: Whether to plot the results, by default False.
    :type plot: bool, optional

    
    """
    cat_coords = np.array([df_catalog['Y'], df_catalog['X']]).T
    source_coords_matched_idx, ref_coords_matched_idx = crossmatch_sources(
        coord, cat_coords, thres_px=dist_thres_px)
    solar_cat_filt = np.abs(df_catalog['SOLAR'])[ref_coords_matched_idx
                     ] < solar_filter
    source_coords_matched_idx = source_coords_matched_idx[solar_cat_filt]
    ref_coords_matched_idx = ref_coords_matched_idx[solar_cat_filt]
    det_mag = -2.5 * np.log10(flux[source_coords_matched_idx] / exptime)
    cat_mag = df_catalog['MAG'][ref_coords_matched_idx].to_numpy()
    inf_nan_mask = np.isfinite(cat_mag) & np.isfinite(det_mag) & ~np.isnan(
        cat_mag) & ~np.isnan(det_mag)
    cat_mag = cat_mag[inf_nan_mask]
    det_mag = det_mag[inf_nan_mask]
    snrs = (flux / noise)[source_coords_matched_idx][inf_nan_mask]
    m = np.argsort(snrs)
    brightest = m[-N:]
    bright_mask = np.zeros(len(cat_mag), dtype=bool)
    bright_mask[brightest] = True
    if len(cat_mag) <= 3:
        zp, ezp, n, min_mag, max_mag = 0, 0, 0, 0, 0
    else:
        y = cat_mag[bright_mask] - det_mag[bright_mask]
        mask = np.abs(y - np.nanmean(y)) < np.nanstd(y)
        if np.sum(mask) > 15:
            reg = RANSACRegressor(random_state=42, residual_threshold=0.05
                                  ).fit(det_mag[bright_mask].reshape([-1, 1])[mask], cat_mag[
                bright_mask].reshape([-1, 1])[mask])
            inlier = reg.inlier_mask_
            if np.sum(inlier) > 10 and np.abs(np.mean(y[mask][inlier]) - np
                    .mean(y[mask])) < np.std(y[mask]):
                zp = np.mean(y[mask][inlier])
                n = np.sum(inlier)
                ezp = np.std(y[mask][inlier]) / np.sqrt(n)
                min_mag = np.min(cat_mag[bright_mask][mask][inlier])
                max_mag = np.max(cat_mag[bright_mask][mask][inlier])
            else:
                zp = np.mean(y[mask])
                n = np.sum(mask)
                ezp = np.std(y[mask]) / np.sqrt(n)
                min_mag = np.min(cat_mag[bright_mask][mask])
                max_mag = np.max(cat_mag[bright_mask][mask])
        else:
            zp = np.mean(y[mask])
            n = np.sum(mask)
            ezp = np.std(y[mask]) / np.sqrt(n)
            min_mag = np.min(cat_mag[bright_mask][mask])
            max_mag = np.max(cat_mag[bright_mask][mask])
    if plot:
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111)
        ax.plot(cat_mag, cat_mag - det_mag - zp, 'k.', alpha=0.6)
        if np.sum(mask) > 15:
            ax.plot(cat_mag[bright_mask][mask], cat_mag[bright_mask][mask] -
                    det_mag[bright_mask][mask] - zp, 'b.', alpha=0.1)
            ax.plot(cat_mag[bright_mask][mask][inlier], cat_mag[bright_mask
            ][mask][inlier] - det_mag[bright_mask][mask][inlier] - zp,
                    'r.', label='zp = {:.3f} +/- {:.3f} (n={})'.format(zp, ezp,
                                                                       n), alpha=0.5)
        else:
            ax.plot(cat_mag[bright_mask][mask], cat_mag[bright_mask][mask] -
                    det_mag[bright_mask][mask] - zp, 'r.', label=
                    'zp = {:.3f} +/- {:.3f} (n={})'.format(zp, ezp, n), alpha=0.5)
        ax.set_xlabel('catalog magnitude')
        ax.set_ylabel('error magnitude')
        ax.legend(frameon=False)
        ax.set_ylim(-0.5, 0.5)
        plt.show()

    return zp, ezp, n, min_mag, max_mag


@hierarchical_debug(logger)
def delete_header_from(header, val):
    """Delete a section from the FITS header.

    :param header: FITS header.
    :type header: dict
    :param val: Value to delete.
    :type val: str

    
    """
    for i, v in enumerate(header.values()):
        if val in str(v):
            idx = i - 1
    for i in range(len(header) - idx):
        del header[idx]

    return header


@hierarchical_debug(logger)
def sample_im(img, nc=50, ns=100):
    """Sample an image.

    :param img: Input image.
    :type img: ndarray
    :param nc: Number of chunks, by default 50.
    :type nc: int, optional
    :param ns: Number of samples, by default 100.
    :type ns: int, optional

    
    """
    delta = np.round((img.shape[0] - 512) / nc).astype(int)
    pi = 512 * 512
    lim = []
    cmax = []
    np.random.seed(42)
    for k in range(ns):
        i = np.random.randint(nc)
        j = np.random.randint(nc)
        ii = img[delta * i:delta * i + 512, delta * j:delta * j + 512]
        ima = np.median(ii)
        ii = np.log(abs(ii) / ima) / 100
        li = ii.shape[0] * ii.shape[1]
        ii = np.hstack([ii.flatten(), [0] * (pi - li)])
        ii[np.isnan(ii)] = 0
        ii[np.isinf(ii)] = 0
        lim = lim + [ii]
        cmax = cmax + [ima]
    iac2 = np.asarray(lim)
    icmax2 = np.asarray(cmax)
    iac2 = iac2.reshape(-1, 512, 512, 1)

    return iac2, icmax2


@hierarchical_debug(logger)
def pred_mof(pred):
    """Predict Moffat parameters.

    :param pred: Predicted values.
    :type pred: ndarray

    
    """
    alpha = pred[:, 1]
    beta = pred[:, 0] * 0.4 + 4.565
    nstar = pred[:, 2] * 200
    fwhm = 2 * alpha * np.sqrt(2 ** (1 / beta) - 1)

    return alpha, beta, nstar, fwhm


# @hierarchical_debug(logger)
# def handler(signum, frame):
#     """
#     Timeout handler for astrometry.
#
#     Parameters
#     ----------
#     signum : int
#         Signal number.
#     frame : frame
#         Stack frame.
#     """
#     logger.error('Astrometrization timeout!')
#
#     raise Exception('end of time')


@hierarchical_debug(logger)
def sigma_clip(img, sclip):
    """

    :param img: 
    :param sclip: 

    """
    img0 = img.copy()
    for i in range(5):
        imed = cp.nanmean(img0)
        rms = cp.nanstd(img0)
        img0[img0 >= imed + sclip * rms] = cp.nan
        img0[img0 <= imed - sclip * rms] = cp.nan
    del img0

    return imed, rms


@hierarchical_debug(logger)
def gen_gauss_filter(fw):
    """

    :param fw: 

    """
    sigma_r = fw / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_r2 = sigma_r * sigma_r
    lk = np.ceil(sigma_r).astype(np.int16) * 4
    k_dim = 2 * lk + 1, 2 * lk + 1
    indi = cp.indices(k_dim)
    r2 = (lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2
    ker = cp.exp(-r2 / 2 / sigma_r2)
    ksum = cp.sum(ker)
    ksum2 = cp.sum(ker * ker)
    n = k_dim[0] ** 2
    k_app = (ker - ksum / n) / (ksum2 - ksum * ksum / n)

    return k_app, lk


@hierarchical_debug(logger)
def detect_gpu(img, sky, rms, sdet, mode='g', fw=1, alpha=0, beta=0, minpix
=4, mincut=10, mem=cp.get_default_pinned_memory_pool()):
    """

    :param img: 
    :param sky: 
    :param rms: 
    :param sdet: 
    :param mode:  (Default value = 'g')
    :param fw:  (Default value = 1)
    :param alpha:  (Default value = 0)
    :param beta:  (Default value = 0)
    :param minpix:  (Default value = 4)
    :param mincut:  (Default value = 10)
    :param mem:  (Default value = cp.get_default_pinned_memory_pool())

    """
    gf, lk = gen_gauss_filter(fw)
    g = convolve(img - sky, gf, origin=(0, 0))
    g1 = (g / rms > sdet).astype(cp.int32)
    del rms
    mem.free_all_blocks()
    label_im, nb_labels = label(g1)
    ids0 = cp.asarray([range(nb_labels + 1)])
    npix = nd_sum(g1, label_im, ids0)
    ids = ids0[(npix > mincut) & (npix > 0)]
    idm = ids0[(npix <= minpix) & (npix > 0)]
    npix = npix[(npix > mincut) & (npix > 0)]
    if minpix == 0:
        mask = False
    else:
        mask = cp.isin(label_im, cp.asarray(idm))
    del g1
    mem.free_all_blocks()
    idx = cp.indices(img.shape, dtype=cp.int16)
    im1 = g * idx
    x = nd_mean(im1[0, :, :], label_im, ids) / nd_mean(g, label_im, ids)
    y = nd_mean(im1[1, :, :], label_im, ids) / nd_mean(g, label_im, ids)
    el = nd_mean(im1[0, :, :] * im1[1, :, :], label_im, ids) / nd_mean(g,
                                                                       label_im, ids)
    el = el - x * y
    coor = x.astype(cp.int), y.astype(cp.int)
    mm = cp.get_default_memory_pool().used_bytes()
    flux = g[coor]
    del g
    mem.free_all_blocks()
    res = np.asarray([y.get(), x.get(), flux.get(), npix.get(), el.get()]
                     ).transpose().reshape((-1, 5))
    df = pd.DataFrame(res, columns=['xcentroid', 'ycentroid', 'flux',
                                    'npix', 'elip'])

    return df, mask, mm


@hierarchical_debug(logger)
def get_peak_image(img, positions, aper_rad):
    """

    :param img: 
    :param positions: 
    :param aper_rad: 

    """
    lk = 2 * aper_rad
    img_m = maximum_filter(img, size=lk)
    positions = cp.array(cp.round(positions)).astype(cp.int32)
    P = img_m[positions[:, 0], positions[:, 1]]
    del img_m, positions

    return P


# @hierarchical_debug(logger)
# def logodds_callback_100(logodds):
#     """
#     Callback function for astrometry.
#
#     Parameters
#     ----------
#     logodds : float
#         Log odds value.
#
#     Returns
#     -------
#     astrometry.Action
#         Action to take (CONTINUE or STOP).
#     """
#     if (logodds[0] > 100.0) | (len(logodds) > 2):
#
#         return astrometry.Action.STOP
#     else:
#
#         return astrometry.Action.CONTINUE


if __name__ == '__main__':
    from astropy.io import fits
    import os

    directory_path = os.path.join(os.path.dirname(__file__), '..', '..',
                                  'tests', 'data')
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.fits'):
                if 'TTT1' in file:
                    image_path = os.path.join(root, file)
                    try:
                        print(f'Processing {image_path}')

                        image_cp = cp.asarray(fits.getdata(image_path))
                        resul = SP_filter_cupy(image_cp)
                        end_time = time.time()

                    except Exception as e:
                        print(f'Error processing {image_path}: {e}')
                        print('Traceback:')
                        traceback.print_exc()
