import gc
import time
import traceback

import cupy as cp
import numpy as np
import pandas as pd
from astropy.wcs import WCS
# import tensorflow as tf
from cupyx.scipy.ndimage import convolve, label, sum as nd_sum, mean as nd_mean, maximum_filter, \
    median_filter, laplace, binary_dilation

from .conv import fill_image, fill_nan_fft, get_aper_kernel, convolve_fft, gen_apm_filter
from .psf import create_coeff_map, create_star_dataset, detect_isolated_stars, detect_sources_psf, fit_moffat, \
    get_eigen_psfs, group_star_dataset, project_all_stars_onto_eigenpsfs, recreate_normed_star
from ..exceptions import InsufficientStarsError, MoffatFitError, capture_cuda_exception, UnableToAstrometrizeError, \
    DataValidationError
from ..instrument_config_parser import HeaderKey, DefaultConfig
from ..logger.hierarchical_logging import setup_logger, hierarchical_debug
from ..phot.background import get_local_background_fft
from ..stats.reduction import stack_sigmaclip
from ..utils.astro import astrometrice2, get_astrometry_params, get_maglim, get_target_snr, get_zeropoint, \
    plate_scale_px
from ..utils.catalog import catalog_results, crossmatch_sources
from ..utils.headers import update_header_with_astrometry, update_header_with_photometry

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


# def init_gpu():
#     """ """
#     logger.debug('Tensorflow version ' + tf.__version__)
#     gpus = tf.config.list_physical_devices('GPU')
#     logger.debug('GPUs:', gpus)
#     tf.config.set_logical_device_configuration(gpus[0], [tf.config.
#                                                LogicalDeviceConfiguration(memory_limit=1024)])


@hierarchical_debug(logger)
def gen_moff_filter(alpha, beta, **kwargs):
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
def get_sky(im_g, fw, qt=90, mem=cp.get_default_memory_pool(), **kwargs):
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
def SP_filter(img, filter_size=3, high_threshold_factor=10,
              low_threshold_factor=5, scaling_factor=1.4826, **kwargs):
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
def CR_filter(img, thres=3, **kwargs):
    """Apply a cosmic ray filter to an image.

    :param img: Input image.
    :type img: cupy.ndarray
    :param thres: Threshold for cosmic ray detection (default is 3).
    :type thres: float, optional

    
    """

    img = cp.asarray(img, dtype=cp.float32)
    mask = laplace(img)  # > 100
    mask = cp.abs(mask - cp.mean(mask)) > thres * cp.std(mask)
    mask = binary_dilation(mask, cp.ones((3, 3)))
    img_filled = img.copy()
    img_filled[mask] = cp.nan
    n = cp.sum(cp.isnan(img_filled))
    while n > 0:
        img_filled = fill_nan_fft(img_filled, 3, 0, min_neighbors=5)
        if n == cp.sum(cp.isnan(img_filled)):
            break
        else:
            n = cp.sum(cp.isnan(img_filled))
    ref = img_filled < cp.percentile(img_filled, 95)

    mask = mask & ref
    img[mask] = cp.nan
    n = cp.sum(cp.isnan(img))
    while n > 0:
        img = fill_nan_fft(img, 3, 0, min_neighbors=5)
        if n == cp.sum(cp.isnan(img)):
            break
        else:
            n = cp.sum(cp.isnan(img))
    return img


@hierarchical_debug(logger)
def gen_moff_filter2(alpha, beta, **kwargs):
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


@hierarchical_debug(logger)
def calculate_aperture_corrections(corr: np.ndarray) -> cp.ndarray:
    """
    Calculates aperture corrections for all stars in an optimized way.

    :param star_dataset_ref: Reference star dataset.
    :param radii: Aperture radii.
    :return: Aperture corrections and errors.
    """

    # correct ouliers
    corr /= np.nanmax(corr, axis=1).reshape(-1, 1)
    corr_fa = np.nanmedian(corr, axis=0)
    corr_e = np.nanstd(corr, axis=0)
    cmask = np.abs(corr - corr_fa) > corr_e
    corr[cmask] = np.nan
    corr_fact = np.nanmean(corr, axis=0)
    corr_err = np.nanstd(corr, axis=0)
    return corr_fact, corr_err


@hierarchical_debug(logger)
def find_aperture_corrections(sources: cp.ndarray, corrections: np.ndarray, correction_errors: np.ndarray,
                              cluster_centers: np.ndarray,
                              opt_rad_idx: np.array = None, **kwargs) -> cp.ndarray:
    """Find the aperture correction for each source.

    :param sources: Array of source coordinates.
    :param corrections: Array of aperture corrections.
    :param tile_centers: Array of tile centers.
    :param opt_rad_idx: Array of optimal radii indices.
    :return: Array of aperture corrections for each source.
    """
    _, tile_idx = crossmatch_sources(sources.get(), cluster_centers, thres_px=int(cp.max(sources)))
    if opt_rad_idx is None:
        aperture_corrections = corrections[tile_idx, :]
        aperture_correction_errors = correction_errors[tile_idx, :]
    else:
        aperture_corrections = corrections[tile_idx, opt_rad_idx]
        aperture_correction_errors = correction_errors[tile_idx, opt_rad_idx]
    return aperture_corrections, aperture_correction_errors


@hierarchical_debug(logger)
def create_aperture_corrections_map(image_shape: tuple, block_size: int, unit_star_dataset: cp.ndarray,
                                    coords: cp.ndarray, radii: np.ndarray, **kwargs):
    """
    Calculates aperture corrections for all stars in an optimized way.

    :param image_shape: Tuple (height, width) of the image.
    :param block_size: Size of the tiles.
    :param coeff_map: Coefficient map for recreating normalized stars.
    :param eigen_psfs: Eigen PSFs for reconstruction.
    :param psf: Base PSF array.
    :param radii: Radii for aperture photometry.
    :return: Dictionary mapping each star to its aperture correction.
    """
    # calculate aperture photometry curves
    positions = cp.array([[unit_star_dataset.shape[1] // 2, unit_star_dataset.shape[2] // 2]])
    aperture_curves, _, _ = batch_aperture_photometry(unit_star_dataset, None, positions, radii)
    aperture_curves = aperture_curves[:, :, 0].get()

    # cluster stars
    avg_group_size = int(
        max(len(coords), len(coords) / (np.prod(image_shape) / min(max(image_shape), block_size) ** 2)))
    labels = group_star_dataset(coords, avg_group_size=avg_group_size, min_group_size=min(avg_group_size, 5))

    unique_labels = np.unique(labels)
    cluster_centers = []
    aperture_corrections = []
    aperture_correction_errors = []
    for label in unique_labels:
        cluster_points = coords[labels == label]
        cluster_aperture_curves = aperture_curves[labels == label]

        aper_corr, aperr_corr_err = calculate_aperture_corrections(cluster_aperture_curves)
        aperture_corrections.append(aper_corr)
        aperture_correction_errors.append(aperr_corr_err)
        cluster_centers.append(np.mean(cluster_points, axis=0))

    # calculate aperture corrections
    aperture_corrections = np.array(aperture_corrections)
    aperture_correction_errors = np.array(aperture_correction_errors)
    cluster_centers = np.array(cluster_centers)

    return aperture_corrections, aperture_correction_errors, cluster_centers


@capture_cuda_exception
@hierarchical_debug(logger)
def process_image(imdata, imheader, header_descriptions=None, **kwargs):
    # parameters from header
    scale = plate_scale_px(imheader[HeaderKey.PXSIZE.value], imheader[HeaderKey.FOCALEN.value]) * imheader[
        HeaderKey.XBINNING.value]
    n_images = imheader[HeaderKey.TOTIMA.value]
    gain = imheader[HeaderKey.GAIN.value]
    try:
        rdnoise = imheader[HeaderKey.GAIN.value] * imheader[HeaderKey.BIASSTD.value]
    except:
        rdnoise = imheader[HeaderKey.RDNOISE.value]
    exptime = imheader[HeaderKey.EXPT1.value]
    satlevel = imheader[HeaderKey.SATLEVEL.value]
    target_ra = imheader[HeaderKey.POINTRA.value] * 15
    target_dec = imheader[HeaderKey.POINTDEC.value]
    try:
        site_elevation = imheader[HeaderKey.SITEELEV.value]
    except:
        site_elevation = imheader[HeaderKey.SITEALT.value]
    site_latitude = imheader[HeaderKey.SITELAT.value]
    site_longitude = imheader[HeaderKey.SITELONG.value]
    date_obs = imheader[HeaderKey.DATE_OBS.value]
    inmodel = imheader[HeaderKey.INMODEL.value]
    filter = imheader[HeaderKey.FILTER.value]

    # default parameters
    default_params = DefaultConfig.DEFAULT_PROCESSING_PARAMS

    # Update default parameters with any provided in kwargs
    params = {**default_params, **kwargs}

    # Call calibrate_image with updated parameters
    dfm, h_wcs, dic_calib = calibrate_image(
        imdata, inmodel, filter,
        scale, gain, rdnoise, exptime, satlevel,
        target_ra, target_dec, n_images=n_images,
        # SP_filt=params['SP_filt'],
        # CR_filt=params['CR_filt'],
        # border=params['border'],
        # center_factor=params['center_factor'],
        # pca_method=params['pca_method'],
        # tile_section=params['tile_section'],
        # max_stars_ref=params['max_stars_ref'],
        # min_snr=params['min_snr'],
        # color_range=params['color_range'],
        # tile_section_psf=params['tile_section_psf'],
        **params
    )

    if dfm is None:
        raise UnableToAstrometrizeError()

    # Update header
    imheader = update_header_with_astrometry(imheader, h_wcs, site_latitude, site_longitude, site_elevation, date_obs,
                                             header_descriptions)
    imheader = update_header_with_photometry(imheader, dic_calib, header_descriptions)

    return dfm, imheader


@hierarchical_debug(logger)
def perform_opt_photometry(img: cp.ndarray, back: cp.ndarray, conv_ima_sigma: cp.ndarray,
                           source_coord: cp.ndarray, isolated_coord: cp.ndarray,
                           tile_section_psf: int, star_dataset: cp.ndarray, fwhm: float,
                           gain: float, rdnoise: float, n_images: int = 1, center_factor: float = 1.0,
                           min_conv_snr: float = 300.0):
    """
    Optimizes the aperture photometry process to find the best radii for signal-to-noise ratio (SNR) for each star.

    :param img: Input image as a CuPy ndarray.
    :param back: Background image as a CuPy ndarray.
    :param conv_ima_sigma: Convolution sigma map as a CuPy ndarray.
    :param source_coord: Coordinates of sources as a CuPy ndarray.
    :param isolated_coord: Coordinates of isolated stars as a CuPy ndarray.
    :param star_dataset:
    :param fwhm: Full-width at half-maximum of the PSF.
    :param gain: Gain value for flux conversion.
    :param rdnoise: Read noise value of the detector.
    :param n_images: Number of images.
    :param center_factor: Fraction defining the central region for selecting isolated stars (default is 1.0).
    :param min_conv_snr: Minimum convolutional SNR for selecting isolated stars (default is 300.0).
    :return: Tuple containing the optimized fluxes, noise and fitting parameters.
    """
    # Memory pool
    mempool = cp.get_default_memory_pool()

    # Select central stars
    center_factor = np.min((center_factor, 1))
    xmin = int(img.shape[1] * 0.5 * (1 - center_factor))
    xmax = int(img.shape[1] * 0.5 * (1 + center_factor))
    ymin = int(img.shape[0] * 0.5 * (1 - center_factor))
    ymax = int(img.shape[0] * 0.5 * (1 + center_factor))

    center_mask = (isolated_coord[:, 0] > ymin) & (isolated_coord[:, 0] < ymax) & \
                  (isolated_coord[:, 1] > xmin) & (isolated_coord[:, 1] < xmax)

    # Obtain convolutional SNR
    conv_snr = conv_ima_sigma[
        cp.round(source_coord[:, 0]).astype(cp.int32), cp.round(source_coord[:, 1]).astype(cp.int32)]
    pos_conv_snr_mask = conv_snr > 0
    source_coord = source_coord[pos_conv_snr_mask]
    conv_snr = conv_snr[pos_conv_snr_mask]
    _, source_coords_matched_idx = crossmatch_sources(
        isolated_coord[center_mask].get(), source_coord.get(), thres_px=3
    )
    conv_snr_isol = conv_ima_sigma[
        cp.round(isolated_coord[:, 0]).astype(cp.int32), cp.round(isolated_coord[:, 1]).astype(cp.int32)]
    conv_snr_mask = conv_snr_isol > min_conv_snr
    if cp.sum(conv_snr_mask) < 3:
        raise InsufficientStarsError(cp.sum(conv_snr_mask))

    # In case PSF is position-invariant
    # if coeff_map is None or eigen_psfs is None or len(source_coords_matched_idx) < 10:
    #     tile_section_psf = int(1.1 * max(img.shape))

    max_radii = int(np.ceil(7 * fwhm) + 1)
    min_radii = int(np.ceil(.75 * fwhm))
    radii = np.arange(min_radii, max_radii, 1)

    # Find aperture corrections
    corrections, correction_errors, cluster_centers = create_aperture_corrections_map(img.shape, tile_section_psf,
                                                                                      star_dataset[conv_snr_mask],
                                                                                      isolated_coord[
                                                                                          conv_snr_mask].get(), radii)
    aperture_corrections, aperture_correction_errors = find_aperture_corrections(source_coord, corrections,
                                                                                 correction_errors, cluster_centers)

    # Get batch photometry
    source_flux, back_flux, area = batch_aperture_photometry(
        img, back, cp.round(source_coord).astype(cp.int32), radii
    )

    # Calculate photometric parameteres
    center_isolated_flux = np.array(source_flux[:, source_coords_matched_idx].get())
    center_isolated_aper_corr = np.array(aperture_corrections[source_coords_matched_idx].T)
    center_isolated_aper_corr_err = np.array(aperture_correction_errors[source_coords_matched_idx].T)

    center_isolated_signal = center_isolated_flux / center_isolated_aper_corr

    center_isolated_back_noise_sq = np.abs(back_flux[:, source_coords_matched_idx]).get() * gain
    center_isolated_read_noise_sq = area.get().reshape(-1, 1) * rdnoise ** 2
    center_isolated_source_noise_sq = center_isolated_flux * gain / center_isolated_aper_corr ** 2
    center_isolated_corr_noise_sq = (
                                            center_isolated_flux * gain * center_isolated_aper_corr_err / center_isolated_aper_corr ** 2) ** 2

    center_isolated_total_noise = np.sqrt(
        center_isolated_source_noise_sq + center_isolated_back_noise_sq + center_isolated_read_noise_sq + center_isolated_corr_noise_sq) / gain / np.sqrt(
        n_images)

    center_isolated_snr = center_isolated_signal / center_isolated_total_noise
    center_conv_snr = conv_snr[source_coords_matched_idx].get()

    # Find optimal aperture radius
    opt_radii_idx = np.argmax(center_isolated_snr, axis=0)
    opt_radii = radii[opt_radii_idx]

    if len(center_conv_snr) == 0 or len(opt_radii) == 0:
        raise DataValidationError("center_conv_snr or opt_radii is empty. Ensure valid data is provided.")

    pov = np.polyfit(np.log10(center_conv_snr), np.log10(opt_radii), 1, cov=False)

    # Calculate optimal flux and noise
    source_opt_rad = np.round(
        np.fmax(np.fmin((10 ** (pov[0] * np.log10(conv_snr.get()) + pov[1])), max_radii), min_radii)).astype(int)
    source_opt_rad_idx = np.searchsorted(radii, source_opt_rad, side='right') - 1

    opt_aperture_corrections = np.array(aperture_corrections)[np.arange(source_coord.shape[0]), source_opt_rad_idx]
    opt_aperture_correction_errors = np.array(aperture_correction_errors)[
        np.arange(source_coord.shape[0]), source_opt_rad_idx]
    opt_flux = np.array(source_flux[source_opt_rad_idx, np.arange(source_coord.shape[0])].get())

    mask = opt_flux > 0
    opt_flux = opt_flux[mask]
    opt_aper_corr = np.array(opt_aperture_corrections)[mask]
    opt_corr_err = np.array(opt_aperture_correction_errors)[mask]

    opt_signal = opt_flux / opt_aper_corr

    opt_back_noise_sq = np.abs(back_flux[source_opt_rad_idx, np.arange(source_coord.shape[0])]).get()[mask] * gain
    opt_read_noise_sq = area[source_opt_rad_idx].get()[mask] * rdnoise ** 2
    opt_source_noise_sq = opt_flux * gain / opt_aper_corr ** 2
    opt_corr_noise_sq = (opt_flux * gain * opt_corr_err / opt_aper_corr ** 2) ** 2

    opt_total_noise = np.sqrt(opt_source_noise_sq / n_images +
                              opt_back_noise_sq / n_images +
                              opt_read_noise_sq / n_images +
                              opt_corr_noise_sq) / gain
    opt_coords = source_coord.get()[mask]

    # Obtain aditional information for header purposes 
    ref_snr = [10, 100, 250, 1000]
    extra_info = {}
    for snr in ref_snr:
        ref_idx = np.argmin(np.abs(opt_signal / opt_total_noise - snr))
        extra_info[f'RAD{snr}'] = int(source_opt_rad[mask][ref_idx])
        extra_info[f'CORR{snr}'] = round(opt_aperture_corrections[mask][ref_idx], 3)

    return opt_signal, opt_total_noise, opt_coords, extra_info


@hierarchical_debug(logger)
def batch_aperture_photometry(img, back, positions, radii, **kwargs):
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
    area = cp.zeros(len(radii))
    image_shape = img.shape
    kernel_shape = 2 * radii[-1] + 1, 2 * radii[-1] + 1
    padding = int((kernel_shape[0] - 1) / 2)

    convolved = None
    kernel = None
    img_c = None

    if len(image_shape) == 3:
        flux = cp.zeros((image_shape[0], len(radii), len(positions)))
        back_flux = cp.zeros((image_shape[0], len(radii), len(positions)))
        new_image_shape = fill_image((image_shape[1] + 2 * padding,
                                      image_shape[2] + 2 * padding))
        for c in range(image_shape[0]):
            img_c = cp.fft.rfft2(img[c], s=new_image_shape)
            if back is not None: back_c = cp.fft.rfft2(back, s=new_image_shape)
            for i, r in enumerate(radii):
                kernel, area[i] = get_aper_kernel(r, size=kernel_shape[0])
                kernel = cp.conj(cp.fft.rfft2(kernel, s=new_image_shape))
                convolved = cp.fft.irfft2(img_c * kernel, s=new_image_shape)
                convolved = cp.roll(convolved, shift=[padding, padding], axis=[0, 1])
                convolved = convolved[:image_shape[1], :image_shape[2]]
                flux[c, i, :] = convolved[positions[:, 0], positions[:, 1]]
                if back is not None:
                    convolved = cp.fft.irfft2(back_c * kernel, s=new_image_shape)
                    convolved = cp.roll(convolved, shift=[padding, padding], axis=[0, 1])
                    convolved = convolved[:image_shape[1], :image_shape[2]]
                    back_flux[c, i, :] = convolved[positions[:, 0], positions[:, 1]]
                else:
                    back_flux = None
    else:
        flux = cp.zeros((len(radii), len(positions)))
        back_flux = cp.zeros((len(radii), len(positions)))
        new_image_shape = fill_image((image_shape[0] + 2 * padding,
                                      image_shape[1] + 2 * padding))
        img_c = cp.fft.rfft2(img, s=new_image_shape)
        if back is not None: back_c = cp.fft.rfft2(back, s=new_image_shape)
        for i, r in enumerate(radii):
            kernel, area[i] = get_aper_kernel(r, size=kernel_shape[0])
            kernel = cp.conj(cp.fft.rfft2(kernel, s=new_image_shape))
            convolved = cp.fft.irfft2(img_c * kernel, s=new_image_shape)
            convolved = cp.roll(convolved, shift=[padding, padding], axis=[0, 1])
            convolved = convolved[:image_shape[0], :image_shape[1]]
            flux[i, :] = convolved[positions[:, 0], positions[:, 1]]
            if back is not None:
                convolved = cp.fft.irfft2(back_c * kernel, s=new_image_shape)
                convolved = cp.roll(convolved, shift=[padding, padding], axis=[0, 1])
                convolved = convolved[:image_shape[0], :image_shape[1]]
                back_flux[i, :] = convolved[positions[:, 0], positions[:, 1]]
            else:
                back_flux = None
    if back is not None: del back_c
    if convolved is not None: del convolved
    if kernel is not None: del kernel
    if img_c is not None: del img_c
    mempool.free_all_blocks()
    gc.collect()

    return flux, back_flux, area


@hierarchical_debug(logger)
def calibrate_image(imdata: np.ndarray, inmodel: str, filter: str, scale: float, gain: float, rdnoise: float,
                    exptime: float, satlevel: float, target_ra: float, target_dec: float = None, n_images: int = 1,
                    SP_filt: bool = True, CR_filt: bool = False, border: int = 10, center_factor: float = 0.7,
                    pca_method: bool = True, tile_section: int = 1000, max_stars_ref: int = 15, min_snr: int = 5,
                    color_range: float = 0.6, tile_section_psf: int = 2500, **kwargs):
    """
    Calibrate an image.

    :param imdata: Image data.
    :param inmodel: Instrument model.
    :param scale: Image scale, in arcsec/pixel.
    :param gain: Gain value, in e-/ADU.
    :param rdnoise: Read noise value, in e-.
    :param exptime: Exposure time, in seconds.
    :param satlevel: Saturation level, in ADU.
    :param target_ra: Target right ascension, in degrees.
    :param target_dec: Target declination, in degrees.
    :param n_images: Number of stacked images.
    :param SP_filt: Whether to apply a Salt-and-Pepper filter.
    :param CR_filt: Whether to apply a cosmic ray filter.
    :param border: Distance from the border where sources are ignored.
    :param center_factor: Factor to select the center of the image for reference calculations.
    :param pca_method: Whether to use PCA for PSF fitting.
    :param tile_section: Tile section size for background estimation.
    :param max_stars_ref: Maximum number of stars to use for reference PSF.
    :param min_snr: Minimum SNR for source detection.
    :param color_range: Color range around B-V = 0.65 for zeropoing calculations.
    :param tile_section_psf: Tile section size for aperture corrections variations.
    :return: Calibration dictionary, astrometry dictionary, photometry dataframe.
    """

    mempool = cp.get_default_memory_pool()
    img_cp = cp.asarray(imdata)

    # Get background
    back, _ = get_local_background_fft(img_cp, scale, get_std=False, **kwargs)
    rms = cp.sqrt(back * gain + rdnoise ** 2) / gain / cp.sqrt(n_images)
    xmin = int(imdata.shape[1] * 0.5 * (1 - 0.3))
    xmax = int(imdata.shape[1] * 0.5 * (1 + 0.3))
    ymin = int(imdata.shape[0] * 0.5 * (1 - 0.3))
    ymax = int(imdata.shape[0] * 0.5 * (1 + 0.3))
    m = cp.median(back[ymin:ymax, xmin:xmax])
    s = cp.std(back[ymin:ymax, xmin:xmax])
    mask = cp.abs(back[ymin:ymax, xmin:xmax] - m) < 3 * s
    m = cp.median(back[ymin:ymax, xmin:xmax][mask])
    fluxsky = np.round(m.get(), 6)
    if fluxsky < 0: logger.warning('Median background flux is negative')
    dic_calib = {'FLUXSKY': fluxsky}

    # Detect isolated stars
    if CR_filt:
        img = CR_filter(img_cp - back)
    elif SP_filt:
        img = SP_filter(img_cp - back)  # change SP_filter_cupy to SP_filter
    else:
        img = img_cp - back
    sources = detect_isolated_stars(img[border:-border, border:-border],
                                    rms[border:-border, border:-border],
                                    scale, sat_lim=satlevel * 0.8, **kwargs)
    sources = sources + border

    if len(sources) < 5:
        logger.error('Less than 5 isolated stars detected')
        raise InsufficientStarsError(num_stars=len(sources))

    star_dataset, coord, scaling = create_star_dataset(img, sources, scale)
    center_factor = np.min((center_factor, 1))
    xmin = int(imdata.shape[1] * 0.5 * (1 - center_factor))
    xmax = int(imdata.shape[1] * 0.5 * (1 + center_factor))
    ymin = int(imdata.shape[0] * 0.5 * (1 - center_factor))
    ymax = int(imdata.shape[0] * 0.5 * (1 + center_factor))

    # Find reference psf
    unit_star_dataset = star_dataset.astype(cp.double) / scaling[:, 3][:, None, None]
    center_mask = (coord[:, 0] > xmin) & (coord[:, 0] < xmax) & (coord[:, 1] > ymin) & (coord[:, 1] < ymax)
    unit_star_dataset_stds = cp.std(unit_star_dataset, axis=(1, 2))
    mask_star_dataset = unit_star_dataset_stds < cp.percentile(cp.std(unit_star_dataset, axis=(1, 2)), 95.4)
    unit_star_dataset = unit_star_dataset[mask_star_dataset]
    coord = coord[mask_star_dataset]

    star_dataset_ref = unit_star_dataset[center_mask[mask_star_dataset]][:max_stars_ref, :, :]
    if star_dataset_ref.shape[0] < 5:
        logger.error('Less than 5 isolated stars detected. Image may be too crowded or too noisy')
        raise InsufficientStarsError(num_stars=star_dataset_ref.shape[0])

    psf, _ = stack_sigmaclip(star_dataset_ref, n=2)
    psf = psf / cp.sum(psf)
    try:
        _, _, _, fwhm, _ = fit_moffat(psf.get())
    except:
        logger.error('Error fitting Moffat to reference PSF')
        raise MoffatFitError()
    if fwhm < 2:
        logger.error('Error fitting Moffat to reference PSF')
        raise MoffatFitError()

    dic_calib['FWHM'] = fwhm
    eigen_psfs, coeff_map = None, None
    fwhm = max(fwhm, 2)

    if pca_method:
        # Create psf deviations dataset
        unit_star_dataset_dev = unit_star_dataset - psf
        unit_star_dataset_dev_norm = (unit_star_dataset_dev - cp.mean(unit_star_dataset_dev, axis=(1, 2))[:, None,
                                                              None]) / cp.std(unit_star_dataset_dev, axis=(1, 2))[:,
                                                                       None, None]

        # get eigen psfs
        eigen_psfs = get_eigen_psfs(unit_star_dataset_dev_norm, n_components=5)
        eigen_psfs = cp.asarray(eigen_psfs)
        coefficients = project_all_stars_onto_eigenpsfs(unit_star_dataset_dev, eigen_psfs)
        coeff_map = create_coeff_map(imdata.shape, coord, coefficients.T, scale, tile_section=tile_section)

        mempool.free_all_blocks()
        gc.collect()

        x1 = int(imdata.shape[1] * 0.25)
        x2 = int(imdata.shape[1] * 0.75)
        y1 = int(imdata.shape[0] * 0.25)
        y2 = int(imdata.shape[0] * 0.75)
        x = np.array([x1, x2, x1, x2])
        y = np.array([y1, y2, y2, y1])
        fwhm_lab = ['FWHMLL', 'FWHMLR', 'FWHMUL', 'FWHMUR']
        # fwhms = np.zeros(len(fwhm_lab))
        for point in range(len(fwhm_lab)):
            psf_l = psf + recreate_normed_star(coeff_map, eigen_psfs, (x[point], y[point]))
            try:
                _, _, _, fwhm_l, _ = fit_moffat(psf_l.get())
                fwhm_l = np.round(fwhm_l, 2)
            except:
                fwhm_l = 0
            dic_calib[fwhm_lab[point]] = fwhm_l

    # Detect sources
    sources, conv_ima_sigma = detect_sources_psf(img, rms, fwhm, psf, eigen_psfs, coeff_map, min_snr=min_snr, **kwargs)
    sources = sources[(sources[:, 0] > border) & (sources[:, 0] < img.shape[0] - border) & (sources[:, 1] > border) & (
            sources[:, 1] < img.shape[1] - border)]

    del coeff_map, img, rms
    mempool.free_all_blocks()

    # Perform optimized photometry

    # optimal_flux, optimal_noise, optimal_coords, extra_info = perform_opt_photometry(img_cp - back, back,
    #                                                                                  conv_ima_sigma, sources,
    #                                                                                  coord, tile_section_psf,
    #                                                                                  star_dataset[mask_star_dataset],
    #                                                                                  fwhm, gain, n_images, rdnoise)

    # TODO: Revisar por Miguel: dejamos center_factor y min_conv_snr con valor dor defecto de la función, o por defecto de DEFAULT_PROCESSING_PARAMS
    optimal_flux, optimal_noise, optimal_coords, extra_info = perform_opt_photometry(
        img=img_cp - back,
        back=back,
        conv_ima_sigma=conv_ima_sigma,
        source_coord=sources,
        isolated_coord=coord,
        tile_section_psf=tile_section_psf,
        star_dataset=star_dataset[mask_star_dataset],
        fwhm=fwhm,
        gain=gain,
        rdnoise=rdnoise,
        n_images=n_images,
        # center_factor=kwargs.get('center_factor', 1.0),
        # min_conv_snr=kwargs.get('min_conv_snr', 300.0)
    )

    del img_cp, back, conv_ima_sigma
    mempool.free_all_blocks()

    dic_calib.update(extra_info)

    dfm = pd.DataFrame({'xcentroid': optimal_coords[:, 1],
                        'ycentroid': optimal_coords[:, 0],
                        'flux': optimal_flux,
                        'noise': optimal_noise,
                        'snr': optimal_flux / optimal_noise})
    dfm_ast = dfm.sort_values('snr', ascending=False).dropna().reset_index(drop=True)

    # Astrometrize
    h_wcs = astrometrice2(dfm_ast, scale, target_ra, target_dec, sip_order=1)
    if h_wcs == {}:
        logger.error('Astrometry failed')
        return dfm, h_wcs, dic_calib

    else:
        # Photometrize
        coocenter, FOV, scale = get_astrometry_params(h_wcs, imdata.shape)
        result, catalog, ref_filter = catalog_results(coocenter, FOV / 2,
                                                      filter, inmodel, maglimit=23, **kwargs)
        dic_calib['CATALOG'] = catalog
        dic_calib['CATBAND'] = ref_filter

        w = WCS(h_wcs)
        ra, dec = w.all_pix2world(dfm.xcentroid.values, dfm.ycentroid.values, 1)
        dfm.loc[:, 'RA'] = ra
        dfm.loc[:, 'DEC'] = dec
        photo_dict = get_zeropoint(result, dfm, exptime, center_lims=(xmin, xmax, ymin, ymax),
                                   solar_filter=color_range, dist_thres_px=1.5 * fwhm * scale / 3600)
        dic_calib.update(photo_dict)

        # Check catalog coincidence
        dfm_idx, catalog_idx = crossmatch_sources(dfm[['RA', 'DEC']].values,
                                                  result[['RA', 'DEC']].values,
                                                  thres_px=1.5 * fwhm * scale / 3600)

        # Add astrometric errors to dfm
        dfm.loc[dfm_idx, 'RAERR'] = dfm.loc[dfm_idx, 'RA'].values - result.loc[catalog_idx, 'RA'].values
        dfm.loc[dfm_idx, 'DECERR'] = dfm.loc[dfm_idx, 'DEC'].values - result.loc[catalog_idx, 'DEC'].values
        dic_calib['RAPREC'] = np.round(np.nanmedian(dfm.RAERR) * 3600, 3)
        dic_calib['DECPREC'] = np.round(np.nanmedian(dfm.DECERR) * 3600, 3)
        dic_calib['RADISP'] = np.round(np.nanstd(dfm.RAERR) * 3600, 3)
        dic_calib['DECDISP'] = np.round(np.nanstd(dfm.DECERR) * 3600, 3)

        # Calculate limiting magnitude
        try:
            mag = dic_calib['ZP'] - 2.5 * np.log10(dfm.flux.values / exptime)
            snr = dfm.snr.values
            maglim3 = get_maglim(mag, snr, 3)
        except Exception as e:
            maglim3 = 0
            logger.warning('Error calculating limiting magnitude: {}'.format(e))
        dic_calib['MAGLIM'] = maglim3

        # Calculate target SNR
        try:
            if target_ra and target_dec:
                target_snr = get_target_snr(dfm, target_ra, target_dec, dist_thres_px=1.5 * fwhm)
            else:
                target_snr = 0
        except Exception as e:
            target_snr = 0
            logger.warning('Error calculating target SNR: {}'.format(e))
        dic_calib['OBJECSNR'] = np.round(target_snr, 2)

    return dfm, h_wcs, dic_calib


@hierarchical_debug(logger)
def aperture_photometry(img, positions, aper_rad, **kwargs):
    """Perform aperture photometry.

    :param img: Image data.
    :type img: ndarray
    :param positions: Positions of sources.
    :type positions: ndarray
    :param aper_rad: Aperture radius.
    :type aper_rad: int

    
    """
    kernel, area = get_aper_kernel(aper_rad)
    conv_ima = convolve_fft(img, kernel, **kwargs)
    positions = cp.array(cp.round(positions)).astype(cp.int32)
    flux = conv_ima[positions[:, 0], positions[:, 1]]
    del conv_ima, kernel, positions

    return flux, area


@hierarchical_debug(logger)
def get_fwhm_mof(model, img, step=50, ns=25, mins=3, **kwargs):
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
def cov_nan(img, nc=10, **kwargs):
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
#     arcsec_per_pixel = plate_scale_px(head[HeaderKey.PXSIZE.value], head[HeaderKey.FOCALEN.value]) * head[
#         HeaderKey.XBINNING.value]
#     signal.signal(signal.SIGALRM, handler)
#     signal.alarm(120)
#     try:
#         solution = get_solver().solve(stars_xs=dfm['xcentroid'], stars_ys=
#         dfm['ycentroid'], size_hint=astrometry.SizeHint(
#             lower_arcsec_per_pixel=arcsec_per_pixel * 0.8,
#             upper_arcsec_per_pixel=arcsec_per_pixel * 1.2), position_hint=
#                                       astrometry.PositionHint(ra_deg=head[HeaderKey.POINTRA.value] * 360 / 24,
#                                                               dec_deg=head[HeaderKey.POINTDEC.value], radius_deg=0.5),
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

#
# @hierarchical_debug(logger)
# def get_zeropoint(df_catalog, flux, noise, coord, exptime, solar_filter=0.3,
#                   dist_thres_px=3, N=50, plot=False):
#     """Calculate the zeropoint for photometry.
#
#     :param df_catalog: Catalog dataframe.
#     :type df_catalog: pd.DataFrame
#     :param flux: Flux values.
#     :type flux: ndarray
#     :param noise: Noise values.
#     :type noise: ndarray
#     :param coord: Coordinates of sources.
#     :type coord: ndarray
#     :param exptime: Exposure time.
#     :type exptime: float
#     :param solar_filter: Solar filter, by default 0.3.
#     :type solar_filter: float, optional
#     :param dist_thres_px: Distance threshold in pixels, by default 3.
#     :type dist_thres_px: int, optional
#     :param N: Number of brightest stars to use, by default 50.
#     :type N: int, optional
#     :param plot: Whether to plot the results, by default False.
#     :type plot: bool, optional
#
#
#     """
#     cat_coords = np.array([df_catalog['Y'], df_catalog['X']]).T
#     source_coords_matched_idx, ref_coords_matched_idx = crossmatch_sources(
#         coord, cat_coords, thres_px=dist_thres_px)
#     solar_cat_filt = np.abs(df_catalog['SOLAR'])[ref_coords_matched_idx
#                      ] < solar_filter
#     source_coords_matched_idx = source_coords_matched_idx[solar_cat_filt]
#     ref_coords_matched_idx = ref_coords_matched_idx[solar_cat_filt]
#     det_mag = -2.5 * np.log10(flux[source_coords_matched_idx] / exptime)
#     cat_mag = df_catalog['MAG'][ref_coords_matched_idx].to_numpy()
#     inf_nan_mask = np.isfinite(cat_mag) & np.isfinite(det_mag) & ~np.isnan(
#         cat_mag) & ~np.isnan(det_mag)
#     cat_mag = cat_mag[inf_nan_mask]
#     det_mag = det_mag[inf_nan_mask]
#     snrs = (flux / noise)[source_coords_matched_idx][inf_nan_mask]
#     m = np.argsort(snrs)
#     brightest = m[-N:]
#     bright_mask = np.zeros(len(cat_mag), dtype=bool)
#     bright_mask[brightest] = True
#     if len(cat_mag) <= 3:
#         zp, ezp, n, min_mag, max_mag = 0, 0, 0, 0, 0
#     else:
#         y = cat_mag[bright_mask] - det_mag[bright_mask]
#         mask = np.abs(y - np.nanmean(y)) < np.nanstd(y)
#         if np.sum(mask) > 15:
#             reg = RANSACRegressor(random_state=42, residual_threshold=0.05
#                                   ).fit(det_mag[bright_mask].reshape([-1, 1])[mask], cat_mag[
#                 bright_mask].reshape([-1, 1])[mask])
#             inlier = reg.inlier_mask_
#             if np.sum(inlier) > 10 and np.abs(np.mean(y[mask][inlier]) - np
#                     .mean(y[mask])) < np.std(y[mask]):
#                 zp = np.mean(y[mask][inlier])
#                 n = np.sum(inlier)
#                 ezp = np.std(y[mask][inlier]) / np.sqrt(n)
#                 min_mag = np.min(cat_mag[bright_mask][mask][inlier])
#                 max_mag = np.max(cat_mag[bright_mask][mask][inlier])
#             else:
#                 zp = np.mean(y[mask])
#                 n = np.sum(mask)
#                 ezp = np.std(y[mask]) / np.sqrt(n)
#                 min_mag = np.min(cat_mag[bright_mask][mask])
#                 max_mag = np.max(cat_mag[bright_mask][mask])
#         else:
#             zp = np.mean(y[mask])
#             n = np.sum(mask)
#             ezp = np.std(y[mask]) / np.sqrt(n)
#             min_mag = np.min(cat_mag[bright_mask][mask])
#             max_mag = np.max(cat_mag[bright_mask][mask])
#     if plot:
#         plt.figure(figsize=(8, 8))
#         ax = plt.subplot(111)
#         ax.plot(cat_mag, cat_mag - det_mag - zp, 'k.', alpha=0.6)
#         if np.sum(mask) > 15:
#             ax.plot(cat_mag[bright_mask][mask], cat_mag[bright_mask][mask] -
#                     det_mag[bright_mask][mask] - zp, 'b.', alpha=0.1)
#             ax.plot(cat_mag[bright_mask][mask][inlier], cat_mag[bright_mask
#             ][mask][inlier] - det_mag[bright_mask][mask][inlier] - zp,
#                     'r.', label='zp = {:.3f} +/- {:.3f} (n={})'.format(zp, ezp,
#                                                                        n), alpha=0.5)
#         else:
#             ax.plot(cat_mag[bright_mask][mask], cat_mag[bright_mask][mask] -
#                     det_mag[bright_mask][mask] - zp, 'r.', label=
#                     'zp = {:.3f} +/- {:.3f} (n={})'.format(zp, ezp, n), alpha=0.5)
#         ax.set_xlabel('catalog magnitude')
#         ax.set_ylabel('error magnitude')
#         ax.legend(frameon=False)
#         ax.set_ylim(-0.5, 0.5)
#         plt.show()
#
#     return zp, ezp, n, min_mag, max_mag


# @hierarchical_debug(logger)
# def delete_header_from(header, val):
#     """Delete a section from the FITS header.
#
#     :param header: FITS header.
#     :type header: dict
#     :param val: Value to delete.
#     :type val: str
#
#
#     """
#     for i, v in enumerate(header.values()):
#         if val in str(v):
#             idx = i - 1
#     for i in range(len(header) - idx):
#         del header[idx]
#
#     return header


@hierarchical_debug(logger)
def sample_im(img, nc=50, ns=100, **kwargs):
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
def pred_mof(pred, **kwargs):
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
def sigma_clip(img, sclip, **kwargs):
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
def gen_gauss_filter(fw, **kwargs):
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
=4, mincut=10, mem=cp.get_default_pinned_memory_pool(), **kwargs):
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
def get_peak_image(img, positions, aper_rad, **kwargs):
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
                        resul = SP_filter(image_cp)
                        end_time = time.time()

                    except Exception as e:
                        print(f'Error processing {image_path}: {e}')
                        print('Traceback:')
                        traceback.print_exc()
