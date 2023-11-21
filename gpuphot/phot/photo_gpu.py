import logging
import os
import signal
from pathlib import Path

import astrometry
import cupy as cp
import numpy as np
import pandas as pd
import tensorflow as tf

# from astropy.stats import sigma_clip as sclip
from astroquery.astrometry_net import AstrometryNet
from cupyx.scipy.ndimage import gaussian_filter, label, convolve, maximum_filter
from cupyx.scipy.ndimage import mean as nd_mean, sum as nd_sum
from sklearn.linear_model import RANSACRegressor

# from tensorflow.keras.models import load_model
from tensorflow import keras

from gpuphot.astrometry.utils import (
    cat_input_from_header,
    get_ccw,
    get_if_header_already_post_processed,
)
from gpuphot.phot.catalog import catalog_results, catalog_match
from gpuphot.stats.utils import free_gpu_mem

# import ttt.equipment.models as db
from .utils import plate_scale_px

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

logger = logging.getLogger("photo_gpu")


### FW Model


def scan_im(im, nc=50):
    img = im
    delta = np.round((img.shape[0] - 512) / nc).astype(int)
    pi = 512 * 512
    lim = []
    cmax = []

    for i in range(nc):
        for j in range(nc):
            ii = img[(delta * i) : (delta * i + 512), (delta * j) : (delta * j + 512)]
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
    return (iac2, icmax2)


def sample_im(img, nc=50, ns=100):
    delta = np.round((img.shape[0] - 512) / nc).astype(int)
    pi = 512 * 512
    lim = []
    cmax = []
    np.random.seed(42)
    for k in range(ns):
        i = np.random.randint(nc)
        j = np.random.randint(nc)
        ii = img[(delta * i) : (delta * i + 512), (delta * j) : (delta * j + 512)]
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
    return (iac2, icmax2)


def pred_mof(pred):
    alpha = pred[:, 1]
    beta = pred[:, 0] * 0.4 + 4.565
    nstar = pred[:, 2] * 200
    fwhm = 2 * alpha * np.sqrt(2 ** (1 / beta) - 1)
    return alpha, beta, nstar, fwhm


def pred_ga(pred):
    fwhm = pred[:, 0]
    nstar = pred[:, 1] * 100
    return nstar, fwhm


def get_fwhm_ga(model, img, step=50, ns=25, mins=3):
    ims, cs = sample_im(img, step, ns)
    pred2 = model.predict(ims)
    nstar, fwhm = pred_ga(pred2)
    fws = fwhm[nstar > np.mean(nstar)]
    return np.mean(fws), np.std(fws)


def get_fwhm_mof(model, img, step=50, ns=25, mins=3):
    ims, cs = sample_im(img, step, ns)
    pred2 = model.predict(ims)
    alpha, beta, nstar, fwhm = pred_mof(pred2)
    fws = fwhm[nstar >= np.mean(nstar)]
    return np.mean(fws), np.std(fws), np.mean(alpha), np.mean(beta)


################ Photometry


def sigma_clip(img, sclip):
    img0 = img.copy()
    for i in range(5):
        imed = cp.nanmean(img0)
        rms = cp.nanstd(img0)
        img0[img0 >= imed + sclip * rms] = cp.nan
        img0[img0 <= imed - sclip * rms] = cp.nan
    del img0
    return (imed, rms)  # FIXME imed and rms are not used


def gen_ap_filter(lk):
    k_dim = (2 * lk + 1, 2 * lk + 1)
    indi = cp.indices(k_dim)
    fw2 = lk**2
    k_app = cp.zeros(k_dim)
    struc = cp.where(((lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2) < fw2)
    k_app[struc] = 1
    return k_app


def gen_apm_filter(lk):
    k_dim = (2 * lk + 1, 2 * lk + 1)
    indi = cp.indices(k_dim)
    fw2 = lk**2
    k_app = cp.zeros(k_dim)
    struc = cp.where(((lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2) < fw2)
    k_app[struc] = 1
    k_app = k_app / k_app.sum()
    return k_app


def gen_gauss_filter(fw):
    sigma_r = fw / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_r2 = sigma_r * sigma_r
    lk = np.ceil(sigma_r).astype(np.int16) * 4
    k_dim = (2 * lk + 1, 2 * lk + 1)
    indi = cp.indices(k_dim)
    r2 = (lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2
    ker = cp.exp(-r2 / 2 / sigma_r2)
    ksum = cp.sum(ker)
    ksum2 = cp.sum(ker * ker)
    n = k_dim[0] ** 2
    k_app = (ker - ksum / n) / (ksum2 - ksum * ksum / n)
    return (k_app, lk)


def gen_moff_filter(alpha, beta):
    fw = alpha * (2 * np.sqrt(2 ** (1 / beta) - 1))
    sigma_r = fw / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    # sigma_r2 = sigma_r*sigma_r
    lk = np.ceil(sigma_r).astype(np.int16) * 4
    k_dim = (2 * lk + 1, 2 * lk + 1)
    indi = cp.indices(k_dim)
    r2 = (lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2
    ker = (1 + r2 / alpha**2) ** (-beta)
    ksum = cp.sum(ker)
    ksum2 = cp.sum(ker * ker)
    n = k_dim[0] ** 2
    k_app = (ker - ksum / n) / (ksum2 - ksum * ksum / n)
    return (k_app, lk)


def gen_moff_filter2(alpha, beta):
    alpha = alpha / 2
    fw = alpha * (2 * np.sqrt(2 ** (1 / beta) - 1))
    sigma_r = fw / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    # sigma_r2 = sigma_r*sigma_r
    lk = np.ceil(sigma_r).astype(np.int16) * 4
    k_dim = (2 * lk + 1, 2 * lk + 1)
    indi = cp.indices(k_dim)
    r2 = (lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2
    ker = (1 + r2 / alpha**2) ** (-beta)
    ksum = cp.sum(ker)
    k_app = (ker) / ksum
    return (k_app, lk)


def cov_nan(img, nc=10):
    delta = np.round((img.shape[0]) / nc).astype(int)
    for i in range(nc):
        for j in range(nc):
            ii = img[(delta * i) : (delta * (i + 1)), (delta * j) : (delta * (j + 1))]
            ii[cp.isnan(ii)] = cp.nanmean(ii)
            img[(delta * i) : (delta * (i + 1)), (delta * j) : (delta * (j + 1))] = ii
    img[cp.isnan(img)] = cp.nanmean(img)
    img[cp.isinf(img)] = cp.nanmean(img)
    return img


def get_sky(im_g, fw, qt=90, mem=cp.get_default_memory_pool()):
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


def daofind_gpu_fast(
    img,
    sky,
    rms,
    sdet,
    mode="g",
    fw=0,
    alpha=0,
    beta=0,
    mem=cp.get_default_pinned_memory_pool(),
):
    thres = sdet * cp.mean(rms)
    sigma_r = fw / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    if mode == "g":
        gf, lk = gen_gauss_filter(sigma_r)
    else:
        gf, lk = gen_moff_filter(alpha, beta)

    sky = gaussian_filter(sky, 3 * lk)
    rms = gaussian_filter(rms, 3 * lk)

    g = convolve((img - sky), gf, origin=(0, 0))
    g1 = ((g) / rms > sdet).astype(cp.int32)
    lbs = label(g1)
    ids = cp.asarray([range(lbs[1])])
    npix = nd_sum(g1, lbs[0], ids)
    del g1
    mem.free_all_blocks()
    idx = cp.indices(img.shape, dtype=cp.int16)
    im1 = g * idx

    x = (nd_mean(im1[0, :, :], lbs[0], ids) / nd_mean(g, lbs[0], ids)) + 1
    y = (nd_mean(im1[1, :, :], lbs[0], ids) / nd_mean(g, lbs[0], ids)) + 1

    coor = (cp.round(x).astype(cp.int), cp.round(y).astype(cp.int))

    lk = np.round(lk * 3).astype(np.int)
    # sk_s = sky[coor]*np.pi*lk**2
    sk_s = (sky[coor] - cp.mean(sky)) / rms[coor]
    del rms
    mem.free_all_blocks()

    k_app = gen_ap_filter(lk)

    fot_p = maximum_filter(g, footprint=k_app, origin=(0, 0))
    mm = cp.get_default_memory_pool().used_bytes()
    flux_p = fot_p[coor] / thres
    del (g, fot_p)
    mem.free_all_blocks()

    mpea = maximum_filter(img - sky, footprint=k_app, origin=(0, 0))
    peak = mpea[coor]
    del mpea
    fot_a = convolve(img - sky, k_app, origin=(0, 0))
    flux_a = fot_a[coor]  # - sky[coor]*np.pi*lk**2
    del (fot_a, k_app)
    mem.free_all_blocks()

    res = (
        np.asarray(
            [
                y.get(),
                x.get(),
                peak.get(),
                flux_a.get(),
                flux_p.get(),
                npix.get(),
                sk_s.get(),
            ]
        )
        .transpose()
        .reshape((-1, 7))
    )
    df = pd.DataFrame(
        res,
        columns=["xcentroid", "ycentroid", "peak", "flux_a", "flux_p", "npix", "sks"],
    )
    df = df[df.flux_a > 0]
    return df, lk


def detect_gpu(
    img,
    sky,
    rms,
    sdet,
    mode="g",
    fw=1,
    alpha=0,
    beta=0,
    minpix=4,
    mincut=10,
    mem=cp.get_default_pinned_memory_pool(),
):
    thres = sdet * cp.mean(rms)
    gf, lk = gen_gauss_filter(fw)
    g = convolve((img - sky), gf, origin=(0, 0))
    # g= img-sky
    g1 = ((g) / rms > sdet).astype(cp.int32)
    del rms
    mem.free_all_blocks()
    label_im, nb_labels = label(g1)
    ids0 = cp.asarray([range(nb_labels + 1)])
    npix = nd_sum(g1, label_im, ids0)

    # remove small regions
    ids = ids0[(npix > mincut) & (npix > 0)]
    idm = ids0[(npix <= minpix) & (npix > 0)]
    npix = npix[(npix > mincut) & (npix > 0)]
    if minpix == 0:
        mask = False
    else:
        mask = cp.isin(label_im, cp.asarray(idm))
    # print(label_im[:10,:10])

    del g1
    mem.free_all_blocks()

    idx = cp.indices(img.shape, dtype=cp.int16)
    im1 = g * idx

    x = nd_mean(im1[0, :, :], label_im, ids) / nd_mean(g, label_im, ids)
    y = nd_mean(im1[1, :, :], label_im, ids) / nd_mean(g, label_im, ids)

    el = nd_mean(im1[0, :, :] * im1[1, :, :], label_im, ids) / nd_mean(g, label_im, ids)
    el = el - x * y
    coor = (x.astype(cp.int), y.astype(cp.int))
    mm = cp.get_default_memory_pool().used_bytes()

    flux = g[coor]
    del g
    mem.free_all_blocks()

    res = (
        np.asarray([y.get(), x.get(), flux.get(), npix.get(), el.get()])
        .transpose()
        .reshape((-1, 5))
    )
    df = pd.DataFrame(res, columns=["xcentroid", "ycentroid", "flux", "npix", "elip"])
    # df = df[df.flux>0]
    return (df, mask, mm)


def init_gpu():
    print("Tensorflow version " + tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_logical_device_configuration(
        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
    )


def get_fwhm_model(model_path=Path(__file__).parent.parent):
    name = f"{model_path}/fwhm/fwhm_3_2_mofatt_ns_mix_100_model"
    if tf.__version__ == "2.4.1":
        name = name + "_old"
    model = keras.models.load_model(name)
    return model


def get_detections(model_mo, im, det=2, gain=1.024, rdnoise=2.3, scale=0.21):
    fw, efw, alpha, beta = get_fwhm_mof(model_mo, im, step=50, ns=50)
    print("FWHM = " + str(fw))

    im_g = cp.asarray(im)
    sky, rms, _ = get_sky(im_g, fw, qt=80)

    print("Sky:", sky.mean(), " RMS: ", rms.mean())

    dfm, lk = daofind_gpu_fast(im_g, sky, rms, det, mode="m", alpha=alpha, beta=beta)
    # dfm = dfm[dfm.sks>0]

    dfm["snr"] = (
        (dfm.flux_a)
        * gain
        / np.sqrt((dfm.flux_a + dfm.sks) * gain + rdnoise**2 * np.pi * lk**2)
    )
    dfm = dfm[dfm.xcentroid == dfm.xcentroid]
    dfm = dfm[dfm.flux_a == dfm.flux_a]
    a = (dfm.peak) / dfm.flux_a
    dfm = dfm[a < scale / 2]
    dfm["idx"] = np.round(dfm.xcentroid) * im.shape[0] + np.round(dfm.ycentroid)
    del im_g
    free_gpu_mem()
    return dfm, sky, rms, fw, efw, 2 * lk + 1


# calculate error in regresion parameters
def err_reg(reg, X, y, yt):
    N = y.shape[0]
    if N < 4:
        return (1.0, 1.0)
    p = len(reg.estimator_.coef_) + 1
    X_with_intercept = np.empty(shape=(N, p), dtype=float)
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = X
    residuals = yt - y
    residual_sum_of_squares = residuals.T @ residuals
    sigma_squared_hat = residual_sum_of_squares / (N - p)
    var_beta_hat = (
        np.linalg.inv(X_with_intercept.T @ X_with_intercept) * sigma_squared_hat
    )
    err = (var_beta_hat[p_, p_] ** 0.5 for p_ in range(p))
    return err


def get_calibration(dfm, head, fw, cat_sources):
    scale = head["SCALEORI"]
    if len(cat_sources) == 0:
        coocenter, FOV, filter, scale = cat_input_from_header(head)
        print(coocenter, FOV, filter, scale)
        cat_sources, cat, labels = catalog_results(coocenter, FOV, filter, maglimit=23)

        print(len(cat_sources))

    fwhm = fw * scale
    df, dfm = catalog_match(dfm, cat_sources, head, fwhm)
    df = df[df.flux_a > 0]
    mask = (df.snr > 10) & (df.mask_solar)
    if sum(mask) < 10:
        mask = df.snr > 0
    dfm["magc"] = -1.0
    zp = 0.0
    ezp = 0.0
    mag_lim = 0.0
    if sum(mask) > 10:
        expt = head["INTEGT"] / head["TOTIMA"]

        X = -2.5 * (np.log10(df.flux_a.values[mask] / expt)).reshape([-1, 1])
        y = df.rmag[mask]

        reg = RANSACRegressor(random_state=42).fit(X, y)
        Xt = -2.5 * (np.log10(dfm.flux_a.values / expt)).reshape([-1, 1])
        yt = reg.predict(Xt)
        yp = reg.predict(X)
        dfm["magc"] = yt

        mask = reg.inlier_mask_

        delta = X[:, 0] - y
        # clipped_val = sclip(delta, sigma=3, masked=True)
        # mask = np.logical_not(clipped_val.mask)

        zp1 = -np.mean(delta[mask])
        ezp1 = np.std(delta[mask])

        zp = reg.estimator_.intercept_
        sl = reg.estimator_.coef_

        ezp, esl = err_reg(reg, X[mask], y[mask], yp[mask])

        print(zp, ezp, sl, esl, zp1, ezp1, 1, 0)

        mask = df.snr > 0
        y = df.rmag[mask]
        X = np.log10(df.snr[mask]).values.reshape([-1, 1])

        reg = RANSACRegressor(random_state=42).fit(X, y)
        mag_lim = reg.predict(np.log10(3).reshape(1, -1)).tolist()[0]

        # mag_lim = df3[df3.snr>3].rmag.max()

    df3 = pd.merge(
        dfm, df[["idx", "objID", "rmag", "RA", "DEC", "xcat", "ycat"]], how="left"
    )

    return df3, cat_sources, scale, zp, ezp, mag_lim


def process_image(model_mo, im, head, det=2, astrom=True, cat=[]):
    gain = head["GAIN"]
    rnois = head["RDNOISE"]
    scale = np.round(plate_scale_px(head["PXSIZE"], head["FOCALEN"]), 3)
    # scale = head['SCALEORI']

    dfm, sky, rms, fw, efw, ap = get_detections(
        model_mo, im, det=det, gain=gain, rdnoise=rnois, scale=scale
    )
    sm = np.mean(sky).tolist()
    rm = np.mean(rms).tolist()

    del sky, rms
    free_gpu_mem()

    logger.info(
        f"Detection: fw:{fw},sky:{sm},rms:{rm}", extra={"fw": fw, "sky": sm, "rms": rm}
    )

    dfm = dfm.sort_values("flux_p", ascending=False)
    dfm = dfm[dfm.npix > fw**2]

    if dfm.shape[0] == 0:
        return None, 0, 0

    #     #Temporary fix for old bugged astrometrizations
    #     px = head['NAXIS1']/2
    #     py = head['NAXIS2']/2
    #     w = WCS(head)
    #     ra, dec = w.wcs_pix2world(px, py, 1)
    #     ra = ra.tolist()
    #     dec = dec.tolist()
    #     head['RA'] = ra
    #     head['DEC'] = dec

    if astrom:
        head = astrometrice2(dfm, head, im.shape)
        # return None,head,None

    if head != None:
        df3, cat, scale, zp, ezp, magl = get_calibration(dfm, head, fw, cat)

        print(zp, ezp, magl)

        nmatches = len(df3[df3.rmag == df3.rmag])
        print("Matches: ", nmatches)

        ccw = get_ccw(head)

        phot_exists = get_if_header_already_post_processed(head, "PHOTOMETRY")
        if not phot_exists:
            head["COMINIT"] = "e"
            head.insert("COMINIT", ("COMMENT", "***************************"))
            head.insert("COMINIT", ("COMMENT", "       PHOTOMETRY          "))
            head.insert("COMINIT", ("COMMENT", "***************************"))

        fovx, fovy = im.shape
        fovx = fovx * scale / 3600
        fovy = fovy * scale / 3600
        if sm > 0:
            m_sky = zp - 2.5 * np.log(sm / head["INTEGT"] * head["TOTIMA"])
        else:
            m_sky = zp

        stats = {
            "FoVx": (fovx, "Image horizontal axis Fiel of View(deg)"),
            "FoVy": (fovy, "Image vertical axis Fiel of View(deg)"),
            "ZP": (zp, "Zero point"),
            "eZP": (ezp, "Zero point's error"),
            "FWHM": (fw, "Full width"),
            "eFWHM": (efw, "Full width's error"),
            "m_lim": (magl, "Limit magnitude (3 sigmas)"),
            "m_sky": (m_sky, "Sky's magnitud"),
            "sky": (sm, "Sky flux"),
            "esky": (rm, "Sky's flux error"),
            "scale": (scale, "Image scale in arcsec"),
            "ccw": (ccw, "Field rotation"),
            "ap_snr": (ap, "Opening radius maximising snr"),
            "ap_flux": (ap, "Opening radius maximising flux"),
        }
        # print(stats)
        for v in stats:
            head[v] = stats[v]
        if not phot_exists:
            del head["COMINIT"]

        return df3, head, cat
    return dfm, 0, 0


def astrometrice(dfm, im_shape, api_key):
    ast = AstrometryNet()
    ast.api_key = api_key
    image_width = im_shape[0]
    image_height = im_shape[1]
    h_wcs = ast.solve_from_source_list(
        dfm["xcentroid"][0:100],
        dfm["ycentroid"][0:100],
        image_width,
        image_height,
        crpix_center=True,
        solve_timeout=120,
    )
    # print(h_wcs)
    return h_wcs
    # for v in h_wcs:
    #     head[v] = h_wcs[v]
    #
    # astro_exists = get_if_header_already_post_processed(head, "ASTROMETRY")
    # if not astro_exists:
    #     head['COMINIT'] = 'e'
    #     head.insert('COMINIT', ('COMMENT', '***************************'))
    #     head.insert('COMINIT', ('COMMENT', '       ASTROMETRY          '))
    #     head.insert('COMINIT', ('COMMENT', '***************************'))
    #
    #     px = head['NAXIS1'] / 2
    #     py = head['NAXIS2'] / 2
    #     w = WCS(h_wcs)
    #     ra, dec = w.wcs_pix2world(px, py, 1)
    #     ra = ra.tolist()
    #     dec = dec.tolist()
    #
    #     head['RA'] = ra
    #     head['DEC'] = dec
    #
    #     c1 = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
    #     c2 = SkyCoord(head['POINTRA'] * 360 / 24 * u.deg, head['POINTDEC'] * u.deg, frame='icrs')
    #     sep = (c2.separation(c1)).arcsecond
    #     print('Error de apuntado: ', sep, ' arcsec')
    #
    # if not astro_exists:
    #     ra_hms, dec_dms = deg_to_hms(ra, dec)
    #
    #     try:
    #         elev = head['SITEALT']
    #     except:
    #         elev = head['SITEALT']
    #
    #     az, alt, airmass, zd = radec_to_altaz(ra, dec, head['SITELAT'], head['SITELONG'], elev, head['DATE-OBS'])
    #     longal, latgal = radec_to_gal(ra, dec)
    #     lonecl, latecl = radec_to_ecl(ra, dec)
    #
    #     head.insert('COMINIT', ('RA', ra, db.Header.objects.get(name='RA').description))
    #     head.insert('COMINIT', ('DEC', dec, db.Header.objects.get(name='DEC').description))
    #     head.insert('COMINIT', ('RAhms', ra_hms, db.Header.objects.get(name='RAhms').description))
    #     head.insert('COMINIT', ('DECdms', dec_dms, db.Header.objects.get(name='DECdms').description))
    #     head.insert('COMINIT', ('AZ', az, db.Header.objects.get(name='AZ').description))
    #     head.insert('COMINIT', ('ALT', alt, db.Header.objects.get(name='ALT').description))
    #     head.insert('COMINIT', ('ZD', zd, db.Header.objects.get(name='ZD').description))
    #     head.insert('COMINIT', ('AIRMASS', airmass, db.Header.objects.get(name='AIRMASS').description))
    #     head.insert('COMINIT', ('LONGAL', longal, db.Header.objects.get(name='LONGAL').description))
    #     head.insert('COMINIT', ('LATGAL', latgal, db.Header.objects.get(name='LATGAL').description))
    #     head.insert('COMINIT', ('LONECL', lonecl, db.Header.objects.get(name='LONECL').description))
    #     head.insert('COMINIT', ('LATECL', latecl, db.Header.objects.get(name='LATECL').description))
    #
    #     del head['COMINIT']
    #
    # # Parche hasta que instroduzca en los raw
    # if 'JD' not in head:
    #     try:
    #         jd, mjd = date_to_jd(head['DATE-OBS'])
    #         head.insert('PCDATE', ('JD-OBS', jd, db.Header.objects.get(name='JD-OBS').description))
    #         head.insert('PCDATE', ('MJD-OBS', mjd, db.Header.objects.get(name='MJD-OBS').description))
    #     except:
    #         pass
    #
    # if 'SCALEORI' not in head:
    #     try:
    #         head.insert('PARITY', ('SCALEORI', np.round(plate_scale_px(head['PXSIZE'], head['FOCALEN']), 3),
    #                                db.Header.objects.get(name='SCALEORI').description))
    #     except:
    #         pass
    #
    # return wcs_header


def logodds_callback3(logodds):
    if len(logodds) < 3:
        return astrometry.Action.CONTINUE
    if logodds[1] > logodds[0] - 10 and logodds[2] > logodds[0] - 10:
        return astrometry.Action.STOP
    return astrometry.Action.CONTINUE


def logodds_callback_100(logodds):
    # print(logodds)
    if (logodds[0] > 200.0) | (len(logodds) > 2):
        return astrometry.Action.STOP
    else:
        return astrometry.Action.CONTINUE


# Register an handler for the timeout
def handler(signum, frame):
    print("Astrometrization timeout!")
    raise Exception("end of time")


def astrometrice2(dfm, header, im_shape, cache="/data/astrometry_cache"):
    if not os.path.exists(cache):
        os.makedirs(cache)

    solver = astrometry.Solver(
        astrometry.series_5200.index_files(
            cache_directory=cache,
            scales={0, 1, 2, 3, 4, 5, 6},
        )
        + astrometry.series_4100.index_files(
            cache_directory=cache,
            scales={7, 8, 9, 10, 11},
        ),
    )

    head = header.copy()
    arcsec_per_pixel = plate_scale_px(head["PXSIZE"], head["FOCALEN"])

    dx0 = dfm.xcentroid.std() / im_shape[0]

    print(dx0)

    if dx0 > 0.2:
        dfm1 = dfm
    else:
        x0 = dfm.xcentroid.mean()
        y0 = dfm.ycentroid.mean()
        dx0 = dfm.xcentroid.std()
        r = np.sqrt((x0 - dfm.xcentroid) ** 2 + (y0 - dfm.ycentroid) ** 2)
        dfm1 = dfm[r > dx0]

    ms = min(dfm1.shape[0], 200)
    dfm1 = dfm1[dfm1.sks < 0.5]

    print(arcsec_per_pixel, dfm1.shape, ms)

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(120)  # Timeout
    try:
        #    if True:
        solution = solver.solve(
            stars_xs=dfm1["xcentroid"][0:ms],
            stars_ys=dfm1["ycentroid"][0:ms],
            size_hint=astrometry.SizeHint(
                # lower_arcsec_per_pixel=arcsec_per_pixel*.9,
                # upper_arcsec_per_pixel=arcsec_per_pixel*1.1,),
                lower_arcsec_per_pixel=0.09,
                upper_arcsec_per_pixel=162,
            ),
            # position_hint = None,
            position_hint=astrometry.PositionHint(
                ra_deg=head["POINTRA"] * 360 / 24,
                dec_deg=head["POINTDEC"],
                radius_deg=1.0,
            ),
            solution_parameters=astrometry.SolutionParameters(
                logodds_callback=logodds_callback_100,
                # logodds_callback = logodds_callback3,
                # sip_order=5,
            ),
        )  # FIXME stars undefined
        nmatches = len(solution.matches)
        print(nmatches)
    except:
        nmatches = 0

    return nmatches
    # signal.alarm(0)
    # if nmatches > 0:
    #     astro_exists = get_if_header_already_post_processed(head, "ASTROMETRY")
    #     if not astro_exists:
    #         head['COMINIT'] = 'e'
    #         head.insert('COMINIT', ('COMMENT', '***************************'))
    #         head.insert('COMINIT', ('COMMENT', '       ASTROMETRY          '))
    #         head.insert('COMINIT', ('COMMENT', '***************************'))
    #     h_wcs = solution.best_match().wcs_fields
    #     # print(h_wcs)
    #     for v in h_wcs:
    #         head[v] = h_wcs[v]
    #
    #     px = head['NAXIS1'] / 2
    #     py = head['NAXIS2'] / 2
    #     w = WCS(h_wcs)
    #     ra, dec = w.wcs_pix2world(px, py, 1)
    #     ra = ra.tolist()
    #     dec = dec.tolist()
    #
    #     head['RA'] = ra
    #     head['DEC'] = dec
    #
    #     c1 = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
    #     c2 = SkyCoord(head['POINTRA'] * 360 / 24 * u.deg, head['POINTDEC'] * u.deg, frame='icrs')
    #     sep = (c2.separation(c1)).arcsecond
    #     print('Error de apuntado: ', sep, ' arcsec')
    #
    #     if not astro_exists:
    #         ra_hms, dec_dms = deg_to_hms(ra, dec)
    #
    #         try:
    #             elev = head['SITEALT']
    #         except:
    #             elev = head['SITEALT']
    #
    #         az, alt, airmass, zd = radec_to_altaz(ra, dec, head['SITELAT'], head['SITELONG'], elev, head['DATE-OBS'])
    #         longal, latgal = radec_to_gal(ra, dec)
    #         lonecl, latecl = radec_to_ecl(ra, dec)
    #
    #         head.insert('COMINIT', ('RA', ra, db.Header.objects.get(name='RA').description))
    #         head.insert('COMINIT', ('DEC', dec, db.Header.objects.get(name='DEC').description))
    #         head.insert('COMINIT', ('RAhms', ra_hms, db.Header.objects.get(name='RAhms').description))
    #         head.insert('COMINIT', ('DECdms', dec_dms, db.Header.objects.get(name='DECdms').description))
    #         head.insert('COMINIT', ('AZ', az, db.Header.objects.get(name='AZ').description))
    #         head.insert('COMINIT', ('ALT', alt, db.Header.objects.get(name='ALT').description))
    #         head.insert('COMINIT', ('ZD', zd, db.Header.objects.get(name='ZD').description))
    #         head.insert('COMINIT', ('AIRMASS', airmass, db.Header.objects.get(name='AIRMASS').description))
    #         head.insert('COMINIT', ('LONGAL', longal, db.Header.objects.get(name='LONGAL').description))
    #         head.insert('COMINIT', ('LATGAL', latgal, db.Header.objects.get(name='LATGAL').description))
    #         head.insert('COMINIT', ('LONECL', lonecl, db.Header.objects.get(name='LONECL').description))
    #         head.insert('COMINIT', ('LATECL', latecl, db.Header.objects.get(name='LATECL').description))
    #
    #         del head['COMINIT']
    #
    #     # Parche hasta que instroduzca en los raw
    #     if 'JD-OBS' not in head:
    #         try:
    #             jd, mjd = date_to_jd(head['DATE-OBS'])
    #             head.insert('PCDATE', ('JD-OBS', jd, db.Header.objects.get(name='JD-OBS').description))
    #             head.insert('PCDATE', ('MJD-OBS', mjd, db.Header.objects.get(name='MJD-OBS').description))
    #         except:
    #             pass
    #
    #     if 'SCALEORI' not in head:
    #         try:
    #             head.insert('PARITY', ('SCALEORI', np.round(plate_scale_px(head['PXSIZE'], head['FOCALEN']), 3),
    #                                    db.Header.objects.get(name='SCALEORI').description))
    #         except:
    #             pass
    #     ######
    #
    #     return head
    # return None
