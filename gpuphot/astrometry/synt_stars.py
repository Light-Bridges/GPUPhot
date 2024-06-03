import os

import numpy as np
from astropy.convolution import discretize_model
from astropy.modeling.models import Moffat2D
from astropy.table import QTable


from gpuphot.stats.io import pruebaPrint

def print_hello():
    pruebaPrint()

def _generate_noise_rng():
    # Set up the random number generator, allowing a seed to be set from the environment
    seed = os.getenv('GUIDE_RANDOM_SEED', None)

    if seed is not None:
        seed = int(seed)

    # This is the generator to use for any image component which changes in each image, e.g. read noise
    # or Poisson error
    return np.random.default_rng(seed)


# Noise functions
# Noise functions
def read_noise(image, amount, gain=1):
    shape = image.shape

    noise = _generate_noise_rng().normal(scale=amount / gain, size=shape)

    return noise


def dark_current(image, current, exposure_time, gain=1.0, hot_pixels=True):
    # dark current for every pixel
    base_current = current * exposure_time / gain
    dark_im = _generate_noise_rng().poisson(base_current, size=image.shape)

    if hot_pixels:
        y_max, x_max = dark_im.shape

        n_hot = int(0.0001 * x_max * y_max)  # 0.01% hot pixels

        rng = np.random.RandomState(16201649)
        hot_x = rng.randint(0, x_max, size=n_hot)
        hot_y = rng.randint(0, y_max, size=n_hot)

        hot_current = 10000 * current

        dark_im[(hot_y, hot_x)] = hot_current * exposure_time / gain
    return dark_im


def sky_background(image, sky_counts, gain=1):
    sky_im = _generate_noise_rng().poisson(sky_counts * gain, size=image.shape) / gain

    return sky_im


def moffat_fwhm(alpha, beta):
    fwhm = 2 * alpha * np.sqrt(2 ** (1 / beta) - 1)
    return fwhm


def make_random_models_table(n_sources, param_ranges, seed=None):
    rng = np.random.default_rng(seed)
    sources = QTable()
    for param_name, (lower, upper, dist) in param_ranges.items():
        # Generate a column for every item in param_ranges, even if it
        # is not in the model (e.g., flux). However, such columns will be
        # ignored when rendering the image.
        if dist == 'uni':
            sources[param_name] = rng.uniform(lower, upper, n_sources)
        elif dist == 'pow':
            sources[param_name] = upper + lower - rng.power(50., n_sources) * upper

    return sources


def make_model_sources_image(shape, model, source_table, oversample=1):
    image = np.zeros(shape, dtype=float)
    yidx, xidx = np.indices(shape)

    params_to_set = []
    for param in source_table.colnames:
        if param in model.param_names:
            params_to_set.append(param)

    # Save the initial parameter values so we can set them back when
    # done with the loop. It's best not to copy a model, because some
    # models (e.g., PSF models) may have substantial amounts of data in
    # them.
    init_params = {param: getattr(model, param) for param in params_to_set}

    try:
        flux = []
        for source in source_table:
            for param in params_to_set:
                setattr(model, param, source[param])

            if oversample == 1:
                s = model(xidx, yidx)
                image += s
                flux.append(np.sum(s))

            else:
                image += discretize_model(model, (0, shape[1]),
                                          (0, shape[0]), mode='oversample',
                                          factor=oversample)
    finally:
        for param, value in init_params.items():
            setattr(model, param, value)

    return image, flux


def stars(image, nstars, alpha, beta, sky=0, max_counts=65000, gain=1):
    model = Moffat2D()

    flux_range = sky, max_counts
    n_sources = nstars
    shape = image.shape
    param_ranges = {'amplitude': [sky, max_counts, 'pow'],
                    'x_0': [5, shape[1] - 5, 'uni'],
                    'y_0': [5, shape[0] - 5, 'uni'],
                    'alpha': [beta, beta, 'uni'],
                    'gamma': [alpha, alpha, 'uni']}

    sources = make_random_models_table(n_sources, param_ranges)

    data, flux = make_model_sources_image(shape, model, sources)
    sources['flux'] = flux

    sources.rename_column('alpha', 'beta')
    sources.rename_column('gamma', 'alpha')

    return data, sources


def create_synt_image(image, gain, rd_noise, texp, dark_cur, sky_level, fwhm, nstars, max_counts):
    # Add readout noise
    image += read_noise(image, rd_noise, gain=gain)

    # Add dark noise
    image += dark_current(image, dark_cur, texp, hot_pixels=True, gain=gain)

    # Add sky background
    image += sky_background(image, sky_level, gain=gain)

    # Add stars
    beta = abs(np.random.rand() * 0.4 + 4.565)  # beta = 4.765 +- 0.2 (Trujillo et al)
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))

    data, sources = stars(image, nstars, alpha, beta, max_counts=max_counts, gain=gain)

    image += data

    return image, alpha, beta, sources

#
# # Set up the random number generator, allowing a seed to be set from the environment
# seed = os.getenv('GUIDE_RANDOM_SEED', None)
#
# if seed is not None:
#     seed = int(seed)
#
# # This is the generator to use for any image component which changes in each image, e.g. read noise
# # or Poisson error
# noise_rng = np.random.default_rng(seed)
#
# N = 10000
# df = pd.DataFrame()
# for i in range(N):
#
#     try:
#         # Create image
#         synthetic_image = np.zeros([512, 512])
#
#         # Parameters
#         gain = 0.77
#         rd_noise = np.random.randn() + 2.5
#         texp = 20
#         dark_cur = 0.01
#         sky_level = np.random.randn() * 100 + 400
#         fwhm = np.random.rand() * 7 + 3
#
#         # if fwhm < 2 or fwhm > 15: continue
#
#         nstars = int(np.random.randint(200) + 3)
#         # nstars = int(np.random.randint(20) )
#         max_counts = np.random.rand() * 2000 + 500
#
#         if nstars == 0:
#             fwhm = 0
#
#         synthetic_image, alpha, beta, _ = create_synt_image(synthetic_image, gain, rd_noise, texp, dark_cur, sky_level,
#                                                             fwhm, nstars, max_counts)
#
#         name = '%i.fits' % (i + 1)
#
#         df = df.append({'name': name, 'gain': gain, 'rd_noise': rd_noise, 'sky_level': sky_level,
#                         'moffat_alpha': alpha, 'moffat_beta': beta, 'fwhm': fwhm, 'nstars': nstars,
#                         'max_counts': max_counts}, ignore_index=True)
#
#         cols = ['name', 'gain', 'rd_noise', 'sky_level', 'moffat_alpha', 'moffat_beta', 'fwhm', 'nstars', 'max_counts']
#         df = df[cols]
#
#         HDU = fits.PrimaryHDU(data=synthetic_image.astype(np.float32))
#         HDU.writeto('/data/train/' + name, overwrite=True)
#
#         if (i % 100) == 0:
#             print(i)
#
#         df.to_csv('/data/df.csv')
#
#     except:
#         continue
