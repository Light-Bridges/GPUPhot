import cupy as cp
import numpy as np


def convolve_fft(image, kernel):
    image_shape = image.shape
    kernel_shape = kernel.shape
    padding = int((kernel_shape[0] - 1) / 2)
    new_image_shape = fill_image((image_shape[0] + 2 * padding, image_shape[1] + 2 * padding))

    # Rellenar la imagen y el kernel
    padded_image = cp.pad(image, ((padding, padding), (padding, padding)))
    padded_kernel = cp.pad(kernel,
                           ((0, new_image_shape[0] - kernel_shape[0]), (0, new_image_shape[1] - kernel_shape[1])))

    # FFT en la imagen y el kernel
    F_image = cp.fft.rfft2(padded_image, s=new_image_shape)
    F_kernel = cp.fft.rfft2(padded_kernel, s=new_image_shape)

    # Convolución en el dominio de Fourier
    F_convolved = F_image * F_kernel

    # Transformada inversa de Fourier
    convolved = cp.fft.irfft2(F_convolved, s=new_image_shape)

    # Recortar la imagen convolucionada
    convolved = convolved[padding:padding + image_shape[0], padding:padding + image_shape[1]]
    return convolved


def get_mean_std(im_g, lk, std=True):
    k_app = gen_apm_filter(lk)
    fot_m = convolve_fft(im_g, k_app)
    if std:
        fot_m2 = convolve_fft(im_g * im_g, k_app)
        fot_m2 = cp.sqrt(fot_m2 - fot_m * fot_m)
    else:
        fot_m2 = None
    del k_app
    return fot_m, fot_m2


def gaussian_kernel(lk, sigma):
    k_dim = (2 * lk + 1, 2 * lk + 1)
    x = cp.linspace(-lk, lk, k_dim[0])
    y = cp.linspace(-lk, lk, k_dim[1])
    xx, yy = cp.meshgrid(x, y)
    kernel = cp.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    del x, y, xx, yy
    return kernel / cp.sum(kernel)


def get_aper_kernel(radius, size=None):
    if size == None:
        size = 2 * radius + 1
    kernel = cp.zeros((size, size))
    y, x = cp.indices(kernel.shape)
    mask = (x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2 <= radius ** 2
    kernel[mask] = 1
    area = cp.sum(kernel)
    return kernel, area


def fill_image(image_shape):
    h, w = image_shape
    new_height = 2 ** int(np.ceil(np.log2(h)))
    new_width = 2 ** int(np.ceil(np.log2(w)))
    return (new_height, new_width)


def gen_apm_filter(lk, li=0, norm=True):
    k_dim = (2 * lk + 1, 2 * lk + 1)
    indi = cp.indices(k_dim)
    fw2 = lk ** 2
    k_app = cp.zeros(k_dim)
    struc = cp.where(((lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2) < fw2)
    k_app[struc] = 1
    if li != 0:
        struc = cp.where(((lk - indi[0, :, :]) ** 2 + (lk - indi[1, :, :]) ** 2) < (fw2 - li ** 2))
        k_app[struc] = 0
    if norm: k_app = k_app / k_app.sum()
    return (k_app)
