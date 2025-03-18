import os

import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from skimage.measure import block_reduce

from gpuphot.image_processor import create_processor
from gpuphot.logger.hierarchical_logging import setup_logger

logger = setup_logger(__name__)

BASE_IMAGES_PATH = os.environ.get('IMAGE_BASE_PATH', '/data/images')
PROCESSED_IMAGE_FOLDER = os.environ.get('PROCESSED_IMAGE_FOLDER', 'gpuphot_processed')


def get_processor(instrument_name=None):
    """
    Create a processor with optional custom instrument and configuration path.

    :param instrument_name: Name of the instrument to use.
    :type instrument_name: str or None
    :return: Configured image processor
    :rtype: ImageProcessor
    """
    instrument_name = instrument_name or os.environ.get('INSTRUMENT_NAME', 'default_instrument')
    config_base_path = os.environ.get('INSTRUMENT_CONFIG_BASE_PATH', '/gpuphot/instrument_configs')
    return create_processor(instrument_name, config_base_path)


def open_image_file(file_path):
    """
    Opens an astronomical image file (FITS or NPY) and returns the data and header.

    :param file_path: Path to the image file.
    :type file_path: str
    :return: Tuple containing image data and header.
    :rtype: tuple(numpy.ndarray, astropy.io.fits.Header)
    :raises ValueError: If the file format is not supported or if the header file is missing for NPY.
    """

    if file_path.endswith('.fits'):
        try:
            with fits.open(file_path) as hdul:
                imdata = hdul[0].data.astype(np.float32)
                imheader = hdul[0].header
        except Exception as e:
            imdata = fits.getdata(file_path).astype(np.float32)
            imheader = fits.getheader(file_path)
    elif file_path.endswith('.npy'):
        imdata = np.load(file_path)
        header_file = file_path.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(header_file):
            with open(header_file, 'r') as f:
                header_content = f.read()
            imheader = fits.Header.fromstring(header_content)
        else:
            raise ValueError(f"Header file not found for NPY file: {file_path}, expected header file: {header_file}")
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Only FITS and NPY files are supported.")

    return imdata, imheader


def save_processed_image(file_path, base_path, imdata, hwcs):
    """
    Saves the processed image data and header as a FITS file in a 'gpuphot_processed' subdirectory.

    :param file_path: Original file path.
    :type file_path: str
    :param base_path: Base path for relative paths.
    :type base_path: str
    :param imdata: Processed image data.
    :type imdata: numpy.ndarray
    :param hwcs: Updated header with WCS information.
    :type hwcs: astropy.io.fits.Header
    :return: Path of the saved FITS file.
    :rtype: str
    :raises ValueError: If there's an issue creating the output directory.
    """
    # Get the relative path
    process_file = os.path.relpath(file_path, base_path)

    # Create the new output path
    output_dir = os.path.join(base_path, PROCESSED_IMAGE_FOLDER, os.path.dirname(process_file))
    os.makedirs(output_dir, exist_ok=True)

    # Change the extension to .fits
    file_name = os.path.splitext(os.path.basename(process_file))[0]
    output_path = os.path.join(output_dir, f"{file_name}.fits")

    # Create and save the FITS file
    photometrized_image = fits.PrimaryHDU(data=imdata.astype(np.float32), header=hwcs)
    photometrized_image.writeto(output_path, overwrite=True)

    return output_path


# def crop_and_bin_image(fits_file, binning, binning_method='sum', crop_size=None, center=None):
#     """
#     Processes a FITS or NPY file: applies binning first, then optionally crops a region of interest.
#     Maintains a detailed history of all processing steps in the FITS header.
#
#     :param fits_file: Path to the FITS or NPY file to process.
#     :type fits_file: str
#     :param binning: Binning factor. If 1, no binning is applied.
#     :type binning: int
#     :param binning_method: Method to apply binning. Can be 'sum' or 'median'.
#     :type binning_method: str
#     :param crop_size: Size of the crop (in pixels). Can be None (no crop), an integer (square), or a tuple (width, height).
#     :type crop_size: int or tuple or None
#     :param center: Coordinates of the crop center (x, y). If not provided and crop_size is not None, the image center is used.
#                 These are pixel coordinates, *not* WCS coordinates.
#     :type center: tuple or None
#     :return: Path of the processed file or original file if no processing was done.
#     :rtype: str
#     :raises ValueError: If inputs are invalid or processing is not possible.
#     """
#     logger.debug(f"Processing file: {fits_file}, binning: {binning}, crop_size: {crop_size}, center: {center}")
#
#     try:
#         imdata, imheader = open_image_file(fits_file)
#
#     except Exception as e:
#         raise ValueError(f"Error opening file {fits_file}: {str(e)}") from e
#
#     original_shape = imdata.shape
#     wcs = WCS(imheader)
#
#     # Validate inputs
#     if not isinstance(binning, int) or binning < 1:
#         raise ValueError("Binning factor must be an integer greater than or equal to 1.")
#
#     # Check if any processing is needed
#     if binning <= 1 and crop_size is None:
#         return fits_file
#
#     # Initialize header comments
#     imheader['COMINIT'] = 'e'
#     imheader.insert('COMINIT', ('COMMENT', '***************************'))
#     imheader.insert('COMINIT', ('COMMENT', '       IMAGE PROCESSING    '))
#     imheader.insert('COMINIT', ('COMMENT', '***************************'))
#
#     # Apply binning if necessary
#     if binning > 1:
#         VALID_METHODS = {'sum', 'median'}
#         binning_method = binning_method.lower()
#         if binning_method not in VALID_METHODS:
#             raise ValueError(f"Invalid binning method: {binning_method}. Valid methods: {VALID_METHODS}")
#
#         bin_func = np.sum if binning_method == 'sum' else np.nanmedian  # Usar nanmedian
#         imdata = block_reduce(imdata, block_size=(binning, binning), func=bin_func)
#
#         imdata = imdata.astype(np.float32)  # Asegurar tipo de dato
#         imdata = np.clip(imdata, 0, 65535).astype(np.float32)  # Asegurar rango
#
#         # **********  ACTUALIZAR PXSIZE  **********
#         imheader['PXSIZE'] = imheader.get('PXSIZE', 1.0) * binning  # ¡CORREGIDO!
#         # ****************************************
#
#         # Update WCS.  Modificamos la matriz CD (o PC) directamente.
#         if wcs.wcs.has_cd():
#             wcs.wcs.cd = wcs.wcs.cd * binning
#         elif wcs.wcs.has_pc():
#             wcs.wcs.pc = wcs.wcs.pc * binning
#             wcs.wcs.cdelt = wcs.wcs.cdelt * binning  # Si existe pc, hay que modificar cdelt
#         else:
#             # Si no tiene ni CD ni PC, *asumimos* que tiene CDELT y CROTA2
#             wcs.wcs.cdelt = wcs.wcs.cdelt * binning
#         # No es necesario hacer wcs.wcs.crpix = wcs.wcs.crpix / binning
#
#         # Update header
#         imheader['BIN-FCTR'] = (binning, 'Binning factor applied')
#         imheader['BIN_ALG'] = (binning_method.upper(), 'Pixel combination method')
#         imheader['BINSTAT'] = ('LINEAR' if binning_method == 'sum' else 'NONLINEAR',
#                                'Linearity of binning operation')
#         imheader['BINFCTR'] = (binning, 'Binning factor in both axes')
#         imheader['BINTYPE'] = ('LINEAR' if binning_method == 'sum' else 'NON_LINEAR')
#
#         if binning_method == 'sum':
#             imheader['GAIN'] = imheader.get('GAIN', 1.0) / (
#                     binning ** 2)  # Gain decreases with sum, CORRECTO
#             imheader['RDNOISE'] = imheader.get('RDNOISE', 0.0) * binning  # RDNOISE increases with sum
#             imheader['SATLEVEL'] = imheader.get('SATLEVEL', 1.0) * (binning ** 2)
#
#         elif binning_method == 'median':
#             # Ver documentación.
#             imheader['GAIN'] = imheader.get('GAIN', 1.0) * np.sqrt(np.pi / 2) / (binning ** 2)
#             imheader['RDNOISE'] = imheader.get('RDNOISE', 0.0) / np.sqrt(binning ** 2 - np.pi / 2 + 1)
#
#         imheader.insert('COMINIT', ('COMMENT', f"BINNING APPLIED - Factor: {binning}, Method: {binning_method}"))
#         if binning_method == 'median':
#             imheader.insert('COMINIT', ('COMMENT',
#                                         f"WARNING: Median binning alters photometric linearity (deviation ~12% at 2x2)"))
#
#     # Apply cropping if crop_size is specified
#     if crop_size is not None:
#         if center is None:
#             center = (imdata.shape[1] // 2, imdata.shape[0] // 2)  # Usa el nuevo tamaño
#         elif isinstance(center, (tuple, list)) and len(center) == 2:
#             center = tuple(center)  # Ensure immutability
#         else:
#             raise ValueError("Center must be a tuple or list of two integers.")
#
#         if isinstance(crop_size, int):
#             crop_size = (crop_size, crop_size)
#         elif isinstance(crop_size, (tuple, list)) and len(crop_size) == 2:
#             crop_size = tuple(crop_size)
#         else:
#             raise ValueError("Crop size must be an integer or a tuple/list of two integers.")
#
#         # Check if the crop is possible
#         half_width, half_height = crop_size[0] // 2, crop_size[1] // 2
#         if (center[0] - half_width < 0 or center[0] + half_width > imdata.shape[1] or
#                 center[1] - half_height < 0 or center[1] + half_height > imdata.shape[0]):
#             raise ValueError(
#                 "The specified crop size and center would result in a region outside the image boundaries.")
#
#         # Usamos Cutout2D, especificando el origen (1, 1) para FITS.
#         cutout = Cutout2D(imdata, center, crop_size, wcs=wcs, mode='strict', origin=1)
#         imdata = cutout.data
#         wcs = cutout.wcs
#
#         crop_size_str = f"{crop_size[0]}x{crop_size[1]}"
#         imheader.insert('COMINIT', ('COMMENT', f"CROP APPLIED - Size: {crop_size_str}, Center: {center}"))
#
#     # Update header after processing
#     imheader['NAXIS1'] = imdata.shape[1]
#     imheader['NAXIS2'] = imdata.shape[0]
#     imheader.update(wcs.to_header())
#
#     imheader['DATAMIN'] = np.min(imdata)
#     imheader['DATAMAX'] = np.max(imdata)
#
#     # Register original dimensions in the header
#     imheader.insert('COMINIT', ('COMMENT', 'Original dimensions of the image before any processing.'))
#     imheader.insert('COMINIT', ('O_NAXIS1', original_shape[1]))
#     imheader.insert('COMINIT', ('O_NAXIS2', original_shape[0]))
#
#     del imheader['COMINIT']
#
#     # Save the new FITS file
#     hdu = fits.PrimaryHDU(imdata, imheader)
#     output_filename = f"{os.path.splitext(os.path.basename(fits_file))[0]}"
#     if binning > 1:
#         output_filename += f"_bin{binning}_{binning_method}"
#     if crop_size:
#         crop_size_str = f"{crop_size[0]}_{crop_size[1]}"
#         output_filename += f"_crop{crop_size_str}"
#     output_filename += ".fits"
#     output_file = os.path.join(os.path.dirname(fits_file), output_filename)
#     hdu.writeto(output_file, overwrite=True)
#
#     logger.debug(f"Binned and/or cropped image saved to: {output_file}")
#
#     return output_file
def crop_and_bin_image(fits_file, binning, binning_method='sum',
                       crop_size=None, center=None):
    """
    Procesa un archivo FITS o NPY: aplica binning y luego, opcionalmente, recorta
    una región de interés. Actualiza el header FITS para reflejar todos los cambios.

    :param fits_file: Ruta al archivo FITS o NPY a procesar.
    :param binning: Factor de binning. Si es 1, no se aplica binning.
    :param binning_method: 'sum' o 'median'.
    :param crop_size: Tamaño del recorte en píxeles (None = sin recorte).
                      Puede ser int (cuadrado) o tuple (ancho, alto).
    :param center: Centro del recorte (x, y) en coordenadas de píxel (no WCS).
                   Si None, se asume el centro de la imagen resultante.
    :return: Ruta al archivo resultante (FITS).
    """
    logger.debug(f"Processing file: {fits_file}, binning: {binning}, "
                 f"crop_size: {crop_size}, center: {center}")

    # 1) Abrir la imagen
    try:
        imdata, imheader = open_image_file(fits_file)
    except Exception as e:
        raise ValueError(f"Error opening file {fits_file}: {str(e)}") from e

    # Guardamos forma original como referencia
    original_shape = imdata.shape
    # Construimos WCS a partir del header, si existe
    wcs = WCS(imheader) if len(imheader) > 0 else None

    # 2) Validar el factor de binning
    if not isinstance(binning, int) or binning < 1:
        raise ValueError("Binning factor must be an integer >= 1.")

    # 3) Comprobamos si no hay que hacer nada
    if binning == 1 and crop_size is None:
        # Si no hay binning ni recorte, devolvemos tal cual
        return fits_file

    # Preparamos algunos comentarios en el header
    imheader['COMINIT'] = 'e'
    imheader.insert('COMINIT', ('COMMENT', '***************************'))
    imheader.insert('COMINIT', ('COMMENT', '       IMAGE PROCESSING    '))
    imheader.insert('COMINIT', ('COMMENT', '***************************'))

    # -------------------------------------------------------------------------
    # BINNING
    # -------------------------------------------------------------------------
    if binning > 1:
        valid_methods = {'sum', 'median'}
        binning_method = binning_method.lower()
        if binning_method not in valid_methods:
            raise ValueError(f"Invalid binning method {binning_method}. "
                             f"Valid are {valid_methods}")

        # Elegimos función de binning
        bin_func = np.sum if binning_method == 'sum' else np.nanmedian
        # Aplicar binning con block_reduce
        binned_data = block_reduce(imdata, block_size=(binning, binning),
                                   func=bin_func)


        imdata = binned_data

        # ---------------------------------------------------------------------
        # Actualizar encabezado y WCS
        # ---------------------------------------------------------------------
        # -- 1) Actualizar tamaño de píxel en el header (si se usa PXSIZE en arcsec)
        #    Asumiendo que PXSIZE era el tamaño en arcsec/píxel, ahora cada
        #    píxel representa binning veces más tamaño angular.
        pxsize_original = imheader.get('PXSIZE', 1.0)
        imheader['PXSIZE'] = pxsize_original * binning

        # -- 2) Ajuste del WCS si existe
        if wcs is not None and wcs.wcs.naxis >= 2:
            # Si el header posee una matriz CD, PC o cdelt, hay que ampliarla
            if wcs.wcs.has_cd():
                # Multiplicamos la matriz CD por el factor
                wcs.wcs.cd *= binning
            elif wcs.wcs.has_pc():
                wcs.wcs.pc *= binning
                if hasattr(wcs.wcs, 'cdelt'):
                    wcs.wcs.cdelt *= binning
            else:
                # Asumimos que usa CDELT si no hay ni CD ni PC
                if hasattr(wcs.wcs, 'cdelt'):
                    wcs.wcs.cdelt *= binning

            # IMPORTANTE: Ajustar CRPIX para que la referencia en el cielo
            # siga apuntando al mismo lugar. Después de binning NxN,
            # la crpix debe escalarse.
            wcs.wcs.crpix /= binning

        # -- 3) Actualizar ganancias y ruidos
        imheader['BIN-FCTR'] = (binning, 'Binning factor applied')
        imheader['BIN_ALG'] = (binning_method.upper(), 'Pixel combination method')
        imheader['BINSTAT'] = ('LINEAR' if binning_method == 'sum' else 'NONLINEAR',
                               'Linearity of binning operation')
        imheader['BINFCTR'] = (binning, 'Binning factor in both axes')
        imheader['BINTYPE'] = ('LINEAR' if binning_method == 'sum' else 'NON_LINEAR')

        if binning_method == 'sum':
            # Ajuste de GAIN, RDNOISE, SATLEVEL, etc. para sum binning
            old_gain = imheader.get('GAIN', 1.0)
            old_rdnoise = imheader.get('RDNOISE', 0.0)
            old_satlevel = imheader.get('SATLEVEL', 65535)

            # Si GAIN estaba en e-/ADU, con sum binning la nueva "ganancia" en e-/ADU
            # se reduce en factor binning^2
            imheader['GAIN'] = old_gain / (binning**2)
            # RDNOISE aumenta ~ N en e- (suponiendo se mide en e-)
            imheader['RDNOISE'] = old_rdnoise * binning
            # Nivel de saturación escala con el total de electrones sumados
            imheader['SATLEVEL'] = old_satlevel * (binning**2)

        else:
            # Ajuste aproximado para median binning (hay variantes)
            old_gain = imheader.get('GAIN', 1.0)
            old_rdnoise = imheader.get('RDNOISE', 0.0)

            # Ajuste recomendado (muy aproximado):
            # Reference: https://iraf.net/forum/viewtopic.php?showtopic=146351
            imheader['GAIN'] = old_gain * np.sqrt(np.pi/2) / (binning**2)
            # Ruido efectivo tras la mediana ~ sigma / sqrt(Npix)
            # Hay estimaciones analíticas más elaboradas, pero sirva de ejemplo
            Npix = binning**2
            imheader['RDNOISE'] = old_rdnoise / np.sqrt(Npix)

            imheader.insert('COMINIT', (
                'COMMENT',
                f"WARNING: Median binning alters photometric linearity (~12% at 2x2)."
            ))

        imheader.insert('COMINIT', (
            'COMMENT', f"BINNING APPLIED - Factor: {binning}, Method: {binning_method}"
        ))

    # -------------------------------------------------------------------------
    # RECORTE (CROP)
    # -------------------------------------------------------------------------
    if crop_size is not None:
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        elif (isinstance(crop_size, (tuple, list)) and len(crop_size) == 2):
            crop_size = tuple(crop_size)
        else:
            raise ValueError("crop_size must be None, an integer, or a tuple of two integers")

        # Centro de recorte: si no se da, usar el centro de la imagen resultante
        if center is None:
            center = (imdata.shape[1] // 2, imdata.shape[0] // 2)
        elif (isinstance(center, (tuple, list)) and len(center) == 2):
            center = tuple(center)
        else:
            raise ValueError("Center must be a tuple/list of two integers (x, y).")

        # Validar que el recorte sea posible
        half_w, half_h = crop_size[0] // 2, crop_size[1] // 2
        x_min = center[0] - half_w
        x_max = center[0] + half_w
        y_min = center[1] - half_h
        y_max = center[1] + half_h

        if x_min < 0 or y_min < 0 or x_max > imdata.shape[1] or y_max > imdata.shape[0]:
            raise ValueError("Crop region is out of image boundaries.")

        # Usar Cutout2D para el recorte
        # origin=0 es la convención por defecto en Python. Si se desea 1, ajustarlo.
        if wcs is not None:
            cutout = Cutout2D(imdata, center, crop_size, wcs=wcs, mode='trim', origin=0)
            imdata = cutout.data
            wcs = cutout.wcs
        else:
            # Si no hay WCS, se hace un recorte básico con slicing
            imdata = imdata[y_min:y_max, x_min:x_max]

        crop_size_str = f"{crop_size[0]}x{crop_size[1]}"
        imheader.insert('COMINIT', (
            'COMMENT', f"CROP APPLIED - Size: {crop_size_str}, Center: {center}"
        ))

    # -------------------------------------------------------------------------
    # ACTUALIZAR HEADER FINAL
    # -------------------------------------------------------------------------
    imheader['NAXIS1'] = imdata.shape[1]
    imheader['NAXIS2'] = imdata.shape[0]

    # Si existe WCS actualizado, volcar sus claves al header
    if wcs is not None:
        imheader.update(wcs.to_header())

    # Actualizar valores estadísticos
    imheader['DATAMIN'] = float(np.nanmin(imdata))
    imheader['DATAMAX'] = float(np.nanmax(imdata))

    # Dejar constancia del tamaño original
    imheader.insert('COMINIT', ('COMMENT', 'Original dimensions before any processing:'))
    imheader.insert('COMINIT', ('O_NAXIS1', original_shape[1]))
    imheader.insert('COMINIT', ('O_NAXIS2', original_shape[0]))

    # Limpiar bandera auxiliar
    del imheader['COMINIT']

    # Guardar la nueva imagen FITS
    hdu = fits.PrimaryHDU(imdata, imheader)
    base_name = os.path.splitext(os.path.basename(fits_file))[0]
    output_filename = base_name
    if binning > 1:
        output_filename += f"_bin{binning}_{binning_method}"
    if crop_size:
        w_str, h_str = str(crop_size[0]), str(crop_size[1])
        output_filename += f"_crop{w_str}x{h_str}"
    output_filename += ".fits"

    output_path = os.path.join(os.path.dirname(fits_file), output_filename)
    hdu.writeto(output_path, overwrite=True)

    logger.debug(f"Processed file saved to: {output_path}")
    return output_path
