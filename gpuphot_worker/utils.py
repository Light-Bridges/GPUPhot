import os

import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from skimage.measure import block_reduce

from gpuphot.image_processor import create_processor
from gpuphot.logger.hierarchical_logging import setup_logger

logger = setup_logger(__name__)

BASE_IMAGES_PATH = '/data/images'
PROCESSED_IMAGE_FOLDER = 'gpuphot_processed'


def get_processor(instrument_name=None):
    """
    Create a processor with optional custom instrument and configuration path.

    :param instrument_name: Name of the instrument to use.
    :type instrument_name: str or None
    :return: Configured image processor
    :rtype: ImageProcessor
    """
    instrument_name = instrument_name or os.environ.get('INSTRUMENT_NAME', 'default_instrument')
    config_base_path = '/gpuphot/instrument_configs'
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


def crop_and_bin_image(fits_file, binning, crop_size=None, center=None):
    """
    Processes a FITS or NPY file: optionally crops a region of interest and then applies binning.
    Maintains a detailed history of all processing steps in the FITS header.

    :param fits_file: Path to the FITS or NPY file to process.
    :type fits_file: str
    :param binning: Binning factor. If 1, no binning is applied.
    :type binning: int
    :param crop_size: Size of the crop (in pixels). Can be None (no crop), an integer (square), or a tuple (width, height).
    :type crop_size: int or tuple or None
    :param center: Coordinates of the crop center (x, y). If not provided and crop_size is not None, the image center is used.
    :type center: tuple or None
    :return: Path of the processed file or original file if no processing was done.
    :rtype: str
    :raises ValueError: If inputs are invalid or processing is not possible.
    """
    logger.debug(f"Processing file: {fits_file}, binning: {binning}, crop_size: {crop_size}, center: {center}")

    try:
        imdata, imheader = open_image_file(fits_file)
    except Exception as e:
        raise ValueError(f"Error opening file {fits_file}: {str(e)}")

    wcs = WCS(imheader)
    image_shape = imdata.shape

    # Validate inputs
    if not isinstance(binning, int) or binning < 1:
        raise ValueError("Binning factor must be an integer greater than or equal to 1.")

    if crop_size is not None:
        if center is None:
            center = (image_shape[1] // 2, image_shape[0] // 2)
        elif isinstance(center, (tuple, list)) and len(center) == 2:
            center = tuple(center)
        else:
            raise ValueError("Center must be a tuple or list of two integers.")

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        elif isinstance(crop_size, (tuple, list)) and len(crop_size) == 2:
            crop_size = tuple(crop_size)
        else:
            raise ValueError("Crop size must be an integer or a tuple/list of two integers.")

        # Check if the crop is possible
        half_width, half_height = crop_size[0] // 2, crop_size[1] // 2
        if (center[0] - half_width < 0 or center[0] + half_width > image_shape[1] or
                center[1] - half_height < 0 or center[1] + half_height > image_shape[0]):
            raise ValueError(
                "The specified crop size and center would result in a region outside the image boundaries.")

    # Check if any processing is needed
    if binning <= 1 and crop_size is None:
        return fits_file

    imheader['COMINIT'] = 'e'
    imheader.insert('COMINIT', ('COMMENT', '***************************'))
    imheader.insert('COMINIT', ('COMMENT', '       IMAGE PROCESSING    '))
    imheader.insert('COMINIT', ('COMMENT', '***************************'))

    if crop_size is not None:
        crop_size_str = f"{crop_size[0]}x{crop_size[1]}"
        imheader.insert('COMINIT', ('COMMENT', f"CROP APPLIED - Size: {crop_size_str}, Center: {center}"))

    if binning > 1:
        imheader.insert('COMINIT', ('COMMENT', f"BINNING APPLIED - Factor: {binning}"))

    # Register original dimensions in the header
    if 'ORIG_NAXIS1' not in imheader:
        imheader.insert('COMINIT', ('COMMENT', 'Original dimensions of the image before any processing.'))
        imheader.insert('COMINIT', ('O_NAXIS1', image_shape[1]))
        imheader.insert('COMINIT', ('O_NAXIS2', image_shape[0]))

    # Apply cropping if crop_size is specified
    if crop_size is not None:
        cutout = Cutout2D(imdata, center, crop_size, wcs=wcs)
        imdata = cutout.data
        wcs = cutout.wcs
        imheader.insert('COMINIT', ('COMMENT', f'Image cropped from {image_shape} to {crop_size}.'))

    # Apply binning if necessary
    if binning > 1:
        imdata = block_reduce(imdata, block_size=(binning, binning), func=np.median)
        imdata = np.clip(imdata, 0, 65535).astype(np.float32)

        if hasattr(wcs.wcs, 'cdelt'):
            wcs.wcs.cdelt *= binning
        elif hasattr(wcs.wcs, 'cd'):
            wcs.wcs.cd *= binning

        if 'GAIN' in imheader:
            imheader['GAIN'] *= binning ** 2
        if 'RDNOISE' in imheader:
            imheader['RDNOISE'] /= binning

    # Update header after processing
    imheader['NAXIS1'] = imdata.shape[1]
    imheader['NAXIS2'] = imdata.shape[0]
    imheader['CDELT1'] = imheader.get('CDELT1', 1) * binning
    imheader['CDELT2'] = imheader.get('CDELT2', 1) * binning

    # imheader.update(wcs.to_header(relax=True))
    imheader['DATAMIN'] = np.min(imdata)
    imheader['DATAMAX'] = np.max(imdata)

    del imheader['COMINIT']

    # Save the new FITS file
    hdu = fits.PrimaryHDU(imdata, imheader)
    output_filename = f"{os.path.splitext(os.path.basename(fits_file))[0]}"
    if crop_size:
        crop_size_str = f"{crop_size[0]}_{crop_size[1]}"
        output_filename += f"_crop{crop_size_str}"
    if binning > 1:
        output_filename += f"_bin{binning}"
    output_filename += ".fits"
    output_file = os.path.join(os.path.dirname(fits_file), output_filename)
    hdu.writeto(output_file, overwrite=True)

    logger.debug(f"Cropped and/or binned image saved to: {output_file}")

    return output_file
