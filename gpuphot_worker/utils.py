import os

import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from skimage.measure import block_reduce

from gpuphot.image_processor import create_processor


def get_processor(instrument_name=None):
    """
    Create a processor with optional custom instrument and configuration path.

    Args:
        instrument_name (str, optional): Name of the instrument to use.

    Returns:
        processor: Configured image processor
    """
    instrument_name = instrument_name or os.environ.get('INSTRUMENT_NAME', 'default_instrument')
    config_base_path = os.environ.get('INSTRUMENT_CONFIG_BASE_PATH', '/app/gpuphot/instrument_configs')
    return create_processor(instrument_name, config_base_path)


def open_image_file(file_path):
    """
    Opens an astronomical image file (FITS or NPY) and returns the data and header.

    Args:
        file_path (str): Path to the image file.

    Returns:
        tuple: (imdata, imheader) where imdata is a numpy array and imheader is a fits.Header object.

    Raises:
        ValueError: If the file format is not supported or if the header file is missing for NPY.
    """
    if file_path.endswith('.fits'):
        with fits.open(file_path) as hdul:
            imdata = hdul[0].data.astype(np.float32)
            imheader = hdul[0].header
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


def save_processed_image(file_path, imdata, hwcs, overwrite):
    """
    Saves the processed image data and header as a FITS file.

    Args:
        file_path (str): Original file path.
        imdata (numpy.ndarray): Processed image data.
        hwcs (fits.Header): Updated header with WCS information.
        overwrite (bool): If True, overwrites the original file if it's a FITS file.

    Returns:
        str: Path of the saved FITS file.

    Raises:
        ValueError: If the file format is not supported.
    """
    # Determine the output path
    if file_path.endswith('.fits') and overwrite:
        output_path = file_path
    else:
        # For any input format, we'll save as FITS
        file_name = os.path.splitext(file_path)[0]
        output_path = f"{file_name}_photometrized.fits"

    # Create and save the FITS file
    photometrized_image = fits.PrimaryHDU(data=imdata.astype(np.float32), header=hwcs)
    photometrized_image.writeto(output_path, overwrite=True)

    return output_path

def crop_and_bin_image(fits_file, binning, crop_size=None, center=None):
    """
    Processes a FITS or NPY file: optionally crops a region of interest and then applies binning.

    Args:
        fits_file (str): Path to the FITS or NPY file to process.
        binning (int): Binning factor. If 1, no binning is applied.
        crop_size (int or tuple, optional): Size of the crop (in pixels).
                                            Can be None (no crop), an integer (square), or a tuple (width, height).
        center (tuple, optional): Coordinates of the crop center (x, y).
                                  If not provided and crop_size is not None, the image center is used.

    Returns:
        str: Path of the processed file or original file if no processing was done.
    """
    try:
        imdata, imheader = open_image_file(fits_file)
    except Exception as e:
        raise ValueError(f"Error opening file {fits_file}: {str(e)}")

    wcs = WCS(imheader)
    image_shape = imdata.shape

    # Check if any processing is needed
    if binning <= 1 and crop_size is None:
        return fits_file  # Return original file path if no processing is needed

    # Apply cropping if crop_size is specified
    if crop_size is not None:
        if center is None:
            center = (image_shape[1] // 2, image_shape[0] // 2)

        cutout = Cutout2D(imdata, center, crop_size, wcs=wcs)
        imdata = cutout.data
        wcs = cutout.wcs

    # Apply binning if necessary
    if binning > 1:
        imdata = block_reduce(imdata, block_size=(binning, binning), func=np.median)
        imdata[imdata < 0] = 0
        imdata[imdata > 2 ** 16 - 1] = 2 ** 16 - 1
        imdata = imdata.astype(np.float32)

        # Update WCS to reflect binning
        wcs = wcs[::binning, ::binning]

    # Update header after processing
    if crop_size is not None or binning > 1:
        # Update dimensions
        imheader['NAXIS1'] = imdata.shape[1]
        imheader['NAXIS2'] = imdata.shape[0]

        # Remove CDELT if CD is present
        if 'CD1_1' in imheader:
            imheader.remove('CDELT1', ignore_missing=True)
            imheader.remove('CDELT2', ignore_missing=True)
        elif binning > 1:
            # Update CDELT only if CD is not present
            imheader['CDELT1'] = (imheader.get('CDELT1', 1) * binning)
            imheader['CDELT2'] = (imheader.get('CDELT2', 1) * binning)

        # Update CTYPE to include SIP if necessary
        if any(key.startswith('A_') or key.startswith('B_') for key in imheader):
            if not imheader['CTYPE1'].endswith('-SIP'):
                imheader['CTYPE1'] += '-SIP'
            if not imheader['CTYPE2'].endswith('-SIP'):
                imheader['CTYPE2'] += '-SIP'

        # Preserve SIP distortion keywords
        sip_keywords = ['A_ORDER', 'B_ORDER', 'AP_ORDER', 'BP_ORDER']
        sip_keywords.extend([f'A_{i}_{j}' for i in range(4) for j in range(4)])
        sip_keywords.extend([f'B_{i}_{j}' for i in range(4) for j in range(4)])
        sip_keywords.extend([f'AP_{i}_{j}' for i in range(4) for j in range(4)])
        sip_keywords.extend([f'BP_{i}_{j}' for i in range(4) for j in range(4)])

        for keyword in sip_keywords:
            if keyword in imheader:
                wcs.wcs.set(keyword, imheader[keyword])

        # Update WCS in header
        imheader.update(wcs.to_header(relax=True))

        # Update image statistics
        imheader['DATAMIN'] = np.min(imdata)
        imheader['DATAMAX'] = np.max(imdata)

        # Add processing history
        if crop_size is not None:
            imheader['HISTORY'] = f'Image cropped to size {imdata.shape}'
        if binning > 1:
            imheader['HISTORY'] = f'Image binned by factor {binning}'

    # Create a new HDU with processed data and updated header
    hdu = fits.PrimaryHDU(imdata, imheader)

    # Generate output filename
    input_dir = os.path.dirname(fits_file)
    output_filename = f"{os.path.splitext(os.path.basename(fits_file))[0]}"
    if crop_size:
        crop_size_str = f"{crop_size[0]}_{crop_size[1]}" if isinstance(crop_size, tuple) else f"{crop_size}_{crop_size}"
        output_filename += f"_crop{crop_size_str}"
    if binning > 1:
        output_filename += f"_bin{binning}"
    output_filename += ".fits"

    output_file = os.path.join(input_dir, output_filename)

    # Save the processed file
    hdu.writeto(output_file, overwrite=True)

    return output_file
