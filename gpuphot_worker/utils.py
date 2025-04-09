import os
from typing import Optional, Tuple, Union

import numpy as np
from astropy.io import fits
from astropy.nddata import block_reduce, Cutout2D
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


def open_image_file(file_path: str) -> Tuple[np.ndarray, fits.Header]:
    """
    Opens an astronomical image file (FITS or NPY) and returns the data and header.

    For NPY files, it first attempts to load metadata from a corresponding '.txt'
    file (same basename). If the '.txt' file is not found or cannot be parsed,
    a minimal FITS header is generated with basic dimension information,
    and a warning is logged.

    :param file_path: Path to the image file (.fits or .npy).
    :return: Tuple containing image data (as float32) and FITS header object.
    :raises FileNotFoundError: If the specified file_path does not exist.
    :raises ValueError: If the file format is not supported (.fits, .npy).
    :raises IOError: If there's an error reading the FITS file or NPY data.
    """
    logger.debug(f"Attempting to open image file: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_lower = file_path.lower()

    if file_lower.endswith(('.fits', '.fit')):
        try:
            # Prefer using fits.open context manager
            with fits.open(file_path) as hdul:
                if not hdul:
                    raise IOError("FITS file is empty or corrupt.")
                # Try common locations for primary data
                hdu_index = 0
                if len(hdul) > 1 and hdul[0].data is None:
                    # Might be in the second HDU (e.g., MEF files)
                    logger.debug("Primary HDU data is None, trying HDU 1.")
                    hdu_index = 1
                if hdul[hdu_index].data is None:
                    raise IOError(f"No image data found in HDU {hdu_index}.")

                imdata = hdul[hdu_index].data
                imheader = hdul[hdu_index].header
                # Ensure data is float32 for processing consistency
                if imdata.dtype != np.float32:
                    imdata = imdata.astype(np.float32)
                logger.debug(f"Successfully opened FITS file: {file_path}")
                return imdata, imheader
        except Exception as e:
            # Catch more specific errors if possible, but keep a general fallback
            logger.error(f"Error opening FITS file {file_path} with fits.open: {e}. "
                         f"Attempting fallback with getdata/getheader.")
            try:
                # Fallback (less efficient, reads file twice)
                imdata = fits.getdata(file_path).astype(np.float32)
                imheader = fits.getheader(file_path)
                logger.debug(f"Successfully opened FITS file (using fallback): {file_path}")
                return imdata, imheader
            except Exception as e_fallback:
                raise IOError(f"Failed to open FITS file {file_path} even with fallback: {e_fallback}") from e_fallback

    elif file_lower.endswith('.npy'):
        try:
            imdata = np.load(file_path)
            # Ensure data is float32
            if imdata.dtype != np.float32:
                imdata = imdata.astype(np.float32)
        except Exception as e:
            raise IOError(f"Error loading NPY data from {file_path}: {e}") from e

        header_file = os.path.splitext(file_path)[0] + '.txt'
        imheader = None
        logger.debug(f"Looking for associated header file: {header_file}")

        try:
            # Try to read the associated text header file
            with open(header_file, 'r') as f:
                header_content = f.read()
            imheader = fits.Header.fromstring(header_content, sep='\n')
            logger.debug(f"Successfully loaded header from associated file: {header_file}")
        except FileNotFoundError:
            logger.warning(f"Header file '{header_file}' not found for NPY file '{file_path}'. "
                           f"Generating minimal default header.")
        except Exception as e:
            logger.warning(f"Error reading or parsing header file '{header_file}': {e}. "
                           f"Generating minimal default header.")

        if imheader is None:
            # Create a minimal header if .txt wasn't found or failed to parse
            imheader = fits.Header()
            imheader['SIMPLE'] = True
            imheader['BITPIX'] = -32  # FITS code for float32
            imheader['NAXIS'] = imdata.ndim
            if imdata.ndim == 2:
                imheader['NAXIS1'] = imdata.shape[1]  # Columns
                imheader['NAXIS2'] = imdata.shape[0]  # Rows
            elif imdata.ndim == 1:
                imheader['NAXIS1'] = imdata.shape[0]
            # Add more axes if needed, though unlikely for typical images
            imheader['EXTEND'] = True  # Standard practice
            imheader.add_comment("Minimal header generated for NPY file.")
            imheader.add_comment(f"Original file: {os.path.basename(file_path)}")

        # Basic validation: Check if header dimensions match data dimensions (if header came from .txt)
        if 'NAXIS1' in imheader and 'NAXIS2' in imheader and imdata.ndim == 2:
            if imheader['NAXIS1'] != imdata.shape[1] or imheader['NAXIS2'] != imdata.shape[0]:
                logger.warning(f"Header dimensions ({imheader.get('NAXIS2', 'N/A')}x{imheader.get('NAXIS1', 'N/A')}) "
                               f"do not match NPY data dimensions ({imdata.shape[0]}x{imdata.shape[1]}). "
                               f"Using NPY data dimensions.")
                # Correct the header to match the actual data
                imheader['NAXIS1'] = imdata.shape[1]
                imheader['NAXIS2'] = imdata.shape[0]

        logger.debug(f"Successfully processed NPY file: {file_path}")
        return imdata, imheader

    else:
        raise ValueError(f"Unsupported file format: {file_path}. Only FITS (.fits, .fit) and NPY (.npy) are supported.")


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

def crop_and_bin_image(image_file: str,
                       binning: int,
                       binning_method: str = 'sum',
                       crop_size: Optional[Union[int, Tuple[int, int]]] = None,
                       center: Optional[Tuple[float, float]] = None) -> str:
    """
    Processes a FITS or NPY file: applies binning and then optionally crops
    a region of interest. Updates/creates a FITS header to reflect all changes.
    The output file is always in FITS format.

    :param image_file: Path to the FITS or NPY file to process.
    :param binning: Binning factor (int >= 1). If 1, no binning is applied.
    :param binning_method: 'sum' or 'median'. Method for combining pixels during binning.
    :param crop_size: Desired output size in pixels after cropping.
                      None = no cropping. Int = square crop. Tuple = (width, height).
    :param center: Center of the crop region (x, y) in pixel coordinates of the
                   image *after* any binning (0-based index). If None, the center
                   of the (potentially binned) image is used. Floats are allowed.
    :return: Path to the resulting FITS file.
    :raises ValueError: If parameters are invalid or the crop region is out of bounds.
    :raises IOError: If there are problems reading/writing files.
    :raises FileNotFoundError: If the input file does not exist.
    """
    print(f"BASE_IMAGES_PATH = {BASE_IMAGES_PATH}")
    relative_path = os.path.relpath(image_file, BASE_IMAGES_PATH) if 'BASE_IMAGES_PATH' in globals() else image_file
    logger.debug(f"Processing file: {relative_path}, "
                 f"binning: {binning}, method: {binning_method}, "
                 f"crop: {crop_size}, center: {center}")

    # 1) Open the image using the updated function
    # This will return data and a header (either from FITS, .txt, or minimal)
    try:
        imdata, imheader = open_image_file(image_file)
    except (FileNotFoundError, ValueError, IOError) as e:
        logger.error(f"Failed to open or read image file {image_file}: {e}")
        raise e  # Re-raise the specific error

    # # Determine if the original header was likely minimal (generated for NPY)
    # # Heuristic: check for more than just the basic keys we add
    # basic_keys = {'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND'}
    # is_minimal_header = all(
    #     key in basic_keys or imheader.comments[key].strip().startswith(('Minimal header generated', 'Original file:'))
    #     for key in imheader) and file_lower.endswith('.npy')

    # if is_minimal_header:
    #     logger.warning(f"Processing NPY file '{relative_path}' with a minimal header. "
    #                    f"WCS, pixel scale, and photometric keywords (GAIN, RDNOISE, etc.) "
    #                    f"are likely missing. Related header updates will be skipped.")

    original_shape = imdata.shape
    was_originally_fits = image_file.lower().endswith(('.fits', '.fit'))

    # Try to build WCS from header, fail gracefully
    wcs = None
    if 'CTYPE1' in imheader:  # A common basic check for WCS presence
        try:
            wcs = WCS(imheader)
            if not wcs.is_celestial:
                logger.debug("WCS found but is not celestial. Ignoring for cropping/updates.")
                wcs = None  # Treat non-celestial WCS as absent for this purpose
        except Exception as e:
            logger.warning(f"Could not build WCS for {relative_path}: {e}. Proceeding without WCS updates.")
            wcs = None

    # 2) Validate binning factor
    if not isinstance(binning, int) or binning < 1:
        raise ValueError("Binning factor must be an integer >= 1.")

    # 3) Check if nothing needs to be done (only if input was FITS and no ops)
    if binning == 1 and crop_size is None and was_originally_fits:
        logger.debug(f"No processing needed for FITS file {relative_path}. Returning original path.")
        return image_file

    # Add initial processing comment to header
    imheader.set('HISTORY', f"Processed by crop_and_bin_image function.")

    processed_data = imdata.copy()  # Work on a copy
    processed_wcs = wcs

    # -------------------------------------------------------------------------
    # BINNING
    # -------------------------------------------------------------------------
    if binning > 1:
        valid_methods = {'sum', 'median'}
        binning_method = binning_method.lower()
        if binning_method not in valid_methods:
            raise ValueError(f"Invalid binning method '{binning_method}'. "
                             f"Valid options are: {valid_methods}")

        # *** Check for divisibility ***
        original_rows, original_cols = processed_data.shape
        if original_rows % binning != 0 or original_cols % binning != 0:
            new_rows = (original_rows // binning) * binning
            new_cols = (original_cols // binning) * binning
            discarded_rows = original_rows - new_rows
            discarded_cols = original_cols - new_cols
            logger.warning(f"Image dimensions ({original_rows}x{original_cols}) are not perfectly "
                           f"divisible by binning factor {binning}. "
                           f"block_reduce will effectively use the top-left "
                           f"{new_rows}x{new_cols} region, discarding {discarded_rows} row(s) "
                           f"and {discarded_cols} column(s) from the bottom/right edges.")
        # ******************************

        bin_func = np.sum if binning_method == 'sum' else np.nanmedian

        try:
            binned_data = block_reduce(processed_data,
                                       block_size=(binning, binning),
                                       func=bin_func,
                                       cval=np.nan)  # Use nan for padding if func handles it
        except Exception as e:
            raise RuntimeError(f"Error during block_reduce binning: {e}") from e

        processed_data = binned_data
        logger.debug(f"Applied {binning}x{binning} binning using '{binning_method}'.")
        imheader.set('HISTORY', f"Applied {binning}x{binning} binning using '{binning_method}'.")

        # ---------------------------------------------------------------------
        # Update Header and WCS (if possible)
        # ---------------------------------------------------------------------
        # These updates only make sense if the original header had the info

        # -- 1) Update pixel scale keywords (if they exist)
        pxsize_updated = False
        if 'PIXSCALE' in imheader:  # Common keyword
            try:
                original_pixscale = float(imheader['PIXSCALE'])
                imheader['PIXSCALE'] = original_pixscale * binning
                imheader.comments['PIXSCALE'] = f"Original pixel scale: {original_pixscale}"
                pxsize_updated = True
            except (ValueError, TypeError):
                logger.warning("Could not parse PIXSCALE value for update.")
        elif 'SECPIX' in imheader:  # Another common one
            try:
                original_secpix = float(imheader['SECPIX'])
                imheader['SECPIX'] = original_secpix * binning
                imheader.comments['SECPIX'] = f"Original pixel scale: {original_secpix}"
                pxsize_updated = True
            except (ValueError, TypeError):
                logger.warning("Could not parse SECPIX value for update.")
        # Add checks for CDELT if no explicit scale keyword found and WCS not present/updated below?
        # Be careful not to double-update if WCS handles CDELT.

        # -- 2) Adjust WCS if it exists and is celestial
        if processed_wcs is not None:  # Already checked for is_celestial earlier
            try:
                # Use WCS slicing for updates - handles CRPIX, CD/PC/CDELT
                processed_wcs = processed_wcs[::binning, ::binning]
                logger.debug("WCS updated for binning using slicing.")
                # WCS object is updated, will be written to header later
            except Exception as e:
                logger.error(f"Failed to update WCS after binning: {e}. WCS info might be incorrect.")
                processed_wcs = None  # Invalidate WCS if update fails

        # -- 3) Update photometric keywords (GAIN, RDNOISE, SATLEVEL) if they exist
        imheader['BINNING'] = (binning, 'Binning factor applied (may differ from detector binning)')
        imheader['BIN_ALG'] = (binning_method.upper(), 'Pixel combination method used in software')

        gain_key = imheader.cards['GAIN'].keyword if 'GAIN' in imheader else None  # Find exact case
        rdnoise_key = imheader.cards['RDNOISE'].keyword if 'RDNOISE' in imheader else None
        sat_key = next((k for k in ['SATURATE', 'SATLEVEL', 'MAXLIN'] if k in imheader),
                       None)  # Try common saturation keys

        bin_sq = binning ** 2

        if binning_method == 'sum':
            if gain_key and isinstance(imheader.get(gain_key), (int, float)):
                imheader[gain_key] /= bin_sq
                imheader.comments[gain_key] = f"Adjusted for {binning}x{binning} sum binning"
            if rdnoise_key and isinstance(imheader.get(rdnoise_key), (int, float)):
                imheader[
                    rdnoise_key] *= binning  # Assuming RDNOISE is in e-, variance adds, std dev adds in quadrature -> sqrt(N)*sigma_pix = binning*sigma_pix
                imheader.comments[rdnoise_key] = f"Adjusted for {binning}x{binning} sum binning (sqrt({bin_sq}) factor)"
            if sat_key and isinstance(imheader.get(sat_key), (int, float)):
                imheader[sat_key] *= bin_sq
                imheader.comments[sat_key] = f"Adjusted for {binning}x{binning} sum binning"
        else:  # median
            if gain_key:
                # Gain adjustment for median is complex and often non-linear
                imheader.add_comment(
                    f"Original {gain_key}={imheader.get(gain_key)} may not be accurate after median binning.")
            if rdnoise_key and isinstance(imheader.get(rdnoise_key), (int, float)):
                # Median reduces noise approx by sqrt(N), where N=bin_sq
                imheader[rdnoise_key] /= binning  # rdnoise / sqrt(bin_sq) = rdnoise / binning
                imheader.comments[
                    rdnoise_key] = f"Adjusted for {binning}x{binning} median binning (approx /sqrt({bin_sq}) factor)"
            if sat_key:
                imheader.add_comment(
                    f"Original {sat_key}={imheader.get(sat_key)} may not be accurate after median binning.")
            imheader.add_comment(f"WARNING: Median binning affects photometric linearity.")

    # -------------------------------------------------------------------------
    # CROPPING
    # -------------------------------------------------------------------------
    if crop_size is not None:
        current_shape = processed_data.shape
        current_h, current_w = current_shape

        # Validate and normalize crop_size
        crop_w: int
        crop_h: int
        if isinstance(crop_size, int):
            if crop_size <= 0: raise ValueError("Crop size must be a positive integer.")
            crop_w, crop_h = crop_size, crop_size
        elif isinstance(crop_size, (tuple, list)) and len(crop_size) == 2:
            crop_w, crop_h = int(crop_size[0]), int(crop_size[1])
            if crop_w <= 0 or crop_h <= 0:
                raise ValueError("Crop width and height must be positive integers.")
        else:
            raise ValueError(
                "crop_size must be None, a positive integer, or a tuple/list of two positive integers (width, height).")

        # Validate crop_size against current image dimensions
        if crop_w > current_w or crop_h > current_h:
            raise ValueError(f"Requested crop size ({crop_w}x{crop_h}) is larger than "
                             f"the current image dimensions ({current_w}x{current_h}) after binning.")

        # Validate and determine center coordinates (using 0-based image coordinates X, Y)
        center_x: float
        center_y: float
        if center is None:
            # Default to the center of the current image
            center_x = (current_w - 1) / 2.0  # Center pixel coordinate X
            center_y = (current_h - 1) / 2.0  # Center pixel coordinate Y
            logger.debug(f"No center provided, using image center: ({center_x:.2f}, {center_y:.2f})")
        elif isinstance(center, (tuple, list)) and len(center) == 2:
            try:
                center_x, center_y = float(center[0]), float(center[1])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid center coordinates: {center}. Must be numbers.") from e
            # Check if center is within bounds (0 <= coord < dim)
            if not (0 <= center_x < current_w and 0 <= center_y < current_h):
                raise ValueError(f"Provided center ({center_x:.2f}, {center_y:.2f}) is outside "
                                 f"the current image boundaries (W={current_w}, H={current_h}).")
        else:
            raise ValueError("Center must be None or a tuple/list of two numbers (x, y).")

        # Perform cropping
        # Cutout2D expects center=(y, x) and size=(h, w) for numpy array indexing
        cutout_center_yx = (center_y, center_x)
        cutout_size_hw = (crop_h, crop_w)

        try:
            if processed_wcs is not None:  # Use Cutout2D if we have a valid WCS
                logger.debug(f"Cropping using Cutout2D with WCS: center={cutout_center_yx}, size={cutout_size_hw}")
                cutout = Cutout2D(processed_data, position=cutout_center_yx, size=cutout_size_hw,
                                  wcs=processed_wcs, mode='trim', copy=True)  # mode='trim' handles edges
                processed_data = cutout.data
                processed_wcs = cutout.wcs  # WCS is automatically updated
                logger.debug(f"Cropped image to {processed_data.shape[1]}x{processed_data.shape[0]} using Cutout2D.")
            else:
                # Manual slicing if no WCS is available
                logger.debug(
                    f"Cropping using numpy slicing: center=({center_x:.1f},{center_y:.1f}), size=({crop_w},{crop_h})")
                # Calculate integer slice indices (0-based)
                y_min = int(np.round(center_y - crop_h / 2.0))
                y_max = y_min + crop_h  # Slice upper bound is exclusive
                x_min = int(np.round(center_x - crop_w / 2.0))
                x_max = x_min + crop_w  # Slice upper bound is exclusive

                # Clip indices to be within the image bounds (important!)
                y_min = max(0, y_min)
                y_max = min(current_h, y_max)
                x_min = max(0, x_min)
                x_max = min(current_w, x_max)

                # Check if the resulting slice has the correct size (it might be smaller if center was near edge)
                actual_h = y_max - y_min
                actual_w = x_max - x_min
                if actual_h != crop_h or actual_w != crop_w:
                    logger.warning(f"Requested crop size was {crop_h}x{crop_w}, but resulting crop "
                                   f"is {actual_h}x{actual_w} due to image boundaries or centering.")

                if actual_h <= 0 or actual_w <= 0:
                    raise ValueError("Calculated crop region has zero or negative size. Check center and crop_size.")

                processed_data = processed_data[y_min:y_max, x_min:x_max]
                # WCS remains None
                logger.debug(f"Cropped image to {processed_data.shape[1]}x{processed_data.shape[0]} using slicing.")

            # Add cropping info to header
            crop_size_str = f"{processed_data.shape[1]}x{processed_data.shape[0]}"  # Use actual final size
            center_str = f"({center_x:.2f}, {center_y:.2f})"  # Use requested center
            imheader.set('HISTORY',
                         f"Cropped to {crop_size_str} around requested center pix {center_str} (X,Y; 0-based)")
            # Optionally add CRPIX adjustment comment if only slicing was done? No, too complex without WCS.

        except Exception as e:
            # Catch errors from Cutout2D or slicing logic
            logger.error(f"Error during cropping operation: {e}")
            raise ValueError(f"Error during cropping: {e}") from e

    # -------------------------------------------------------------------------
    # FINAL HEADER UPDATES & SAVE (Always as FITS)
    # -------------------------------------------------------------------------
    final_shape = processed_data.shape

    # Update NAXIS keywords to final dimensions
    imheader['NAXIS'] = 2  # Assuming 2D image data
    imheader['NAXIS1'] = final_shape[1]  # Width
    imheader['NAXIS2'] = final_shape[0]  # Height

    # Write the updated WCS object to the header (if it exists)
    if processed_wcs is not None:
        try:
            # Remove potentially conflicting old WCS keywords before updating
            # This is safer than just `imheader.update(processed_wcs.to_header())`
            # which might leave obsolete keywords
            header_keys_to_remove = []
            wcs_keywords_obj = processed_wcs.to_header()
            for key in imheader:
                # Standard WCS keywords + potentially related (like EQUINOX, RADESYS)
                if key.startswith(('CRVAL', 'CRPIX', 'CDELT', 'CTYPE', 'CUNIT', 'CD', 'PC', 'PV', 'LONGPOLE', 'LATPOLE',
                                   'EQUINOX', 'RADESYS', 'WCSAXES')):
                    # Only remove if not present in the new WCS header to avoid removing unrelated keywords
                    # if key not in wcs_keywords_obj: # Be careful, this might keep old PC if new uses CD
                    header_keys_to_remove.append(key)

            # A safer approach might be to always remove known WCS keyword patterns
            known_wcs_patterns = ('CRVAL', 'CRPIX', 'CDELT', 'CTYPE', 'CUNIT', 'CD1_', 'CD2_', 'PC1_', 'PC2_', 'PV')
            header_keys_to_remove = [k for k in imheader if k.startswith(known_wcs_patterns)]
            # Add specific keys
            for k in ['EQUINOX', 'RADESYS', 'WCSAXES', 'LONGPOLE', 'LATPOLE']:
                if k in imheader: header_keys_to_remove.append(k)

            logger.debug(f"Removing old WCS keys: {header_keys_to_remove}")
            for key in set(header_keys_to_remove):  # Use set for uniqueness
                try:
                    del imheader[key]
                except KeyError:
                    pass  # Ignore if already removed

            # Now update with the new WCS information
            imheader.update(wcs_keywords_obj)
            logger.debug("WCS keywords updated in header.")

        except Exception as e:
            logger.error(f"Failed to write updated WCS to header: {e}. Header WCS might be inconsistent.")
            imheader.add_comment("ERROR: Failed to update WCS keywords in header during processing.")

    # Update data statistics (handle all-NaN case)
    if np.any(np.isfinite(processed_data)):
        imheader['DATAMIN'] = float(np.nanmin(processed_data))
        imheader['DATAMAX'] = float(np.nanmax(processed_data))
    else:
        logger.warning("Final image data contains only NaNs or infinite values.")
        imheader['DATAMIN'] = 0.0  # Or np.nan? FITS standard doesn't specify behavior here well.
        imheader['DATAMAX'] = 0.0  # Or np.nan?

    # Record original dimensions
    imheader.set('O_NAXIS1', original_shape[1], 'Original NAXIS1 before processing')
    imheader.set('O_NAXIS2', original_shape[0], 'Original NAXIS2 before processing')
    imheader.set('O_FILENA', os.path.basename(image_file), 'Original input filename')

    # Add comment if input was NPY
    if not was_originally_fits:
        imheader.set('HISTORY', f'Input file was NPY: {os.path.basename(image_file)}')
        # if is_minimal_header:
        #     imheader.set('HISTORY', 'A minimal FITS header was generated for the NPY input.')

    # Construct output filename
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    output_filename = base_name
    if binning > 1:
        output_filename += f"_bin{binning}_{binning_method}"
    if crop_size:
        # Use the actual final dimensions for the filename crop tag
        output_filename += f"_crop{final_shape[1]}x{final_shape[0]}"  # WxH

    output_filename += ".fits"
    output_path = os.path.join(os.path.dirname(image_file), output_filename)

    # Create the primary HDU with the processed data and header
    # Ensure data type is float32 for saving
    if processed_data.dtype != np.float32:
        processed_data = processed_data.astype(np.float32)
        imheader['BITPIX'] = -32  # Ensure BITPIX matches data type

    hdu = fits.PrimaryHDU(processed_data, header=imheader)

    # Save the new FITS file
    try:
        hdu.writeto(output_path, overwrite=True, output_verify='fix')
        output_relative_path = os.path.relpath(output_path,
                                               BASE_IMAGES_PATH) if 'BASE_IMAGES_PATH' in globals() else output_path
        logger.debug(f"Processed image saved successfully to: {output_relative_path}")
    except Exception as e:
        logger.error(f"Error writing processed FITS file to {output_path}: {e}")
        raise IOError(f"Error writing processed FITS file to {output_path}: {e}") from e

    return output_path
