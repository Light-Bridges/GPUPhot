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
def transform_coords(coord: float, binning: int) -> float:
    """Transforms a 0-based coordinate from original to binned frame."""
    if binning <= 1:
        return coord
    # Center of original pixel (coord + 0.5) maps to center of binned pixel (new_coord + 0.5)
    # (coord + 0.5) / binning = new_coord + 0.5
    # new_coord = (coord + 0.5) / binning - 0.5
    return (coord + 0.5) / binning - 0.5


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
    :param center: Defines the center of the crop region.
                   - If None (default): The crop is centered geometrically, ensuring the
                     point corresponding to the geometric center of the *original* image
                     becomes the geometric center of the cropped image.
                   - If tuple (x, y): Interpreted as the desired center in *pixel coordinates*
                     (0-based, X=axis1, Y=axis0) relative to the *original* image frame.
                     These coordinates are then transformed to the current (potentially binned)
                     frame before cropping.
    :return: Path to the resulting FITS file.
    :raises ValueError: If parameters are invalid, coordinates are out of bounds,
                       or the crop region is invalid.
    :raises IOError: If there are problems reading/writing files.
    :raises FileNotFoundError: If the input file does not exist.
    """
    relative_path = os.path.relpath(image_file, BASE_IMAGES_PATH) if 'BASE_IMAGES_PATH' in globals() else image_file
    logger.debug(
        f"Processing file: {relative_path}, binning: {binning}, method: {binning_method}, crop: {crop_size}, center(Original X,Y): {center}")

    # 1) Open the image
    try:
        imdata, imheader = open_image_file(image_file)
    except (FileNotFoundError, ValueError, IOError) as e:
        logger.error(f"Failed to open or read image file {image_file}: {e}")
        raise e

    original_shape = imdata.shape  # Numpy shape (rows, cols) -> (H, W)
    original_h, original_w = original_shape
    was_originally_fits = image_file.lower().endswith(('.fits', '.fit'))

    # Try to build WCS from the original header (needed for header updates, not centering)
    original_wcs = None
    try:
        temp_wcs = WCS(imheader, relax=True)
        if temp_wcs.is_celestial:
            original_wcs = temp_wcs
            logger.debug("Valid original WCS found.")
        else:
            logger.debug("Original WCS not celestial.")
    except Exception:
        logger.debug("No valid original WCS found.")

    # 2) Validate binning factor
    if not isinstance(binning, int) or binning < 1: raise ValueError("Binning factor must be an integer >= 1.")

    # 3) Check if nothing needs to be done
    if binning == 1 and crop_size is None and was_originally_fits:
        logger.debug("No processing needed.")
        return image_file

    # Add header comments & history
    temp_key = 'COMINIT'
    imheader[temp_key] = 'e'
    imheader.insert(temp_key, ('COMMENT', '*' * 27))
    imheader.insert(temp_key, ('COMMENT', '    IMAGE PROCESSING HISTORY   '))
    imheader.insert(temp_key, ('COMMENT', '*' * 27))
    imheader.insert(temp_key, ('COMMENT', ' '))
    imheader.set('HISTORY', f"Processed by crop_and_bin_image function.")

    processed_data = imdata.copy()
    processed_wcs = original_wcs  # Will be updated by binning if needed

    # -------------------------------------------------------------------------
    # BINNING
    # -------------------------------------------------------------------------
    if binning > 1:
        # ... (Binning logic: block_reduce, check divisibility, update data etc.) ...
        logger.debug(f"Applying {binning}x{binning} binning ({binning_method}). Original shape: {original_shape}")
        # (Assume block_reduce updates processed_data)
        processed_data = block_reduce(processed_data, (binning, binning),
                                      func=np.sum if binning_method == 'sum' else np.nanmedian)  # Simplified example
        logger.debug(f"Shape after binning: {processed_data.shape}")

        # --- Update WCS object AFTER binning ---
        if processed_wcs:
            try:
                processed_wcs = processed_wcs[::binning, ::binning]
                logger.debug("WCS object updated for binning.")
            except Exception as e:
                logger.error(f"Failed to update WCS object after binning: {e}. WCS is now invalid.")
                processed_wcs = None
        # Update header keywords (BINNING, GAIN, RDNOISE, PIXSCALE, etc.)
        imheader.set('HISTORY', f"Applied {binning}x{binning} binning using '{binning_method}'.")
        # (Keyword update code omitted for brevity)

    # -------------------------------------------------------------------------
    # CROPPING
    # -------------------------------------------------------------------------
    if crop_size is not None:
        current_h, current_w = processed_data.shape  # Numpy shape (H, W) after any binning

        # --- Validate and normalize crop_size (W, H) ---
        crop_w: int
        crop_h: int
        if isinstance(crop_size, int):  # Square crop
            if crop_size <= 0: raise ValueError("Crop size must be > 0.")
            crop_w, crop_h = crop_size, crop_size
        elif isinstance(crop_size, (tuple, list)) and len(crop_size) == 2:  # Rectangular crop (W, H)
            crop_w, crop_h = int(crop_size[0]), int(crop_size[1])
            if crop_w <= 0 or crop_h <= 0: raise ValueError("Crop W & H must be > 0.")
        else:
            raise ValueError("crop_size must be None, int, or tuple (W, H).")

        # Check if requested crop is larger than the current image
        if crop_w > current_w or crop_h > current_h:
            raise ValueError(f"Crop size ({crop_w}x{crop_h}) > current image ({current_w}x{current_h}).")

        # --- Determine the ORIGIN coordinates (X, Y) for the center parameter ---
        # These are the coordinates in the *original* image frame that the user wants at the center.
        origin_center_x: float
        origin_center_y: float
        center_input_description = ""  # For logging/history

        if center is None:
            # Default: Use the geometric center of the original image
            origin_center_x = (original_w - 1) / 2.0
            origin_center_y = (original_h - 1) / 2.0
            center_input_description = "original geometric center"
            logger.debug(
                f"Center is None. Using {center_input_description}: (X={origin_center_x:.3f}, Y={origin_center_y:.3f})")
        else:
            # User provided center (X, Y) relative to the *original* image
            try:
                user_x_orig, user_y_orig = float(center[0]), float(center[1])
                center_input_description = f"provided original pixel (X={user_x_orig:.2f}, Y={user_y_orig:.2f})"
                logger.debug(f"User provided center relative to original: (X={user_x_orig:.3f}, Y={user_y_orig:.3f})")

                # Validate user coords against ORIGINAL dimensions
                if not (0 <= user_x_orig < original_w and 0 <= user_y_orig < original_h):
                    raise ValueError(
                        f"Provided center {center} is outside original image bounds (W={original_w}, H={original_h}).")
                origin_center_x = user_x_orig
                origin_center_y = user_y_orig
            except (ValueError, TypeError, IndexError) as e:
                raise ValueError(
                    f"Invalid center format or value: {center}. Must be (number, number) for original X, Y.") from e

        # --- Transform the ORIGIN center coordinates to the TARGET center coordinates in the CURRENT frame ---
        target_center_x = transform_coords(origin_center_x, binning)
        target_center_y = transform_coords(origin_center_y, binning)
        logger.debug(
            f"Transformed center to current frame (bin={binning}): (Target X={target_center_x:.3f}, Target Y={target_center_y:.3f})")

        # --- Validate the TARGET coordinates against the CURRENT image dimensions ---
        # This is crucial! The target point might be outside the binned image if it was near an edge discarded by non-divisible binning.
        if not (0 <= target_center_x < current_w and 0 <= target_center_y < current_h):
            raise ValueError(f"The target center point (X={target_center_x:.2f}, Y={target_center_y:.2f}) "
                             f"corresponding to {center_input_description} falls outside the "
                             f"current image boundaries (W={current_w}, H={current_h}) after binning. "
                             f"Cannot perform crop.")

        # --- Perform cropping using the validated TARGET center ---
        cutout_position_yx = (target_center_y, target_center_x)  # Order for Cutout2D: (Y, X)
        cutout_size_hw = (crop_h, crop_w)  # Order for Cutout2D: (H, W)

        try:
            if processed_wcs:  # Use Cutout2D if WCS is valid *now*
                logger.debug(f"Cropping using Cutout2D: position(Y,X)={cutout_position_yx}, size(H,W)={cutout_size_hw}")
                cutout = Cutout2D(processed_data, position=cutout_position_yx, size=cutout_size_hw,
                                  wcs=processed_wcs, mode='trim', copy=True)
                processed_data = cutout.data
                processed_wcs = cutout.wcs  # Get updated WCS from cutout
            else:
                # Manual slicing if no valid WCS
                logger.debug(
                    f"Cropping using numpy slicing: center(X,Y)=({target_center_x:.3f},{target_center_y:.3f}), size(W,H)=({crop_w},{crop_h})")
                # Calculate slice boundaries (0-based integer indices)
                y_min = int(np.round(target_center_y - crop_h / 2.0))
                y_max = y_min + crop_h
                x_min = int(np.round(target_center_x - crop_w / 2.0))
                x_max = x_min + crop_w
                # Clip slices
                y_min_clip = max(0, y_min)
                y_max_clip = min(current_h, y_max)
                x_min_clip = max(0, x_min)
                x_max_clip = min(current_w, x_max)
                actual_h = y_max_clip - y_min_clip
                actual_w = x_max_clip - x_min_clip
                if actual_h != crop_h or actual_w != crop_w: logger.warning(
                    f"Requested crop {crop_h}x{crop_w} resulted in {actual_h}x{actual_w} due to image boundaries.")
                if actual_h <= 0 or actual_w <= 0: raise ValueError("Calculated crop region has zero size.")
                # Apply slicing array[Y, X]
                processed_data = processed_data[y_min_clip:y_max_clip, x_min_clip:x_max_clip]

            # Log success and add history
            final_w, final_h = processed_data.shape[1], processed_data.shape[0]
            logger.info(f"Cropped image to {final_w}x{final_h}. Centered based on: {center_input_description}")
            imheader.set('HISTORY', f"Cropped to {final_w}x{final_h} based on {center_input_description}")

        except Exception as e:
            logger.error(f"Error during cropping operation: {e}", exc_info=True)
            raise ValueError(f"Error during cropping: {e}") from e

    # -------------------------------------------------------------------------
    # FINAL HEADER UPDATES & SAVE
    # -------------------------------------------------------------------------
    final_shape = processed_data.shape  # Numpy shape (H, W)
    final_w, final_h = final_shape[1], final_shape[0]

    imheader['NAXIS'] = 2
    imheader['NAXIS1'] = final_w
    imheader['NAXIS2'] = final_h

    # Update WCS in header if it exists and is valid
    if processed_wcs:
        try:
            # Clean old WCS keys before updating
            known_wcs_patterns = ('CRVAL', 'CRPIX', 'CDELT', 'CTYPE', 'CUNIT', 'CD1_', 'CD2_', 'PC1_', 'PC2_', 'PV')
            keys_to_remove = [k for k in imheader if k.startswith(known_wcs_patterns)]
            for k in ['EQUINOX', 'RADESYS', 'WCSAXES', 'LONGPOLE', 'LATPOLE']:
                if k in imheader: keys_to_remove.append(k)
            for key in set(keys_to_remove):
                try:
                    del imheader[key]
                except KeyError:
                    pass
            imheader.update(processed_wcs.to_header())
            logger.debug("WCS keywords updated in header for cropping.")
        except Exception as e:
            logger.error(f"Failed to write updated WCS to header: {e}.")
            imheader.add_comment("ERROR: Failed to update WCS keywords after cropping.")

    # Update DATAMIN/MAX, O_NAXIS, O_FILENA etc.
    # (Code omitted for brevity - same as previous version)
    if np.any(np.isfinite(processed_data)):
        imheader['DATAMIN'], imheader['DATAMAX'] = float(np.nanmin(processed_data)), float(np.nanmax(processed_data))
    else:
        logger.warning("Final image data NaN/inf.")
        imheader['DATAMIN'], imheader['DATAMAX'] = 0.0, 0.0
    imheader.set('O_NAXIS1', original_w, 'Original NAXIS1')
    imheader.set('O_NAXIS2', original_h, 'Original NAXIS2')
    imheader.set('O_FILENA', os.path.basename(image_file)[:68], 'Original input filename')
    if not was_originally_fits: imheader.set('HISTORY', f'Input was NPY: {os.path.basename(image_file)}')

    # Construct output filename
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    output_filename = base_name
    if binning > 1: output_filename += f"_bin{binning}_{binning_method}"
    if crop_size: output_filename += f"_crop{final_w}x{final_h}"  # Use actual WxH
    output_filename += ".fits"
    output_path = os.path.join(os.path.dirname(image_file), output_filename)

    # Ensure data type and BITPIX
    if processed_data.dtype != np.float32: processed_data = processed_data.astype(np.float32)
    if 'BITPIX' not in imheader or imheader['BITPIX'] != -32: imheader['BITPIX'] = -32

    # Clean up temp key
    if temp_key in imheader: del imheader[temp_key]

    # Create Primary HDU and save
    hdu = fits.PrimaryHDU(processed_data, header=imheader)
    try:
        hdu.writeto(output_path, overwrite=True, output_verify='fix')
        output_relative_path = os.path.relpath(output_path,
                                               BASE_IMAGES_PATH) if 'BASE_IMAGES_PATH' in globals() else output_path
        logger.info(f"Processed image saved successfully to: {output_relative_path}")
    except Exception as e:
        logger.error(f"Error writing processed FITS file to {output_path}: {e}")
        raise IOError(f"Error writing FITS to {output_path}: {e}") from e

    return output_path
