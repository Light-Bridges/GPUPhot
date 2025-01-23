import hashlib
import hmac
import os

import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from skimage.measure import block_reduce
from sqlalchemy import create_engine, exc, text

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

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        elif not isinstance(crop_size, tuple) or len(crop_size) != 2:
            raise ValueError("Crop size must be an integer or a tuple of two integers.")

        # Check if the crop is possible
        half_width, half_height = crop_size[0] // 2, crop_size[1] // 2
        if (center[0] - half_width < 0 or center[0] + half_width > image_shape[1] or
                center[1] - half_height < 0 or center[1] + half_height > image_shape[0]):
            raise ValueError(
                "The specified crop size and center would result in a region outside the image boundaries.")

    # Check if any processing is needed
    if binning <= 1 and crop_size is None:
        return fits_file

    # Apply cropping if crop_size is specified
    if crop_size is not None:
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
        if hasattr(wcs.wcs, 'cdelt'):
            wcs.wcs.cdelt *= binning
        elif hasattr(wcs.wcs, 'cd'):
            wcs.wcs.cd *= binning

        # Update GAIN and RDNOISE if present
        if 'GAIN' in imheader:
            imheader['GAIN'] *= binning ** 2
        if 'RDNOISE' in imheader:
            imheader['RDNOISE'] /= binning

    # Update header after processing
    if crop_size is not None or binning > 1:
        imheader['NAXIS1'] = imdata.shape[1]
        imheader['NAXIS2'] = imdata.shape[0]

        if 'CD1_1' in imheader:
            imheader.remove('CDELT1', ignore_missing=True)
            imheader.remove('CDELT2', ignore_missing=True)
        elif binning > 1:
            imheader['CDELT1'] = imheader.get('CDELT1', 1) * binning
            imheader['CDELT2'] = imheader.get('CDELT2', 1) * binning

        # Handle SIP distortion
        try:
            if any(key.startswith('A_') or key.startswith('B_') for key in imheader):
                for ctype in ['CTYPE1', 'CTYPE2']:
                    if ctype in imheader and not imheader[ctype].endswith('-SIP'):
                        imheader[ctype] += '-SIP'

                sip_keywords = ['A_ORDER', 'B_ORDER', 'AP_ORDER', 'BP_ORDER']
                sip_keywords.extend([f'{p}_{i}_{j}' for p in 'ABAPBP' for i in range(4) for j in range(4)])

                # Solo intentar procesar SIP si el objeto WCS lo soporta
                if hasattr(wcs, 'sip') and wcs.sip is not None:
                    for keyword in sip_keywords:
                        if keyword in imheader:
                            value = imheader[keyword]
                            if keyword.startswith(('A_', 'B_')) and keyword not in ['A_ORDER', 'B_ORDER']:
                                value /= binning ** (int(keyword.split('_')[1]) - 1)
                            setattr(wcs.sip, keyword.lower(), value)

        except Exception as e:
            pass

        # Update WCS in header
        imheader.update(wcs.to_header(relax=True))

        imheader['DATAMIN'] = np.min(imdata)
        imheader['DATAMAX'] = np.max(imdata)

        if crop_size is not None:
            imheader.add_history(f'Image cropped to size {imdata.shape}')
        if binning > 1:
            imheader.add_history(f'Image binned by factor {binning}')

    # Create a new HDU and save
    hdu = fits.PrimaryHDU(imdata, imheader)
    output_filename = f"{os.path.splitext(os.path.basename(fits_file))[0]}"
    if crop_size:
        crop_size_str = f"{crop_size[0]}_{crop_size[1]}" if isinstance(crop_size, tuple) else f"{crop_size}_{crop_size}"
        output_filename += f"_crop{crop_size_str}"
    if binning > 1:
        output_filename += f"_bin{binning}"
    output_filename += ".fits"
    output_file = os.path.join(os.path.dirname(fits_file), output_filename)
    hdu.writeto(output_file, overwrite=True)

    return output_file


def __generate_connection_string():
    """
    Generate a PostgreSQL connection string using environment variables.

    :return: PostgreSQL connection string.
    :rtype: str
    """

    # Get database connection parameters from environment variables
    db_name = os.getenv('POSTGRES_DB', 'GPUPhotDB')
    user = os.getenv('POSTGRES_USER', 'admin')
    password = os.getenv('POSTGRES_PASSWORD', 'gpuphot')
    host = 'postgres'
    port = '5432'

    # Create the connection string
    connection_string = f'postgresql://{user}:{password}@{host}:{port}/{db_name}'

    return connection_string


def insert_dataframe_to_postgres(df, tbl_name, unique_col='id'):
    """
    Insert a DataFrame into a PostgreSQL table, replacing existing records.

    :param df: DataFrame to insert.
    :type df: pandas.DataFrame
    :param tbl_name: Name of the target table.
    :type tbl_name: str
    :param unique_col: Name of the column with unique values.
    :type unique_col: str
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        # Get database connection parameters from environment variables
        connection_string = __generate_connection_string()

        # Create an SQLAlchemy engine
        engine = create_engine(connection_string)

        # Proceed only if DataFrame is not empty
        if not df.empty:
            unique_values = df[unique_col].unique()

            with engine.begin() as connection:  # Use transaction context manager
                # Delete existing records in one go
                delete_query = text(f"DELETE FROM {tbl_name} WHERE {unique_col} IN :values")
                connection.execute(delete_query, {"values": tuple(unique_values)})

                # Insert the DataFrame into the PostgreSQL table in chunks
                chunk_size = 5000  # Adjust based on your needs
                for start in range(0, len(df), chunk_size):
                    end = start + chunk_size
                    df.iloc[start:end].to_sql(tbl_name, con=connection, if_exists='append', index=False)

        return True

    except exc.SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error: {str(e)}")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

    return False


def queryStrAdd(query: str, toAdd: str) -> str:
    """
    Add a string value to an SQL query.

    :param query: Existing SQL query.
    :type query: str
    :param toAdd: String to add to the query.
    :type toAdd: str
    :return: Updated SQL query.
    :rtype: str
    """
    return query + "'" + toAdd + "', "


def queryAdd(query: str, toAdd) -> str:
    """
    Add a non-string value to an SQL query.

    :param query: Existing SQL query.
    :type query: str
    :param toAdd: Value to add to the query.
    :type toAdd: Any
    :return: Updated SQL query.
    :rtype: str
    """
    return query + str(toAdd) + ", "


# Takes a FITS header and formats it into a PSQL-compatible HStore fragment.
def headerToHstore(header):
    """
    Convert a FITS header to a PostgreSQL HStore-compatible string.

    :param header: FITS header.
    :type header: astropy.io.fits.Header
    :return: HStore-compatible string.
    :rtype: str
    """
    fragment = ""
    for key, value in header.items():
        if key != "COMMENT" and not isinstance(value, fits.header._HeaderCommentaryCards):
            # Convertir todo a string y escapar las comillas dobles
            key_str = str(key).replace('"', '\\"')
            value_str = str(value).replace('"', '\\"')
            fragment += f'"{key_str}" => "{value_str}", '
    return fragment[:-2]


def insert_header(header, file_path):
    """
    Generate an SQL query to insert or update a FITS header in the database.

    :param header: FITS header.
    :type header: astropy.io.fits.Header
    :param file_path: Path of the FITS file.
    :type file_path: str
    :return: SQL query string.
    :rtype: str
    """
    query_parts = ["INSERT INTO imastats (id, file_path, "]

    columns = [
        "naxis1", "naxis2", "telescop", "instrume", "camera", "filter",
        "date_obs", "exptime", "object", "ra", "dec", "fwhm", "maglim", "header"
    ]
    query_parts.append(", ".join(columns))
    query_parts.append(") VALUES (")

    query_parts.append(f"'{header.get('GPUPHOTI', '')}', '{file_path}', ")

    keys = [
        "NAXIS1", "NAXIS2", "TELESCOP", "INSTRUME", "CAMERA", "FILTER",
        "DATE-OBS", "EXPTIME", "OBJECT", "RA", "DEC", "FWHM", "MAGLIM"
    ]

    for key in keys:
        value = header.get(key, 'NULL')
        if value == 'NULL' or value is None:
            query_parts.append("NULL, ")
        elif isinstance(value, (int, float)):
            query_parts.append(f"{value}, ")
        else:
            query_parts.append(f"'{str(value)}', ")

    hstore_fragment = headerToHstore(header)
    query_parts.append(f"'{hstore_fragment}'")

    query = ''.join(query_parts) + ") ON CONFLICT (id) DO UPDATE SET "

    update_parts = [f"{col} = EXCLUDED.{col}" for col in columns]
    query += ", ".join(update_parts)

    return query


def populate_ima_stats(gpuphotid, file_path, header, delete_prev=True):
    """
    Insert or update image statistics in the database.

    :param gpuphotid: Unique identifier for the image.
    :type gpuphotid: str
    :param file_path: Path of the image file.
    :type file_path: str
    :param header: FITS header of the image.
    :type header: astropy.io.fits.Header
    :param delete_prev: Whether to delete previous entries for this image.
    :type delete_prev: bool
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    try:
        connection_string = __generate_connection_string()
        engine = create_engine(connection_string)

        with engine.begin() as connection:
            if delete_prev:
                delete_query = text(f"DELETE FROM imastats WHERE id = '{gpuphotid}'")
                connection.execute(delete_query)

            q = insert_header(header, file_path)
            connection.execute(text(q))
        return True
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    return False


def generate_gpuphotid(process_file):
    """
    Generate a unique identifier for a processed file.

    :param process_file: Path of the processed file.
    :type process_file: str
    :return: Unique identifier.
    :rtype: str
    """
    h = hmac.new('GPUPHOT_KEY'.encode(), process_file.encode(), hashlib.sha1)
    return str(h.hexdigest())
