import os
from datetime import datetime
from glob import glob

import numpy as np
import pytz
from astropy.io import fits
from celery import shared_task

from gpuphot.image_processor import create_processor
from gpuphot.logger.hierarchical_logging import setup_logger
from gpuphot_worker.celery_exceptions import BaseTaskWithFailureHandling, SerializableTaskError

logger = setup_logger(__name__)


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


@shared_task
def process_directory_task(path=None, filename=None, instrument_name=None, overwrite=False):
    """
        Process astronomical images (FITS and NPY) with flexible search and configuration options.

        This task searches for images based on the given criteria and initiates individual
        processing tasks for each image found.

        Task Usage Examples:
        1. Process all images in default path:
           process_directory_task.delay()

        2. Process images in a specific directory:
           process_directory_task.delay(path='today')

        3. Process images matching a specific pattern:
           process_directory_task.delay(path='today/camera1', filename='image.fits')

        4. Process images containing a specific name pattern:
           process_directory_task.delay(filename='NEO')

        5. Process with custom instrument configuration:
           process_directory_task.delay(
               path='today',
               instrument_name='other_instrument'
           )

        6. Process and overwrite original files:
           process_directory_task.delay(path='today', overwrite=True)

        Args:
            path (str, optional): Subdirectory to search for images.
                                  If None, searches in base image path.
            filename (str, optional): Specific filename or pattern to match.
                                      Supports partial matches and wildcards.
            instrument_name (str, optional): Override default instrument name.
            overwrite (bool, optional): If True, overwrites original files.
                                        If False, creates new files with '_photometrized' suffix.
                                        Defaults to False.

        Returns:
            dict: A dictionary containing 'task_ids', a list of task IDs for the
                  individual image processing tasks that were initiated.
        """

    base_path = os.environ.get('IMAGE_BASE_PATH', '/app/images')

    if path:
        path = path.lstrip('/')
        search_path = os.path.join(base_path, path)
    else:
        search_path = base_path

    if filename:
        if '*' not in filename:
            filename = f'*{filename}*'
        search_pattern = os.path.join(search_path, '**', filename)
    else:
        search_pattern = os.path.join(search_path, '**', '*')

    image_files = glob(search_pattern, recursive=True)
    image_files = [f for f in image_files if f.endswith(('.fits', '.npy'))]

    results = []
    for file_path in image_files:
        relative_path = os.path.relpath(file_path, base_path)
        task = process_image_task.delay(relative_path, instrument_name, overwrite)
        results.append(task.id)

    return {'task_ids': results}


@shared_task(bind=True, base=BaseTaskWithFailureHandling)  # Usar bind=True para acceder a self
def process_image_task(self, image_path, instrument_name=None, overwrite=False):
    """
    Process a single astronomical image file (FITS or NPY).

    This task is designed to be called by process_directory_task for each individual image,
    but can also be used independently to process a single image file.

    Args:
        image_path (str): Path to the image file to be processed.
        instrument_name (str, optional): Name of the instrument to use for processing.
                                         If None, uses the default instrument.
        overwrite (bool, optional): If True, overwrites the original file.
                                    If False, creates a new file with '_photometrized' suffix.
                                    Defaults to False.

    Returns:
        dict: A dictionary containing the processing results or error information.
              If successful, the dictionary includes:
                - 'file': Relative path of the original file
                - 'output_file': Relative path of the processed file
                - 'dfm': Processed image data
                - 'header': Dictionary of the image header
              If an error occurs, the dictionary includes:
                - 'file': Relative path of the file that caused the error
                - 'error': Description of the error
                - 'status': 'failed'

    Raises:
        Exception: Raises an exception with error details if processing fails.

    Note:
        This task uses the instrument configuration specified by instrument_name
        to process the image. It can handle both FITS and NPY file formats.
    """
    base_path = os.environ.get('IMAGE_BASE_PATH', '/app/images')
    processor = get_processor(instrument_name)

    file_path = os.path.join(base_path, image_path)

    try:
        if file_path.endswith('.fits'):
            with fits.open(file_path) as hdul:
                imdata = hdul[0].data
                imheader = hdul[0].header
        elif file_path.endswith('.npy'):
            imdata = np.load(file_path)
            header_file = file_path.rsplit('.', 1)[0] + '.txt'
            if os.path.exists(header_file):
                with open(header_file, 'r') as f:
                    header_content = f.read()
                imheader = fits.Header.fromstring(header_content)
            else:
                raise ValueError(
                    f"Header file not found for NPY file: {file_path}, expected header file: {header_file}")
        else:
            raise ValueError(f"Unsupported file format: {file_path}. Only FITS and NPY files are supported.")

        phot_df, hwcs = processor.process_image(imdata, imheader)

        dateproc = datetime.now().replace(tzinfo=pytz.UTC)
        hwcs['DATEPROC'] = (dateproc.strftime('%Y-%m-%dT%H:%M:%S.%f'), 'Date and time of processing')

        if file_path.endswith('.fits'):
            if overwrite:
                output_path = file_path
            else:
                file_name, file_extension = os.path.splitext(file_path)
                output_path = f"{file_name}_photometrized{file_extension}"

            # Create reduced image fits
            photometrized_image = fits.PrimaryHDU(data=imdata.astype(np.float32), header=hwcs)
            photometrized_image.writeto(output_path, overwrite=overwrite)

        elif file_path.endswith('.npy'):
            if overwrite:
                output_path = file_path
                header_output_path = file_path.rsplit('.', 1)[0] + '.txt'
            else:
                file_name, file_extension = os.path.splitext(file_path)
                output_path = f"{file_name}_photometrized{file_extension}"
                header_output_path = f"{file_name}_photometrized.txt"

            # Save the processed numpy array
            np.save(output_path, imdata.astype(np.float32))

            # Save the updated header
            hwcs.totextfile(header_output_path, overwrite=True)

        else:
            raise ValueError(f"Unsupported file format: {file_path}. Only FITS and NPY files are supported.")

        return {
            'file': image_path,
            'output_file': os.path.relpath(output_path, base_path),
            'phot_df': phot_df,
            'hwcs': dict(hwcs)
        }
    except Exception as e:
        error_message = f"Error processing file {image_path}: {str(e)}"
        self.update_state(
            state="FAILURE",
            meta={
                "exc_type": e.__class__.__name__,
                "error_message": error_message,
            },
        )
        raise SerializableTaskError(error_message, exc_type=e.__class__.__name__)
