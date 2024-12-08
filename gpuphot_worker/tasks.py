import os
from glob import glob

import numpy as np
from astropy.io import fits
from celery import shared_task
from celery.exceptions import Ignore

from gpuphot.image_processor import create_processor
from gpuphot.logger.hierarchical_logging import setup_logger, hierarchical_debug

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


@hierarchical_debug(logger)
@shared_task
def process_directory_task(path=None, filename=None, instrument_name=None):
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

    Args:
        path (str, optional): Subdirectory to search for images.
                               If None, searches in base image path.
        filename (str, optional): Specific filename or pattern to match.
                                  Supports partial matches and wildcards.
        instrument_name (str, optional): Override default instrument name.

    Returns:
        dict: A dictionary containing 'task_ids', a list of task IDs for the
              individual image processing tasks that were initiated.
    """
    base_path = os.environ.get('IMAGE_BASE_PATH', '/app/images')

    if path:
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
        task = process_image_task.delay(relative_path, instrument_name)
        results.append(task.id)

    return {'task_ids': results}


@hierarchical_debug(logger)
@shared_task(bind=True)  # Usar bind=True para acceder a self
def process_image_task(self, image_path, instrument_name=None):
    """
    Process a single astronomical image file (FITS or NPY).

    This task is designed to be called by process_directory_task for each individual image,
    but can also be used independently to process a single image file.

    Args:
        image_path (str): Path to the image file to be processed.
        instrument_name (str, optional): Name of the instrument to use for processing.
                                         If None, uses the default instrument.

    Returns:
        dict: A dictionary containing the processing results or error information.
              If successful, the dictionary includes:
                - 'file': Relative path of the processed file
                - 'dfm': Processed image data
                - 'header': Dictionary of the image header
              If an error occurs, the dictionary includes:
                - 'file': Relative path of the file that caused the error
                - 'error': Description of the error

    Raises:
        No exceptions are raised as they are caught and returned in the result dictionary.

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
                header_descriptions = {k: v for k, v in hdul[0].header.cards}
        elif file_path.endswith('.npy'):
            imdata = np.load(file_path)
            header_file = file_path.rsplit('.', 1)[0] + '.txt'
            if os.path.exists(header_file):
                with open(header_file, 'r') as f:
                    header_content = f.read()
                imheader = fits.Header.fromstring(header_content)
                header_descriptions = {k: v for k, v in imheader.cards}
            else:
                imheader = fits.Header()
                header_descriptions = {}

        dfm, original_header = processor.process_image(imdata, imheader, header_descriptions)
        return {
            'file': image_path,
            'dfm': dfm,
            'header': dict(original_header)
        }
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")

        result = {
            'file': image_path,
            'error': str(e),
            'status': 'failed'
        }

        self.update_state(state='FAILURE', meta=result)

        raise Ignore()
