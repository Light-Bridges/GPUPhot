import os
import re
from datetime import datetime
from glob import glob

import numpy as np
import pytz
from celery import shared_task

from gpuphot.instrument_config_parser import ImageReduction
from gpuphot.logger.hierarchical_logging import setup_logger
from gpuphot.utils.gpu import reset_cupy_allocators
from gpuphot_worker.celery_exceptions import SerializableTaskError, BaseTaskWithFailureHandling
from gpuphot_worker.database_insert_utils import insert_dataframe_to_postgres, populate_ima_stats, generate_gpuphotid
from gpuphot_worker.header_descriptions import HEADER_DESCRIPTIONS
from gpuphot_worker.utils import get_processor, open_image_file, save_processed_image, crop_and_bin_image, \
    BASE_IMAGES_PATH, PROCESSED_IMAGE_FOLDER

logger = setup_logger(__name__)


def task_error_handler(task, e, image_path):
    """
    Handles errors that occur during the processing of a task.

    This function logs the error details, updates the task state to "FAILURE",
    and raises a SerializableTaskError with relevant information.

    :param task: The task instance that encountered an error.
    :type task: celery.Task
    :param e: The exception that was raised during the task execution.
    :type e: Exception
    :param image_path: The path of the image file that caused the error.
    :type image_path: str
    :raises SerializableTaskError: An error that includes the original exception type
                                    and a message describing the error.
    """
    error_message = f"Error processing file {image_path}: {str(e)}"
    logger.error(error_message)

    # Check if the task has a valid task_id before updating its state
    if task and hasattr(task, 'id') and task.id is not None:
        try:
            # Capture the full traceback
            import traceback
            tb = traceback.format_exc()

            # Update the task state with more detailed information
            task.update_state(
                state="FAILURE",
                meta={
                    "exc_type": e.__class__.__name__,
                    "error_message": error_message,
                    "traceback": tb,  # Include the full traceback
                    "additional_info": {
                        "image_path": image_path,
                        "original_exception": str(e),
                        "cause": str(e.__cause__) if e.__cause__ else None,
                        "context": str(e.__context__) if e.__context__ else None,
                    }
                },
            )
        except Exception as update_error:
            logger.error(f"Failed to update task state: {update_error}")

    raise SerializableTaskError(error_message, exc_type=e.__class__.__name__)


@shared_task
def process_directory_task(path=None, filename=None, instrument_name=None, exclude_pattern=None, reprocess=False):
    """
    Processes astronomical images (FITS and NPY) with customizable search and configuration options.

    This task searches for images based on specified criteria and initiates individual
    processing tasks for each image found.

    :param path: Subdirectory to search for images. If None, searches in the base image path.
    :type path: str, optional
    :param filename: Specific filename or pattern to match. Supports partial matches and wildcards.
    :type filename: str, optional
    :param instrument_name: Overrides the default instrument name.
    :type instrument_name: str, optional
    :param exclude_pattern: Regular expression pattern to exclude certain filenames.
    :type exclude_pattern: str, optional
    :param reprocess: If True, processes all found images; if False, skips images already processed.
                      Default is .
    :type reprocess: bool, optional
    :return: A dictionary containing 'task_ids', which is a list of task IDs for the individual image processing tasks initiated.
    :rtype: dict

    Examples
    --------
    1. Process all images in the default path:
       >>> process_directory_task.delay()

    2. Process images in a specific directory:
       >>> process_directory_task.delay(path='today')

    3. Process images matching a specific filename:
       >>> process_directory_task.delay(path='today/camera1', filename='image.fits')

    4. Process images containing a specific name pattern:
       >>> process_directory_task.delay(filename='NEO')

    5. Process with a custom instrument configuration:
       >>> process_directory_task.delay(path='today', instrument_name='other_instrument')

    6. Process without reprocessing already processed files:
       >>> process_directory_task.delay(path='today', reprocess=False)
    """

    base_path = BASE_IMAGES_PATH

    if path:
        path = path.lstrip('/')
        search_path = os.path.join(base_path, path)
    else:
        search_path = base_path

    logger.info(
        f"Processing directory: {path}, filename: {filename}, instrument: {instrument_name}, reprocess: {reprocess}")

    if filename:
        if '*' not in filename:
            filename = f'*{filename}*'
        search_pattern = os.path.join(search_path, '**', filename)
    else:
        search_pattern = os.path.join(search_path, '**', '*')

    # Search for files using glob
    image_files = glob(search_pattern, recursive=True)
    processed_dir = os.path.normpath(os.path.join(base_path, PROCESSED_IMAGE_FOLDER))

    # Filter out processed files from the list
    image_files = [f for f in image_files if not f.startswith(processed_dir)]

    # Filter out files that are already processed if reprocess is False
    if not reprocess:
        image_files = [f for f in image_files if
                       not os.path.exists(os.path.join(processed_dir, os.path.relpath(f, base_path)))]

    # Further filter to only include FITS and NPY files
    image_files = [f for f in image_files if f.endswith(('.fits', '.npy'))]

    # Apply exclusion filter using regular expressions
    if exclude_pattern:
        regex = re.compile(exclude_pattern)
        image_files = [f for f in image_files if not regex.search(os.path.basename(f))]

    logger.info(f"Found {len(image_files)} images matching the search criteria")

    results = {}
    for file_path in image_files:
        relative_path = os.path.relpath(file_path, base_path)

        # Determine the processed file path by changing the extension to .fits
        processed_file_path = os.path.join(processed_dir, os.path.splitext(relative_path)[0] + '.fits')

        # Check if the file exists in the processed directory
        if not reprocess or not os.path.exists(processed_file_path):
            task = process_image_task.delay(relative_path, instrument_name)
            results[relative_path] = task.id

    return results


@shared_task(bind=True, base=BaseTaskWithFailureHandling)
def process_image_task(self, image_path, instrument_name=None):
    """
    Processes a single astronomical image file (FITS or NPY).

    This task is designed to be called by `process_directory_task` for each individual image,
    but it can also be used independently to process a single image file.

    Key Features:
    - Supports configurable image processing and reduction strategies.
    - Handles different image reduction methods.
    - Generates and stores processing metadata.

    :param image_path: The path to the image file to be processed.
    :type image_path: str
    :param instrument_name: The name of the instrument for processing. If None, the default instrument is used.
    :type instrument_name: str or None

    :return: Processing results containing various keys depending on success or failure.
    :rtype: dict

    :raises SerializableTaskError: A serializable exception with error details.
    :raises MemoryError: If memory issues occur during processing.

    On successful processing, the returned dictionary contains:

    - input_file (str): Relative path of the original file.
    - process_file (str): Relative path of the processed file.
    - output_file (str): Output path of the processed file.
    - imaphot (dict): A dictionary with photometric data:
        - stored (bool): Indicates if photometric data is stored in PostgreSQL.
        - objets (int): Number of objects detected.
        - transients (int): Number of transient objects detected.
    - imastats (dict): A dictionary with image statistics:
        - stored (bool): Indicates if image statistics are stored in PostgreSQL.

    On error, the returned dictionary contains:

    - input_file (str): Path of the file that caused the error.
    - error (str): Error description.
    - status (str): 'failed'.

    Reduction Strategies:
    - 'never': Default behavior; no reduction applied.
    - 'always': Apply reduction unconditionally.
    - 'on_failure': Apply reduction only if initial processing fails due to memory issues.

    Example return value on success::

        {
            "input_file": "path/to/image.fits",
            "process_file": "processed/path/to/image.fits",
            "output_file": "/data/images/processed/image.fits",
            "imaphot": {
                "stored": True,
                "objets": 100,
                "transients": 5,
            },
            "imastats": {
                "stored": True,
            }
        }

    Example return value on failure::

        {
            "input_file": "path/to/image.fits",
            "error": "MemoryError during processing",
            "status": "failed"
        }
    """

    base_path = BASE_IMAGES_PATH
    if image_path:
        image_path = image_path.lstrip('/')

    logger.info(f"Processing image: {image_path} with instrument: {instrument_name}")

    processor = get_processor(instrument_name)

    reduction_config = processor.config['image_reduction']
    apply_reduction = reduction_config['apply_reduction']

    def call_process_image(file_path_call):
        logger.debug(f"Processing file: {file_path_call}")
        reset_cupy_allocators()

        try:
            # Attempt to process the image normally
            imdata, imheader = open_image_file(file_path_call)
            phot_df, hwcs = processor.process_image(imdata, imheader, header_descriptions=HEADER_DESCRIPTIONS)

            process_file = os.path.relpath(file_path_call, base_path)
            gpuphotid = str(generate_gpuphotid(str(process_file)))
            hwcs['GPUPHOTI'] = (gpuphotid, HEADER_DESCRIPTIONS['GPUPHOTI'])

            # Add processing date to header
            dateproc = datetime.now().replace(tzinfo=pytz.UTC)
            hwcs['DATEPROC'] = (dateproc.strftime('%Y-%m-%dT%H:%M:%S.%f'), HEADER_DESCRIPTIONS['DATEPROC'])

            # Save the processed image
            output_path = save_processed_image(file_path_call, base_path, imdata, hwcs)

            # Create a new DataFrame with necessary transformations for photometry
            df_imaphot = (
                phot_df
                .assign(trans=lambda x: np.isnan(x['RAERR']))  # Set 'trans' based on RAERR
                .rename(columns={'RA': 'ra', 'DEC': 'dec', 'noise': 'dflux'})  # Rename columns
            )

            # Add image ID and select relevant columns
            df_imaphot['id'] = gpuphotid
            df_imaphot = df_imaphot[['id', 'ra', 'dec', 'flux', 'dflux', 'trans']]

            # Insert into PostgreSQL and capture results
            result_imastats = populate_ima_stats(gpuphotid, str(os.path.relpath(output_path, base_path)), hwcs, True)
            result_imaphot = insert_dataframe_to_postgres(df_imaphot)
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise
        # finally:
        #     reset_cupy_allocators()

        return {
            'input_file': image_path,
            'process_file': process_file,
            'output_file': os.path.relpath(output_path, base_path),
            'imaphot': {
                'objets': len(df_imaphot.index) if df_imaphot is not None else 0,
                'transients': len(df_imaphot[df_imaphot.trans == True].index) if df_imaphot is not None else 0,
                'stored': result_imaphot
            },
            'imastats': {
                'stored': result_imastats
            }
        }

    file_path = os.path.join(base_path, image_path)
    binning_config = reduction_config.get('binning', 1)
    if isinstance(binning_config, int):
        factor = binning_config
        method = 'sum'
    else:
        factor = binning_config['factor']
        method = binning_config.get('method', 'sum')

    try:

        if apply_reduction == ImageReduction.ALWAYS.value:
            logger.info(f"Applying image reduction to: {image_path}")
            result = call_process_image(
                file_path_call=crop_and_bin_image(
                    fits_file=file_path,
                    binning=factor,
                    binning_method=method,
                    crop_size=reduction_config['crop_size'],
                    center=reduction_config['center']
                )
            )

        else:
            result = call_process_image(file_path)

        reset_cupy_allocators()
        return result
    except MemoryError as e:
        reset_cupy_allocators()
        if apply_reduction == ImageReduction.ON_FAILURE.value:
            try:
                logger.warning(f"Memory error processing image: {image_path}")
                logger.info(f"Applying image reduction to: {image_path}")
                return call_process_image(
                    file_path_call=crop_and_bin_image(
                        fits_file=file_path,
                        binning=factor,
                        binning_method=method,
                        crop_size=reduction_config['crop_size'],
                        center=reduction_config['center']
                    )
                )
            except Exception as e:
                task_error_handler(self, e, image_path)

        else:
            task_error_handler(self, e, image_path)
    except Exception as e:
        task_error_handler(self, e, image_path)
