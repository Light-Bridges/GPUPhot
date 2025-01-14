import os
import re
from datetime import datetime
from glob import glob

import numpy as np
import pytz
from celery import shared_task

import gpuphot.utils.gpu
import gpuphot.utils.gpu
from gpuphot.instrument_config_parser import ImageReduction
from gpuphot.logger.hierarchical_logging import setup_logger
from gpuphot_worker.celery_exceptions import SerializableTaskError, BaseTaskWithFailureHandling
from gpuphot_worker.utils import get_processor, open_image_file, save_processed_image, crop_and_bin_image, \
    insert_dataframe_to_postgres, populate_ima_stats, generate_gpuphotid, BASE_IMAGES_PATH, PROCESSED_IMAGE_FOLDER

logger = setup_logger(__name__)


def task_error_handler(task, e, image_path):
    """
    Handles errors that occur during the processing of a task.

    This function logs the error details, updates the task state to "FAILURE",
    and raises a SerializableTaskError with relevant information.

    Args:
        task (Task): The task instance that encountered an error.
        e (Exception): The exception that was raised during the task execution.
        image_path (str): The path of the image file that caused the error.

    Raises:
        SerializableTaskError: An error that includes the original exception type
                                and a message describing the error.
    """
    error_message = f"Error processing file {image_path}: {str(e)}"
    logger.error(error_message)

    # Check if the task has a valid task_id before updating its state
    if task and hasattr(task, 'id') and task.id is not None:
        try:
            task.update_state(
                state="FAILURE",
                meta={
                    "exc_type": e.__class__.__name__,
                    "error_message": error_message,
                },
            )
        except Exception as update_error:
            logger.error(f"Failed to update task state: {update_error}")

    raise SerializableTaskError(error_message, exc_type=e.__class__.__name__)


@shared_task
def process_directory_task(path=None, filename=None, instrument_name=None, exclude_pattern=None, reprocess=True):
    """
    Processes astronomical images (FITS and NPY) with customizable search and configuration options.

    This task searches for images based on specified criteria and initiates individual
    processing tasks for each image found.

    Task Usage Examples:
        1. Process all images in the default path:
           process_directory_task.delay()

        2. Process images in a specific directory:
           process_directory_task.delay(path='today')

        3. Process images matching a specific filename:
           process_directory_task.delay(path='today/camera1', filename='image.fits')

        4. Process images containing a specific name pattern:
           process_directory_task.delay(filename='NEO')

        5. Process with a custom instrument configuration:
           process_directory_task.delay(
               path='today',
               instrument_name='other_instrument'
           )

        6. Process without reprocesing processed files:
           process_directory_task.delay(path='today', reprocess=False)

    Args:
        path (str, optional): Subdirectory to search for images.
                              If None, searches in the base image path.
        filename (str, optional): Specific filename or pattern to match.
                                  Supports partial matches and wildcards.
        instrument_name (str, optional): Overrides the default instrument name.
        exclude_pattern (str, optional): Regular expression pattern to exclude certain filenames.
        reprocess (bool, optional): If True, processes all found images; if False,
                                    skips images already processed.

    Returns:
        dict: A dictionary containing 'task_ids', which is a list of task IDs for the
              individual image processing tasks that were initiated.
    """

    logger.info(
        f"Processing directory: {path}, filename: {filename}, instrument: {instrument_name}, reprocess: {reprocess}")

    base_path = BASE_IMAGES_PATH

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

    # Search for files using glob
    image_files = glob(search_pattern, recursive=True)
    processed_dir = os.path.normpath(os.path.join(base_path, PROCESSED_IMAGE_FOLDER))

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

    Args:
        image_path (str): The path to the image file to be processed.
        instrument_name (str, optional): The name of the instrument for processing.
                                          If None, the default instrument is used.

    Returns:
        dict: Processing results containing the following keys:
            On successful processing:
            - 'input_file': Relative path of the original file.
            - 'process_file': Relative path of the processed file.
            - 'output_file': Output path of the processed file.
            - 'imaphot': A dictionary with photometric data:
                - 'stored': Indicates if photometric data is stored in PostgreSQL.
                - 'objets': Number of objects detected.
                - 'transients': Number of transient objects detected.
            - 'imastats': A dictionary with image statistics:
                - 'stored': Indicates if image statistics are stored in PostgreSQL.

            On error:
            - 'input_file': Path of the file that caused the error.
            - 'error': Error description.
            - 'status': 'failed'.

    Raises:
        SerializableTaskError: A serializable exception with error details.
        MemoryError: If memory issues occur during processing.

    Reduction Strategies:
    - 'never': Default behavior; no reduction applied.
    - 'always': Apply reduction unconditionally.
    - 'on_failure': Apply reduction only if initial processing fails due to memory issues.
    """

    logger.info(f"Processing image: {image_path} with instrument: {instrument_name}")
    base_path = BASE_IMAGES_PATH
    processor = get_processor(instrument_name)

    reduction_config = processor.config['image_reduction']
    apply_reduction = reduction_config['apply_reduction']

    def call_process_image(file_path_call):
        # Attempt to process the image normally
        imdata, imheader = open_image_file(file_path_call)
        phot_df, hwcs = processor.process_image(imdata, imheader)

        # Add processing date to header
        dateproc = datetime.now().replace(tzinfo=pytz.UTC)
        hwcs['DATEPROC'] = (dateproc.strftime('%Y-%m-%dT%H:%M:%S.%f'), 'Date and time of processing')

        process_file = os.path.relpath(file_path_call, base_path)
        gpuphotid = str(generate_gpuphotid(process_file))
        hwcs['GPUPHOTI'] = (gpuphotid, 'Unique identifier for GPUPhot processing')

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
        result_imastats = populate_ima_stats(gpuphotid, str(output_path), hwcs, 'imastats')
        result_imaphot = insert_dataframe_to_postgres(df_imaphot, 'imaphot')

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

    try:
        file_path = os.path.join(base_path, image_path)

        if apply_reduction == ImageReduction.ALWAYS.value:
            logger.info(f"Applying image reduction to: {image_path}")
            return call_process_image(
                file_path_call=crop_and_bin_image(file_path, reduction_config['binning'],
                                                  reduction_config['crop_size'], reduction_config['center'])
            )

        return call_process_image(file_path)

    except MemoryError as e:
        gpuphot.utils.gpu.free_gpu_mem()
        if apply_reduction == ImageReduction.ON_FAILURE.value:
            try:
                logger.warning(f"Memory error processing image: {image_path}")
                logger.info(f"Applying image reduction to: {image_path}")
                return call_process_image(
                    file_path_call=crop_and_bin_image(image_path, reduction_config['binning'],
                                                      reduction_config['crop_size'], reduction_config['center'])
                )
            except Exception as e:
                task_error_handler(self, e, image_path)

        else:
            task_error_handler(self, e, image_path)
    except Exception as e:
        task_error_handler(self, e, image_path)
