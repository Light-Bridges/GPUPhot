import os
import re
from datetime import datetime
from glob import glob

import numpy as np
import pytz
from celery import shared_task

import gpuphot.utils.gpu
from gpuphot.instrument_config_parser import ImageReduction
from gpuphot.logger.hierarchical_logging import setup_logger
from gpuphot_worker.celery_exceptions import SerializableTaskError, BaseTaskWithFailureHandling
from gpuphot_worker.utils import get_processor, open_image_file, save_processed_image, crop_and_bin_image, \
    insert_dataframe_to_postgres

logger = setup_logger(__name__)


def task_error_handler(task, e, image_path):
    error_message = f"Error processing file {image_path}: {str(e)}"
    logger.error(error_message)

    task.update_state(
        state="FAILURE",
        meta={
            "exc_type": e.__class__.__name__,
            "error_message": error_message,
        },
    )
    raise SerializableTaskError(error_message, exc_type=e.__class__.__name__)


@shared_task
def process_directory_task(path=None, filename=None, instrument_name=None, overwrite=False, exclude_pattern=None):
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
        exclude_pattern (str, optional): Regular expression pattern to exclude certain filenames.

    Returns:
        dict: A dictionary containing 'task_ids', a list of task IDs for the
              individual image processing tasks that were initiated.
    """

    logger.info(f"Processing directory: {path}, filename: {filename}, instrument: {instrument_name}")

    base_path = '/data/images'

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

    # Buscar archivos usando glob
    image_files = glob(search_pattern, recursive=True)
    image_files = [f for f in image_files if f.endswith(('.fits', '.npy'))]

    # Aplicar filtro de exclusión usando expresiones regulares
    if exclude_pattern:
        regex = re.compile(exclude_pattern)
        image_files = [f for f in image_files if not regex.search(os.path.basename(f))]

    logger.info(f"Found {len(image_files)} images matching the search criteria")
    results = []
    for file_path in image_files:
        relative_path = os.path.relpath(file_path, base_path)
        task = process_image_task.delay(relative_path, instrument_name, overwrite)
        results.append(task.id)

    return {'task_ids': results}


@shared_task(bind=True, base=BaseTaskWithFailureHandling)
def process_image_task(self, image_path, instrument_name=None, overwrite=False):
    """
    Process a single astronomical image file (FITS or NPY).

    This task is designed to be called by process_directory_task for each individual image,
    but can also be used independently to process a single image file.

    Key Features:
    - Supports image processing with configurable reduction
    - Handles different image reduction strategies
    - Generates processing metadata

    Args:
        image_path (str): Path to the image file to be processed.
        instrument_name (str, optional): Name of the instrument for processing.
                                         If None, uses the default instrument.
        overwrite (bool, optional):
            - True: Overwrites the original file
            - False: Creates a new file with '_photometrized' suffix
            Default: False

    Returns:
        dict: Processing results with the following keys:
            On successful processing:
            - 'input_file': Relative path of the original file
            - 'process_file': Relative path of the processed file
            - 'output_file': Output path of the processed file
            - 'phot_df': Photometric data DataFrame
            - 'hwcs': WCS header dictionary

            On error:
            - 'input_file': Path of the file that caused the error
            - 'error': Error description
            - 'status': 'failed'

    Raises:
        SerializableTaskError: Serializable exception with error details
        MemoryError: If memory issues occur during processing

    Reduction Strategies:
    - 'never': Default behavior, no reduction applied
    - 'always': Apply reduction always
    - 'on_failure': Apply reduction only if initial processing fails by memory
    """
    logger.info(f"Processing image: {image_path} with instrument: {instrument_name}")
    base_path = '/data/images'
    processor = get_processor(instrument_name)

    reduction_config = processor.config['image_reduction']
    apply_reduction = reduction_config['apply_reduction']

    def call_process_image(file_path_call):
        # Intenta procesar la imagen normalmente
        imdata, imheader = open_image_file(file_path_call)
        phot_df, hwcs = processor.process_image(imdata, imheader)

        # Añade la fecha de procesamiento al encabezado
        dateproc = datetime.now().replace(tzinfo=pytz.UTC)
        hwcs['DATEPROC'] = (dateproc.strftime('%Y-%m-%dT%H:%M:%S.%f'), 'Date and time of processing')

        # Guarda la imagen procesada
        output_path = save_processed_image(file_path_call, imdata, hwcs, overwrite)

        # Store photometry results in PostgreSQL
        process_file = os.path.relpath(file_path_call, base_path)

        # Create a new DataFrame with necessary transformations
        df_imaphot = (
            phot_df
            .assign(trans=lambda x: np.isnan(x['RAERR']))  # Set 'trans' based on RAERR
            .rename(columns={'RA': 'ra', 'DEC': 'dec', 'noise': 'dflux'})  # Rename columns
        )

        # Add imageid and select relevant columns
        df_imaphot['imageid'] = str(process_file)
        df_imaphot = df_imaphot[['imageid', 'ra', 'dec', 'flux', 'dflux', 'trans']]

        # Insert into PostgreSQL and capture result
        result_postgress = insert_dataframe_to_postgres(df_imaphot)

        return {
            'input_file': image_path,
            'process_file': process_file,
            'output_file': os.path.relpath(output_path, base_path),
            'imaphot': {
                'objets': len(phot_df.index) if phot_df is not None else 0,
                'transients': len(phot_df[phot_df.trans == True].index) if phot_df is not None else 0,
                'saved': result_postgress
            },
            'hwcs': dict(hwcs)
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
        # error_message = f"Error processing file {image_path}: {str(e)}"
        # self.update_state(
        #     state="FAILURE",
        #     meta={
        #         "exc_type": e.__class__.__name__,
        #         "error_message": error_message,
        #     },
        # )
        # raise SerializableTaskError(error_message, exc_type=e.__class__.__name__)
