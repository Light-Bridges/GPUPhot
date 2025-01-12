import os
import re
from datetime import datetime, timedelta
from glob import glob

import numpy as np
import pytz
from celery import shared_task, current_app as app

import gpuphot.utils.gpu
import gpuphot.utils.gpu
from gpuphot.instrument_config_parser import ImageReduction
from gpuphot.logger.hierarchical_logging import setup_logger
from gpuphot_worker.celery_exceptions import SerializableTaskError, BaseTaskWithFailureHandling
from gpuphot_worker.utils import get_processor, open_image_file, save_processed_image, crop_and_bin_image, \
    insert_dataframe_to_postgres, populate_ima_stats, generate_gpuphotid, BASE_IMAGES_PATH, PROCESSED_IMAGE_FOLDER

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
def process_directory_task(path=None, filename=None, instrument_name=None, exclude_pattern=None):
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

        6. Process images in a specific directory:
           process_directory_task.delay(path='today')

    Args:
        path (str, optional): Subdirectory to search for images.
                              If None, searches in base image path.
        filename (str, optional): Specific filename or pattern to match.
                                  Supports partial matches and wildcards.
        instrument_name (str, optional): Override default instrument name.
        exclude_pattern (str, optional): Regular expression pattern to exclude certain filenames.

    Returns:
        dict: A dictionary containing 'task_ids', a list of task IDs for the
              individual image processing tasks that were initiated.
    """

    logger.info(f"Processing directory: {path}, filename: {filename}, instrument: {instrument_name}")

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

    # Buscar archivos usando glob
    image_files = glob(search_pattern, recursive=True)
    processed_dir = os.path.normpath(os.path.join(base_path, PROCESSED_IMAGE_FOLDER))
    image_files = [f for f in image_files if
                   f.endswith(('.fits', '.npy')) and not os.path.normpath(f).startswith(processed_dir)]

    # Aplicar filtro de exclusión usando expresiones regulares
    if exclude_pattern:
        regex = re.compile(exclude_pattern)
        image_files = [f for f in image_files if not regex.search(os.path.basename(f))]

    logger.info(f"Found {len(image_files)} images matching the search criteria")

    results = {}
    for file_path in image_files:
        relative_path = os.path.relpath(file_path, base_path)
        task = process_image_task.delay(relative_path, instrument_name)
        results[relative_path] = task.id

    return results


@shared_task(bind=True, base=BaseTaskWithFailureHandling)
def process_image_task(self, image_path, instrument_name=None):
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

    Returns:
        dict: Processing results with the following keys:
            On successful processing:
            - 'input_file': Relative path of the original file
            - 'process_file': Relative path of the processed file
            - 'output_file': Output path of the processed file
            - 'imaphot.stored': If Photometric data stored in PostgreSQL
            - 'imaphot.objets': Number of objects detected
            - 'imaphot.transients': Number of transient objects detected
            - 'imastats.stored': If Image statistics stored in PostgreSQL

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
    base_path = BASE_IMAGES_PATH
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

        process_file = os.path.relpath(file_path_call, base_path)
        gpuphotid = str(generate_gpuphotid(process_file))
        hwcs['GPUPHOTI'] = (gpuphotid, 'Unique identifier for GPUPhot processing')

        # Guarda la imagen procesada
        output_path = save_processed_image(file_path_call, base_path, imdata, hwcs)

        # Store photometry results in PostgreSQL

        # Create a new DataFrame with necessary transformations
        df_imaphot = (
            phot_df
            .assign(trans=lambda x: np.isnan(x['RAERR']))  # Set 'trans' based on RAERR
            .rename(columns={'RA': 'ra', 'DEC': 'dec', 'noise': 'dflux'})  # Rename columns
        )

        # Add imageid and select relevant columns
        df_imaphot['id'] = gpuphotid
        df_imaphot = df_imaphot[['id', 'ra', 'dec', 'flux', 'dflux', 'trans']]

        # Insert into PostgreSQL and capture result
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

#
# # Tarea dummy que se ejecutará cada minuto
# @shared_task
# def dummy_task(message):
#     print(message)
#
#
# @shared_task
# def remove_task(message):
#     print(message)
#
#
# @shared_task
# def update_tasks():
#     # Obtén la hora actual en UTC
#     now = datetime.now(tz=pytz.utc)
#
#     # Calcula el tiempo objetivo para ejecutar la tarea dentro de un minuto
#     target_time = now + timedelta(minutes=1, seconds=5)
#     target_time_remove = now + timedelta(minutes=2, seconds=5)
#
#     # Añade una nueva tarea dummy que se ejecutará a la hora específica
#     scheduled_tasks = app.control.inspect().scheduled()
#     task_number = sum(len(task_list) for task_list in scheduled_tasks.values()) if scheduled_tasks else 0
#
#     # Formatea el tiempo objetivo como una cadena legible
#     formatted_time = target_time.strftime('%Y-%m-%dT%H:%M:%S.%f')
#     formatted_time_remove = target_time_remove.strftime('%Y-%m-%dT%H:%M:%S.%f')
#
#     # Ejecuta la tarea dummy a la hora específica usando eta
#     dummy_task.apply_async(args=[f'Soy tarea dummy {task_number} y me ejecuto a las {formatted_time}'], eta=target_time)
#     remove_task.apply_async(args=[f'Soy tarea remove {task_number} y me ejecuto a las {formatted_time_remove}'],
#                             eta=target_time_remove)
#
#     print(f"Tarea dummy añadida para ejecución a las {formatted_time} con argumento 'Soy tarea {task_number}'")
#     print(f"Tarea remove añadida para ejecución a las {formatted_time_remove} con argumento 'Soy tarea {task_number}'")
#
#     if scheduled_tasks:
#         print("Tareas programadas:")
#         for worker, task_list in scheduled_tasks.items():
#             print(f"Worker: {worker} - {len(task_list)} tareas programadas")
#             for task_info in task_list:
#                 print(
#                     f"  - Tarea: {task_info['request']['name']}, ETA: {task_info['eta']}, Prioridad: {task_info['priority']}")
#
#                 # Revoca la tarea si es una tarea 'remove'
#                 if 'remove' in task_info['request']['name']:
#                     # Usa el nombre de la tarea y los argumentos para revocar
#                     app.control.revoke(task_info['request']['id'],
#                                        terminate=True)  # Cambia esto según cómo obtengas el ID si es necesario
#                     print(f"Tarea revocada: {task_info['request']['name']}")
#
#     else:
#         print("No hay tareas programadas.")
