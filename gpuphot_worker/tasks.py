# SPDX-License-Identifier: MIT
"""
Celery tasks provided by gpuphot_worker.

This module contains the Celery tasks used to process astronomical images
(found under `BASE_IMAGES_PATH`) and persist results. The main tasks are:

- process_directory_task: search a directory (with optional filters) and queue
  processing tasks for every image found.
- process_image_task: process a single image file and store photometry and
  image statistics in PostgreSQL.

The module attempts to use GPU-accelerated arrays when available (cupynumeric).
All docstrings are written in English and aim to describe side effects and
returned metadata for remote callers.
"""

import os
import re
from datetime import datetime
from glob import glob

try:
    import cupynumeric as np
except ImportError:
    import numpy as np
except Exception:
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
    and raises a :class:`SerializableTaskError` with relevant information.

    Parameters
    ----------
    task : celery.Task
        The task instance that encountered an error.
    e : Exception
        The exception that was raised during the task execution.
    image_path : str
        The path of the image file that caused the error.

    Raises
    ------
    SerializableTaskError
        Encapsulating the original exception type and a descriptive message.
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
def probe_cuml():
    """Return a human-readable string describing cuML availability and cuML config."""
    import os
    lines = []
    try:
        import cuml
        import cupy as cp
        n_gpus = cp.cuda.runtime.getDeviceCount()
        lines.append(f"cuML {cuml.__version__} available — {n_gpus} GPU(s) detected.")
    except ImportError:
        lines.append("cuML not installed.")
    except Exception as e:
        lines.append(f"cuML installed but could not initialise: {type(e).__name__}: {e}")

    use_cuml = os.getenv("GPUPHOT_USE_CUML_CROSSMATCH", "0")
    min_src  = os.getenv("GPUPHOT_CUML_MIN_SOURCES", "0")
    max_src  = os.getenv("GPUPHOT_CUML_MAX_SOURCES", "0")
    if use_cuml == "1":
        lines.append("Mode: cuML ALWAYS ON (GPUPHOT_USE_CUML_CROSSMATCH=1)")
    elif min_src != "0" and max_src != "0":
        lines.append(f"Mode: ADAPTIVE — cuML active for {min_src}–{max_src} sources")
    else:
        lines.append("Mode: DISABLED — cKDTree used for all crossmatches")
    return "\n".join(lines)


@shared_task
def process_directory_task(path=None, filename=None, instrument_name=None, exclude_pattern=None, reprocess=False):
    """
    Orchestrator task that scans a directory and queues image processing tasks.

    This task acts as a generator, searching for FITS/NPY files in the specified
    directory (relative to `BASE_IMAGES_PATH`) and dispatching a `process_image_task`
    for each valid file found.

    It supports filtering by filename patterns (glob) and exclusion regexes.
    It also checks if the output file already exists to avoid redundant processing
    (unless `reprocess=True`).

    Parameters
    ----------
    path : str, optional
        Subdirectory to search for images (e.g., '2023-10-25'). If None, searches root.
    filename : str, optional
        Glob pattern to match filenames (e.g., '*.fits', 'target_A*').
    instrument_name : str, optional
        Name of the instrument configuration to use (e.g., 'telescope_A').
        Passed down to `process_image_task`.
    exclude_pattern : str, optional
        Regex pattern to exclude specific filenames (e.g., '.*bias.*').
    reprocess : bool, optional
        If True, forces reprocessing of images even if the output file exists.
        Default is False.

    Returns
    -------
    dict
        A dictionary mapping relative input file paths to their corresponding
        Celery task IDs. Example::

            {
                '2023/image1.fits': 'task-uuid-1',
                '2023/image2.fits': 'task-uuid-2'
            }
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
    Core task to process a single astronomical image.

    This task executes the full photometry pipeline for a given image file.
    It handles:

    1. Loading the image and instrument configuration.
    2. Executing the GPU-accelerated processing (`processor.process_image`).
    3. Persisting results to PostgreSQL (photometry and stats).
    4. Saving the processed FITS file with updated headers.
    5. **Automatic Fallback**: If a `MemoryError` occurs (OOM), it can automatically
       retry processing with a reduced version of the image (binned/cropped),
       depending on the instrument configuration (`image_reduction`).

    Parameters
    ----------
    image_path : str
        Relative path to the image file (from `BASE_IMAGES_PATH`).
    instrument_name : str, optional
        Name of the instrument configuration to use.

    Returns
    -------
    dict
        A summary of the processing result, including:

        - 'input_file': Original file path.
        - 'process_file': Path of the file actually processed (original or reduced).
        - 'imaphot': Statistics of detected objects and transients.
        - 'imastats': Database insertion status.

    Raises
    ------
    SerializableTaskError
        Wraps any unhandled exception for Celery serialization.
    """

    base_path = BASE_IMAGES_PATH
    if image_path:
        image_path = image_path.lstrip('/')

    logger.info(f"Processing image: {image_path} with instrument: {instrument_name}")

    processor = get_processor(instrument_name)

    reduction_config = processor.config['image_reduction']
    apply_reduction = reduction_config['apply_reduction']

    def call_process_image(file_path_call):
        """
        Internal helper to execute the processing pipeline on a specific file.
        This allows reusing the logic for both the original image and the
        reduced (binned/cropped) version.
        """
        process_file = os.path.relpath(file_path_call, base_path)
        logger.debug(f"Processing file: {process_file}")
        reset_cupy_allocators()

        try:
            # Attempt to process the image normally
            imdata, imheader = open_image_file(file_path_call)
            phot_df, hwcs = processor.process_image(imdata, imheader, header_descriptions=HEADER_DESCRIPTIONS)


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
        # Strategy: Check if reduction is forced ('always') or conditional ('on_failure')
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
            # Try processing the original image first
            result = call_process_image(file_path)

        reset_cupy_allocators()
        return result

    except MemoryError as e:
        # Handle Out-Of-Memory errors by attempting reduction if configured
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
                # If reduction also fails, log and raise
                task_error_handler(self, e, image_path)

        else:
            # If reduction is not enabled for failures, just fail
            task_error_handler(self, e, image_path)
    except Exception as e:
        # Handle generic exceptions
        task_error_handler(self, e, image_path)
