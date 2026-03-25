# GPUPhot Worker

**Version:** 1.0
**Date:** 2024-08-02

## 1. Overview

The `gpuphot_worker` directory contains all the components necessary for running `GPUPhot` in a distributed and asynchronous manner using [Celery](https://docs.celeryq.dev/en/stable/). It is designed to handle large-scale image processing by distributing tasks to one or more worker nodes.

The core of this component is a Celery application that exposes high-level tasks for processing individual images or entire directories. These tasks orchestrate the core `gpuphot` library, handle data persistence to a database, and manage image reduction strategies in case of failures (e.g., out-of-memory errors).

## 2. Directory Structure

-   `worker_app.py`: Defines and configures the main Celery application instance.
-   `celeryconfig.py`: Contains the configuration for the Celery app, such as broker URL, result backend, and task routing.
-   `tasks.py`: **(Most important for users)** Defines the Celery tasks that can be called remotely to process data.
-   `database_insert_utils.py`: Contains utility functions for inserting photometry results and image statistics into the PostgreSQL database.
-   `database_search_utils.py`: Read-only database helper functions for searching and querying photometry results.
-   `utils.py`: Provides helper functions for the worker, such as opening image files and managing processed image paths.
-   `celery_exceptions.py`: Defines custom, serializable exceptions for robust error handling across the distributed system.
-   `header_descriptions.py`: Stores the descriptions for custom FITS header keywords added by the pipeline.

## 3. Available Celery Tasks

These tasks are the main entry points for users wanting to process images asynchronously.

### 3.1. `process_directory_task`

This is a high-level orchestration task that scans a directory for images and queues an individual `process_image_task` for each one found.

**Signature:**
```python
process_directory_task.delay(
    path: str = None,
    filename: str = None,
    instrument_name: str = None,
    exclude_pattern: str = None,
    reprocess: bool = False
)
```

**Parameters:**

-   `path` (str, optional): The subdirectory within the main image data folder (`IMAGE_BASE_PATH`) to search. If `None`, it searches from the root of the data folder.
-   `filename` (str, optional): A specific filename or a pattern with wildcards (e.g., `*object_A*.fits`) to match.
-   `instrument_name` (str, optional): The name of the instrument configuration to use for processing. If `None`, the default configuration is used.
-   `exclude_pattern` (str, optional): A regular expression to exclude certain filenames from processing.
-   `reprocess` (bool, optional): If `True`, all matching images will be processed, even if they have been processed before. If `False` (default), it skips images that already have a corresponding output file.

**Returns:**
A dictionary mapping the relative path of each queued image to the Celery task ID of its `process_image_task`.

**Example Usage:**
```python
from gpuphot_worker.tasks import process_directory_task

# Process all .fits files in the 'night_2024-08-01' directory
task = process_directory_task.delay(
    path='night_2024-08-01',
    filename='*.fits'
)

print(f"Directory processing task started with ID: {task.id}")

# To get the result (a dict of queued tasks) later:
# results_dict = task.get()
```

### 3.2. `process_image_task`

This task performs the complete processing pipeline for a single image file. It is typically called by `process_directory_task` but can be invoked directly.

**Signature:**
```python
process_image_task.delay(
    image_path: str,
    instrument_name: str = None
)
```

**Parameters:**

-   `image_path` (str): The path to the image file, relative to the `IMAGE_BASE_PATH` directory.
-   `instrument_name` (str, optional): The instrument configuration to use.

**Behavior:**

1.  Loads the image data and header.
2.  Calls the core `gpuphot.process_image` function.
3.  If processing is successful, it saves the photometry and statistics to the database and writes the processed FITS file to the output directory.
4.  **Automatic Image Reduction**: If the initial processing fails with a `MemoryError`, and the instrument configuration allows it (`"apply_reduction": "on_failure"`), the task will automatically attempt to re-process the image after applying binning and/or cropping as defined in the configuration.

**Returns:**
A dictionary containing metadata about the processing run, including input/output file paths and database insertion results.

**Example Usage:**
```python
from gpuphot_worker.tasks import process_image_task

# Process a single, specific image
task = process_image_task.delay(
    image_path='raw_data/image_001.fits',
    instrument_name='my_telescope_config'
)

print(f"Image processing task started with ID: {task.id}")

# To get the result dictionary later:
# result = task.get()
```

## 4. Configuration

The worker's behavior is configured through environment variables, which are loaded by `celeryconfig.py`. Key variables include:

-   `CELERY_BROKER_URL`: Celery message broker URL (e.g. `amqp://user:pass@host:5672/`). The `celeryconfig.py` loads all `CELERY_`-prefixed env vars dynamically.
-   `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_DB`, `POSTGRES_PORT`: Credentials for the results database.
-   `IMAGE_BASE_PATH`: The absolute path to the root directory where raw image data is stored (default: `/data/images`).
-   `INSTRUMENT_CONFIG_BASE_PATH`: The path to the directory containing instrument JSON configuration files.
-   `INSTRUMENT_NAME`: The name of the default instrument configuration to use if none is specified (default: `default_instrument`).

Ensure these are set correctly in your `.env` file or your deployment environment before starting the worker.
