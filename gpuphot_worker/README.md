# gpuphot_worker

This directory contains the Celery worker implementation and supporting helpers
used by the GPUPhot project to process astronomical images and persist
photometry and image metadata. It provides the end-to-end worker system that
runs in a Celery environment (workers + optional beat scheduler) and integrates
with the project's image processing pipeline and PostgreSQL backend.

Purpose and scope
-----------------
The `gpuphot_worker` package groups everything required to run asynchronous
processing tasks for GPUPhot. It focuses on three responsibilities:

1. Task orchestration and error handling (Celery tasks and custom exceptions).
2. Image file input/output and pre-/post-processing helpers (open, bin/crop,
   save processed images, WCS/header handling).
3. Persistence helpers to store photometry results and image metadata in
   PostgreSQL, and read-only search helpers for monitoring and analysis.

Architecture overview
---------------------
- Celery tasks are defined in `tasks.py`. The main public tasks are:
  - `process_directory_task`: find images under the configured images base
    path and queue processing for each image found.
  - `process_image_task`: process a single file, save the processed FITS and
    store photometry and statistics in the database.

- A Celery application instance and example periodic schedule are provided in
  `worker_app.py` (used by `celery` command and `celery beat`).

- `utils.py` contains file I/O and image-processing helpers used by tasks,
  including robust FITS/NPY opening and a `crop_and_bin_image` function that
  preserves and adjusts WCS/header keywords.

- `database_insert_utils.py` and `database_search_utils.py` contain helper
  functions to write and read the PostgreSQL tables used by the project
  (`imaphot`, `imastats`, ...). Insert functions use SQLAlchemy; search
  helpers use psycopg2 and return pandas DataFrames for convenience.

- `celery_exceptions.py` defines a serializable exception and a task base
  class that stores structured failure metadata in the Celery task state, which
  helps remote callers and monitoring tools inspect failures.

- `header_descriptions.py` provides a lookup dictionary with human-friendly
  descriptions for common FITS header keywords used by the pipeline.

Key modules (short)
-------------------
- `celery_exceptions.py` — custom SerializableTaskError and Task base class
  that sets structured failure metadata.
- `celeryconfig.py` — Celery defaults; supports overriding values using
  environment variables prefixed with `CELERY_`.
- `tasks.py` — task entrypoints and orchestration logic.
- `utils.py` — image open/save, binning/cropping and WCS/header maintenance.
- `database_insert_utils.py` — insert photometry and image stats into Postgres.
- `database_search_utils.py` — read-only queries returning pandas DataFrames.
- `header_descriptions.py` — description mapping for FITS header keywords.
- `worker_app.py` — Celery app configuration + example beat schedule.
- `generators/` — notebooks and helper scripts useful for onboarding and
  demonstrations related to this worker package.

Configuration (environment variables)
-------------------------------------
The worker relies on a set of well-known environment variables. Important
examples used across the package:

- IMAGE_BASE_PATH — Base path where raw images are stored (default: `/data/images`).
- PROCESSED_IMAGE_FOLDER — Relative folder name for processed FITS (default: `gpuphot_processed`).
- INSTRUMENT_CONFIG_BASE_PATH — Base path for instrument configs used by the processor.

- Celery specific (prefix `CELERY_`) — any Celery setting can be overridden by
  exporting `CELERY_<SETTING>` (for example `CELERY_BROKER_URL`,
  `CELERY_RESULT_BACKEND`, `CELERY_TIMEZONE`). See `celeryconfig.py`.

- PostgreSQL connection (read/write)
  - POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT
  - Read-only variants used by search utilities: POSTGRES_USER_READ,
    POSTGRES_PASSWORD_READ

Runtime flow (high level)
-------------------------
1. A scheduler or manual invocation calls `process_directory_task` to discover
   image files and queue `process_image_task` for each path.
2. `process_image_task` opens the image, calls the processor (instrument
   specific) to compute photometry and produce an updated header/WCS.
3. The processed image is saved to a `PROCESSED_IMAGE_FOLDER` path as a FITS
   file and a unique ID (GPUPHOTI) is generated and stored in the header.
4. Photometry rows are converted into a DataFrame and inserted into the
   `imaphot` table; image metadata is stored/updated in `imastats`.
5. On failure, tasks raise `SerializableTaskError` (or the task base class
   stores structured failure metadata) so the remote client can inspect
   `task.info()` or Celery backend metadata for details.

Operational notes
-----------------
- The package attempts to use GPU-accelerated arrays (cupy) when available
  and falls back to NumPy otherwise. This is transparent to most of the code.

- Persistent data (processed FITS and database entries) are saved under the
  configured `IMAGE_BASE_PATH` and PostgreSQL. Make sure worker processes have
  the correct file system and DB access.

- For debugging, task states and `meta` information (including tracebacks)
  can be inspected via Celery's result backend (e.g. Redis) or via the Celery
  monitoring tools.

Developer notes
---------------
- Docstrings are written in English and use NumPy-style conventions.
- Source files in this package include SPDX headers identifying the license
  (consistent with the project top-level license).
- If you plan to evolve header definitions, consider generating a
  human-readable `HEADERS_INDEX.md` from `header_descriptions.py`.

Where to look next
------------------
- `tasks.py` to understand the asynchronous processing flow and retry/failure handling.
- `utils.py` for detailed image handling (important: WCS and header updates).
- `database_insert_utils.py` and `database_search_utils.py` for persistence
  semantics and sample queries.

If you prefer this README in Spanish, or want a shorter developer-oriented
version (API, examples), tell me which style you prefer and I will update it.

## Notebook generators

This package includes a small collection of generator scripts under
`gpuphot_worker/generators/` that programmatically create example Jupyter
notebooks used for onboarding and demonstrations. The notebooks are intended
for interactive use inside the project's JupyterLab environment.

Generators available
--------------------
- `0_System_Overview_Notebook.py` — creates `0_System_Overview_Notebook.ipynb`.
  Provides a high-level overview of the services and architecture used by
  GPUPhot (RabbitMQ, Redis, Celery, worker, PostgreSQL, JupyterLab, Flower).

- `1_Setup_Notebook.py` — creates `1_Setup_Notebook.ipynb`.
  Guides the user through environment variables, `.env` example, and
  instructions for downloading astrometry index files required by the solver.

- `2_Instrument_Configuration_Notebook.py` — creates `2_Instrument_Configuration_Notebook.ipynb`.
  Demonstrates how to inspect, generate and customize instrument configuration
  JSON files (forced values, filter mapping, default configuration).

- `3_Task_Execution_Notebook.py` — creates `3_Task_Execution_Notebook.ipynb`.
  Examples for submitting Celery tasks, monitoring AsyncResults, scheduling
  periodic tasks with RedBeat and using `app.send_task`.

- `4_Database_Query_Notebook.py` — creates `4_Database_Query_Notebook.ipynb`.
  Shows how to query the PostgreSQL database for photometric results,
  generate light curves, and explore image statistics.

How the generators are used
--------------------------
The repository contains an initializer script `initialize_notebooks.sh` which
is executed when the Jupyter-based container starts. The initializer:

1. Creates symbolic links from host data directories (instrument configs and
   images) into the Jupyter work directory.
2. Executes each generator script once to populate the user's Jupyter
   `work` directory with example notebooks.

The initializer calls the generators using the Python interpreter installed in
`/app/venv/bin/python` and passes `--output-dir` to control where the
notebooks are written. The marker file `.notebooks_generated` in the work
folder prevents re-generation on subsequent container starts.

Notes and suggestions
---------------------
- The generators are intentionally simple and write plain notebooks; you can
  adapt or extend them to include additional examples or environment checks.
- If you prefer a different layout for notebooks (for example, inside
  `docs/notebooks/`), consider moving the generated files and updating the
  initializer script accordingly. If you want, I can help reorganize the
  directory structure before generating additional documentation.

