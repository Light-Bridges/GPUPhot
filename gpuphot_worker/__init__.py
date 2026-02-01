"""
gpuphot_worker package
======================

Utilities, Celery tasks and helpers used by the GPUPhot worker service.
This package contains task definitions, configuration helpers and database
utilities required to process astronomical images and persist results.

Public modules
- celery_exceptions: Custom Celery exceptions and task base class.
- celeryconfig: Celery configuration loaded from environment variables.
- database_insert_utils: Helpers to insert processing results into PostgreSQL.
- database_search_utils: Read-only database helper functions (search queries).
- header_descriptions: Human-friendly descriptions for FITS header keys.
- tasks: Celery tasks that perform image processing.
- utils: Local helper functions used by tasks (file I/O, saving results).
- worker_app: Celery application definition and scheduled tasks.

The package follows the project documentation guidelines: docstrings are
written in English and follow NumPy-style conventions where applicable.
"""

# Expose a minimal public API for importing the package
__all__ = [
    'celery_exceptions',
    'celeryconfig',
    'database_insert_utils',
    'database_search_utils',
    'header_descriptions',
    'tasks',
    'utils',
    'worker_app',
]
