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

# Patch multiprocessing to allow cuda.pathfinder to spawn subprocesses
# inside Celery daemon workers. Python 3.12 strictly enforces that daemon
# processes cannot create children, but cuda.pathfinder needs to spawn a
# subprocess to locate CUDA headers for kernel compilation.
#
# Two patches are needed:
# 1. Clear daemon flag during Process.start() to bypass Python 3.12 assertion
# 2. Allow billiard's AuthenticationString to be pickled, because
#    cuda.pathfinder uses 'spawn' context which requires pickling the
#    process state, and billiard blocks pickling AuthenticationString
import multiprocessing.process as _mp_process

_original_process_start = _mp_process.BaseProcess.start


def _patched_process_start(self):
    current = _mp_process.current_process()
    was_daemon = current._config.get('daemon')
    if was_daemon:
        current._config['daemon'] = False
    try:
        _original_process_start(self)
    finally:
        if was_daemon:
            current._config['daemon'] = was_daemon


_mp_process.BaseProcess.start = _patched_process_start

try:
    from billiard.process import AuthenticationString as _BilliardAuthString
    _BilliardAuthString.__reduce__ = lambda self: (_BilliardAuthString, (bytes(self),))
except ImportError:
    pass

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
