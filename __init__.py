# Patch multiprocessing to allow cuda.pathfinder to spawn subprocesses
# inside Celery daemon workers. Python 3.12 strictly enforces that daemon
# processes cannot create children, but cuda.pathfinder needs to spawn a
# subprocess to locate CUDA headers for kernel compilation.
# This patch temporarily clears the daemon flag during Process.start().
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

from . import gpuphot
from . import tests

__all__ = ['gpuphot', 'tests']
