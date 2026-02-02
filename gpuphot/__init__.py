# SPDX-License-Identifier: MIT
"""
Top-level gpuphot package initializer.

This module performs minor runtime patches (a safe rmtree for TemporaryDirectory)
and exposes commonly used subpackages. It intentionally avoids heavy runtime
initialization (like forcing a specific GPU) to keep import-time side-effects
minimal.
"""

import os
import subprocess
import tempfile
from types import MethodType


def _super_safe_rmtree(cls, name, ignore_errors=False, onerror=None):
    """
    Custom _rmtree for tempfile.TemporaryDirectory that handles:
    1. The top-level directory 'name' being a symbolic link.
    2. Avoids calling shutil.rmtree on the top-level directory if it's a plain directory,
       using 'rm -rf' instead as a fallback due to potential shutil.rmtree bugs/issues.
    """

    # Internal onerror handler (kept simple because we prefer an explicit rm -rf fallback)
    def _internal_onerror(func, path, exc_info):

        # If the caller provided a custom onerror, call it
        if onerror:
            onerror(func, path, exc_info)
        # If we should not ignore errors, re-raise
        elif not ignore_errors:
            exc_type, exc_value, tb = exc_info
            raise exc_value.with_traceback(tb)

    # --- Main logic of _super_safe_rmtree ---
    try:
        # First, check existence using lexists (do not follow symlinks)
        if not os.path.lexists(name):
            return

        # If it's a symlink, unlink it
        if os.path.islink(name):
            os.unlink(name)

        # If it's a directory (and not a symlink), fall back to a shell rm -rf
        elif os.path.isdir(name):
            cmd = ['rm', '-rf', name]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                if not ignore_errors:
                    raise OSError(f"'rm -rf' failed for {name}: {result.stderr}")

        # Otherwise, if it exists, treat it as a regular file and remove
        elif os.path.exists(name):
            os.remove(name)

    except Exception as e:
        if not ignore_errors:
            raise
        # If ignore_errors is True, suppress the exception here


# Apply the super enhanced patch to TemporaryDirectory if available
if hasattr(tempfile.TemporaryDirectory, '_rmtree'):
    tempfile.TemporaryDirectory._rmtree = MethodType(_super_safe_rmtree, tempfile.TemporaryDirectory)

# Optional: cupy float64 patch is provided in gpuphot.patch_cupy
# from . import patch_cupy

# Import commonly used subpackages (deferred heavy initialization)
from . import image_processor
from . import instrument_config_parser
from . import logger
from . import phot
from . import stats
from . import utils

# GPU configuration hints are intentionally commented out to avoid import-time side-effects
# gpu_id = os.environ.get('GPUPHOT_GPU_ID', '0')
# try:
#     from .logger.hierarchical_logging import setup_logger
#     logger = setup_logger(__name__)
#     cp.cuda.Device(int(gpu_id)).use()
#     logger.info(f"Using GPU: {cp.cuda.get_device_id()}")
# except Exception as e:
#     logger.error(f"Could not set the GPU {gpu_id}. Using GPU 0 by default.")
#     cp.cuda.Device(0).use()

__all__ = ['logger', 'phot', 'stats', 'utils', 'instrument_config_parser', 'image_processor']
