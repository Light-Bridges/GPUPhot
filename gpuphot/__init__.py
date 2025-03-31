import os
import shutil
import tempfile
from types import MethodType


def _enhanced_safe_rmtree(cls, name, ignore_errors=False, onerror=None):
    """
    Custom _rmtree for tempfile.TemporaryDirectory that handles:
    1. The top-level directory 'name' being a symbolic link.
    2. Symbolic links encountered *inside* the directory 'name' during shutil.rmtree.
    """

    def _rmtree_onerror(func, path, exc_info):
        """
        Error handler for shutil.rmtree.
        Attempts to remove symbolic links that cause errors.
        """
        exc_type, exc_value, tb = exc_info

        # Check if the error is related to a symbolic link
        # Often OSError during listdir/remove/rmdir on a link, or if islink fails due to permissions
        # Let's specifically check if the path is a link when an error occurs
        if os.path.islink(path):
            try:
                os.unlink(path)
                # Important: Return here to indicate the error was handled (if possible)
                # so shutil.rmtree might continue if ignore_errors=True allows it.
                # However, standard shutil behavior stops on error unless ignore_errors=True.
                # We handled *this specific* link error.
                return  # Signal that we handled this specific error case
            except OSError as e:
                pass
                # If unlinking fails, let the original error propagate below.
            except Exception as e:
                pass
                # Catch other potential errors during unlink

        # If the error wasn't handled (not a link, or unlink failed)
        # and we are NOT ignoring errors, we should let the exception propagate.
        # The original onerror passed by the user (if any) or the default
        # behavior of rmtree (raising the exception if ignore_errors=False) should take over.
        # If the user supplied an 'onerror', we should call it.
        # Otherwise, if ignore_errors is False, the exception should be raised.
        if onerror is not None:
            onerror(func, path, exc_info)  # Call original onerror if provided
        elif not ignore_errors:
            # Re-raise the original exception correctly
            raise exc_value.with_traceback(tb)

    # --- Main logic of _enhanced_safe_rmtree ---
    try:
        if os.path.islink(name):
            os.unlink(name)
        elif os.path.exists(name):  # Only call rmtree if it exists and isn't a link
            # Pass our custom handler to the nested rmtree call
            shutil.rmtree(name, ignore_errors=ignore_errors, onerror=_rmtree_onerror)

    except Exception as e:
        if not ignore_errors:
            # If errors are not ignored at the top level either, re-raise
            raise
        # If ignore_errors is True, suppress the exception at this level too


# Apply the enhanced patch
if hasattr(tempfile.TemporaryDirectory, '_rmtree'):
    # Store original for safety, although we don't use it here
    # tempfile.TemporaryDirectory._rmtree_original = tempfile.TemporaryDirectory._rmtree
    tempfile.TemporaryDirectory._rmtree = MethodType(_enhanced_safe_rmtree, tempfile.TemporaryDirectory)

# Resto de imports y configuración
from . import image_processor
from . import instrument_config_parser
from . import logger
from . import phot
from . import stats
from . import utils

# Configuración de GPU (descomenta si es necesario)
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
