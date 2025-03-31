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
        Attempts to remove symbolic links that cause errors. NOW WITH MORE DEBUGGING!
        """
        exc_type, exc_value, tb = exc_info

        # --- NUEVO DEBUGGING ---
        print(f"DEBUG ONERROR: Triggered for func={func.__name__}, path={repr(path)}", flush=True)
        print(f"DEBUG ONERROR: Exception type={exc_type.__name__}, value={exc_value}", flush=True)

        path_exists = False
        path_is_link = False
        path_is_dir = False
        path_is_file = False

        try:
            path_exists = os.path.exists(path)  # Usa os.path.exists (sigue enlaces)
            path_l_exists = os.path.lexists(path)  # Usa os.path.lexists (NO sigue enlaces)
            print(f"DEBUG ONERROR: os.path.exists({repr(path)}) = {path_exists}", flush=True)
            print(f"DEBUG ONERROR: os.path.lexists({repr(path)}) = {path_l_exists}", flush=True)
            if path_l_exists:  # Solo intenta islink/isdir/isfile si lexists es True
                path_is_link = os.path.islink(path)
                path_is_dir = os.path.isdir(path)  # isdir sigue enlaces por defecto
                path_is_file = os.path.isfile(path)  # isfile sigue enlaces por defecto
                print(f"DEBUG ONERROR: os.path.islink({repr(path)}) = {path_is_link}", flush=True)
                print(f"DEBUG ONERROR: os.path.isdir({repr(path)}) = {path_is_dir}", flush=True)
                print(f"DEBUG ONERROR: os.path.isfile({repr(path)}) = {path_is_file}", flush=True)
            else:
                print(f"DEBUG ONERROR: Path {repr(path)} does not lexist.", flush=True)

        except Exception as e_stat:
            print(f"DEBUG ONERROR: Error checking path status for {repr(path)}: {e_stat}", flush=True)
        # --- FIN NUEVO DEBUGGING ---

        # Check if the error is related to a symbolic link
        # Let's rely on os.path.islink now that we have debugged it
        if path_is_link:  # Usamos la variable que ya comprobamos
            print(f"DEBUG ONERROR: Path {repr(path)} IS a link. Attempting os.unlink().", flush=True)
            try:
                os.unlink(path)
                print(f"DEBUG ONERROR: Successfully unlinked {repr(path)}.", flush=True)
                # Signal that we handled this specific error case
                return
            except OSError as e:
                print(f"DEBUG ONERROR: Failed to unlink symlink {repr(path)}: {e}. Propagating original error.",
                      flush=True)
                # If unlinking fails, let the original error propagate below.
            except Exception as e:
                print(
                    f"DEBUG ONERROR: Unexpected error unlinking symlink {repr(path)}: {e}. Propagating original error.",
                    flush=True)
        else:
            print(f"DEBUG ONERROR: Path {repr(path)} is NOT detected as a link by os.path.islink.", flush=True)

        # If the error wasn't handled (not a link, or unlink failed)
        print(f"DEBUG ONERROR: Error for {repr(path)} not handled by symlink logic. Checking ignore_errors/onerror.",
              flush=True)
        if onerror is not None:
            print(f"DEBUG ONERROR: Calling user-provided onerror.", flush=True)
            onerror(func, path, exc_info)  # Call original onerror if provided
        elif not ignore_errors:
            print(f"DEBUG ONERROR: ignore_errors is False. Re-raising original exception.", flush=True)
            # Re-raise the original exception correctly
            # NOTE: In Python 3, just 'raise' might be enough to re-raise the active exception
            # but explicitly using exc_value.with_traceback(tb) is safer.
            raise exc_value.with_traceback(tb)
        else:
            print(f"DEBUG ONERROR: ignore_errors is True. Suppressing error.", flush=True)

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
