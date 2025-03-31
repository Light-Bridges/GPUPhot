import sys
import time
from datetime import timedelta
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
    print(f"DEBUG: _enhanced_safe_rmtree called for: '{name}'")

    def _rmtree_onerror(func, path, exc_info):
        """
        Error handler for shutil.rmtree.
        Attempts to remove symbolic links that cause errors.
        """
        exc_type, exc_value, tb = exc_info
        print(f"DEBUG: _rmtree_onerror triggered for path '{path}' by function {func.__name__} with error: {exc_value}")

        # Check if the error is related to a symbolic link
        # Often OSError during listdir/remove/rmdir on a link, or if islink fails due to permissions
        # Let's specifically check if the path is a link when an error occurs
        if os.path.islink(path):
            print(f"DEBUG: Error involves a symbolic link: '{path}'. Attempting os.unlink().")
            try:
                os.unlink(path)
                print(f"DEBUG: Successfully unlinked '{path}'.")
                # Important: Return here to indicate the error was handled (if possible)
                # so shutil.rmtree might continue if ignore_errors=True allows it.
                # However, standard shutil behavior stops on error unless ignore_errors=True.
                # We handled *this specific* link error.
                return # Signal that we handled this specific error case
            except OSError as e:
                print(f"DEBUG: Failed to unlink symlink '{path}': {e}. Original error will propagate.")
                # If unlinking fails, let the original error propagate below.
            except Exception as e:
                print(f"DEBUG: Unexpected error unlinking symlink '{path}': {e}. Original error will propagate.")
                # Catch other potential errors during unlink

        # If the error wasn't handled (not a link, or unlink failed)
        # and we are NOT ignoring errors, we should let the exception propagate.
        # The original onerror passed by the user (if any) or the default
        # behavior of rmtree (raising the exception if ignore_errors=False) should take over.
        # If the user supplied an 'onerror', we should call it.
        # Otherwise, if ignore_errors is False, the exception should be raised.
        print(f"DEBUG: Error for '{path}' not handled by symlink logic.")
        if onerror is not None:
             print(f"DEBUG: Calling user-provided onerror for '{path}'.")
             onerror(func, path, exc_info) # Call original onerror if provided
        elif not ignore_errors:
            print(f"DEBUG: Re-raising original exception for '{path}' as ignore_errors=False.")
            # Re-raise the original exception correctly
            raise exc_value.with_traceback(tb)
        else:
            print(f"DEBUG: Suppressing error for '{path}' as ignore_errors=True.")
            # If ignore_errors is True and we didn't handle it (or user onerror didn't raise),
            # execution continues within shutil.rmtree

    # --- Main logic of _enhanced_safe_rmtree ---
    try:
        if os.path.islink(name):
            print(f"DEBUG: Top-level path '{name}' is a link. Unlinking.")
            os.unlink(name)
        elif os.path.exists(name): # Only call rmtree if it exists and isn't a link
             print(f"DEBUG: Top-level path '{name}' is not a link. Calling shutil.rmtree.")
             # Pass our custom handler to the nested rmtree call
             shutil.rmtree(name, ignore_errors=ignore_errors, onerror=_rmtree_onerror)
        else:
             print(f"DEBUG: Top-level path '{name}' does not exist. Nothing to remove.")

    except Exception as e:
         print(f"DEBUG: Exception during _enhanced_safe_rmtree for '{name}': {e}")
         if not ignore_errors:
             # If errors are not ignored at the top level either, re-raise
             raise
         # If ignore_errors is True, suppress the exception at this level too


# Apply the enhanced patch
if hasattr(tempfile.TemporaryDirectory, '_rmtree'):
    # Store original for safety, although we don't use it here
    # tempfile.TemporaryDirectory._rmtree_original = tempfile.TemporaryDirectory._rmtree
    tempfile.TemporaryDirectory._rmtree = MethodType(_enhanced_safe_rmtree, tempfile.TemporaryDirectory)
    print("DEBUG: Enhanced monkey patch for tempfile.TemporaryDirectory._rmtree applied.")
else:
    print("DEBUG WARNING: tempfile.TemporaryDirectory does not have _rmtree method to patch.")



# Asegúrese de que todas las rutas estén correctamente configuradas
sys.path.append('/app')

# Importe su función principal DESPUÉS del parche
from gpuphot_worker.tasks import process_image_task


def main():
    if len(sys.argv) < 2:
        print("Uso: python profile_image_processing.py <ruta_imagen> [instrument_name]")
        sys.exit(1)

    file_path = sys.argv[1]
    instrument_name = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('INSTRUMENT_NAME', 'default')

    print(f"Procesando imagen: {file_path} con instrumento: {instrument_name}")

    start_time = time.time()
    try:
        result = process_image_task(file_path, instrument_name=instrument_name)
        print("Resultado del procesamiento:") # Añadido para claridad
        print(result)
    except Exception as e:
        # Imprime el traceback completo para más detalles
        import traceback
        print(f"Error durante process_image_task:")
        traceback.print_exc()
        # El logger ya debería haber capturado esto, pero una impresión extra puede ayudar
        print(f"Mensaje de error resumido: {e}")
    finally: # Asegura que el tiempo se mida incluso si hay error
        end_time = time.time()
        elapsed_time = end_time - start_time
        readable_time = str(timedelta(seconds=elapsed_time))
        print(f"Time elapsed: {readable_time}")


if __name__ == "__main__":
    main()