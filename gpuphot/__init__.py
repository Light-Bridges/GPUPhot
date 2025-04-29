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

    # Definimos el onerror interno por si acaso lo necesitáramos (aunque ahora no lo usamos directamente)
    # No necesitamos un onerror complejo si vamos a usar 'rm -rf' para directorios
    def _internal_onerror(func, path, exc_info):

        # Si el usuario proveyó un onerror, lo llamamos
        if onerror:
            onerror(func, path, exc_info)
        # Si no debemos ignorar errores, relanzamos
        elif not ignore_errors:
            exc_type, exc_value, tb = exc_info
            raise exc_value.with_traceback(tb)

    # --- Lógica Principal de _super_safe_rmtree ---
    try:
        # Primero, comprobar si existe usando lexists (no sigue enlaces)
        if not os.path.lexists(name):
            return  # No existe, no hacemos nada

        # Comprobar si es un enlace
        if os.path.islink(name):
            os.unlink(name)

        # Comprobar si es un directorio (y NO un enlace, ya comprobado arriba)
        elif os.path.isdir(name):
            # Usar subprocess.run es más seguro que os.system
            cmd = ['rm', '-rf', name]
            # Usamos check=False porque queremos manejar errores nosotros mismos si ignore_errors es True
            # Capturamos stdout/stderr para depuración
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                # Si no debemos ignorar errores, lanzamos una excepción
                if not ignore_errors:
                    # Podríamos crear una OSError más específica
                    raise OSError(f"'rm -rf' failed for {name}: {result.stderr}")


        # Comprobar si es cualquier otra cosa (un archivo normal)
        elif os.path.exists(name):  # Si no es link ni dir, pero existe, debe ser un archivo
            os.remove(name)

    except Exception as e:
        if not ignore_errors:
            # If errors are not ignored at the top level either, re-raise
            raise
        # If ignore_errors is True, suppress the exception at this level too


# Apply the super enhanced patch
if hasattr(tempfile.TemporaryDirectory, '_rmtree'):
    tempfile.TemporaryDirectory._rmtree = MethodType(_super_safe_rmtree, tempfile.TemporaryDirectory)

# Use this if you want to use cupy float64 patch
# from . import patch_cupy

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
