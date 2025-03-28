import os
import shutil
import tempfile
from types import MethodType


# Monkeypatch para resolver el error de symbolic links
def _safe_rmtree(cls, name, ignore_errors=False):
    """Custom _rmtree que maneja enlaces simbólicos correctamente"""

    def onerror(func, path, exc_info):
        if not ignore_errors:
            raise

    if os.path.islink(name):
        os.unlink(name)
    else:
        shutil.rmtree(name, onerror=onerror)


# Aplicar el parche a TemporaryDirectory
tempfile.TemporaryDirectory._rmtree = MethodType(_safe_rmtree, tempfile.TemporaryDirectory)

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
