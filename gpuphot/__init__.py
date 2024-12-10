import os

import cupy as cp

from . import image_processor
from . import instrument_config_parser
from . import logger
from . import phot
from . import stats
from . import utils

gpu_id = os.environ.get('GPUPHOT_GPU_ID', '0')

try:
    from .logger.hierarchical_logging import setup_logger

    logger = setup_logger(__name__)

    # Attempt to get the GPU ID from an environment variable

    cp.cuda.Device(int(gpu_id)).use()
    logger.info(f"Using GPU: {cp.cuda.get_device_id()}")

except Exception as e:
    logger.error(f"Could not set the GPU {gpu_id}. Using GPU 0 by default.")
    cp.cuda.Device(0).use()

__all__ = ['logger', 'phot', 'stats', 'utils', 'instrument_config_parser', 'image_processor']
