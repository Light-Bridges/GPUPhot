import gc
import logging
import time

import cupy as cp

from ..logger.hierarchical_logging import setup_logger, hierarchical_debug
logger = setup_logger(__name__)

@hierarchical_debug(logger)
def free_gpu_mem():
    logger.debug(f'Iniciando función free_gpu_mem()')
    start_time = time.time()
    """
    Frees all allocated GPU memory blocks and triggers garbage collection.

    This function frees all blocks in the default memory pool and the default
    pinned memory pool of CuPy, and then runs the garbage collector to free up
    any additional resources that might be held.

    Returns
    -------
    None

    Example
    -------
    To use this function, simply call it whenever you need to free up GPU memory:

    >>> free_gpu_mem()

    Notes
    -----
    This function is useful for managing GPU memory in scenarios where multiple
    large GPU arrays are used and memory needs to be explicitly freed to avoid
    out-of-memory errors.
    """
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    gc.collect()
    logger.debug(
        f'Función free_gpu_mem completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos'
    )
