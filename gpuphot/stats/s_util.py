import gc

import cupy as cp

from ..logger.hierarchical_logging import setup_logger, hierarchical_debug

logger = setup_logger(__name__)

@hierarchical_debug(logger)
def free_gpu_mem():

    """Frees all allocated GPU memory blocks and triggers garbage collection.
    
    This function frees all blocks in the default memory pool and the default
    pinned memory pool of CuPy, and then runs the garbage collector to free up
    any additional resources that might be held.


    >>> free_gpu_mem()
    """
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    gc.collect()
