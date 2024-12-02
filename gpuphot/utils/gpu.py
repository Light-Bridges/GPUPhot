import gc

import cupy as cp

from ..logger.hierarchical_logging import setup_logger, hierarchical_debug

logger = setup_logger(__name__)


@hierarchical_debug(logger)
def free_gpu_mem() -> None:
    """Liberates GPU memory by freeing all memory blocks allocated by the default memory pool."""
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    gc.collect()


@hierarchical_debug(logger)
def init_gpu(**kwargs) -> None:
    """Inicializa y registra información de la GPU usando CuPy"""

    # Versión de CuPy
    logger.debug('CuPy version ' + cp.__version__)

    # Información de la GPU
    num_gpus = cp.cuda.runtime.getDeviceCount()
    logger.debug(f'Number of GPUs: {num_gpus}')

    for i in range(num_gpus):
        gpu_props = cp.cuda.runtime.getDeviceProperties(i)
        logger.debug(f'GPU {i}: {gpu_props["name"]}')
        logger.debug(f'  Total memory: {gpu_props["totalGlobalMem"] / (1024 ** 3):.2f} GB')

    # pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    # cp.cuda.set_allocator(pool.malloc)
    #
    # # Limitar el pool a 1024 MB (1 GB)
    # pool.set_limit(size=1024 * 1024 * 1024)  # 1 GB en bytes
    #
    # logger.debug(f'GPU memory pool limited to 1 GB')

    # def init_gpu():
    #     """ """
    # try:
    #     # Intentar liberar toda la memoria posible
    #     import tensorflow as tf
    #     logger.debug('Tensorflow version ' + tf.__version__)
    #     gpus = tf.config.list_physical_devices('GPU')
    #     logger.debug('GPUs:', gpus)
    #
    #     memory_limit = kwargs.get('memory_limit', 1024)
    #
    #     tf.config.set_logical_device_configuration(gpus[0], [tf.config.
    #                                                LogicalDeviceConfiguration(memory_limit=memory_limit)])
    # except Exception:
    #     pass
