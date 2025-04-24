import gc
import time

import cupy as cp
import nvtx

from ..logger.hierarchical_logging import setup_logger

logger = setup_logger(__name__)


@nvtx.annotate('human_readable_size', category='utils.gpu')
def human_readable_size(bytes_size):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(bytes_size) < 1024.0:
            return f"{bytes_size:.1f}{unit}B"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}YiB"


@nvtx.annotate('free_gpu_mem', category='utils.gpu')
def free_gpu_mem() -> None:
    """
    Liberate GPU memory by freeing all memory blocks allocated by the default memory pool.

    This function frees all memory blocks in both the default memory pool and the default pinned memory pool,
    and then calls the garbage collector to clean up any remaining objects.
    """
    # Get memory pools
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    # Get free memory before liberation
    # memory_before = mempool.free_bytes()

    # Free all memory blocks
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    # Force garbage collection
    gc.collect()

    # Get free memory after liberation
    # memory_after = mempool.free_bytes()

    # Calculate freed memory
    # memory_freed = memory_after - memory_before

    # Display freed memory
    # logger.debug(f"Memory freed: {human_readable_size(memory_freed)}")


@nvtx.annotate('force_free_gpu_memory', category='utils.gpu')
def force_free_gpu_memory():
    free_gpu_mem()

    # Sincronizar todos los streams de CUDA
    cp.cuda.Stream.null.synchronize()

    # Reiniciar el entorno de CuPy
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    # Forzar la limpieza de caché de kernels
    cp.fft.config.get_plan_cache().clear()


@nvtx.annotate('use_gpu', category='utils.gpu')
def use_gpu(gpu_id=0):
    num_gpus = cp.cuda.runtime.getDeviceCount()
    if gpu_id < num_gpus:
        cp.cuda.Device(gpu_id).use()
    else:
        cp.cuda.Device(0).use()
        logger.warning(f"GPU {gpu_id} not found. Using GPU 0 by default.")


@nvtx.annotate('init_gpu', category='utils.gpu')
def init_gpu(**kwargs) -> None:
    """
    Initialize and log information about the GPU using CuPy.

    This function performs the following tasks:
    1. Logs the CuPy version.
    2. Determines the number of available GPUs.
    3. For each GPU, logs its name and total memory.

    :param kwargs: Additional keyword arguments (currently unused).
    """
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


@nvtx.annotate('reset_cupy_allocators', category='utils.gpu')
def reset_cupy_allocators():
    # Clear all existing memory pools
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    # Create new memory pools
    mempool = cp.cuda.MemoryPool()
    pinned_mempool = cp.cuda.PinnedMemoryPool()

    # Set the new pools as default
    cp.cuda.set_allocator(mempool.malloc)
    cp.cuda.set_pinned_memory_allocator(pinned_mempool.malloc)

    # Force garbage collection
    gc.collect()

    # Synchronize CUDA streams
    cp.cuda.Stream.null.synchronize()

    # Clear kernel caches
    cp.fft.config.get_plan_cache().clear()


@nvtx.annotate('maybe_free_arrays', category='utils.gpu')
def maybe_free_arrays(arrays, mempool, threshold=0.2):
    """
    Free arrays if free GPU memory is below threshold fraction of total.
    """
    free_mem, total_mem = cp.cuda.Device(0).mem_info
    if free_mem / total_mem < threshold:
        for arr in arrays:
            del arr
        mempool.free_all_blocks()
        cp.cuda.Stream.null.synchronize()


@nvtx.annotate('get_memory_usage_ratio', category='utils.gpu')
def get_memory_usage_ratio(mempool):
    """Returns the ratio of used GPU memory to total GPU memory."""
    try:
        return mempool.used_bytes() / mempool.total_bytes()
    except ZeroDivisionError:
        return 0.0


@nvtx.annotate('adaptive_memory_management', category='utils.gpu')
def adaptive_memory_management(mempool, thresholds=None,
                               force_free: bool = False):
    """
    Checks memory usage against thresholds.

    :param mempool: CuPy memory pool.
    :param thresholds: Dict with keys 'safe', 'warning', 'critical' and ratio values.
    :param force_free: If True, always call free_all_blocks.
    :return: Cleanup level (0: safe, 1: warning, 2: critical).
    """
    if thresholds is None:
        thresholds = {'safe': 0.6, 'warning': 0.75, 'critical': 0.85}

    ratio = get_memory_usage_ratio(mempool)
    level = 0
    if ratio >= thresholds['critical']:
        level = 2
    elif ratio >= thresholds['warning']:
        level = 1

    logger.debug(f"AMM Memory Check: Ratio={ratio:.3f} (Allocated/Total), Determined Level={level}")

    if force_free or level >= 2:
        if level >= 2:
            logger.warning(f"AMM CRITICAL memory pressure (Ratio={ratio:.3f}). Forcing free_all_blocks.")
        else:
            logger.info(f"AMM Forcing free_all_blocks (force_free=True).")

        start_free = time.time()
        mempool.free_all_blocks()
        gc.collect()
        end_free = time.time()
        ratio_after = get_memory_usage_ratio(mempool)
        logger.info(
            f"AMM free_all_blocks took {end_free - start_free:.2f}s. Memory after: Ratio={ratio_after:.3f}")
    # elif level == 1:
    #     logger.warning(f"{log_prefix} WARNING memory pressure (Ratio={ratio:.3f}). Consider targeted cleanup/offload.")

    return level
