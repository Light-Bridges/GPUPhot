import gc

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

@nvtx.annotate('adaptive_memory_management', category='utils.gpu')
def adaptive_memory_management(mempool, threshold_warning=0.6, threshold_critical=0.8):
    """Retorna nivel de presión: 0 (normal), 1 (advertencia), 2 (crítico)"""
    try:
        dev = cp.cuda.Device()
        free_physical, total_physical_mem = dev.mem_info
        used_mem = mempool.used_bytes() # Memoria usada por el pool de CuPy

        if total_physical_mem == 0: return 0
        memory_usage_ratio = used_mem / total_physical_mem if total_physical_mem > 0 else 0

        action_taken = False
        level = 0

        if memory_usage_ratio > threshold_critical:
            logger.warning(f"Memory Pressure CRITICAL: Usage {memory_usage_ratio:.2%} > {threshold_critical:.0%}. "
                           f"Used: {human_readable_size(used_mem)}, Free Phys: {human_readable_size(free_physical)}. Aggressive cleanup.")
            mempool.free_all_blocks()
            gc.collect()
            # cp.cuda.Stream.null.synchronize() # Evitar sync si no es estrictamente necesario por rendimiento
            action_taken = True
            level = 2
        elif memory_usage_ratio > threshold_warning:
            logger.info(f"Memory Pressure WARNING: Usage {memory_usage_ratio:.2%} > {threshold_warning:.0%}. "
                        f"Used: {human_readable_size(used_mem)}, Free Phys: {human_readable_size(free_physical)}. Moderate cleanup.")
            mempool.free_all_blocks() # Liberar bloques no usados
            action_taken = True
            level = 1

        # Log final state after potential cleanup
        # if action_taken:
        #    free_after, _ = dev.mem_info
        #    logger.info(f"Memory state after cleanup: Used={human_readable_size(mempool.used_bytes())}, Free Phys={human_readable_size(free_after)}")
        # else:
        #    logger.debug(f"Memory Pressure NORMAL: Usage {memory_usage_ratio:.2%}. ")

        return level
    except cp.cuda.runtime.CUDARuntimeError as e:
        logger.error(f"Error getting CUDA memory info: {e}")
        return 0
    except Exception as e:
         logger.error(f"Unexpected error in adaptive_memory_management: {e}", exc_info=True)
         return 0