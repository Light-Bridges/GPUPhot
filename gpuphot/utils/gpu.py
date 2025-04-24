import gc
import time
from typing import Dict, Optional

import cupy as cp
import nvtx

from ..logger.hierarchical_logging import setup_logger

logger = setup_logger(__name__)

# Define constants for clarity and maintainability
SAFE_LEVEL = 0
WARNING_LEVEL = 1
CRITICAL_LEVEL = 2

# Define default thresholds in a single place
DEFAULT_THRESHOLDS = {'warning': 0.75, 'critical': 0.85}


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
def get_memory_usage_ratio(mempool: cp.cuda.MemoryPool) -> float:
    """Returns the ratio of used GPU memory to total GPU memory."""
    try:
        used = mempool.used_bytes()
        total = mempool.total_bytes()
        # Handle case where total_bytes might be 0 initially or after reset
        return used / total if total > 0 else 0.0
    except Exception as e:
        logger.error(f"Error getting memory usage ratio: {e}", exc_info=True)
        # Return a value indicating potential issue, e.g., 1.0 to trigger cleanup
        return 1.0


@nvtx.annotate('adaptive_memory_management', category='utils.gpu')
def adaptive_memory_management(
        mempool: cp.cuda.MemoryPool,
        thresholds: Optional[Dict[str, float]] = None,
        force_free: bool = False
) -> int:
    """
    Checks GPU memory usage against thresholds and potentially frees memory.

    Determines the memory pressure level based on the ratio of used bytes to
    total bytes in the provided CuPy memory pool. If the 'critical' threshold
    is exceeded or `force_free` is True, it attempts to free all blocks
    in the memory pool, runs garbage collection, and synchronizes the default stream.

    Args:
        mempool: The CuPy memory pool to monitor (e.g., cp.get_default_memory_pool()).
        thresholds: A dictionary with optional keys 'warning' and 'critical', mapping
                    to memory usage ratios (0.0 to 1.0). If None or keys are missing,
                    defaults defined in DEFAULT_THRESHOLDS are used.
                    Values must be between 0.0 and 1.0, and warning < critical.
        force_free: If True, always trigger the memory freeing process,
                    regardless of the current usage ratio.

    Returns:
        An integer representing the determined memory pressure level:
        - SAFE_LEVEL (0): Usage is below the 'warning' threshold.
        - WARNING_LEVEL (1): Usage is at or above 'warning' but below 'critical'.
        - CRITICAL_LEVEL (2): Usage is at or above the 'critical' threshold.

    Raises:
        ValueError: If the provided threshold values are invalid (e.g., outside [0,1]
                    or warning >= critical).
    """
    # Establish effective thresholds, validating inputs
    effective_thresholds = DEFAULT_THRESHOLDS.copy()
    if thresholds is not None:
        for key, value in thresholds.items():
            if key in effective_thresholds:
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"Invalid threshold value for '{key}': {value}. Must be between 0.0 and 1.0."
                    )
                effective_thresholds[key] = value
            else:
                logger.warning(f"Ignoring unknown threshold key: '{key}'")
        # Validate relationship between thresholds after merging
        if effective_thresholds['warning'] >= effective_thresholds['critical']:
            raise ValueError(
                f"Invalid thresholds: 'warning' ({effective_thresholds['warning']:.2f}) "
                f"must be less than 'critical' ({effective_thresholds['critical']:.2f})."
            )

    # Get current memory usage ratio
    try:
        ratio = get_memory_usage_ratio(mempool)
    except Exception:
        # Error already logged in get_memory_usage_ratio
        # Assume critical state if ratio cannot be determined
        ratio = 1.0  # Force critical level

    # Determine memory pressure level
    level = SAFE_LEVEL
    if ratio >= effective_thresholds['critical']:
        level = CRITICAL_LEVEL
    elif ratio >= effective_thresholds['warning']:
        level = WARNING_LEVEL

    log_prefix = "AMM"  # Adaptive Memory Management prefix for logs
    logger.debug(
        f"{log_prefix} Check: Ratio={ratio:.3f}, Level={level} "
        f"(Thresholds: W={effective_thresholds['warning']:.2f}, C={effective_thresholds['critical']:.2f})"
    )

    # Log specific warnings if threshold is breached but not critical yet
    if level == WARNING_LEVEL:
        logger.warning(
            f"{log_prefix} WARNING memory pressure detected (Ratio={ratio:.3f}). "
            f"Usage exceeds threshold {effective_thresholds['warning']:.2f}."
        )

    # Decide whether to perform cleanup
    perform_cleanup = force_free or level >= CRITICAL_LEVEL

    if perform_cleanup:
        if level >= CRITICAL_LEVEL:
            logger.warning(
                f"{log_prefix} CRITICAL memory pressure (Ratio={ratio:.3f} >= "
                f"{effective_thresholds['critical']:.2f}). Forcing free_all_blocks."
            )
        else:  # force_free must be True
            logger.info(f"{log_prefix} Forcing free_all_blocks (force_free=True). Current Ratio={ratio:.3f}.")

        try:
            start_time = time.monotonic()
            # Free CuPy memory blocks
            mempool.free_all_blocks()
            # Run Python garbage collector to release references
            gc.collect()
            # Ensure GPU operations related to freeing are complete before proceeding
            # or measuring memory again. This helps get a more accurate "after" state.
            cp.cuda.Stream.null.synchronize()
            end_time = time.monotonic()

            # Measure and log memory state *after* cleanup
            ratio_after = get_memory_usage_ratio(mempool)
            duration = end_time - start_time
            logger.info(
                f"{log_prefix} free_all_blocks completed in {duration:.3f}s. "
                f"Memory ratio after cleanup: {ratio_after:.3f}"
            )
        except Exception as e:
            logger.error(f"{log_prefix} Error during memory cleanup: {e}", exc_info=True)
            # Even if cleanup failed, the level remains critical or was forced
            # Return the determined level, but the state might be uncertain.
            # Consider re-raising if cleanup failure is fatal for the application.
            # raise e # Optional: re-raise the exception

    return level
