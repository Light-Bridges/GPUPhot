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
DEFAULT_THRESHOLDS = {'warning': 0.5, 'critical': 0.75}


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


@nvtx.annotate('offload_to_cpu', category='mem_mgmt')
def offload_to_cpu(var_name: str, data_dict: dict, data_location: dict, mempool: cp.cuda.MemoryPool):
    """Mueve un array CuPy a CPU (si existe en GPU) y actualiza el estado."""
    if data_location.get(var_name) == 'GPU' and var_name in data_dict:
        if data_dict[var_name] is None:  # Check if None before proceeding
            logger.warning(f"Attempted to offload '{var_name}', but it was None.")
            data_location[var_name] = 'None'  # Mark as None explicitly
            return False
        try:
            logger.info(f"Offloading '{var_name}' from GPU to CPU due to memory pressure.")
            nvtx_range = nvtx.start_range(f'offload_{var_name}', category='mem_mgmt')
            size_gb = data_dict[var_name].nbytes / (1024 ** 3)
            logger.debug(f"Offloading '{var_name}' (Size: {size_gb:.3f} GB)")
            start_time = time.monotonic()
            data_dict[var_name + '_cpu'] = data_dict[var_name].get()
            del data_dict[var_name]
            mempool.free_all_blocks()
            cp.cuda.Stream.null.synchronize()
            duration = time.monotonic() - start_time
            data_location[var_name] = 'CPU'
            logger.info(f"Successfully offloaded '{var_name}' in {duration:.3f}s.")
            nvtx.end_range(nvtx_range)
            return True
        except Exception as e:
            logger.error(f"Error offloading '{var_name}': {e}", exc_info=True)
            data_location[var_name] = 'GPU_OFFLOAD_FAILED'
            if var_name in data_dict: del data_dict[var_name]
            if var_name + '_cpu' in data_dict: del data_dict[var_name + '_cpu']
            mempool.free_all_blocks()
            cp.cuda.Stream.null.synchronize()
            return False
    elif var_name not in data_dict or data_location.get(var_name) != 'GPU':
        logger.debug(f"Variable '{var_name}' not on GPU or not found, skipping offload.")
        return False


@nvtx.annotate('load_to_gpu', category='mem_mgmt')
def load_to_gpu(var_name: str, data_dict: dict, data_location: dict, mempool: cp.cuda.MemoryPool,
                force_if_critical: bool = False):
    """Carga un array de CPU a GPU (si existe en CPU y es necesario) y actualiza estado."""
    cpu_var_name = var_name + '_cpu'
    if data_location.get(var_name) == 'CPU' and cpu_var_name in data_dict:
        if data_dict[cpu_var_name] is None:  # Check if None
            logger.warning(f"Attempted to load '{var_name}' from CPU, but its value was None.")
            data_location[var_name] = 'None'  # Mark as None
            return False
        logger.info(f"Attempting to load '{var_name}' from CPU back to GPU.")
        level_before_load = adaptive_memory_management(mempool,
                                                       force_free=False)  # Pasar thresholds si son personalizados

        if level_before_load < CRITICAL_LEVEL or force_if_critical:
            if level_before_load >= WARNING_LEVEL:
                logger.warning(f"Loading '{var_name}' to GPU while memory is at WARNING level.")
            try:
                nvtx_range = nvtx.start_range(f'load_{var_name}', category='mem_mgmt')
                size_gb = data_dict[cpu_var_name].nbytes / (1024 ** 3)
                logger.debug(f"Loading '{var_name}' (Size: {size_gb:.3f} GB)")
                start_time = time.monotonic()
                data_dict[var_name] = cp.asarray(data_dict[cpu_var_name])
                del data_dict[cpu_var_name]  # Borrar copia CPU
                duration = time.monotonic() - start_time
                data_location[var_name] = 'GPU'
                logger.info(f"Successfully loaded '{var_name}' to GPU in {duration:.3f}s.")
                nvtx.end_range(nvtx_range)
                return True
            except (MemoryError, cp.cuda.runtime.CUDARuntimeError) as e:
                logger.error(f"OOM Error loading '{var_name}' back to GPU: {e}", exc_info=True)
                data_location[var_name] = 'CPU_LOAD_FAILED'
                reset_cupy_allocators()
                return False
            except Exception as e:
                logger.error(f"Unexpected error loading '{var_name}': {e}", exc_info=True)
                data_location[var_name] = 'CPU_LOAD_FAILED'
                return False
        else:
            logger.error(f"CRITICAL memory pressure ({level_before_load}). Cannot load '{var_name}' back to GPU now.")
            return False
    elif cpu_var_name not in data_dict and data_location.get(var_name) == 'CPU':
        logger.error(f"Inconsistency: State for '{var_name}' is CPU, but '{cpu_var_name}' not found in data_dict.")
        data_location[var_name] = 'Unknown'
        return False
    elif data_location.get(var_name) != 'CPU':
        logger.debug(
            f"Variable '{var_name}' not on CPU (State: {data_location.get(var_name)}), skipping load function.")
        return False


@nvtx.annotate('get_gpu_var_reactive', category='mem_mgmt')
def get_gpu_var_reactive(var_name: str, data_dict: dict, data_location: dict, mempool: cp.cuda.MemoryPool,
                         force_load_if_critical: bool = False) -> cp.ndarray:
    """
    Obtiene una variable asegurándose de que esté en la GPU.
    Intenta cargarla desde CPU si fue descargada previamente.
    Lanza MemoryError o RuntimeError si no se puede obtener en GPU.
    """
    loc = data_location.get(var_name)

    if loc == 'GPU':
        if var_name in data_dict and data_dict[var_name] is not None:
            logger.debug(f"Accessing '{var_name}' directly from GPU.")
            return data_dict[var_name]
        else:
            raise RuntimeError(f"State inconsistency: Variable '{var_name}' marked as GPU but not found or is None.")
    elif loc == 'CPU':
        logger.info(f"Variable '{var_name}' is on CPU, attempting reactive load to GPU.")
        if load_to_gpu(var_name, data_dict, data_location, mempool, force_if_critical=force_load_if_critical):
            if var_name in data_dict and data_dict[var_name] is not None:
                logger.info(f"Reactive load successful for '{var_name}'.")
                return data_dict[var_name]
            else:
                raise RuntimeError(f"State inconsistency after successful load_to_gpu for '{var_name}'.")
        else:
            raise MemoryError(f"Failed to reactively load '{var_name}' from CPU back to GPU.")
    elif loc == 'None' or loc is None:
        raise RuntimeError(f"Variable '{var_name}' needed but its state is '{loc}' (deleted or never existed).")
    else:  # Estados de error
        raise RuntimeError(f"Variable '{var_name}' needed but is in unexpected/error state: {loc}")

@nvtx.annotate('cleanup_cupy', category='utils.gpu')
def cleanup_cupy(*args):
    """Deletes CuPy arrays passed as arguments and runs GC and free_all_blocks."""
    mempool = cp.get_default_memory_pool()
    for arr in args:
        del arr
    # gc.collect() # Colectar referencias Python
    # mempool.free_all_blocks() # Liberar bloques CuPy (hacer con cuidado)
    # cp.cuda.Stream.null.synchronize() # Esperar a que la GPU termine
    maybe_free_arrays([], mempool) # Usar tu función existente si prefieres
