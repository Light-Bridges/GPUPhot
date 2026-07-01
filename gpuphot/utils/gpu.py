# SPDX-License-Identifier: MIT
"""
GPU helper utilities for gpuphot.

This module centralizes GPU memory management helpers and utility functions
used across the project. It provides functions to inspect memory usage,
free memory pools, adaptively manage memory pressure, and safely offload
or load arrays between host (CPU) and device (GPU).

All changes in this file are documentation-only: comments and log messages
are expressed in English. No behavioral changes were made.
"""

import gc
import time
from typing import Dict, Optional

import cupy as cp
import nvtx

from ..exceptions import capture_cuda_exception

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
    """
    Convert a byte count to a human-readable string (e.g. ``1.5 GiB``).

    :param bytes_size: Size in bytes.
    :type bytes_size: int or float
    :return: Human-readable size string.
    :rtype: str
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(bytes_size) < 1024.0:
            return f"{bytes_size:.1f}{unit}B"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}YiB"


@nvtx.annotate('free_gpu_mem', category='utils.gpu')
def free_gpu_mem() -> None:
    """
    Free GPU memory by releasing all blocks in CuPy's default memory pools.

    This function frees both the default pooled memory and the pinned memory
    pool used by CuPy, then runs Python garbage collection to remove any
    remaining Python references to GPU arrays.
    """
    # Get memory pools
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    # Free all memory blocks
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    # Force garbage collection
    gc.collect()


@nvtx.annotate('force_free_gpu_memory', category='utils.gpu')
def force_free_gpu_memory():
    """
    Aggressively free GPU memory and synchronize CUDA streams.

    This function attempts to clear all memory pools, synchronize the default
    stream and clear CuPy's FFT plan cache. Use with care: this can be
    disruptive if executed while other GPU work is in-flight.
    """
    free_gpu_mem()

    # Synchronize CUDA streams to ensure all kernels are finished
    cp.cuda.Stream.null.synchronize()

    # Ensure pools are cleared
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    # Clear kernel/plan caches
    try:
        cp.fft.config.get_plan_cache().clear()
    except Exception:
        # Some CuPy versions may not expose plan cache; ignore failure
        pass


@nvtx.annotate('use_gpu', category='utils.gpu')
def use_gpu(gpu_id=0):
    """
    Select a GPU device to be used by CuPy.

    If the requested device is not available, device 0 is selected as fallback.
    """
    num_gpus = cp.cuda.runtime.getDeviceCount()
    if gpu_id < num_gpus:
        cp.cuda.Device(gpu_id).use()
    else:
        cp.cuda.Device(0).use()
        logger.warning(f"GPU {gpu_id} not found. Using GPU 0 by default.")


@nvtx.annotate('init_gpu', category='utils.gpu')
def init_gpu(**kwargs) -> None:
    """
    Initialize and log GPU information using CuPy.

    Logs CuPy version, number of GPUs, and per-GPU properties. Kept lightweight
    to avoid heavy runtime side-effects.
    """
    logger.debug('CuPy version ' + cp.__version__)

    num_gpus = cp.cuda.runtime.getDeviceCount()
    logger.debug(f'Number of GPUs: {num_gpus}')

    for i in range(num_gpus):
        gpu_props = cp.cuda.runtime.getDeviceProperties(i)
        logger.debug(f'GPU {i}: {gpu_props["name"]}')
        logger.debug(f'  Total memory: {gpu_props["totalGlobalMem"] / (1024 ** 3):.2f} GB')


@capture_cuda_exception
@nvtx.annotate('reset_cupy_allocators', category='utils.gpu')
def reset_cupy_allocators():
    """
    Reset CuPy allocators by creating fresh memory pools and setting them as default.
    """
    # Clear existing pools
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    # Create new memory pools and set allocators
    mempool = cp.cuda.MemoryPool()
    pinned_mempool = cp.cuda.PinnedMemoryPool()

    cp.cuda.set_allocator(mempool.malloc)
    cp.cuda.set_pinned_memory_allocator(pinned_mempool.malloc)

    gc.collect()
    cp.cuda.Stream.null.synchronize()

    try:
        cp.fft.config.get_plan_cache().clear()
    except Exception:
        pass


@nvtx.annotate('maybe_free_arrays', category='utils.gpu')
def maybe_free_arrays(arrays, mempool, threshold=0.2):
    """
    Free provided arrays if free GPU memory drops below a specified threshold.
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
        return used / total if total > 0 else 0.0
    except Exception as e:
        logger.error(f"Error getting memory usage ratio: {e}", exc_info=True)
        return 1.0


@nvtx.annotate('get_memory_stats', category='utils.gpu')
def get_memory_stats(mempool: cp.cuda.MemoryPool) -> Dict[str, float]:
    """
    Retrieve memory statistics from CuPy pool and the CUDA device.

    Returns a dictionary with pool and device usage and ratios.
    """
    try:
        pool_used = mempool.used_bytes()
        pool_total = mempool.total_bytes()
        pool_ratio = pool_used / pool_total if pool_total > 0 else 0.0

        device_free, device_total = cp.cuda.Device().mem_info
        device_used = device_total - device_free
        device_ratio = device_used / device_total if device_total > 0 else 0.0

        return {
            "pool_used_bytes": pool_used,
            "pool_total_bytes": pool_total,
            "pool_ratio": pool_ratio,
            "device_free_bytes": device_free,
            "device_total_bytes": device_total,
            "device_used_bytes": device_used,
            "device_ratio": device_ratio,
        }
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}", exc_info=True)
        return {
            "pool_ratio": 1.0,
            "device_ratio": 1.0,
        }


@nvtx.annotate('adaptive_memory_management', category='utils.gpu')
def adaptive_memory_management(
        mempool: cp.cuda.MemoryPool,
        thresholds: Optional[Dict[str, float]] = None,
        force_free: bool = False
) -> int:
    """
    Check GPU memory usage against thresholds and free memory if necessary.

    The function monitors both the CuPy memory pool and the device memory and
    uses the worst-case ratio to decide whether to log a warning or force a
    cleanup.

    :param mempool: CuPy memory pool to inspect.
    :type mempool: cp.cuda.MemoryPool
    :param thresholds: Optional custom thresholds for ``'warning'`` and ``'critical'`` keys.
    :type thresholds: dict, optional
    :param force_free: If ``True``, always attempt to free memory regardless of usage.
    :type force_free: bool
    :return: Memory pressure level — ``0`` (safe), ``1`` (warning), ``2`` (critical).
    :rtype: int
    """
    effective_thresholds = DEFAULT_THRESHOLDS.copy()
    if thresholds is not None:
        for key, value in thresholds.items():
            if key in effective_thresholds:
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"Invalid threshold value for '{key}': {value}. Must be between 0.0 and 1.0.")
                effective_thresholds[key] = value
        if effective_thresholds['warning'] >= effective_thresholds['critical']:
            raise ValueError(
                f"Invalid thresholds: 'warning' ({effective_thresholds['warning']:.2f}) "
                f"must be less than 'critical' ({effective_thresholds['critical']:.2f})."
            )

    stats = get_memory_stats(mempool)
    pool_ratio = stats.get("pool_ratio", 1.0)
    device_ratio = stats.get("device_ratio", 1.0)

    effective_ratio = max(pool_ratio, device_ratio)

    level = SAFE_LEVEL
    if effective_ratio >= effective_thresholds['critical']:
        level = CRITICAL_LEVEL
    elif effective_ratio >= effective_thresholds['warning']:
        level = WARNING_LEVEL

    logger.debug(
        f"Mem Check: Effective Ratio={effective_ratio:.3f}, Level={level} "
        f"(Device: {human_readable_size(stats.get('device_used_bytes', 0))}/{human_readable_size(stats.get('device_total_bytes', 0))} [{device_ratio:.3f}], "
        f"Pool: {human_readable_size(stats.get('pool_used_bytes', 0))}/{human_readable_size(stats.get('pool_total_bytes', 0))} [{pool_ratio:.3f}])"
    )

    perform_cleanup = force_free or level >= CRITICAL_LEVEL

    if perform_cleanup:
        if level >= CRITICAL_LEVEL:
            logger.warning(
                f"CRITICAL MEMORY PRESSURE (Ratio={effective_ratio:.3f} >= "
                f"{effective_thresholds['critical']:.2f}). Forcing free_all_blocks."
            )
        else:
            logger.debug(f"Forcing free_all_blocks (force_free=True). Current ratio={effective_ratio:.3f}.")

        try:
            start_time = time.monotonic()
            mempool.free_all_blocks()
            gc.collect()
            cp.cuda.Stream.null.synchronize()
            end_time = time.monotonic()

            stats_after = get_memory_stats(mempool)
            duration = end_time - start_time
            logger.debug(
                f"free_all_blocks completed in {duration:.3f}s. "
                f"Device ratio after cleanup: {stats_after.get('device_ratio', -1.0):.3f}"
            )
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}", exc_info=True)

    return level


@nvtx.annotate('check_memory_availability', category='utils.gpu')
def check_memory_availability(required_bytes: int, safety_margin: float = 0.10) -> bool:
    """
    Check if enough free device memory exists for a requested allocation.

    :param required_bytes: Number of bytes required for the allocation.
    :type required_bytes: int
    :param safety_margin: Fractional safety margin to keep free (e.g. ``0.10`` = 10 %).
    :type safety_margin: float
    :return: ``True`` if the allocation is likely to fit, ``False`` otherwise.
    :rtype: bool
    """
    try:
        free_mem, total_mem = cp.cuda.Device().mem_info

        available_for_alloc = free_mem - (total_mem * safety_margin)

        if required_bytes < available_for_alloc:
            logger.debug(
                f"Memory available for allocation of {human_readable_size(required_bytes)}. "
                f"(Free: {human_readable_size(free_mem)}, Required: {human_readable_size(required_bytes)})"
            )
            return True
        else:
            logger.warning(
                f"INSUFFICIENT MEMORY for allocation of {human_readable_size(required_bytes)}. "
                f"(Available with margin: {human_readable_size(available_for_alloc)}, Required: {human_readable_size(required_bytes)})"
            )
            return False
    except Exception as e:
        logger.error(f"Failed to check memory availability: {e}", exc_info=True)
        return False


@nvtx.annotate('offload_to_cpu', category='mem_mgmt')
def offload_to_cpu(var_name: str, data_dict: dict, data_location: dict, mempool: cp.cuda.MemoryPool):
    """Move a CuPy array to CPU (if present on GPU) and update state.

    On success, the function creates a new key `<var_name>_cpu` in `data_dict`
    that contains the host-side NumPy array and updates `data_location[var_name]`
    to the string 'CPU'. The original GPU array is deleted to free device memory.
    """
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
    """Load a CPU-side array back to GPU (if exists) and update state."""
    cpu_var_name = var_name + '_cpu'
    if data_location.get(var_name) == 'CPU' and cpu_var_name in data_dict:
        if data_dict[cpu_var_name] is None:  # Check if None
            logger.warning(f"Attempted to load '{var_name}' from CPU, but its value was None.")
            data_location[var_name] = 'None'  # Mark as None
            return False
        logger.info(f"Attempting to load '{var_name}' from CPU back to GPU.")
        level_before_load = adaptive_memory_management(mempool,
                                                       force_free=False)  # Pass thresholds if custom

        if level_before_load < CRITICAL_LEVEL or force_if_critical:
            if level_before_load >= WARNING_LEVEL:
                logger.warning(f"Loading '{var_name}' to GPU while memory is at WARNING level.")
            try:
                nvtx_range = nvtx.start_range(f'load_{var_name}', category='mem_mgmt')
                size_gb = data_dict[cpu_var_name].nbytes / (1024 ** 3)
                logger.debug(f"Loading '{var_name}' (Size: {size_gb:.3f} GB)")
                start_time = time.monotonic()
                data_dict[var_name] = cp.asarray(data_dict[cpu_var_name])
                del data_dict[cpu_var_name]  # Remove CPU copy
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
    Ensure a variable is present on the GPU and return it.

    If the variable was offloaded to CPU, attempt to load it back to GPU. Raises
    MemoryError or RuntimeError if the operation fails.
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
    else:  # error states
        raise RuntimeError(f"Variable '{var_name}' needed but is in unexpected/error state: {loc}")


@nvtx.annotate('cleanup_cupy', category='utils.gpu')
def cleanup_cupy(*args):
    """Deletes CuPy arrays passed as arguments and runs GC and free_all_blocks."""
    mempool = cp.get_default_memory_pool()
    for arr in args:
        del arr
    # Note: free_all_blocks and GC are intentionally left commented; caller may choose when to force.
    maybe_free_arrays([], mempool)  # Use existing helper if preferred
