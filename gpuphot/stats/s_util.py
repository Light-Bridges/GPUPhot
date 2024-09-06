import gc
import logging

import cupy as cp

logger = logging.getLogger(__name__)

from . import common


def free_gpu_mem():
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


# def set_dtype(dtype):
#     """
#     Change the default dtype used in all functions
#     and classes in this package.
#     Parameters
#     ----------
#     dtype : str or dtype
#     """
#
#     common.default_dtype = judge_dtype(dtype).name


# def judge_dtype(dtype):
#     if dtype is None:
#         dtype = common.default_dtype
#
#     dtype = cp.dtype(dtype)
#
#     if dtype.kind == 'f':
#         return dtype
#     else:
#         raise TypeError('dtype must be floating point')

#
# def reduction(image, bias, dark, flat, out=None, dtype=None):
#     """
#     This function is equal to the equation:
#     out = (image - bias - dark) / flat, but needs less memory.
#     Therefore, each inputs must be broadcastable shape.
#     Parameters
#     ----------
#     image : ndarray
#     bias : ndarray
#     dark : ndarray
#     flat : ndarray
#     out : cupy.ndarray, default None
#         Alternate output array in which to place the result. The default
#         is ``None``; if provided, it must have the same shape as the
#         expected output, but the type will be cast if necessary.
#     dtype : str or dtype, default 'float32'
#         dtype of array used internally
#         If None, use eclair.common.default_dtype.
#         If the input dtype is different, use a casted copy.
#
#     Returns
#     -------
#     out : cupy.ndarray
#     """
#     dtype = judge_dtype(dtype)
#     asarray = lambda x: cp.asarray(x, dtype=dtype)
#
#     image = asarray(image)
#     bias = asarray(bias)
#     dark = asarray(dark)
#     flat = asarray(flat)
#
#     if out is None:
#         out = cp.empty(
#             cp.broadcast(image, bias, dark, flat).shape,
#             dtype=dtype
#         )
#
#     reduction_kernel(image, bias, dark, flat, out)
#
#     return out

#
# reduction_kernel = cp.ElementwiseKernel(
#     in_params='T x, T b, T d, T f',
#     out_params='F z',
#     operation='z = (x - b - d) / f',
#     name='reduction'
# )
#
# checkfinite = cp.ElementwiseKernel(
#     in_params='T x, T f',
#     out_params='T z',
#     operation='''
#         int flag = isfinite(x) & isfinite(f);
#         z = (flag ? f : 0);
#     ''',
#     name='checkfinite'
# )
#
# replace_kernel = cp.ElementwiseKernel(
#     in_params='T input, T before, T after',
#     out_params='T output',
#     operation='''
#         output = (
#             (input==before) ? after:input
#         )
#     ''',
#     name='replace'
# )
#
# ternary_operation = cp.ElementwiseKernel(
#     in_params='I condition, T t, T f',
#     out_params='T output',
#     operation='output = (condition ? t : f)',
#     name='ternary_operation'
# )
#
# elementwise_not = cp.ElementwiseKernel(
#     in_params='T m',
#     out_params='T f',
#     operation='f = (m==0)',
#     name='elementwise_not'
# )
