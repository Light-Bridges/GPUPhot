import cupy as cp

from . import common
from ..logger.hierarchical_logging import setup_logger, hierarchical_debug

logger = setup_logger(__name__)

@hierarchical_debug(logger)
def judge_dtype(dtype):
    if dtype is None:
        dtype = common.default_dtype

    dtype = cp.dtype(dtype)

    if dtype.kind == 'f':
        return dtype
    else:
        raise TypeError('dtype must be floating point')
