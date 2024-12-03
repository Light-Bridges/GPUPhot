import cupy as cp

from . import common
from ..logger.hierarchical_logging import setup_logger

logger = setup_logger(__name__)


def judge_dtype(dtype):
    if dtype is None:
        dtype = common.default_dtype

    dtype = cp.dtype(dtype)

    if dtype.kind == 'f':
        return dtype
    else:
        raise TypeError('dtype must be floating point')
