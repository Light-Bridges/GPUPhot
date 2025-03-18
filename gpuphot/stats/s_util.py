import cupy as cp

from . import common
from ..logger.hierarchical_logging import setup_logger, hierarchical_debug

logger = setup_logger(__name__)


### # @hierarchical_debug(logger)
def judge_dtype(dtype):
    """
    Validate and return a floating point dtype.

    :param dtype: The input dtype to be validated.
    :type dtype: None or dtype-like object
    :return: A valid floating point dtype.
    :rtype: cupy.dtype
    :raises TypeError: If the input dtype is not a floating point type.
    """
    if dtype is None:
        dtype = common.default_dtype

    dtype = cp.dtype(dtype)

    if dtype.kind == 'f':
        return dtype
    else:
        raise TypeError('dtype must be floating point')
