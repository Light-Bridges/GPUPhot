# SPDX-License-Identifier: MIT
"""
Optional runtime patch to make common CuPy array creators use float64 by default.

This module wraps a small set of CuPy array creation functions (array, zeros,
empty, zeros_like, empty_like) and ensures that when no dtype is provided the
default dtype becomes cp.float64. This is a non-invasive, optional patch and
is applied on import. Keep in mind that changing default dtype can increase
GPU memory usage.
"""

import cupy as cp
from functools import wraps
import inspect

CRITICAL_FUNCTIONS = ['array', 'zeros', 'empty', 'zeros_like', 'empty_like']


def apply_float64_patch():
    _originals = {}

    def wrap_array_creator(original):
        # Inspect signature of original function to find 'dtype' position
        sig = inspect.signature(original)
        params = list(sig.parameters.values())

        dtype_pos = None
        for i, p in enumerate(params):
            if p.name == 'dtype':
                dtype_pos = i
                break

        @wraps(original)
        def wrapped(*args, **kwargs):
            # Determine if 'dtype' is provided positionally or as keyword
            dtype_provided = (
                    (dtype_pos is not None and len(args) > dtype_pos) or
                    'dtype' in kwargs
            )

            # Add dtype=cp.float64 only if not provided
            if not dtype_provided:
                kwargs['dtype'] = cp.float64

            return original(*args, **kwargs)

        return wrapped

    for func_name in CRITICAL_FUNCTIONS:
        original = getattr(cp, func_name)
        _originals[func_name] = original
        setattr(cp, func_name, wrap_array_creator(original))

    return _originals


# Apply the patch at import time
_original_cupy_funcs = apply_float64_patch()
