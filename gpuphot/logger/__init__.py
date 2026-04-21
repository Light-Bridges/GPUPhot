# SPDX-License-Identifier: MIT
"""
Light-weight logging utilities used by the GPUPhot project.

This package exposes the `hierarchical_logging` utilities which provide a
singleton logger factory, an IndentFormatter to keep readable nested logs,
and the `hierarchical_debug` decorator that captures function arguments,
execution time and structured system information for debugging and telemetry.
"""
from . import hierarchical_logging

__all__ = ['hierarchical_logging']
