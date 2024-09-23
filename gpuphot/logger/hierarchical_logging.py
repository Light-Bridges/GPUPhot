import functools
import inspect
import logging
import threading
import time

import cupy as cp
import numpy as np
from astropy.io.fits import Header


class IndentFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.indent_levels = {}

    def format(self, record):
        thread_id = threading.get_ident()
        indent = self.indent_levels.get(thread_id, 0)
        record.indent = '  ' * indent
        return super().format(record)


def format_arg(arg):
    if isinstance(arg, (np.ndarray, cp.ndarray)):
        return f"{type(arg).__name__}(shape={arg.shape}, dtype={arg.dtype})"
    elif isinstance(arg, (list, tuple)):
        if len(arg) > 3:
            return f"{type(arg).__name__}(len={len(arg)}, [{arg[0]!r}, ..., {arg[-1]!r}])"
        else:
            return f"{arg!r}"
    elif isinstance(arg, Header):
        # Mostrar un resumen del Header
        keys = list(arg.keys())
        if len(keys) > 5:
            key_summary = f"{keys[:3]} ... {keys[-2:]} (total: {len(keys)})"
        else:
            key_summary = keys
        return f"Header(keys={key_summary})"
    elif hasattr(arg, 'shape') and hasattr(arg, 'dtype'):
        # Para manejar otros tipos de arrays similares
        return f"{type(arg).__name__}(shape={arg.shape}, dtype={arg.dtype})"
    else:
        return f"{arg!r}:{type(arg).__name__}"


def hierarchical_debug(logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            thread_id = threading.get_ident()
            indent_levels = logger.handlers[0].formatter.indent_levels
            current_level = indent_levels.get(thread_id, 0)

            indent_levels[thread_id] = current_level + 1

            # Obtener información sobre los parámetros
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            arg_info = []

            for param_name, param in sig.parameters.items():
                if param_name in bound_args.arguments:
                    arg = bound_args.arguments[param_name]
                    arg_info.append(f"{param_name}={format_arg(arg)}")
                elif param.default is not param.empty:
                    arg_info.append(f"{param_name}={format_arg(param.default)}")
                else:
                    arg_info.append(f"{param_name}")

            arg_str = ", ".join(arg_info)
            logger.debug(f'Iniciando función {func.__name__}({arg_str})')

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                logger.debug(
                    f'Finalizando función {func.__name__} - Tiempo de ejecución: {execution_time:.6f} segundos')
                indent_levels[thread_id] = max(0, current_level)

        return wrapper

    return decorator


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = IndentFormatter('%(asctime)s - %(levelname)s - %(indent)s%(message)s')
    handler.setFormatter(formatter)

    logger.handlers = [handler]
    return logger
