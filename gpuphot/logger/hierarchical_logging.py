import functools
import inspect
import logging
import os
import threading
import time
import traceback

import cupy as cp
import numpy as np
from astropy.io.fits import Header

# Try to import Logstash handlers, but don't fail if not available
try:
    from logstash_async.handler import AsynchronousLogstashHandler
    from logstash_async.formatter import LogstashFormatter
except ImportError:
    pass

class IndentFormatter(logging.Formatter):
    """Custom formatter to add indentation to log messages."""
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.indent_levels = {}

    def format(self, record):
        """

        :param record: 

        """
        thread_id = threading.get_ident()
        indent = self.indent_levels.get(thread_id, 0)
        record.indent = '  ' * indent
        return super().format(record)

def format_arg(arg):
    """Format function arguments for logging.

    :param arg: 

    """
    if isinstance(arg, (np.ndarray, cp.ndarray)):
        return f"{type(arg).__name__}(shape={arg.shape}, dtype={arg.dtype})"
    elif isinstance(arg, (list, tuple)):
        if len(arg) > 3:
            return f"{type(arg).__name__}(len={len(arg)}, [{arg[0]!r}, ..., {arg[-1]!r}])"
        else:
            return f"{arg!r}"
    elif isinstance(arg, Header):
        keys = list(arg.keys())
        if len(keys) > 5:
            key_summary = f"{keys[:3]} ... {keys[-2:]} (total: {len(keys)})"
        else:
            key_summary = keys
        return f"Header(keys={key_summary})"
    elif hasattr(arg, 'shape') and hasattr(arg, 'dtype'):
        return f"{type(arg).__name__}(shape={arg.shape}, dtype={arg.dtype})"
    else:
        return f"{arg!r}:{type(arg).__name__}"

def hierarchical_debug(logger):
    """Decorator for hierarchical debugging and exception handling.

    :param logger: 

    """
    def decorator(func):
        """

        :param func: 

        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """

            :param *args: 
            :param **kwargs: 

            """
            thread_id = threading.get_ident()
            indent_levels = logger.handlers[0].formatter.indent_levels
            current_level = indent_levels.get(thread_id, 0)

            indent_levels[thread_id] = current_level + 1

            # Format function arguments for logging
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
            logger.debug(f'Starting function {func.__name__}({arg_str})')

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Exception in {func.__name__}: {str(e)}\n{tb}",
                             extra={
                                 'exception': str(e),
                                 'traceback': tb,
                                 'function': func.__name__
                             })
                raise
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                logger.debug(
                    f'Finishing function {func.__name__} - Execution time: {execution_time:.6f} seconds')
                indent_levels[thread_id] = max(0, current_level)

        return wrapper
    return decorator

def setup_logstash_handler(logger):
    """Set up Logstash handler if environment variables are set.

    :param logger: 

    """
    logstash_host = os.environ.get('LOGSTASH_HOST')
    logstash_port = os.environ.get('LOGSTASH_PORT', 5000)

    if logstash_host and logstash_port:
        logstash_handler = AsynchronousLogstashHandler(
            logstash_host,
            int(logstash_port),
            database_path=None,
            transport='logstash_async.transport.TcpTransport',
            level=logging.DEBUG
        )
        formatter = LogstashFormatter(
            message_type='gpuphot',
            extra_prefix='extra',
            extra={"index_name": "gpuphot"}
        )
        logstash_handler.setFormatter(formatter)
        logger.addHandler(logstash_handler)

def setup_logger(name):
    """Set up logger with custom formatter and Logstash handler.

    :param name: 

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = IndentFormatter('%(asctime)s - %(levelname)s - %(indent)s%(message)s')
    handler.setFormatter(formatter)

    logger.handlers = [handler]

    setup_logstash_handler(logger)

    return logger

# Example usage
if __name__ == "__main__":
    logger = setup_logger(__name__)

    @hierarchical_debug(logger)
    def example_function(a, b):
        """

        :param a: param b:
        :param b: 

        """
        return a / b

    try:
        example_function(10, 0)
    except Exception as e:
        print(f"Caught an exception: {e}")