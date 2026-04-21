import atexit
import functools
import inspect
import logging
import os
import platform
import sys
import threading
import time
import traceback
from datetime import datetime

import cupy as cp
try:
    import cupynumeric as np
except ImportError:
    import numpy as np
except Exception:
    import numpy as np

from astropy.io.fits import Header
from dotenv import load_dotenv

# Try to import Logstash handlers, but don't fail if not available
try:
    from logstash_async.handler import AsynchronousLogstashHandler
    from logstash_async.formatter import LogstashFormatter
    # from elasticsearch import Elasticsearch
except ImportError:
    AsynchronousLogstashHandler = None
    LogstashFormatter = None

load_dotenv()

LOGGER_NAME = "gpuphot"

_NA_VALUES = ('[N/A]', '[Not Supported]', 'N/A', '')

def _safe_float(val):
    """Parse a float from nvidia-smi output, returning None on failure."""
    if val is None or val in _NA_VALUES:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def _safe_int(val):
    """Parse an int from nvidia-smi output, returning None on failure."""
    if val is None or val in _NA_VALUES:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None

def _safe_str(val):
    """Return a string from nvidia-smi output, or None for N/A values."""
    if val is None or val in _NA_VALUES:
        return None
    return str(val)

def _safe_percent(val):
    """Parse a utilization percentage (0-100) into a 0.0-1.0 float."""
    f = _safe_float(val)
    return f / 100.0 if f is not None else None


class IndentFormatter(logging.Formatter):
    """Custom formatter to add indentation to log messages."""

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.indent_levels = {}

    def format(self, record):
        thread_id = threading.get_ident()
        indent = self.indent_levels.get(thread_id, 0)
        record.indent = ' ' * indent
        return super().format(record)


class NotifyingHandler(logging.Handler):
    """Handler that forwards structured log data to a supplied callback.

    The callback receives a dictionary containing fields such as `message`,
    `level`, `timestamp` and optional custom extras (event, function_name,
    arguments, execution_time, exception, traceback).
    """

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        log_entry = {
            'message': record.getMessage(),
            'level': record.levelname,
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'logger_name': record.name
        }
        if hasattr(record, 'event'):
            log_entry['event'] = record.event
        if hasattr(record, 'function_name'):
            log_entry['function_name'] = record.function_name
        if hasattr(record, 'arguments'):
            log_entry['arguments'] = record.arguments
        if hasattr(record, 'execution_time'):
            log_entry['execution_time'] = record.execution_time
        if hasattr(record, 'exception'):
            log_entry['exception'] = record.exception
        if hasattr(record, 'traceback'):
            log_entry['traceback'] = record.traceback
        self.callback(log_entry)


def format_arg(arg):
    """Return a concise representation of function arguments for logging.

    Handles numpy/cupy arrays, astropy Header objects and common containers.
    """
    try:
        if isinstance(arg, (np.ndarray, cp.ndarray)):
            return f"{type(arg).__name__}(shape={arg.shape}, dtype={arg.dtype})"
        elif isinstance(arg, (list, tuple)):
            if len(arg) > 3:
                return f"{type(arg).__name__}(len={len(arg)}, [{arg[0]!r}, ..., {arg[-1]!r}])"
            else:
                return f"{arg!r}"
        elif isinstance(arg, Header):
            header_dict = {}
            for card in arg.cards:
                key = card.keyword
                value = card.value
                if key not in ['COMMENT', 'HISTORY']:
                    if key in header_dict:
                        if isinstance(header_dict[key], list):
                            header_dict[key].append(value)
                        else:
                            header_dict[key] = [header_dict[key], value]
                    else:
                        header_dict[key] = value

            for key, value in header_dict.items():
                if isinstance(value, list):
                    header_dict[key] = np.array(value)

            return f"Header({header_dict}), total: {len(header_dict)})"
        elif hasattr(arg, 'shape') and hasattr(arg, 'dtype'):
            return f"{type(arg).__name__}(shape={arg.shape}, dtype={arg.dtype})"
        else:
            return f"{arg!r}:{type(arg).__name__}"
    except Exception:
        return f"{str(arg)!r}:{type(arg).__name__}"


class SystemInfo:
    """Singleton collecting system metadata to include in structured logs.

    The instance caches values such as the Cupy version, OS, Python version,
    GPU information obtained via nvidia-smi and an attempt to read the local Git
    commit id for traceability.
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Collect system information
        self.system_info = {
            'cupy_version': getattr(cp, '__version__', 'n/a'),
            'os': platform.system(),
            'os_version': platform.version(),
            'kernel_version': platform.release(),
            'python_version': sys.version,
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'commit_info': self.get_git_commit_id(),
            'gpu': self.get_gpu_info()
        }

    def get_git_commit_id(self):
        try:
            script_dir = os.path.abspath(os.path.dirname(__file__))

            git_dir = os.path.join(script_dir, '..', '..', '.git')

            with open(os.path.join(git_dir, 'HEAD'), 'r') as head_file:
                head_content = head_file.read().strip()

            if head_content.startswith('ref:'):
                branch_ref = head_content.split(' ')[-1]
                branch_name = branch_ref.split('/')[-1]

                with open(os.path.join(git_dir, branch_ref), 'r') as branch_file:
                    commit_hash = branch_file.read().strip()
            else:
                commit_hash = head_content.split(' ')[-1]
                branch_name = None

            return {
                'commit_hash': commit_hash,
                'branch_name': branch_name,
            }

        except Exception as e:
            # Return None if Git information cannot be retrieved.
            return None

    # nvidia-smi fields queried in a single call (avoids duplicate subprocess calls)
    _NVML_QUERY_FIELDS = (
        'index', 'name', 'driver_version', 'uuid',
        'memory.total', 'memory.free', 'memory.used',
        'temperature.gpu', 'utilization.gpu',
        'power.draw', 'power.limit', 'power.max_limit', 'power.default_limit',
    )

    def get_gpu_info(self):
        """Query all GPU info in a single nvidia-smi call.

        Each field is handled individually so that a single [N/A] value
        does not discard the rest of the GPU data.
        """
        import subprocess
        try:
            result = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=' + ','.join(self._NVML_QUERY_FIELDS),
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return [{'id': -1, 'name': "No GPUs found", 'error': None}]

            gpus = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(', ')]
                raw = {}
                for i, field in enumerate(self._NVML_QUERY_FIELDS):
                    raw[field] = parts[i] if i < len(parts) else None

                info = {
                    'id':              _safe_int(raw.get('index')),
                    'name':            _safe_str(raw.get('name')),
                    'driver_version':  _safe_str(raw.get('driver_version')),
                    'uuid':            _safe_str(raw.get('uuid')),
                    'memory_total':    _safe_float(raw.get('memory.total')),
                    'memory_free':     _safe_float(raw.get('memory.free')),
                    'memory_used':     _safe_float(raw.get('memory.used')),
                    'temperature':     _safe_float(raw.get('temperature.gpu')),
                    'load':            _safe_percent(raw.get('utilization.gpu')),
                    'power_draw_w':    _safe_float(raw.get('power.draw')),
                    'power_limit_w':   _safe_float(raw.get('power.limit')),
                    'power_max_w':     _safe_float(raw.get('power.max_limit')),
                    'power_default_w': _safe_float(raw.get('power.default_limit')),
                }
                gpus.append(info)
            return gpus if gpus else [{'id': -1, 'name': "No GPUs found", 'error': None}]
        except Exception as e:
            return [{'id': -1, 'name': "Error retrieving GPU info", 'error': str(e)}]

    def refresh_gpu_info(self):
        # Update only the GPU information in the cached system info
        self.system_info['gpu'] = self.get_gpu_info()


def hierarchical_debug(logger_name):
    """
    Decorator factory that instruments a function with structured entry/exit logs.

    Wraps the decorated function so that every call emits a DEBUG log on entry
    (with a summary of arguments) and on exit (with elapsed time). Exceptions
    are logged automatically before being re-raised.

    :param logger_name: Logger name passed to :func:`setup_logger`.
    :type logger_name: str
    :return: Decorator that wraps a function with hierarchical debug logging.
    :rtype: callable
    """
    logger = SingletonLogger.get_logger(logger_name)
    system_info_instance = SystemInfo.get_instance()
    pid = os.getpid()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            thread_id = threading.get_ident()
            indent_levels = logger.handlers[0].formatter.indent_levels
            current_level = indent_levels.get(thread_id, 0)
            indent_levels[thread_id] = current_level + 1

            # Format function arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            arg_info = [f"{param_name}={format_arg(arg)}" for param_name, arg in bound_args.arguments.items()]
            arg_str = ", ".join(arg_info)

            # Create a separate logger for critical events that always goes to Logstash.
            critical_logger = logging.getLogger(LOGGER_NAME)
            critical_logger.setLevel(logging.DEBUG)  # Ensure this logger captures all levels.

            # Clear existing handlers to avoid showing logs on the console
            if critical_logger.handlers:
                critical_logger.handlers.clear()
            setup_logstash_handler(critical_logger)

            start_time = time.time()
            try:
                # Log the start of the function
                system_info_instance.refresh_gpu_info()
                critical_logger.debug(f'Starting function {func.__name__} (PID: {pid})', extra={
                    'function_name': func.__name__,
                    'arguments': arg_str,
                    'event': 'function_start',
                    'system_info': system_info_instance.system_info
                })

                # Call the function
                result = func(*args, **kwargs)
                return_info = format_arg(result)

                # Log successful completion
                execution_time = time.time() - start_time
                critical_logger.debug(f'Finishing function {func.__name__} (PID: {pid})', extra={
                    'function_name': func.__name__,
                    'arguments': arg_str,
                    'execution_time': execution_time,
                    'event': 'function_end',
                    'success': True,
                    'return_value': return_info,
                    'system_info': system_info_instance.system_info
                })

                return result
            except Exception as e:
                # Log the exception
                tb = traceback.format_exc()
                system_info_instance.refresh_gpu_info()
                critical_logger.error(f"Exception in {func.__name__} (PID: {pid})", extra={
                    'function_name': func.__name__,
                    'arguments': arg_str,
                    'exception': str(e),
                    'traceback': tb,
                    'event': 'function_exception',
                    'execution_time': time.time() - start_time,
                    'system_info': system_info_instance.system_info
                })
                raise  # Re-raise the exception after logging it.
            finally:
                try:
                    if isinstance(critical_logger.handlers[0], AsynchronousLogstashHandler):
                        critical_logger.handlers[0].flush()  # Ensure logs are sent.
                except Exception as e:
                    logger.debug(f"Error sending logs: {str(e)}")

                indent_levels[thread_id] = max(0, current_level)

        return wrapper

    return decorator


def setup_logstash_handler(logger):
    logstash_host = os.environ.get('LOGSTASH_HOST', 'localhost')
    logstash_port = int(os.environ.get('LOGSTASH_PORT', 5000))

    if os.environ.get('LOGSTASH_LOGGING', 'False').lower() == 'true':
        try:
            formatter = LogstashFormatter(
                extra_prefix='extra',
                extra={
                    "environment": os.environ.get('GPUPHOT_ENVIRONMENT', 'production'),
                    "application": "gpuphot"
                }
            )
            #
            # class DebugAsynchronousLogstashHandler(AsynchronousLogstashHandler):
            #     def emit(self, record):
            #         if self.formatter:
            #             formatted_message = self.formatter.format(record)
            #             print(f"Enviando log a Logstash: {formatted_message}")
            #         else:
            #             print(f"Enviando log a Logstash (sin formateador): {record.getMessage()}")
            #         super().emit(record)

            logstash_handler = AsynchronousLogstashHandler(
                host=logstash_host,
                port=logstash_port,
                database_path=None,
                transport='logstash_async.transport.TcpTransport',
                ssl_enable=False,
                ssl_verify=False,
                keyfile=None,
                certfile=None,
                ca_certs=None,
                level=logging.DEBUG
            )

            # Assign formatter to handler
            logstash_handler.setFormatter(formatter)

            logger.addHandler(logstash_handler)
            # logger.debug(f"Logstash handler configured successfully for {logstash_host}:{logstash_port}")

            # # Setup Elasticsearch if enabled in environment variables.
            # setup_elasticsearch(logger)

            return logstash_handler
        except Exception as e:
            logger.error(f"Failed to set up Logstash handler: {str(e)}")
    else:
        logger.debug("Logstash logging is disabled")

    return None


class SingletonLogger:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    @classmethod
    def get_logger(cls, name):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls._setup_logger(name)
        return cls._instance

    @classmethod
    def _setup_logger(cls, name):
        logger = logging.getLogger(name)
        logger.propagate = False

        if cls._initialized:
            return logger

        # Set the logging level based on the environment variable.
        if os.getenv('GPUPHOT_LOG_LEVEL', '').upper() == 'DEBUG':
            logger.setLevel(logging.DEBUG)
        elif os.getenv('GPUPHOT_LOG_LEVEL', '').upper() == 'INFO':
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.ERROR)  # Default to ERROR level.

        # Silence LogProcessingWorker
        log_processing_worker_logger = logging.getLogger("LogProcessingWorker")
        log_processing_worker_logger.setLevel(logging.CRITICAL)  # Ignore logs below CRITICAL

        # Configure StreamHandler.
        handler = logging.StreamHandler()
        formatter = IndentFormatter('%(asctime)s - %(levelname)s - %(indent)s%(message)s')
        handler.setFormatter(formatter)

        # Clear existing handlers and add new one.
        logger.handlers.clear()
        logger.addHandler(handler)

        # Setup Logstash handler.
        setup_logstash_handler(logger)

        def shutdown_logger(timeout=5):
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)

        atexit.register(shutdown_logger)

        cls._initialized = True
        return logger


def setup_logger(name):
    """
    Return the package-wide singleton logger instance for the given module name.

    This is the main entry point for obtaining a logger in GPUPhot. All
    returned loggers share the same underlying ``gpuphot`` logger with
    indented console formatting and optional Logstash output.

    :param name: Module name, typically ``__name__``.
    :type name: str
    :return: Configured logger instance.
    :rtype: logging.Logger
    """
    return SingletonLogger.get_logger(name)


# Example usage
if __name__ == "__main__":
    logger = setup_logger(__name__)


    ### # @hierarchical_debug(logger)
    def example_function(a, b):
        """
        An example function to demonstrate the hierarchical_debug decorator.
        :param a: First parameter
        :type a: int
        :param b: Second parameter
        :type b: int
        :return: Result of division a/b
        :rtype: float
        :raises ZeroDivisionError: If b is zero.
        """
        return a / b


    try:
        example_function(10, 0)
    except Exception as e:
        print(f"Caught an exception: {e}")

    example_function(100, 1)
    logger.debug("Finished running example function")
