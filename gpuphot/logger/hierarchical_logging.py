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

import GPUtil
import cupy as cp
import numpy as np
from astropy.io.fits import Header
from dotenv import load_dotenv

# Try to import Logstash handlers, but don't fail if not available
try:
    from logstash_async.handler import AsynchronousLogstashHandler
    from logstash_async.formatter import LogstashFormatter
    # from elasticsearch import Elasticsearch
except ImportError:
    pass

load_dotenv()

LOGGER_NAME = "gpuphot"


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
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Recopilar información del sistema
        self.system_info = {
            'cupy_version': cp.__version__,
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
            # print(f"Error al obtener la información de Git: {e}")
            return None

    def get_gpu_info(self):
        try:
            # Obtener información de las GPUs usando GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return [{
                    'id': gpu.id,
                    'name': gpu.name,
                    'driver_version': gpu.driver,
                    'memory_total': gpu.memoryTotal,
                    'memory_free': gpu.memoryFree,
                    'memory_used': gpu.memoryUsed,
                    'temperature': gpu.temperature,
                    'load': gpu.load,
                } for gpu in gpus]
            else:
                return [{'id': -1, 'name': "No GPUs found", 'error': None}]
        except Exception as e:
            return [{'id': -1, 'name': "Error retrieving GPU info", 'error': str(e)}]

    def refresh_gpu_info(self):
        # Método para actualizar solo la información de la GPU
        self.system_info['gpu'] = self.get_gpu_info()


def hierarchical_debug(logger_name):
    logger = SingletonLogger.get_logger(logger_name)
    system_info_instance = SystemInfo.get_instance()

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
                critical_logger.debug(f'Starting function {func.__name__}', extra={
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
                critical_logger.debug(f'Finishing function {func.__name__}', extra={
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
                critical_logger.error(f"Exception in {func.__name__}", extra={
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
                        critical_logger.handlers[0].flush()  # Asegúrate de que se envíen los logs.
                except Exception as e:
                    logger.debug(f"Error al enviar logs: {str(e)}")

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

            # Asignar el formateador al manejador
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

        if cls._initialized:
            return logger

        # Set the logging level based on the environment variable.
        if os.getenv('GPUPHOT_LOG_LEVEL', '').upper() == 'DEBUG':
            logger.setLevel(logging.DEBUG)
        elif os.getenv('GPUPHOT_LOG_LEVEL', '').upper() == 'INFO':
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.ERROR)  # Default to ERROR level.

        # Silenciar LogProcessingWorker
        log_processing_worker_logger = logging.getLogger("LogProcessingWorker")
        log_processing_worker_logger.setLevel(logging.CRITICAL)  # Ignorar logs menores a CRITICAL

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
