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
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# Try to import Logstash handlers, but don't fail if not available
try:
    from logstash_async.handler import AsynchronousLogstashHandler
    from logstash_async.formatter import LogstashFormatter
except ImportError:
    pass

load_dotenv()

INDEX_NAME = "gpuphot"


class IndentFormatter(logging.Formatter):
    """Custom formatter to add indentation to log messages."""

    def __init__(self, fmt=None, datefmt=None):
        """
        Initialize the IndentFormatter.

        :param fmt: Format string for the log message
        :type fmt: str
        :param datefmt: Format string for the date/time
        :type datefmt: str
        """
        super().__init__(fmt, datefmt)
        self.indent_levels = {}

    def format(self, record):
        """
        Format the specified record as text.

        :param record: A LogRecord instance
        :type record: logging.LogRecord
        :return: Formatted log record
        :rtype: str
        """
        thread_id = threading.get_ident()
        indent = self.indent_levels.get(thread_id, 0)
        record.indent = ' ' * indent
        return super().format(record)


def format_arg(arg):
    """
    Format function arguments for logging.

    :param arg: The argument to format
    :type arg: Any
    :return: Formatted string representation of the argument
    :rtype: str
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
    """
    Decorator for hierarchical debugging and exception handling.

    :param logger: Logger instance to use for logging
    :type logger: logging.Logger
    :return: Decorator function
    :rtype: function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapper function that adds logging and exception handling.

            :param args: Positional arguments of the decorated function
            :param kwargs: Keyword arguments of the decorated function
            :return: Result of the decorated function
            :raises: Any exception raised by the decorated function
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
    """
    Set up Logstash handler if environment variables are set.

    :param logger: Logger instance to add the Logstash handler to
    :type logger: logging.Logger
    """
    logstash_host = os.environ.get('LOGSTASH_HOST', 'localhost')
    logstash_port = int(os.environ.get('LOGSTASH_PORT', 5000))

    if os.environ.get('LOGSTASH_LOGGING', 'True').lower() == 'true':
        try:
            formatter = LogstashFormatter(
                message_type=INDEX_NAME,
                extra_prefix='extra',
                extra={
                    "index_name": INDEX_NAME,
                    "environment": os.environ.get('ENVIRONMENT', 'production')
                }
            )

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
                formatter=formatter,
                level=logging.DEBUG
            )

            logger.addHandler(logstash_handler)
            logger.debug(f"Logstash handler configured successfully for {logstash_host}:{logstash_port}")

            # Configurar Elasticsearch si está habilitado en las variables de entorno
            setup_elasticsearch(logger)

        except Exception as e:
            logger.error(f"Failed to set up Logstash handler: {str(e)}")
    else:
        logger.info("Logstash logging is disabled")


def setup_elasticsearch(logger):
    es_enabled = os.getenv('ELASTICSEARCH_ENABLED', 'False').lower() == 'true'
    if es_enabled:
        try:
            es_host = os.environ.get('ELASTICSEARCH_HOST', 'localhost')
            es_port = int(os.environ.get('ELASTICSEARCH_PORT', 9200))
            es_user = os.environ.get('ELASTICSEARCH_USER')
            es_password = os.environ.get('ELASTICSEARCH_PASSWORD')

            es_client = Elasticsearch(
                f"{es_host}:{es_port}",
                http_auth=(es_user, es_password) if es_user and es_password else None
            )

            # Aquí puedes añadir lógica para crear o verificar el índice INDEX_NAME
            # Por ejemplo:
            if not es_client.indices.exists(index="gpuphot"):
                es_client.indices.create(index="gpuphot")
                logger.debug(f"Created {INDEX_NAME} index in Elasticsearch")
            else:
                logger.debug(f"{INDEX_NAME} index already exists in Elasticsearch")

        except Exception as e:
            logger.error(f"Failed to set up Elasticsearch: {str(e)}")
    else:
        logger.info("Elasticsearch integration is disabled")


def setup_logger(name):
    """
    Set up logger with custom formatter and Logstash handler.

    :param name: Name of the logger
    :type name: str
    :return: Configured logger instance
    :rtype: logging.Logger
    """
    logger = logging.getLogger(name)

    if os.getenv('DEBUG', '').lower() == 'true':
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

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
        An example function to demonstrate the hierarchical_debug decorator.

        :param a: First parameter
        :type a: int
        :param b: Second parameter
        :type b: int
        :return: Result of division a/b
        :rtype: float
        :raises ZeroDivisionError: If b is zero
        """
        return a / b


    try:
        example_function(10, 0)
    except Exception as e:
        print(f"Caught an exception: {e}")
