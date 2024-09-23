import functools
import logging
import threading


class IndentFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.indent_levels = {}

    def format(self, record):
        thread_id = threading.get_ident()
        indent = self.indent_levels.get(thread_id, 0)
        record.indent = '  ' * indent
        return super().format(record)

def hierarchical_debug(logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            thread_id = threading.get_ident()
            indent_levels = logger.handlers[0].formatter.indent_levels
            current_level = indent_levels.get(thread_id, 0)

            indent_levels[thread_id] = current_level + 1
            logger.debug(f'Iniciando función {func.__name__}()')

            try:
                return func(*args, **kwargs)
            finally:
                logger.debug(f'Finalizando función {func.__name__}()')
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
