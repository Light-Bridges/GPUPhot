# SPDX-License-Identifier: MIT
"""
Simple executor wrapper to run functions with a timeout using a separate process pool.

Used by catalog query helpers to enforce external network/time-consuming calls to
respect a maximum wall-time.
"""

import multiprocess as mp


class TimeoutExecutor:
    """
    Run a callable in a separate process and enforce a wall-time limit.

    Used by catalog query helpers to prevent slow or unresponsive network
    calls from blocking the pipeline indefinitely.

    :param timeout: Maximum seconds to wait for the callable to complete.
        ``None`` or ``0`` disables the timeout.
    :type timeout: float or None
    """

    class TimeoutError(Exception):
        """Raised when the callable does not complete within the allowed time."""
        pass

    def __init__(self, timeout=5):
        self.timeout = timeout if timeout is not None and timeout > 0 else None

    def execute(self, func, *args, **kwargs):
        """
        Call *func* with the given arguments, raising ``TimeoutError`` if it
        exceeds the configured timeout.

        :param func: Callable to execute.
        :param args: Positional arguments forwarded to *func*.
        :param kwargs: Keyword arguments forwarded to *func*.
        :return: Return value of *func*.
        :raises TimeoutExecutor.TimeoutError: If the call exceeds the timeout.
        """
        with mp.Pool(processes=1) as pool:
            result = pool.apply_async(func, args=args, kwds=kwargs)
            try:
                actual_timeout = self.timeout if self.timeout > 0 else None
                return result.get(timeout=actual_timeout)
            except mp.TimeoutError:
                raise self.TimeoutError(f"Function {func.__name__} exceeded {self.timeout}s")
            except Exception as e:
                raise e
