# SPDX-License-Identifier: MIT
"""
Simple executor wrapper to run functions with a timeout using a separate process pool.

Used by catalog query helpers to enforce external network/time-consuming calls to
respect a maximum wall-time.
"""

import multiprocess as mp


class TimeoutExecutor:
    class TimeoutError(Exception):
        pass

    def __init__(self, timeout=5):
        self.timeout = timeout if timeout is not None and timeout > 0 else None

    def execute(self, func, *args, **kwargs):
        with mp.Pool(processes=1) as pool:
            result = pool.apply_async(func, args=args, kwds=kwargs)
            try:
                actual_timeout = self.timeout if self.timeout > 0 else None
                return result.get(timeout=actual_timeout)
            except mp.TimeoutError:
                raise self.TimeoutError(f"Function {func.__name__} exceeded {self.timeout}s")
            except Exception as e:
                raise e
