import multiprocess as mp


class TimeoutExecutor:
    class TimeoutError(Exception):
        pass

    def __init__(self, timeout=5):
        self.timeout = timeout

    def execute(self, func, *args, **kwargs):
        with mp.Pool(processes=1) as pool:
            result = pool.apply_async(func, args=args, kwds=kwargs)
            try:
                return result.get(timeout=self.timeout)
            except mp.TimeoutError:
                raise self.TimeoutError(f"Function {func.__name__} exceeded {self.timeout}s")
            except Exception as e:
                raise e
