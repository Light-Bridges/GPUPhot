import multiprocessing


class TimeoutExecutor:
    class TimeoutError(Exception):
        pass

    def __init__(self, timeout=5):
        self.timeout = timeout

    def execute(self, func, *args, **kwargs):
        def wrapper(queue):
            try:
                result = func(*args, **kwargs)
                queue.put(result)
            except Exception as e:
                queue.put(e)

        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=wrapper, args=(queue,))
        process.start()
        process.join(self.timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            raise self.TimeoutError(f"Function {func.__name__} exc {self.timeout}s")

        if not queue.empty():
            result = queue.get()
            if isinstance(result, Exception):
                raise result
            return result
        else:
            raise Exception("No result returned")
