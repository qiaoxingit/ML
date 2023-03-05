import multiprocessing
from multiprocessing.pool import AsyncResult
from typing import List


class WorkerPool:
    _instance = None

    def run_task(self, task, args=(), kwds={}, callback=None, error_callback=None) -> AsyncResult:
        async_result = self.workers.apply_async(func=task, args=args, kwds=kwds, callback=callback, error_callback=error_callback)
        self.results.append(async_result)
        return async_result
    
    def wait_for_all(self):
        for result in self.results:
            result.wait()

    def __init__(self):
        self.workers = multiprocessing.Pool()
        self.results: List[AsyncResult] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance