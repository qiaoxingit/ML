import multiprocessing
from multiprocessing.pool import AsyncResult
from typing import List


class WorkerPool:
    _instance = None

    def run_task(self, task):
        self.results.append(self.workers.apply_async(func=task))
    
    def wait_for_all(self):
        for result in self.results:
            result.wait()

    def __init__(self):
        self.workers = multiprocessing.Pool(16)
        self.results: List[AsyncResult] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance