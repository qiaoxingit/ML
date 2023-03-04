import os

import flip_flop
import four_peaks
import nn
import travel_salesman
from worker_pool import WorkerPool

if __name__ == "__main__":
    if not os.path.exists('images'):
        os.makedirs('images')
    
    workers = WorkerPool()

    flip_flop.run_flip_flop()
    four_peaks.run_four_peaks()
    travel_salesman.run_travel_salesman()

    workers.run_task(nn.run_varied_algorithms)

    workers.wait_for_all()