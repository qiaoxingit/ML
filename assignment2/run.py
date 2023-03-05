import os

import flip_flop
import four_peaks
import travel_salesman
import worker_pool

if __name__ == "__main__":
    if not os.path.exists('images'):
        os.makedirs('images')

    flip_flop.run_flip_flop()
    four_peaks.run_four_peaks()
    travel_salesman.run_travel_salesman()

    worker_pool.workers.close()
    worker_pool.workers.join()