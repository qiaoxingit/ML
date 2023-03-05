import os

import flip_flop
import four_peaks
import travel_salesman
import nn
from helper import log

if __name__ == "__main__":
    if not os.path.exists('images'):
        os.makedirs('images')

    # log('start run_flip_flop')
    # flip_flop.run_flip_flop()

    # log('start run_four_peaks')
    # four_peaks.run_four_peaks()

    # log('start run_travel_salesman')
    # travel_salesman.run_travel_salesman()

    log('start nn')
    nn.run_varied_algorithms()