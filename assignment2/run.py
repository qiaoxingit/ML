import flip_flop, four_peaks, travel_salesman,nn
import os
if __name__ == "__main__":
    if not os.path.exists('images'):
        os.makedirs('images')

    flip_flop.run_flip_flop()
    four_peaks.run_four_peaks()
    travel_salesman.run_travel_salesman()
    nn.run_varied_algorithms()
