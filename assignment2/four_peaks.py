import matplotlib.pyplot as plt
import mlrose_hiive as mlr
import numpy as np
import pandas as pd
from helper import log
from mlrose_hiive import ExpDecay
from mlrose_hiive.algorithms import (genetic_alg, mimic, random_hill_climb,
                                     simulated_annealing)
from Optimals import (GA_pop_breed_pec_optimal, GA_popsize_optimal, SA_optimal,
                      mimic_keepPct_optimal, mimic_popsize_optimal,
                      run_problems)

fitness = mlr.FourPeaks(t_pct=0.1)
problem = mlr.DiscreteOpt(100, fitness_fn = fitness, maximize=True, max_val=2)
problem.set_mimic_fast_mode = True

random_seeds = [2636 + 8 * i for i in range(3)]

def run_four_peaks():

    log('start to run four picks sa')
    SA_optimal(problem = problem, decay_rates=np.arange(0.005, 0.2, 0.005),random_seeds=random_seeds, label = "Four Peaks SA ExpDecay rates", title = "Four Peaks SA ExpDecay rates optimization")

    log('start to run four picks ga popsize')
    GA_popsize_optimal(problem=problem, pop_sizes=np.arange(100, 2000, 100), random_seeds=random_seeds, label = "Four Peaks GA pop size", title = "Four Peaks GA pop size optimization")

    log('start to run four picks ga breed')
    GA_pop_breed_pec_optimal(problem=problem, pop_breed_pecs=np.arange(0.05, 1, 0.1), random_seeds=random_seeds, label = "Four Peaks GA pop breed pec", title = "Four Peaks GA pop breed pec optimization")

    log('start to run four picks mimic popsize')
    mimic_popsize_optimal(problem=problem, pop_sizes=np.arange(100, 2000, 100), random_seeds=random_seeds, label = "Four Peaks mimic pop size", title = "Four Peaks mimic pop size optimization")

    log('start to run four picks mimic keepec')
    mimic_keepPct_optimal(problem=problem, keep_pcts=np.arange(0.05, 1, 0.1), random_seeds=random_seeds, label = "Four Peaks mimic keep pct", title = "Four Peaks mimic keep pct optimization")

    log('start to run run four picks')
    run_problems(problem=problem, rhc_max_attempts=8000, decay_rate=0.02, sa_max_attempts=8000, ga_max_attempts=300, GA_popSize=1000, GA_pop_breed_pec=0.1, mimic_max_attempts=300, mimic_popSize=1000, mimic_keepPct=0.2, random_seeds=random_seeds, title = "Four Peaks")


if __name__ == "__main__":

    run_four_peaks()