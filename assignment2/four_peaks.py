import mlrose_hiive as mlr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from mlrose_hiive.algorithms import random_hill_climb, simulated_annealing, genetic_alg, mimic
from mlrose_hiive import ExpDecay
from Optimals import SA_optimal, GA_pop_breed_pec_optimal, GA_popsize_optimal, mimic_keepPct_optimal, mimic_popsize_optimal, run_problems

fitness = mlr.FourPeaks(t_pct=0.1)
problem = mlr.DiscreteOpt(100, fitness_fn = fitness, maximize=True, max_val=2)
random_seeds = [2636 + 8 * i for i in range(6)]

def run_four_peaks():

    SA_optimal(problem = problem, decay_rates=np.arange(0.005, 0.2, 0.005),random_seeds=random_seeds, label = "Four Peaks SA ExpDecay rates", title = "Four Peaks SA ExpDecay rates optimization")

    GA_popsize_optimal(problem=problem, pop_sizes=np.arange(100, 2000, 100), random_seeds=random_seeds, label = "Four Peaks GA pop size", title = "Four Peaks GA pop size optimization")

    GA_pop_breed_pec_optimal(problem=problem, pop_breed_pecs=np.arange(0, 1, 0.05), random_seeds=random_seeds, label = "Four Peaks GA pop breed pec", title = "Four Peaks GA pop breed pec optimization")

    mimic_popsize_optimal(problem=problem, pop_sizes=np.arange(100, 2000, 100), random_seeds=random_seeds, label = "Four Peaks mimic pop size", title = "Four Peaks mimic pop size optimization")

    mimic_keepPct_optimal(problem=problem, keep_pcts=np.arange(0, 1, 0.05), random_seeds=random_seeds, label = "Four Peaks mimic keep pct", title = "Four Peaks mimic keep pct optimization")

    run_problems(problem=problem, rhc_max_attempts=8000, decay_rate=0.02, sa_max_attempts=8000, ga_max_attempts=300, GA_popSize=1000, GA_pop_breed_pec=0.1, mimic_max_attempts=300, mimic_popSize=1000, mimic_keepPct=0.2, random_seeds=random_seeds, title = "Four Peaks")


if __name__ == "__main__":

    run_four_peaks()