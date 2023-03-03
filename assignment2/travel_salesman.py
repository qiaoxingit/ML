import mlrose_hiive as mlr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from mlrose_hiive.algorithms import random_hill_climb, simulated_annealing, genetic_alg, mimic
from mlrose_hiive import ExpDecay
from Optimals import SA_optimal, GA_mutationProb_optimal, GA_popsize_optimal, mimic_keepPct_optimal, mimic_popsize_optimal, run_problems

dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426), 
             (0, 5, 5.3852), (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000), 
             (1, 3, 2.8284), (1, 4, 2.0000), (1, 5, 4.1231), (1, 6, 4.2426), 
             (1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361), (2, 5, 4.4721), 
             (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056), 
             (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623), 
             (4, 7, 2.2361), (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]
fitness = mlr.TravellingSales(distances=dist_list)
problem = mlr.TSPOpt(length = 8, fitness = fitness, maximize=False)
random_seeds = [2636 + 8 * i for i in range(6)]

def run_travel_salesman():

    SA_optimal(problem = problem, decay_rates=np.arange(0.005, 0.1, 0.005),random_seeds=random_seeds, label = "Travel Salesmen SA ExpDecay rates", title = "Travel Salesmen SA ExpDecay rates optimization")

    GA_popsize_optimal(problem=problem, pop_sizes=np.arange(100, 2000, 100), random_seeds=random_seeds, label = "Travel Salesmen GA pop size", title = "Travel Salesmen GA pop size optimization")

    GA_mutationProb_optimal(problem=problem, mutation_probs=np.arange(0, 1, 0.05), random_seeds=random_seeds, label = "fTravel Salesmen GA mutation prob", title = "Travel Salesmen GA mutation prob optimization")

    mimic_popsize_optimal(problem=problem, pop_sizes=np.arange(100, 2000, 100), random_seeds=random_seeds, label = "Travel Salesmen mimic pop size", title = "Travel Salesmen mimic pop size optimization")

    mimic_keepPct_optimal(problem=problem, keep_pcts=np.arange(0, 1, 0.05), random_seeds=random_seeds, label = "Travel Salesmen mimic keep pct", title = "Travel Salesmen mimic keep pct optimization")

    run_problems(problem=problem, rhc_max_attempts=800, decay_rate=0.02, sa_max_attempts=800, ga_max_attempts=100, GA_popSize=100, GA_mutationProb=0.1, mimic_max_attempts=30, mimic_popSize=700, mimic_keepPct=0.2, random_seeds=random_seeds, title = "Travel Salesmen")


if __name__ == "__main__":

    run_travel_salesman()