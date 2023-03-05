import mlrose_hiive as mlr
import numpy as np
from helper import log
from Optimals import (GA_pop_breed_pec_optimal, GA_popsize_optimal, SA_optimal,
                      mimic_keepPct_optimal, mimic_popsize_optimal,
                      run_problems)

dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426), 
             (0, 5, 5.3852), (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000), 
             (1, 3, 2.8284), (1, 4, 2.0000), (1, 5, 4.1231), (1, 6, 4.2426), 
             (1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361), (2, 5, 4.4721), 
             (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056), 
             (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623), 
             (4, 7, 2.2361), (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]
fitness = mlr.TravellingSales(distances=dist_list)
problem = mlr.TSPOpt(length = 8, fitness_fn = fitness, maximize=False)
problem.set_mimic_fast_mode = True
random_seeds = [2636 + 8 * i for i in range(3)]

def run_task1():
    log('start to run tsm sa')
    SA_optimal(problem = problem, decay_rates=np.arange(0.005, 0.1, 0.005),random_seeds=random_seeds, label = "Travel Salesmen SA ExpDecay rates", title = "Travel Salesmen SA ExpDecay rates optimization")

def run_task2():
    log('start to run tsm ga popsize')
    GA_popsize_optimal(problem=problem, pop_sizes=np.arange(100, 2000, 100), random_seeds=random_seeds, label = "Travel Salesmen GA pop size", title = "Travel Salesmen GA pop size optimization")

def run_task3():
    log('start to run tsm ga breed')
    GA_pop_breed_pec_optimal(problem=problem, pop_breed_pecs=np.arange(0.05, 1, 0.1), random_seeds=random_seeds, label = "fTravel Salesmen GA  pop breed pec", title = "Travel Salesmen GA  pop breed pec optimization")

def run_task4():
    log('start to run tsm mimic popsize')
    mimic_popsize_optimal(problem=problem, pop_sizes=np.arange(100, 2000, 100), random_seeds=random_seeds, label = "Travel Salesmen mimic pop size", title = "Travel Salesmen mimic pop size optimization")

def run_task5():
    log('start to run tsm mimic keepec')
    mimic_keepPct_optimal(problem=problem, keep_pcts=np.arange(0.05, 1, 0.1), random_seeds=random_seeds, label = "Travel Salesmen mimic keep pct", title = "Travel Salesmen mimic keep pct optimization")

def run_task6():
    log('start to run run tsm')
    run_problems(problem=problem, rhc_max_attempts=800, decay_rate=0.02, sa_max_attempts=800, ga_max_attempts=100, GA_popSize=100, GA_pop_breed_pec=0.1, mimic_max_attempts=30, mimic_popSize=300, mimic_keepPct=0.2, random_seeds=random_seeds, title = "Travel Salesmen")

def run_travel_salesman():
    pass
    # workers.run_task(run_task1)
    #workers.run_task(run_task2)
    # workers.run_task(run_task3)
    # workers.run_task(run_task4)
    # workers.run_task(run_task5)
    # worker_pool.run_task(run_task6)
