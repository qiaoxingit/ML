import mlrose_hiive as mlr
import numpy as np
from helper import log
from Optimals import (GA_pop_breed_pec_optimal, GA_popsize_optimal, SA_optimal,
                      mimic_keepPct_optimal, mimic_popsize_optimal,
                      run_problems)
from worker_pool import WorkerPool

fitness = mlr.FlipFlop()
problem = mlr.DiscreteOpt(100, fitness_fn = fitness, maximize=True, max_val=2)

problem.set_mimic_fast_mode = True
random_seeds = [2636 + 8 * i for i in range(3)]

def run_SA_optimal():
    log('start to run flip flop sa')
    SA_optimal(problem = problem, decay_rates=np.arange(0.005, 1.5, 0.05),random_seeds=random_seeds, label = "flip_flop SA ExpDecay rates", title = "flip_flop SA ExpDecay rates optimization")

def run_GA_popsize_optimal():
    log('start to run flip flop ga popsize')
    GA_popsize_optimal(problem=problem, pop_sizes=np.arange(100, 2000, 100), random_seeds=random_seeds, label = "flip_flop GA pop size", title = "flip_flop GA pop size optimization")

def run_GA_pop_breed_pec_optimal():
    log('start to run flip flop ga breed pec')
    GA_pop_breed_pec_optimal(problem=problem, pop_breed_pecs=np.arange(0.05, 1, 0.1), random_seeds=random_seeds, label = "flip_flop GA pop breed pec", title = "flip_flop GA pop breed pec optimization")

def run_mimic_popsize_optimal():
    log('start to run flip flop mimic popsize')
    mimic_popsize_optimal(problem=problem, pop_sizes=np.arange(100, 2000, 100), random_seeds=random_seeds, label = "flip_flop mimic pop size", title = "flip_flop mimic pop size optimization")

def run_mimic_keepPct_optimal():
    log('start to run flip flop mimic keepct')
    mimic_keepPct_optimal(problem=problem, keep_pcts=np.arange(0.05, 1, 0.1), random_seeds=random_seeds, label = "flip_flop mimic keep pct", title = "flip_flop mimic keep pct optimization")

def run_run_problems():
    log('start to run run flip flop')
    run_problems(problem=problem, rhc_max_attempts=1000, decay_rate=0.3, sa_max_attempts=1000, ga_max_attempts=500, GA_popSize=300, GA_pop_breed_pec=0.2, mimic_max_attempts=50, mimic_popSize=1500, mimic_keepPct=0.4, random_seeds=random_seeds, title = "Flip Flop")

def run_flip_flop():
    workers = WorkerPool()
    workers.run_task(run_SA_optimal)
    workers.run_task(run_GA_popsize_optimal)
    workers.run_task(run_GA_pop_breed_pec_optimal)
    workers.run_task(run_mimic_popsize_optimal)
    workers.run_task(run_mimic_keepPct_optimal)
    workers.run_task(run_run_problems)

