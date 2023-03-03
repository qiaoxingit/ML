import mlrose_hiive as mlr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from mlrose_hiive.algorithms import random_hill_climb, simulated_annealing, genetic_alg, mimic
from mlrose_hiive import ExpDecay
import util

#tune SA
def SA_optimal(problem, decay_rates, random_seeds, label, title):

    sa_fitness = []

    for decay_rate in decay_rates:
        exp_decay = ExpDecay(init_temp=100, exp_const=decay_rate, min_temp=0.001)

        best_fitnesses=[]
        for random_seed in random_seeds:
            best_state, best_fitness, _=simulated_annealing(problem, schedule=exp_decay, max_attempts=100, max_iters=300, curve=False,random_state=random_seed)

            best_fitnesses.append(best_fitness)

        sa_fitness.append(best_fitnesses)

    util.plot_figure(x=decay_rates, y=np.array(sa_fitness), xlabel="SA Exponential Decay Rate", ylabel="fitness", label = label, title = title)

def GA_popsize_optimal(problem, pop_sizes, random_seeds, label, title):

    ga_fitness = []

    for pop_size in pop_sizes:

        best_fitnesses=[]
        for random_seed in random_seeds:
            best_state, best_fitness, _=genetic_alg(problem, pop_size=pop_size, max_attempts=100, max_iters=300, curve=False,random_state=random_seed)

            best_fitnesses.append(best_fitness)

        ga_fitness.append(best_fitnesses)

    util.plot_figure(x=pop_sizes, y=np.array(ga_fitness), xlabel=" GA pop size", ylabel="fitness", label = label, title = title)


def GA_mutationProb_optimal(problem, mutation_probs, random_seeds, label, title):

    ga_fitness = []

    for mutation_prob in mutation_probs:

        best_fitnesses=[]
        for random_seed in random_seeds:
            best_state, best_fitness, _=genetic_alg(problem, mutation_prob=mutation_prob, max_attempts=100, max_iters=300, curve=False,random_state=random_seed)

            best_fitnesses.append(best_fitness)

        ga_fitness.append(best_fitnesses)

    util.plot_figure(x=mutation_probs, y=np.array(ga_fitness), xlabel="GA mutation prob", ylabel="fitness", label = label, title = title)


def mimic_popsize_optimal(problem, pop_sizes, random_seeds, label, title):

    mimic_fitness = []

    for pop_size in pop_sizes:

        best_fitnesses=[]
        for random_seed in random_seeds:
            best_state, best_fitness, _=mimic(problem, pop_size=pop_size, max_attempts=100, max_iters=300, curve=False,random_state=random_seed, fast_mimic = True)

            best_fitnesses.append(best_fitness)

        mimic_fitness.append(best_fitnesses)

    util.plot_figure(x=pop_sizes, y=np.array(mimic_fitness), xlabel="mimic pop size", ylabel="fitness", label = label, title = title)


def mimic_keepPct_optimal(problem, keep_pcts, random_seeds, label, title):

    mimic_fitness = []

    for keep_pct in keep_pcts:

        best_fitnesses=[]
        for random_seed in random_seeds:
            best_state, best_fitness, _=mimic(problem, keep_pct=keep_pct, max_attempts=100, max_iters=300, curve=False,random_state=random_seed, fast_mimic = True)

            best_fitnesses.append(best_fitness)

        mimic_fitness.append(best_fitnesses)

    util.plot_figure(x=keep_pcts, y=np.array(mimic_fitness), xlabel="mimic keep pec", ylabel="fitness", label = label, title = title)


def run_problems(problem, rhc_max_attempts, decay_rate, sa_max_attempts, ga_max_attempts, GA_popSize, GA_mutationProb, mimic_max_attempts, mimic_popSize, mimic_keepPct, random_seeds, title):

    rhc_fitness, sa_fitness, ga_fitness, mimic_fitness = [], [], [], []
    rhc_time, sa_time, ga_time, mimic_time =[], [], [], []

    exp_decay = ExpDecay(init_temp=100, exp_const=decay_rate, min_temp=0.001)

    for random_seed in random_seeds:

        start_time = time.time()
        best_state, best_fitness, fitness_curve = random_hill_climb(problem, max_attempts=rhc_max_attempts, max_iters=rhc_max_attempts, curve = True, random_state=random_seed)
        rhc_time.append(time.time() - start_time)
        rhc_fitness.append(fitness_curve)

        start_time =time.time()
        best_state, best_fitness, fitness_curve = simulated_annealing(problem, schedule=exp_decay, max_attempts=sa_max_attempts, max_iters=sa_max_attempts, curve=True, random_state=random_seed)
        sa_time.append(time.time() - start_time)
        sa_fitness.append(fitness_curve)

        start_time =time.time()
        best_state, best_fitness, fitness_curve = genetic_alg(problem, pop_size=GA_popSize, mutation_prob=GA_mutationProb, max_attempts=ga_max_attempts, max_iters=ga_max_attempts, curve=True, random_state=random_seed)
        ga_time.append(time.time() - start_time)
        ga_fitness.append(fitness_curve)

        start_time =time.time()
        best_state, best_fitness, fitness_curve = mimic(problem, pop_size=mimic_popSize, keep_pct=mimic_keepPct, max_attempts=mimic_max_attempts, max_iters=mimic_max_attempts, curve=True, random_state=random_seed)
        mimic_time.append(time.time() - start_time)
        mimic_fitness.append(fitness_curve)
        
    plt.figure()
    plt.title(title +" fitness curve")
    plt.xlabel("fitness vs iterations")
    plt.ylabel("fitness")
    plt.grid()
    plt.plot(np.arange(1, 1000), np.array(rhc_fitness), label = 'RHC')
    plt.plot(np.arange(1, 1000), np.array(sa_fitness), label = 'SA')
    plt.plot(np.arange(1, 1000), np.array(ga_fitness), label = 'GA')
    plt.plot(np.arange(1, 1000), np.array(mimic_fitness), label = 'MIMIC')
    plt.legend(loc="best")
    plt.savefig('images/'+title+" fitness curve")

    plt.figure()
    plt.title(title+" time curve")
    plt.xlabel("time vs iterations")
    plt.ylabel("time")
    plt.grid()
    plt.plot(np.arange(1, 1000), np.array(rhc_time), label = 'RHC')
    plt.plot(np.arange(1, 1000), np.array(sa_time), label = 'SA')
    plt.plot(np.arange(1, 1000), np.array(ga_time), label = 'GA')
    plt.plot(np.arange(1, 1000), np.array(mimic_time), label = 'MIMIC')
    plt.legend(loc="best")
    plt.savefig('images/'+title +" time curve")
