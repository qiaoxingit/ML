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
            best_state, best_fitness, _=mimic(problem, pop_size=pop_size, max_attempts=100, max_iters=300, curve=False,random_state=random_seed)

            best_fitnesses.append(best_fitness)

        mimic_fitness.append(best_fitnesses)

    util.plot_figure(x=pop_sizes, y=np.array(mimic_fitness), xlabel="mimic pop size", ylabel="fitness", label = label, title = title)


def mimic_keepPct_optimal(problem, keep_pcts, random_seeds, label, title):

    mimic_fitness = []

    for keep_pct in keep_pcts:

        best_fitnesses=[]
        for random_seed in random_seeds:
            best_state, best_fitness, _=mimic(problem, keep_pct=keep_pct, max_attempts=100, max_iters=300, curve=False,random_state=random_seed)

            best_fitnesses.append(best_fitness)

        mimic_fitness.append(best_fitnesses)

    util.plot_figure(x=keep_pcts, y=np.array(mimic_fitness), xlabel="mimic keep pec", ylabel="fitness", label = label, title = title)
