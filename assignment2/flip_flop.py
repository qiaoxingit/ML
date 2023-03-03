import mlrose_hiive as mlr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from mlrose_hiive.algorithms import random_hill_climb, simulated_annealing, genetic_alg, mimic
from mlrose_hiive import ExpDecay

fitness = mlr.FlipFlop()
problem = mlr.DiscreteOpt(100, fitness)


    


