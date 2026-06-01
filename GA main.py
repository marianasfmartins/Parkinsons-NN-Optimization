import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import importlib
import random
from GA_operators.crossover import *
from GA_operators.selector import *
from GA_operators.mutators import *
from GA_operators.selector import *
from GA_operators.population import *
from algorithms import genetic_algorithm
from utils import run_multiple_times
from NN import get_predictions, get_n_weights
from project_data import X_train, X_val, X_test, Y_train, Y_val, Y_test
from utils import fitness_misclassification
from GA_operators.population import initialize_population, he_normal_ind, he_uniform_ind, xavier_normal_ind

if __name__ == "__main__":

    ga_params = {
        "population": initialize_population,
        "init_method": "he_normal",
        "pop_size": 100,
        "n_weights": 2401,
        "fit_func": lambda ind: fitness_misclassification(ind, X_train, Y_train, X_val, Y_val), # <-- WRAP IT HERE!
        "selector": tournament_selection,
        "mutator": gaussian_mutation,
        "xover_operator": arithmetic_crossover,
        "p_mut": 0.01,
        "p_xover": 0.8,
        "n_gens": 100,
        "pool_size": 3,
        "mutation_strength": 0.05,
        "n_in": 21,
        "n_out": 1
    }

    best = genetic_algorithm(**ga_params)
    print(best)
