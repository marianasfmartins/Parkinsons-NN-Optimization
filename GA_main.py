import numpy as np
from project_data import X_train, X_val, X_test, Y_train, Y_val, Y_test
from utils import fitness_misclassification, plot_history
from GA_operators.crossover import *
from GA_operators.selector import *
from GA_operators.mutators import *
from GA_operators.population import *
from algorithms import genetic_algorithm
from utils import run_multiple_times
from NN import get_predictions
from project_data import X_train, X_val, X_test, Y_train, Y_val, Y_test
from utils import fitness_misclassification, plot_history
from GA_operators.population import initialize_population, he_normal_ind, he_uniform_ind, xavier_normal_ind
import matplotlib.pyplot as plt

if __name__ == "__main__":

    ga_params = {
        "population": initialize_population,
        "init_method": "he_uniform",
        "pop_size": 100,
        "n_weights": 2401,
        "fit_func": lambda ind: fitness_misclassification(ind, X_train, Y_train, X_val, Y_val),
        "selector": tournament_selection,
        "mutator": gaussian_mutation,
        "xover_operator": blend_crossover,
        "p_mut": 0.1,
        "p_xover": 0.8,
        "n_gens": 30,
        "pool_size": 5,
        "mutation_strength": 0.1,
        "n_in": 21,
        "n_out": 2
    }

    best_individual, fitness_history = genetic_algorithm(**ga_params)
    print("Best Individual fitness:", fitness_history[-1])
    plot_history(fitness_history, title="GA Fitness Evolution")
    plt.show()
  
