from project_data import X_train, X_val, X_test, Y_train, Y_val, Y_test
from utils import fitness_misclassification, plot_history
from algorithms import grey_wolf_optimizer
from GA_operators.population import initialize_population
import matplotlib.pyplot as plt

if __name__ == '__main__':

    gwo_params = {
        'population': initialize_population,
        "init_method": "he_normal",
        'num_wolves': 50,
        'n_weights': 2401,
        'max_iter': 30,
        'fitness_func': lambda ind: fitness_misclassification(ind, X_train, Y_train, X_val, Y_val),
        'visualize': False,
        "n_in": 21,
        "n_out": 2
    }
    best_wolf, fitness_history = grey_wolf_optimizer(**gwo_params)
    print("Best Individual fitness:", fitness_history[-1])
    plot_history(fitness_history, title="GWO Fitness Evolution")
    plt.show()

