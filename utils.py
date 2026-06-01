import numpy as np
import random
import matplotlib.pyplot as plt
from NN import *
def fitness_function(solution, X_train, Y_train, X_val, Y_val): # minimization: makes sense to minimize prediction error
    # Adicionado: Gera as previsões usando a estrutura da NN e os pesos do lobo atual
    prediction = get_predictions(solution, X_train, Y_train, X_val, Y_val)
    actual = Y_val  # O teu "actual" são os valores reais de validação
    
    # O teu código original exatamente igual:
    fitness = 0
    for i in range(len(prediction)):
        error = ((actual[i] - prediction[i]) ** 2) ** 0.5
        fitness += error
    return fitness


#A more intuitive approach with the same results
def fitness_misclassification(solution, X_train, Y_train, X_val, Y_val):
    # 1. Get the discrete class predictions (0 or 1) from the NN
    Y_pred = get_predictions(solution, X_train, Y_train, X_val, Y_val)
    
    # 2. Count how many patients the wolf guessed incorrectly
    wrong_predictions = np.sum(Y_val != Y_pred)
    
    return wrong_predictions

def run_multiple_times(algorithm_fn, params, n_runs=30):
    all_histories = []
    for _ in range(n_runs):
        _, history = algorithm_fn(**params)
        all_histories.append(history)

    n_iter = len(all_histories[0])
    avg_history = []
    for i in range(n_iter):
        total = 0
        for run in range(n_runs):
            total += all_histories[run][i]
        avg_history.append(total / n_runs)

    return avg_history
