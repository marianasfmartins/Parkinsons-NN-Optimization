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
    #getting predictions
    Y_pred = get_predictions(solution, X_train, Y_train, X_val, Y_val)
    
    #Counting how many patients were wrongly predicted
    wrong_predictions = np.sum(Y_val != Y_pred)
    
    return wrong_predictions

'''def run_multiple_times(algorithm_fn, params, n_runs=5):
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

    return avg_history'''

def run_multiple_times(algorithm_fn, params, n_runs=5):
    all_histories = []
    final_errors = []
    for _ in range(n_runs):
        # Run the algorithm passed as a function reference
        best_sol, history = algorithm_fn(**params)
        all_histories.append(history)
        final_errors.append(history[-1]) # Save the final generation's error
    
    avg_history = np.mean(all_histories, axis=0)

    # Return everything your evaluation script needs!
    return avg_history, final_errors, all_histories

def plot_history(history, title="Fitness over Iterations"):
    plt.plot(history)
    
    # Get the best (minimum) fitness and its index
    min_val = min(history)
    min_idx = history.index(min_val)
    
    # Plot a red star at the minimum point
    plt.plot(min_idx, min_val, 'r*', markersize=12, label=f'Best: {min_val:.4f}')
    plt.legend()
    
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title(title)
    return plt.gcf()
