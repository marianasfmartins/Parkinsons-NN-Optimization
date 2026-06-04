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

def run_multiple_times(algorithm_fn, params, n_runs=5):
    all_histories = []
    final_errors = []
    for _ in range(n_runs):
        # Run the algorithm passed as a function reference
        best_sol, history = algorithm_fn(**params)
        all_histories.append(history)
        final_errors.append(int(history[-1])) # Save the final generation's error
    
    avg_history = np.mean(all_histories, axis=0)

    # Return everything your evaluation script needs!
    return avg_history, final_errors, all_histories

def plot_history(history, title="Fitness over Iterations", L2=None):
    plt.plot(history)
    
    # Get the best (minimum) fitness and its index
    min_val = min(history)
    min_idx = history.index(min_val)
    
    # Plot a red star at the minimum point
    if L2 is not None:
        # CORRECTED: Display the passed L2 norm value of the actual weights vector
        plt.plot(min_idx, min_val, 'r*', markersize=12, label=f'Best: {min_val:.1f} ($L_2$ Norm: {L2:.4f})')
    else:
        plt.plot(min_idx, min_val, 'r*', markersize=12, label=f'Best: {min_val:.1f}')
        
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title(title)
    
    return plt.gcf()


def plot_comparison(hist_ga, champion_ga, hist_gwo, champion_gwo):
    """
    Plots a combined convergence chart for GA and GWO with minimum markers 
    and L2 norm annotations in the legend. Handles NumPy arrays safely.
    """
    plt.figure(figsize=(10, 6))
    
    # 1. Plot GA Curve & Find its minimum point using NumPy arrays
    plt.plot(hist_ga, color="mediumblue", linewidth=2, alpha=0.8)
    min_ga = np.min(hist_ga)
    idx_ga = np.argmin(hist_ga) # CORREÇÃO: np.argmin para encontrar o índice no NumPy array
    l2_ga = np.sum(np.array(champion_ga) ** 2)
    
    # Place a marker for GA champion
    plt.plot(idx_ga, min_ga, 'b*', markersize=12, 
             label=f'GA Best: {min_ga:.1f} ($L_2$: {l2_ga:.2f})')
    
    # 2. Plot GWO Curve & Find its minimum point using NumPy arrays
    plt.plot(hist_gwo, color="darkorange", linewidth=2, alpha=0.8)
    min_gwo = np.min(hist_gwo)
    idx_gwo = np.argmin(hist_gwo) # CORREÇÃO: np.argmin para encontrar o índice no NumPy array
    l2_gwo = np.sum(np.array(champion_gwo) ** 2)
    
    # Place a marker for GWO champion
    plt.plot(idx_gwo, min_gwo, 'r*', markersize=12, 
             label=f'GWO Best: {min_gwo:.1f} ($L_2$: {l2_gwo:.2f})')
    
    # 3. Styling & Presentation setup
    plt.title("Neural Network Weight Optimization: Metaheuristic Convergence Profiles", fontweight='bold')
    plt.xlabel("Generation / Iteration Index")
    plt.ylabel("Validation Misclassification Error Count")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Save a high-resolution asset for the final project report
    plt.savefig("metaheuristic_system_comparison.png", dpi=300, bbox_inches='tight')
    print("\n[System Asset Saved] Combined chart saved as 'metaheuristic_system_comparison.png'")
    
    return plt.gcf()
