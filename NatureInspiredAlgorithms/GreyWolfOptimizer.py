import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
# TODO: Corrigir InitializationMethos para retornar arrays NumPy em vez de PyTorch Layers
# from GA_operators.InitializationMethos import HeNormalInitializationMethod
# from GA_operators.InitializationMethos import HeUniformInitializationMethod
# import torch.nn as nn



sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import fitness_function
from GA_operators.population import generate_random_solution


def initialize_population(num_wolves=30, n_weights=10, method='random'): # we generate 30 possible vectors
    population = []
    
    for _ in range(num_wolves):
        
        # TODO: Descomentar quando InitializationMethos retornar arrays NumPy
        # if method == 'he_uniform':
        #     weights = HeUniformInitializationMethod(n_weights)
        # elif method == 'he_normal':
        #     weights = HeNormalInitializationMethod(n_weights)
        # elif method == 'random':
        #     weights = generate_random_solution(n_weights)
        
        weights = generate_random_solution(n_weights)  # por agora usa sempre random
        
        wolf = np.array(weights)
        population.append(wolf)
    
    return population, num_wolves, n_weights


def grey_wolf_optimizer(population, num_wolves, n_weights, max_iter=50, fitness_func=fitness_function, visualize=True):
    """
    Grey Wolf Optimizer Algorithm
    
    Args:
        population: Initial wolf population
        num_wolves: Number of wolves
        n_weights: Number of weights per wolf
        max_iter: Number of iterations
        fitness_func: Fitness function (receives a wolf/solution, returns a scalar)
        visualize: Save plots to PNG
    
    Returns:
        best_wolf, fitness_history
    """
    
    # Initialize leaders and their scores
    alpha_pos = None
    alpha_score = float('inf')
    
    beta_pos = None
    beta_score = float('inf')
    
    delta_pos = None
    delta_score = float('inf')
    
    # Evaluate initial population and find alpha, beta, delta
    #alpha best, beta second best, delta third best 
    for wolf in population:
        fitness = fitness_func(wolf)

        if fitness < alpha_score:
            # Current alpha becomes beta, beta becomes delta
            delta_score = beta_score
            delta_pos = beta_pos
            beta_score = alpha_score
            beta_pos = alpha_pos
            alpha_pos = wolf.copy() #new best solution found, update the rest 
            alpha_score = fitness
        elif fitness < beta_score:
            delta_score = beta_score
            delta_pos = beta_pos
            beta_pos = wolf.copy()
            beta_score = fitness
        elif fitness < delta_score:
            delta_pos = wolf.copy()
            delta_score = fitness
    
    fitness_history = [alpha_score]
    weight_updates = []
    
    # Main loop
    for iteration in range(max_iter):
        # Linearly decrease from 2 to 0 will control exploration and exploitation 
        # Wolf will become closer to the best solution (triangle)
        a = 2 - iteration * (2 / max_iter) #learning rate??
        new_population = []
        iteration_updates = []
        
        for i in range(num_wolves):
            # Vectorized updates for huge performance boost instead of looping over each weight
            
            # Update based on alpha
            # r1: when and where to update, r2: how much to update 
            r1, r2 = np.random.random(n_weights), np.random.random(n_weights)
            A1 = 2 * a * r1 - a  # direction and magnitude of direction towards alpha
            C1 = 2 * r2  # ensures exploration by adding randomness
            D_alpha = np.abs(C1 * alpha_pos - population[i])  # distance between current wolf and alpha
            X1 = alpha_pos - A1 * D_alpha  # actualized position based on alpha influence
            
            # Update based on beta
            r1, r2 = np.random.random(n_weights), np.random.random(n_weights)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta_pos - population[i])
            X2 = beta_pos - A2 * D_beta
            
            # Update based on delta
            r1, r2 = np.random.random(n_weights), np.random.random(n_weights)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta_pos - population[i])
            X3 = delta_pos - A3 * D_delta
            
            # Average the three influences
            new_wolf = (X1 + X2 + X3) / 3  # find the center of the triangle
            
            iteration_updates.append(np.mean(np.abs(new_wolf - population[i])))
            new_population.append(new_wolf)
        
        population = new_population  # everytime a new population is generated we update it 
        
        # Re-evaluate and update leaders
        for wolf in population:
            fitness = fitness_func(wolf)
            if fitness < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos
                beta_score = alpha_score
                beta_pos = alpha_pos
                alpha_pos = wolf.copy()
                alpha_score = fitness
            elif fitness < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos
                beta_pos = wolf.copy()
                beta_score = fitness
            elif fitness < delta_score:
                delta_pos = wolf.copy()
                delta_score = fitness
        
        fitness_history.append(alpha_score)  # for the plot of fitness evolution 
        weight_updates.append(np.mean(iteration_updates) if iteration_updates else 0)  # if wolves are updated we get the mean for the plot to see if they are converging or not 
        
        print(f"Iteration {iteration + 1}/{max_iter} - Fitness: {alpha_score:.6f} - Update: {weight_updates[-1]:.6f}")
    
    if visualize:
        plot_history(fitness_history, weight_updates)
    
    return alpha_pos, fitness_history

#PLOT HISTORY FUNCTION
def plot_history(fitness_history, weight_updates):
    """Plot optimization results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Fitness evolution
    axes[0].plot(fitness_history, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Best Fitness', fontsize=12)
    axes[0].set_title('Fitness Evolution (GWO)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    if max(fitness_history) > 0 and min(fitness_history) > 0:
        axes[0].set_yscale('log')
    
    # Plot 2: Weight updates
    axes[1].plot(weight_updates, 'r-', linewidth=2, marker='s', markersize=4)
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Avg Weight Update', fontsize=12)
    axes[1].set_title('Weight Update Magnitude', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gwo_results.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved to: gwo_results.png")



if __name__ == '__main__':
    from NN import get_predictions, get_n_weights, X_train, X_val, X_test, Y_train, Y_val, Y_test
    from utils import fitness_function

    print("Iniciando GWO para treinamento da Rede Neural...")

    '''def nn_fitness_wrapper(pesos):
        # Gera as previsoes usando a rede neural com os pesos do lobo
        previsoes = get_predictions(pesos, X_train, Y_train, X_val, Y_val)
        # Calcula a fitness (erro)
        try: #isto só tem haver com se é array ou dataframe
            valores_reais = Y_val.values
        except AttributeError:
            valores_reais = Y_val
        return fitness_function(previsoes, valores_reais)

    n_weights = get_n_weights(X_train, Y_train)  # calcula automaticamente (coefs + biases)
    print(f"Total de pesos a otimizar: {n_weights}")
    num_wolves = 30

    print("Inicializando população...")
    population, _, _ = initialize_population(num_wolves, n_weights, method='random')

    print("Otimizando pesos com GWO...")
    best_pos, hist = grey_wolf_optimizer(
        population, num_wolves, n_weights,
        max_iter=20, fitness_func=nn_fitness_wrapper, visualize=True
    )

    print("\nOtimização concluída!")
    print(f"Melhor erro (fitness): {hist[-1]:.8f}")'''