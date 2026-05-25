import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from GA_operators.InitializationMethos import HeNormalInitializationMethod
from GA_operators.InitializationMethos import HeUniformInitializationMethod
import torch.nn as nn



sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import generate_random_solution, fitness_function


def initialize_population(num_wolves=30, n_weights=10, method='he_uniform'):
    population = []
    
    for _ in range(num_wolves):
        
        if method == 'he_uniform':
            weights = HeUniformInitializationMethod(n_weights)      # ← aqui, em vez de generate_random_solution
        
        elif method == 'he_normal':
            weights = HeNormalInitializationMethod(n_weights)        # ← aqui, em vez de generate_random_solution
        
        elif method == 'random':
            weights = generate_random_solution(n_weights)  # comportamento atual
        
        learning_rate = [random.uniform(0.0001, 0.1)]
        wolf = np.array(weights + learning_rate)
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
        fitness_func: Fitness function
        visualize: Save plots to PNG
    
    Returns:
        best_wolf, fitness_history
    """
    
    #if fitness_func is None:
        #fitness_func = fitness_function
    
    # Find initial alpha, beta, delta wolves
    best_fitness = float('inf')
    alpha = None
    beta = None
    delta = None
    
    #Let's evaluate each wolf in our population and determine the best three based on their fitness
    for wolf in population:
        fitness = fitness_func(wolf, wolf)

        #alpha best, beta secound best, delta third best 
        if fitness < best_fitness:
            delta = beta
            beta = alpha
            alpha = wolf.copy() #new best solution found, update the rest 
            best_fitness = fitness

        # only for the begining when we dont have solution for beta and delta so we just update
        elif beta is None or fitness < fitness_func(beta, beta):
            delta = beta
            beta = wolf.copy()
        elif delta is None or fitness < fitness_func(delta, delta):
            delta = wolf.copy()
    
    fitness_history = [best_fitness]
    weight_updates = []
    
    # Main loop
    for iteration in range(max_iter):
        # Linearly decrease from 2 to 0 will control exploration and exploitation 
        # Wolf will become clorer to the best solution (triangle)
        a = 2 - iteration * (2 / max_iter)  
        new_population = []
        iteration_updates = []
        
        for i in range(num_wolves):

            new_wolf = np.zeros_like(population[i]) # store learning rate and weights here to not mess with the original population during updates
            
            # update weights and learning rate based on alpha beta and delta 
            for j in range(n_weights + 1):
                # Update based on alpha
                # r1: when and where to update, r2: how much to update 
                r1, r2 = random.random(), random.random()
                A1 = 2 * a * r1 - a # direction and magnitude of direction towards alpha 
                C1 = 2 * r2 # ensures explration by adding randomness to the influeactual location of alpha
                D_alpha = abs(C1 * alpha[j] - population[i][j]) # distance between current wolf and alpha, influenced by C1
                X1 = alpha[j] - A1 * D_alpha #actualized position based on alpha influence 
                
                # Update based on beta
                r1, r2 = random.random(), random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta[j] - population[i][j])
                X2 = beta[j] - A2 * D_beta
                
                # Update based on delta
                r1, r2 = random.random(), random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta[j] - population[i][j])
                X3 = delta[j] - A3 * D_delta
                
                # Average the three influences
                new_wolf[j] = (X1 + X2 + X3) / 3 #find the center of the triangle
                iteration_updates.append(abs(new_wolf[j] - population[i][j]))
            
            new_population.append(new_wolf)
        
        population = new_population #everytime a new population is generated we update it 
        
        # Re-evaluate and update leaders
        for wolf in population:
            fitness = fitness_func(wolf, wolf)
            if fitness < best_fitness:
                delta = beta
                beta = alpha
                alpha = wolf.copy()
                best_fitness = fitness
            elif beta is None or fitness < fitness_func(beta, beta):
                delta = beta
                beta = wolf.copy()
            elif delta is None or fitness < fitness_func(delta, delta):
                delta = wolf.copy()
        
        fitness_history.append(best_fitness) #for the plot of fitness evolution 
        weight_updates.append(np.mean(iteration_updates) if iteration_updates else 0) # if wolfs are updates we get the mean for the plot to see if they are converging or not 
        
        print(f"Iteration {iteration + 1}/{max_iter} - Fitness: {best_fitness:.6f} - Update: {weight_updates[-1]:.6f}")
    
    if visualize:
        plot_results(fitness_history, weight_updates)
    
    return alpha, fitness_history


def plot_results(fitness_history, weight_updates):
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
