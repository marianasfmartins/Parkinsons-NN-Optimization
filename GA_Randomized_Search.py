from GA_operators.selector import tournament_selection
from utils import fitness_misclassification
from project_data import X_train, X_val, Y_train, Y_val
import random
from GA_operators.population import initialize_population
from GA_operators.crossover import blend_crossover
from GA_operators.mutators import gaussian_mutation
from algorithms import genetic_algorithm 

# 1. Define your parameter pools
init_methods = ['he_normal', 'he_uniform']
p_xovers = [0.7, 0.8, 0.9]
p_muts = [0.01, 0.02, 0.05, 0.1]
mutation_strengths = [0.01, 0.05, 0.1]
pool_sizes = [3, 4, 5]

best_overall_fitness = float('inf')
best_hyperparameters = {}

# 2. Set a strict time budget (e.g., only run 8 experiments total)
MAX_EXPERIMENTS = 8

for experiment in range(1, MAX_EXPERIMENTS + 1):
    # Randomly pick a configuration for this run
    init_method = random.choice(init_methods)
    p_xover = random.choice(p_xovers)
    p_mut = random.choice(p_muts)
    mutation_strength = random.choice(mutation_strengths)
    pool_size = random.choice(pool_sizes)
    
    print(f"\n--- Random Experiment {experiment}/{MAX_EXPERIMENTS} ---")
    print(f"Testing: init={init_method}, p_xover={p_xover}, p_mut={p_mut}, strength={mutation_strength}, pool={pool_size}")
    
    # Run your GA 
    best_weights, fitness_history = genetic_algorithm(
        population=initialize_population,
        init_method=init_method,
        pop_size=100,
        n_weights=2401,
        fit_func=lambda ind: fitness_misclassification(ind, X_train, Y_train, X_val, Y_val), 
        selector=tournament_selection,      
        mutator=gaussian_mutation,          
        xover_operator=blend_crossover,    
        p_mut=p_mut,
        p_xover=p_xover,
        n_gens=30, 
        pool_size=pool_size,
        mutation_strength=mutation_strength,
        elitism=True
    )
    
    final_fitness = fitness_history[-1]
    
    if final_fitness < best_overall_fitness:
        best_overall_fitness = final_fitness
        best_hyperparameters = {
            'init_method': init_method,
            'p_xover': p_xover,
            'p_mut': p_mut,
            'mutation_strength': mutation_strength,
            'pool_size': pool_size
        }

print("\n================ RANDOM SEARCH COMPLETED ================")
print(f"Best Achieved Fitness Score: {best_overall_fitness}")
print(f"Optimal Hyperparameter Set: {best_hyperparameters}")