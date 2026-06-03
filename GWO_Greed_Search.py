from project_data import X_train, X_val, X_test, Y_train, Y_val, Y_test
from utils import fitness_misclassification, plot_history
from GA_operators.population import initialize_population
from algorithms import grey_wolf_optimizer

# 1. Define your GWO parameter lists
init_methods = ['he_normal', 'he_uniform']
num_wolves_list = [30, 50, 100]
max_iters = [30, 50]  # Modest iterations for screening combinations

best_overall_fitness = float('inf')
best_hyperparameters = {}
experiment_count = 0

# 2. Loop manually through every single combination
for init_method in init_methods:
    for num_wolves in num_wolves_list:
        for max_iter in max_iters:
            
            experiment_count += 1
            print(f"\n--- Running GWO Experiment {experiment_count} ---")
            print(f"Params: init={init_method}, num_wolves={num_wolves}, max_iter={max_iter}")
            
            # Run your GWO with the current loop variables
            best_wolf, fitness_history = grey_wolf_optimizer(
                population=initialize_population, # Function reference
                init_method=init_method,
                n_in=21,
                n_out=1,
                num_wolves=num_wolves,
                n_weights=2401,
                max_iter=max_iter,
                fitness_func=lambda ind: fitness_misclassification(ind, X_train, Y_train, X_val, Y_val), # Ensure this is imported
                visualize=False                         # Turn off plots during search to avoid spam
            )
            
            # GWO updates history at each iteration; grab the final alpha score
            final_fitness = fitness_history[-1]
            
            # 3. Check if this alpha wolf is the new global leader
            if final_fitness < best_overall_fitness:
                best_overall_fitness = final_fitness
                best_hyperparameters = {
                    'init_method': init_method,
                    'num_wolves': num_wolves,
                    'max_iter': max_iter
                }

print("\n================ GWO GRID SEARCH COMPLETED ================")
print(f"Total combinations tested: {experiment_count}")
print(f"Best Achieved Fitness Score (Lowest Error): {best_overall_fitness}")
print(f"Optimal Hyperparameter Set: {best_hyperparameters}")