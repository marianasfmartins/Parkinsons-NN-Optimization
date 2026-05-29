# 1. Run your Genetic Algorithm
best_chrom, ga_hist = genetic_algorithm(population, fitness_func=gwo_fitness)

# 2. Run your Grey Wolf Optimizer
best_wolf, gwo_hist = grey_wolf_optimizer(population, fitness_func=gwo_fitness)

# 3. Plot them together to see who won the hunt!
plot_comparison_history(ga_hist, gwo_hist)