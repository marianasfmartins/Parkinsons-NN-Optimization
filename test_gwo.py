#!/usr/bin/env python3
"""Simple Grey Wolf Optimizer Example"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from NatureInspiredAlgorithms.NatureInspiredAlgorithms import (
    initialize_population, grey_wolf_optimizer
)

# Initialize population
print("Creating population...")
population, num_wolves, n_weights = initialize_population(num_wolves=10, n_weights=10)
print(f"Population: {num_wolves} wolves, {n_weights} weights\n")

# Run GWO
print("Running Grey Wolf Optimizer...\n")
best_solution, fitness_history = grey_wolf_optimizer(
    population=population,
    num_wolves=num_wolves,
    n_weights=n_weights,
    max_iter=10,
    visualize=True
)

# Results
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Initial Fitness: {fitness_history[0]:.6f}")
print(f"Final Fitness:   {fitness_history[-1]:.6f}")
print(f"Improvement:     {fitness_history[0] - fitness_history[-1]:.6f}")
print("="*60)
