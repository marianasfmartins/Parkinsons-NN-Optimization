import os
import sys
import random

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import generate_random_solution, fitness_function

def initialize_particles(n_particles, n_dimensions, init_range=(-0.05, 0.05), seed=None):
    """Create initial positions and velocities for a PSO swarm."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    lower, upper = init_range
    positions = [np.array([random.uniform(lower, upper) for _ in range(n_dimensions)])
                 for _ in range(n_particles)]
    velocity_range = upper - lower
    velocities = [np.array([random.uniform(-velocity_range, velocity_range) for _ in range(n_dimensions)])
                  for _ in range(n_particles)]
    return positions, velocities


def particle_swarm_optimizer(
    n_dimensions,
    n_particles=30,
    fitness_func=None,
    max_iter=100,
    init_range=(-0.05, 0.05),
    w=0.7,
    c1=1.5,
    c2=1.5,
    v_max=None,
    bounds=None,
    visualize=True,
    seed=None,
):
    """Run Particle Swarm Optimization.

    Args:
        n_dimensions: Number of decision variables.
        n_particles: Swarm size.
        fitness_func: Callable that returns a scalar fitness. If it accepts one argument, the
                      candidate vector is passed directly. If it accepts two, the candidate is
                      passed twice to preserve compatibility with the current utils.fitness_function.
        max_iter: Number of optimization iterations.
        init_range: Initial position range for each dimension.
        w: Inertia weight.
        c1: Cognitive coefficient.
        c2: Social coefficient.
        v_max: Maximum velocity magnitude per dimension.
        bounds: Optional tuple (lower, upper) for position clamping.
        visualize: Whether to plot results after optimization.
        seed: Random seed for reproducibility.

    Returns:
        best_position: Best solution found.
        best_fitness: Best fitness value.
        fitness_history: List of best fitness values per iteration.
    """
    if fitness_func is None:
        fitness_func = fitness_function

    positions, velocities = initialize_particles(
        n_particles, n_dimensions, init_range=init_range, seed=seed
    )

    if bounds is None:
        bounds = init_range

    lower_bound, upper_bound = bounds
    if v_max is None:
        v_max = 0.2 * (upper_bound - lower_bound)

    pbest_positions = [pos.copy() for pos in positions]
    pbest_scores = [_evaluate_fitness(pos, fitness_func) for pos in positions]

    best_index = int(np.argmin(pbest_scores))
    gbest_position = pbest_positions[best_index].copy()
    gbest_score = pbest_scores[best_index]

    fitness_history = [gbest_score]
    velocity_history = []

    for iteration in range(1, max_iter + 1):
        iteration_velocities = []

        for i in range(n_particles):
            r1 = np.random.rand(n_dimensions)
            r2 = np.random.rand(n_dimensions)

            cognitive = c1 * r1 * (pbest_positions[i] - positions[i])
            social = c2 * r2 * (gbest_position - positions[i])
            velocities[i] = w * velocities[i] + cognitive + social
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            positions[i] = positions[i] + velocities[i]
            positions[i] = np.clip(positions[i], lower_bound, upper_bound)

            score = _evaluate_fitness(positions[i], fitness_func)
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = positions[i].copy()

            if score < gbest_score:
                gbest_score = score
                gbest_position = positions[i].copy()

            iteration_velocities.append(np.linalg.norm(velocities[i]))

        fitness_history.append(gbest_score)
        velocity_history.append(np.mean(iteration_velocities))
        print(
            f"Iteration {iteration}/{max_iter} - Best Fitness: {gbest_score:.6f} - "
            f"Avg Velocity: {velocity_history[-1]:.6f}"
        )

    if visualize:
        _plot_results(fitness_history, velocity_history)

    return gbest_position, gbest_score, fitness_history


def _plot_results(fitness_history, velocity_history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(fitness_history, marker='o', color='tab:blue')
    axes[0].set_title('PSO Best Fitness Over Iterations')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Best Fitness')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(velocity_history, marker='s', color='tab:orange')
    axes[1].set_title('Average Velocity Magnitude')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Average Velocity')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pso_results.png', dpi=300, bbox_inches='tight')
    print('Plots saved to: pso_results.png')


def sphere_function(candidate):
    """Simple sphere benchmark function for testing PSO."""
    return float(np.sum(np.square(candidate)))


if __name__ == '__main__':
    best_position, best_score, history = particle_swarm_optimizer(
        n_dimensions=10,
        n_particles=40,
        max_iter=50,
        fitness_func=sphere_function,
        visualize=True,
        seed=42,
    )
    print('\nBest score found:', best_score)
    print('Best position:', best_position)
