import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random

from GA_operators.InitializationMethos import HeNormalInitializationMethod
from GA_operators.InitializationMethos import HeUniformInitializationMethod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import generate_random_solution, fitness_function


def initialize_swarm(n_particles=30, n_weights=10, method='he_uniform'):
    """
    Initialize the PSO swarm with positions and velocities.

    Args:
        n_particles: Number of particles in the swarm.
        n_weights: Number of weights per particle.
        method: Initialization method for positions ('he_uniform', 'he_normal', 'random').

    Returns:
        positions: List of particle position arrays.
        velocities: List of particle velocity arrays (initialized to zero).
        n_particles: Number of particles.
        n_weights: Number of weights.
    """
    positions = []

    for _ in range(n_particles):
        if method == 'he_uniform':
            weights = HeUniformInitializationMethod(n_weights)
        elif method == 'he_normal':
            weights = HeNormalInitializationMethod(n_weights)
        elif method == 'random':
            weights = generate_random_solution(n_weights)

        positions.append(np.array(weights))

    # Velocities start at zero — particles begin stationary
    velocities = [np.zeros(n_weights) for _ in range(n_particles)]

    return positions, velocities, n_particles, n_weights


def particle_swarm_optimizer(
    positions,
    velocities,
    n_particles,
    n_weights,
    max_iter=50,
    fitness_func=fitness_function,
    w=0.7,
    c1=1.5,
    c2=1.5,
    v_max=0.2,
    visualize=True,
):
    """
    Particle Swarm Optimization Algorithm.

    Each particle represents a candidate weight vector for the neural network.
    Particles update their velocity and position based on their personal best
    and the global best found by the swarm.

    Velocity update rule:
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        x = x + v

    Args:
        positions: Initial particle positions (from initialize_swarm).
        velocities: Initial particle velocities (from initialize_swarm).
        n_particles: Number of particles.
        n_weights: Number of weights per particle.
        max_iter: Number of iterations.
        fitness_func: Fitness function — must accept (candidate, candidate) like GWO.
        w: Inertia weight — controls how much of the previous velocity is kept.
        c1: Cognitive coefficient — attraction towards personal best.
        c2: Social coefficient — attraction towards global best.
        v_max: Maximum velocity per dimension (prevents explosion).
        visualize: Save plots to PNG.

    Returns:
        gbest_position: Best solution found.
        fitness_history: List of best fitness per iteration.
    """

    # Evaluate initial fitness for all particles
    pbest_positions = [pos.copy() for pos in positions]
    pbest_scores = [fitness_func(pos, pos) for pos in positions]

    # Global best is the particle with the lowest fitness
    best_index = int(np.argmin(pbest_scores))
    gbest_position = pbest_positions[best_index].copy()
    gbest_score = pbest_scores[best_index]

    fitness_history = [gbest_score]
    velocity_history = []

    for iteration in range(max_iter):
        iteration_velocities = []

        for i in range(n_particles):
            r1 = np.array([random.random() for _ in range(n_weights)])
            r2 = np.array([random.random() for _ in range(n_weights)])

            # Velocity update
            cognitive = c1 * r1 * (pbest_positions[i] - positions[i])
            social = c2 * r2 * (gbest_position - positions[i])
            velocities[i] = w * velocities[i] + cognitive + social

            # Clamp velocity
            velocities[i] = np.clip(velocities[i], -v_max, v_max)

            # Position update
            positions[i] = positions[i] + velocities[i]

            # Evaluate new position
            score = fitness_func(positions[i], positions[i])

            # Update personal best
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = positions[i].copy()

            # Update global best
            if score < gbest_score:
                gbest_score = score
                gbest_position = positions[i].copy()

            iteration_velocities.append(np.linalg.norm(velocities[i]))

        fitness_history.append(gbest_score)
        avg_velocity = np.mean(iteration_velocities)
        velocity_history.append(avg_velocity)

        print(
            f"Iteration {iteration + 1}/{max_iter} - "
            f"Fitness: {gbest_score:.6f} - "
            f"Avg Velocity: {avg_velocity:.6f}"
        )

    if visualize:
        plot_results(fitness_history, velocity_history)

    return gbest_position, fitness_history


def plot_results(fitness_history, velocity_history):
    """Plot optimization results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(fitness_history, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Best Fitness', fontsize=12)
    axes[0].set_title('Fitness Evolution (PSO)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    if max(fitness_history) > 0 and min(fitness_history) > 0:
        axes[0].set_yscale('log')

    axes[1].plot(velocity_history, 'r-', linewidth=2, marker='s', markersize=4)
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Avg Velocity', fontsize=12)
    axes[1].set_title('Average Velocity Magnitude (PSO)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pso_results.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved to: pso_results.png")
