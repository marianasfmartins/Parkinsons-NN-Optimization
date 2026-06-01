import random
import numpy as np
def tournament_selection(population, fitnesses, pool_size=3):
    '''
    Performs Tournament Selection for Minimization (lower fitness/error is better).
    Selects pool_size individuals randomly and returns the one with the lowest fitness.

    Parameters
    ----------
    population: List of solutions.
    fitnesses: List of fitness values for each solution in the population.
    pool_size: Number of individuals participating in each tournament.

    Returns
    -------
    best_solution: Selected candidate solution.
    '''
    # Select random indexes from the population
    pool_indexes = random.sample(range(len(population)), pool_size)
    # Get fitness values corresponding to the selected pool indexes
    pool_fit = [fitnesses[i] for i in pool_indexes]

    # Find the index of the candidate solution with the lowest fitness in the pool (minimization)
    index = pool_indexes[np.argmin(pool_fit)]
   
    # Return the selected candidate solution
    return population[index]