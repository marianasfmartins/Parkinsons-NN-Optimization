import random
import numpy as np
from GA_operators.population import initialize_population
from utils import fitness_misclassification, plot_history

def genetic_algorithm(population,
                      init_method,
                      pop_size,
                      n_weights,
                      fit_func,
                      selector,
                      mutator,
                      xover_operator,
                      p_mut,
                      p_xover,
                      n_gens,
                      pool_size=3,
                      mutation_strength=0.05,
                      n_in=21,
                      n_out=1,
                      elitism=True):
    """
    Classical Genetic Algorithm adapted for Neural Network weight minimization with Elitism.
    
    Args:
        population (function): Reference to the population initialization function.
        init_method (str): Statistical initialization method name (e.g., 'he_uniform').
        pop_size (int): Total number of individuals in the population.
        n_weights (int): Chromosome length (total weights + biases in the NN).
        fit_func (function): Evaluation function returning a prediction error score.
        selector (function): Parent selection operator (e.g., tournament selection).
        mutator (function): Gene perturbation operator tailored for continuous real values.
        xover_operator (function): Crossover operator adapted for real-valued vectors.
        p_mut (float): Probability of mutating a single gene.
        p_xover (float): Probability of executing crossover between two parents.
        n_gens (int): Total number of evolutionary iterations (generations).
        pool_size (int): Tournament size for the selector operator. Default = 3.
        mutation_strength (float): Standard deviation (sigma) for the mutation step noise. Default = 0.05.
        n_in (int): Number of input features for weight dimension scaling. Default = 21.
        n_out (int): Number of network outputs. Default = 1.
        elitism (bool): Flag to enable/disable elite preservation across generations. Default = True.
        
    Returns:
        best_individual (numpy.ndarray): Optimized weight vector minimizing the network error.
        fitness_history (list): Log of the best fitness score achieved at each generation.
    """
    
    # --- Generation 0: Initialization & Setup ---
    
    # Call the passed function reference to generate the initial matrix of random solutions
    population = population(pop_size, n_weights, init_method, n_in, n_out)
    
    # Calculate fitness for every individual (lower values mean less error)
    pop_fits = [fit_func(ind) for ind in population]

    # Track global historical performance (Minimization: lower error is better)
    best_fitness = min(pop_fits)
    best_individual = population[np.argmin(pop_fits)].copy()
    
    # Initialize the performance log history for later convergence plotting
    fitness_history = [best_fitness]

    print(f"[GA] Generation 0/{n_gens} — Initial Best Fitness (Lowest Error): {best_fitness:.6f}")

    # --- Evolutionary Optimization Loop ---
    for generation in range(n_gens):
        offspring = []

        # Generate a new candidate population pool (mating and variation)
        while len(offspring) < pop_size:

            # Select two parent candidates via tournament brackets
            parent1 = selector(population=population, fitnesses=pop_fits, pool_size=pool_size)
            parent2 = selector(population=population, fitnesses=pop_fits, pool_size=pool_size)

            # Probabilistic Variation: Crossover vs. Direct Replication
            if random.random() <= p_xover:
                child1, child2 = xover_operator(parent1, parent2)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            # Apply mutation noise to both children based on mutation rate and strength
            child1 = mutator(child1, p_mut=p_mut, mutation_strength=mutation_strength)
            child2 = mutator(child2, p_mut=p_mut, mutation_strength=mutation_strength)

            # Append the modified children to the offspring pool while enforcing pop_size boundaries
            offspring.append(child1)
            if len(offspring) < pop_size:
                offspring.append(child2)

        # Execute full generational replacement (offspring replaces current population)
        population = [child for child in offspring]
        pop_fits = [fit_func(ind) for ind in population]

        # --- Algorithmic Elitism Step ---
        if elitism:
            # If the best historical individual outperforms (has lower error than)
            # every single solution in this newly generated generation pool:
            if best_fitness < min(pop_fits):
                # Locate the index of the absolute worst performer (highest error) in the new generation
                worst_idx = np.argmax(pop_fits)
                
                # Overwrite that worst solution with a clean memory copy of our historical elite
                population[worst_idx] = best_individual.copy()
                pop_fits[worst_idx] = best_fitness

        # --- Statistical Tracking Update ---
        
        # Identify the best individual in the current generation state
        gen_best_fitness = min(pop_fits)
        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_individual = population[np.argmin(pop_fits)].copy()

        # Save progress markers for plotting and report logging
        fitness_history.append(best_fitness)
        print(f"generation {generation + 1}/{n_gens} — best fitness: {best_fitness:.6f}")

    # Evolution concluded; return the optimal parameters and convergence array
    return best_individual, fitness_history


def grey_wolf_optimizer(population, init_method, n_in, n_out, num_wolves, n_weights, max_iter=50, fitness_func=fitness_misclassification, visualize=True):
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
    
    population = population(num_wolves, n_weights, init_method, n_in, n_out)

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