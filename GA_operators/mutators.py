import random

def gaussian_mutation(child, p_mut, mutation_strength):
    """
    Mutates genes using a Gaussian (Normal) distribution.
    mutation_strength acts as the standard deviation (sigma).
    """
    for i in range(len(child)):
        if random.random() < p_mut:  # Check per-gene mutation probability
            # random.gauss(0, mutation_strength) can be positive or negative
            child[i] += random.gauss(0, mutation_strength)
            
    return child

def polynomial_mutation(child, p_mut, lower_bound, upper_bound, eta=20):
    """eta controls the shape of the perturbation distribution:
      - low eta  (e.g. 5)  → wide perturbations, more exploration
      - high eta (e.g. 20) → narrow perturbations, more exploitation
    Stays within [lower_bound, upper_bound] by design.
    """
    for i in range(len(child)):
        if random.random() < p_mut:
            u = random.random()
            delta_i = upper_bound - lower_bound
            if u < 0.5:
                delta = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            child[i] += delta * delta_i
            # clamp to valid range
            child[i] = max(lower_bound, min(upper_bound, child[i]))
    return child