from random import random

def arithmetic_crossover(parent1, parent2): #select how much of the parents the children will inherit
    alpha = random()  # Randomly select a value between 0 and 1
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2 #two children to maintain the population size and find the solution faster 

def blend_crossover(parent1, parent2, alpha=0.3):

    child1 = []
    child2 = []
    for g1, g2 in zip(parent1, parent2):
        lo = min(g1, g2)
        hi = max(g1, g2)
        span = hi - lo
        # extend the sampling range by alpha on each side
        lo_ext = lo - alpha * span
        hi_ext = hi + alpha * span
        child1.append(lo_ext + random() * (hi_ext - lo_ext))
        child2.append(lo_ext + random() * (hi_ext - lo_ext))
    return child1, child2


