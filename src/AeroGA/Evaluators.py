"""
Functions dedicated to calculating the fitness of individuals in the population.
"""

import multiprocessing

def parallel_fitness(population = list, fitness_fn = None, num_processes = int):
    """Calculate the fitness of each individual in the population."""
    with multiprocessing.Pool(num_processes) as pool:
        fitness_values = pool.map(fitness_fn, population)
    return fitness_values

# Fitness function without using multi threads
def fitness(population = list, fitness_fn = None):
    """Calculate the fitness of each individual in the population."""
    return [fitness_fn(ind) for ind in population]   