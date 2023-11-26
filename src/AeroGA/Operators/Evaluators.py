"""
Functions dedicated to calculating the fitness of individuals in the population.
"""

import multiprocessing
from AeroGA.Classes.Error import ErrorType, Log

# Setting error log file
ErrorLog = Log("error.log", 'Evaluators')

def parallel_fitness(population = list, fitness_fn = None, num_processes = int):
    """Calculate the fitness of each individual in the population."""
    try:
        with multiprocessing.Pool(num_processes) as pool:
            fitness_values = pool.map(fitness_fn, population)
        return fitness_values
    except Exception as e:
        ErrorLog.error(str(e))
        return ErrorType("danger", str(e), 'parallel_fitness')

# Fitness function without using multi threads
def fitness(population = list, fitness_fn = None):
    """Calculate the fitness of each individual in the population."""
    try:
        return [fitness_fn(ind) for ind in population]
    except Exception as e:
        ErrorLog.error(str(e))
        return ErrorType("danger", str(e), 'fitness')