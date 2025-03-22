"""
Functions dedicated to calculating the fitness of individuals in the population.
"""

from AeroGA.Classes.Error import ErrorType
from joblib import Parallel, delayed

def evaluate(population = list, history = dict, fitness_fn = None, num_processes = int):
    """Evaluate the fitness of the population."""
    
    # Verificar se o indivíduo já está no histórico
    def get_fitness(individual, history):
        if individual in history["ind"]:
            index = history["ind"].index(individual)
            return history["fit"][index]
        else:
            return fitness_fn(individual)

    try:
        fitness_values = []
        if num_processes != 0:
            fitness_values = Parallel(n_jobs=num_processes)(delayed(get_fitness)(individual, history) for individual in population)
        else:
            fitness_values = [get_fitness(individual, history) for individual in population]

        fitness_values_float = [float(x) for x in fitness_values]

        return fitness_values_float
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'fitness')
        return error.message