import random
from bisect import bisect_left

def roulette_selection(population = list, fitness_values = list):
    """Select two parents using roulette wheel selection."""
    total_fitness = sum(fitness_values)
    pick = random.uniform(0, 1/total_fitness)

    current = 0
    for i, ind in enumerate(population):
        current += 1/fitness_values[i]
        if current > pick:
            parent = ind
            break
    
    return parent

def tournament_selection(population = list, fitness_values = list, tournament_size = int):
    """Select two parents using tournament selection."""
    if len(population) < tournament_size:
        tournament_pop = population
    else:
        tournament_pop = random.sample(population, tournament_size)
    tournament_fitness = [fitness_values[population.index(ind)] for ind in tournament_pop]
    parent = tournament_pop[tournament_fitness.index(min(tournament_fitness))]

    return parent

def rank_selection(population = list, fitness_values = list):
    """Select two parents using rank selection."""
    n = len(population)
    fitness_ranks = list(reversed(sorted(range(1, n+1), key=lambda x: fitness_values[x-1])))
    cumulative_prob = [sum(fitness_ranks[:i+1])/sum(fitness_ranks) for i in range(n)]
    parent = population[bisect_left(cumulative_prob, random.random())]

    return parent