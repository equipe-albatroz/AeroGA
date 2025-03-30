"""
Functions dedicated to the genetic algorithm's selection operators.
"""

import random
from bisect import bisect_left

def apply_selection_operator(selection: str, population: list, fitness_values: list):
    if selection == 'tournament':
        parent1, parent2 = tournament_selection(population, fitness_values, tournament_size=3)

    elif selection == 'rank':
        parent1, parent2 = rank_selection(population, fitness_values)

    elif selection == 'roulette':
        parent1, parent2 = roulette_selection(population, fitness_values)
    
    return parent1, parent2


def tournament_selection(population: list, fitness_values: list, tournament_size: int):
    """Seleciona dois pais usando seleção por torneio, onde menor fitness é melhor."""

    def select_one():
        # Seleciona aleatoriamente 'tournament_size' indivíduos
        tournament = random.sample(list(zip(population, fitness_values)), tournament_size)
        # Retorna o indivíduo com menor fitness (melhor solução)
        return min(tournament, key=lambda x: x[1])[0]

    parent1 = select_one()
    parent2 = select_one()
    
    return parent1, parent2

def roulette_selection(population: list, fitness_values: list):
    """Seleciona dois pais usando roleta viciada, onde menor fitness é melhor."""

    # Evita divisão por zero (substitui fitness 0 por um pequeno valor)
    adjusted_fitness = [f if f > 0 else 1e-6 for f in fitness_values]

    # Inverte os valores de fitness para que o menor fitness tenha maior chance
    inv_fitness = [1/f for f in adjusted_fitness]
    total_fitness = sum(inv_fitness)

    # Normaliza as probabilidades
    probabilities = [f/total_fitness for f in inv_fitness]

    # Seleciona dois pais com base nas probabilidades acumuladas
    parent1 = random.choices(population, weights=probabilities, k=1)[0]
    parent2 = random.choices(population, weights=probabilities, k=1)[0]

    return parent1, parent2

def rank_selection(population: list, fitness_values: list):
    """Seleciona dois pais usando seleção por ranking, onde menor fitness é melhor."""
    
    n = len(population)

    # Ordena população e fitness em ordem crescente de fitness (menor fitness = melhor)
    sorted_pop_fit = sorted(zip(population, fitness_values), key=lambda x: x[1])
    sorted_population, sorted_fitness = zip(*sorted_pop_fit)

    # Atribui rankings (o melhor indivíduo recebe o maior ranking)
    fitness_ranks = list(range(n, 0, -1))  # Ranking de N até 1 (menor fitness = maior ranking)

    # Calcula probabilidades acumuladas
    total_rank = sum(fitness_ranks)
    cumulative_prob = [sum(fitness_ranks[:i+1]) / total_rank for i in range(n)]

    # Seleciona dois pais usando roleta viciada no ranking
    parent1 = sorted_population[bisect_left(cumulative_prob, random.random())]
    parent2 = sorted_population[bisect_left(cumulative_prob, random.random())]

    return parent1, parent2