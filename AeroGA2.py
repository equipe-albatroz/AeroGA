import time
import random
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
from bisect import bisect_left 
import pandas as pd
import plotly.express as px
from ypstruct import structure

class Genes:
    def __init__(self, genes):
        self.genes = genes

# #####################################################################################
# ###################################### Main #########################################
# #####################################################################################

def genetic_algorithm(methods, num_variables, min_values, max_values, population_size, mutation_rate, eta, std_dev, num_generations, crossover_rate, alpha, tournament_size, fitness_fn):
    """Perform the genetic algorithm to find an optimal solution."""

    t_inicial = time.time()

    population = generate_population(population_size, num_variables, min_values, max_values)
    history = [population]
    best_fitness = float('inf')
    best_individual = population[0]
    for generation in range(num_generations):
        fitness_values = fitness(population, fitness_fn, methods.n_threads)
        best_fitness_in_gen = min(fitness_values)
        best_individual = population[fitness_values.index(best_fitness_in_gen)]
        if best_fitness_in_gen < best_fitness:
            best_fitness = best_fitness_in_gen
            best_individual = population[fitness_values.index(best_fitness_in_gen)]
        print("Generation: {} | Best Fitness: {}".format(generation+1, best_fitness))
        new_population = []
        for i in range(0, population_size, 2):
            
            if methods.selection == 'tournament':
                parent1, parent2 = tournament_selection(population, fitness_values, tournament_size)
            elif methods.selection == 'rank':
                parent1, parent2 = rank_selection(population, fitness_values)
            elif methods.selection == 'roulette':
                parent1, parent2 = roulette_selection(population, fitness_values)

            if random.uniform(0, 1) < crossover_rate:
                offspring1, offspring2 = arithmetic_crossover(parent1, parent2, alpha)
                new_population.append(offspring1)
                new_population.append(offspring2)
            else:
                new_population.append(parent1)
                new_population.append(parent2)
        
        if methods.mutation == 'polynomial':
            population = [polynomial_mutation(ind, min_values, max_values, eta) if random.uniform(0, 1) < mutation_rate else ind for ind in new_population]
        elif methods.mutation == 'gaussian':
            population = [gaussian_mutation(ind, min_values, max_values, std_dev) if random.uniform(0, 1) < mutation_rate else ind for ind in new_population]
                
        history.append(population)
    
    print("Best Individual: {}".format(best_individual))
    print(f"Tempo de Execução: {time.time() - t_inicial}")
    return population, history, best_individual

# #####################################################################################
# ##################################### Fitness #######################################
# #####################################################################################
    
# def fitness(population, fitness_fn):
#     """Calculate the fitness of each individual in the population."""
#     return [fitness_fn(ind) for ind in population]

def fitness(population, fitness_fn, n_threads):
    """Calculate the fitness of each individual in the population."""
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        fitness_values = list(executor.map(fitness_fn, population))
    return fitness_values


# #####################################################################################
# #################################### Init Pop #######################################
# #####################################################################################

# define the genetic algorithm functions
def generate_population(size, num_variables, min_values, max_values):
    """Generate a population of random genes."""
    population = [[random.uniform(min_values[i], max_values[i]) if isinstance(min_values[i],float) else random.randint(min_values[i], max_values[i]) for i in range(num_variables)] for _ in range(size)]
    return population

# #####################################################################################
# ################################### Selection #######################################
# #####################################################################################

def roulette_selection(population, fitness_values):
    """Select two parents using roulette wheel selection."""
    total_fitness = sum(fitness_values)
    pick1 = random.uniform(0, total_fitness)
    pick2 = random.uniform(0, total_fitness)
    current = 0
    for i, ind in enumerate(population):
        current += fitness_values[i]
        if current > pick1:
            parent1 = ind
            break
    current = 0
    for i, ind in enumerate(population):
        current += fitness_values[i]
        if current > pick2:
            parent2 = ind
            break
    return parent1, parent2

def tournament_selection(population, fitness_values, tournament_size):
    """Select two parents using tournament selection."""
    tournament_pop = random.sample(population, tournament_size)
    tournament_fitness = [fitness_values[population.index(ind)] for ind in tournament_pop]
    parent1 = tournament_pop[tournament_fitness.index(min(tournament_fitness))]
    tournament_pop.remove(parent1)
    tournament_fitness.remove(min(tournament_fitness))
    parent2 = tournament_pop[tournament_fitness.index(min(tournament_fitness))]
    return parent1, parent2

def rank_selection(population, fitness_values):
    """Select two parents using rank selection."""
    n = len(population)
    fitness_ranks = list(reversed(sorted(range(1, n+1), key=lambda x: fitness_values[x-1])))
    cumulative_prob = [sum(fitness_ranks[:i+1])/sum(fitness_ranks) for i in range(n)]
    parent1 = population[bisect_left(cumulative_prob, random.random())]
    parent2 = population[bisect_left(cumulative_prob, random.random())]
    return parent1, parent2

# #####################################################################################
# ################################### Crossover #######################################
# #####################################################################################

def arithmetic_crossover(parent1, parent2, alpha):
    """Apply arithmetic crossover to produce two offspring."""
    offspring1 = []
    offspring2 = []
    for i in range(len(parent1)):
        offspring1.append(alpha*parent1[i] + (1-alpha)*parent2[i])
        offspring2.append(alpha*parent2[i] + (1-alpha)*parent1[i])
        if not isinstance(parent1[i], float):
            offspring1[i] = int(offspring1[i])
            offspring2[i] = int(offspring2[i])
    return offspring1, offspring2

# #####################################################################################
# #################################### Mutation #######################################
# #####################################################################################

def gaussian_mutation(individual, min_values, max_values, std_dev):
    """Perform gaussian mutation on an individual."""
    mutated_genes = []
    for i in range(len(individual)):
        if isinstance(individual[i], int):
            mutated_gene = min(max(round(random.gauss(individual[i], std_dev)),min_values[i]), max_values[i])
        else:
            mutated_gene = min(max(random.gauss(individual[i], std_dev),min_values[i]), max_values[i])
        mutated_genes.append(mutated_gene)
    return mutated_genes

def polynomial_mutation(individual, min_values, max_values, eta):
    """Perform polynomial mutation on an individual."""
    mutated_genes = []
    for i in range(len(individual)):
        if isinstance(individual[i], int):
            if random.uniform(0, 1) < 0.5:
                if ((individual[i] - min_values[i]) / (max_values[i] - min_values[i])) < 0.5:
                    delta = (2 * (individual[i] - min_values[i]) / (max_values[i] - min_values[i])) ** (1 + (eta + random.uniform(0,1))) - 1
                else:
                    delta = 1 - (2 * (max_values[i] - individual[i]) / (max_values[i] - min_values[i])) ** (1 + (eta + random.uniform(0,1)))
                mutated_gene = max(min_values[i], min(individual[i] + delta, max_values[i]))
            else:
                mutated_gene = individual[i]
        else:
            if random.uniform(0, 1) < 0.5:
                if ((individual[i] - min_values[i]) / (max_values[i] - min_values[i])) < 0.5:
                    delta = (2 * (individual[i] - min_values[i]) / (max_values[i] - min_values[i])) ** (1 + (eta + random.uniform(0,1))) - 1
                else:
                    delta = 1 - (2 * (max_values[i] - individual[i]) / (max_values[i] - min_values[i])) ** (1 + (eta + random.uniform(0,1)))
                mutated_gene = max(min_values[i], min(individual[i] + delta, max_values[i]))
            else:
                mutated_gene = individual[i]
        mutated_genes.append(mutated_gene)
    return mutated_genes


# #####################################################################################
# #################################### Graphs #########################################
# #####################################################################################

def create_boxplot(history):
    """Create a boxplot for each variable in the population history"""
    num_generations = len(history)
    num_variables = len(history[0][0])
    data = [[history[gen][ind][var] for ind in range(len(history[gen]))] for gen in range(num_generations) for var in range(num_variables)]
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data, patch_artist = True, notch = 'True', vert = 0)
    colors = ['pink', 'lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_yticklabels(['variable '+str(i+1) for i in range(num_variables)]*num_generations)
    ax.set_xlabel('Value')
    ax.set_ylabel('Variable')
    ax.set_title('Boxplot of variables over generations')
    plt.show()


def create_plotfit(history):
    # Extract the fitness values of the best gene for each generation
    fitness_values = [min(fitness(population)) for population in history]

    # Plot the fitness values over the number of generations
    plt.plot(fitness_values)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()

def parallel_coordinates(history):
    """Create a parallel coordinates graph of the population history."""
    history_df = pd.DataFrame(history)
    fig = px.parallel_coordinates(history_df, color='generation')
    fig.show()

