import time
import random
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
from bisect import bisect_left 
import pandas as pd
import plotly.express as px
from statistics import mean 
from ypstruct import structure

class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = None

    def calculate_fitness(self, fitness_fn):
        self.fitness = fitness_fn(self.genes)

# #####################################################################################
# ###################################### Main #########################################
# #####################################################################################

def genetic_algorithm(methods, num_variables, min_values, max_values, population_size, mutation_rate, eta, std_dev, num_generations, crossover_rate, alpha, tournament_size, fitness_fn, elite_count):
    """Perform the genetic algorithm to find an optimal solution."""

    t_inicial = time.time()

    # Generating initial population
    population = generate_population(population_size, num_variables, min_values, max_values)

    # Creating history, metrics and best/avg lists
    history = [population]; best_fit = []; avg_fit = []; metrics = []

    # Initial value for the best fitness
    best_fitness = float('inf')
    best_individual = population[0]

    # Initializing the main loop
    for generation in range(num_generations):

        # Calculating the fitness values
        fitness_values = fitness(population, fitness_fn, methods.n_threads)

        # Population sorted by the fitness value
        population = [x for _,x in sorted(zip(fitness_values,population))]
        fitness_values = sorted(fitness_values)

        # Best and average fitness and best individual at the generation
        best_fitness_in_gen = min(fitness_values)
        avg_fitness_in_gen = mean(fitness_values)
        best_individual = population[fitness_values.index(best_fitness_in_gen)]

        # Checking if the best fit is better than previus generations
        if best_fitness_in_gen < best_fitness:
            best_fitness = best_fitness_in_gen
            best_individual = population[fitness_values.index(best_fitness_in_gen)]

        # Saving these values in lists
        best_fit.append(best_fitness)
        avg_fit.append(avg_fitness_in_gen)
        metrics.append(diversity_metric(population))
        
        # Applying the online parameter control
        MUTPB_LIST, CXPB_LIST = online_parameter(True, num_generations)

        print("Generation: {} | Best Fitness: {} | Diversity Metric: {}".format(generation+1, best_fitness, metrics[generation]))

        # Creating new population and aplying elitist concept
        new_population = []
        new_population = population[:elite_count]

        for i in range(0, population_size - elite_count, 2):
            
            if methods.selection == 'tournament':
                parent1, parent2 = tournament_selection(population, fitness_values, tournament_size)
            elif methods.selection == 'rank':
                parent1, parent2 = rank_selection(population, fitness_values)
            elif methods.selection == 'roulette':
                parent1, parent2 = roulette_selection(population, fitness_values)

            # Applying crossover to the individuals
            if random.uniform(0, 1) <= CXPB_LIST[generation]: 
                offspring1, offspring2 = arithmetic_crossover(parent1, parent2, alpha)
                new_population.append(offspring1)
                new_population.append(offspring2)
            else:
                new_population.append(parent1)
                new_population.append(parent2)
        
        # Applying mutation to the new population
        if methods.mutation == 'polynomial':
            population = [polynomial_mutation(ind, min_values, max_values, eta) if random.uniform(0, 1) <= MUTPB_LIST[generation] else ind for ind in new_population]
        elif methods.mutation == 'gaussian':
            population = [gaussian_mutation(ind, min_values, max_values, std_dev) if random.uniform(0, 1) <= MUTPB_LIST[generation] else ind for ind in new_population]

        # Saving new population in history
        history.append(population)
    
    print("Best Individual: {}".format(best_individual))
    print(f"Tempo de Execução: {time.time() - t_inicial}")

    # Listing outputs
    out = dict(population = population, 
               history = history, 
               best_individual = best_individual, 
               best_fit = best_fit, 
               avg_fit = avg_fit, 
               metrics = metrics)

    return out

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
                mutated_gene = round(max(min_values[i], min(individual[i] + delta, max_values[i])))
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
# #################################### Metrics ########################################
# #####################################################################################

def diversity_metric(population):
    diversity = 0
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            ind1 = Individual(population[i])
            ind2 = Individual(population[j])
            diversity += sum((ind1.genes[k] - ind2.genes[k])**2 for k in range(len(ind1.genes)))
    return diversity


# #####################################################################################
# ################################ Online Parameters ##################################
# #####################################################################################

def online_parameter(Use, num_generations):

    # MUTPB_LIST: Mutation Probability
    # CXPB_LIST: Crossover Probability

    if Use == True:
        line_x = np.linspace(start=1, stop=50, num=num_generations)
        MUTPB_LIST = (-(np.log10(line_x) - np.log10(line_x[0]))/(np.log10(line_x[-1]) - np.log10(line_x[0])) + 1) * 0.2
        
        line_x = np.linspace(start=1, stop=5, num=num_generations)
        CXPB_LIST = (np.log10(line_x) - np.log10(line_x[0]))/(np.log10(line_x[-1]) - np.log10(line_x[0]))
    else:
        MUTPB_LIST = [0.2]*num_generations
        CXPB_LIST = [1.0]*num_generations

    return MUTPB_LIST, CXPB_LIST


# #####################################################################################
# #################################### Graphs #########################################
# #####################################################################################

def sensibility(individual, fitness_fn, increment, min_values, max_values):
    """Calculate the fitness of an individual for each iteration, where one variable is incremented by a given value within the range of min and max values.
    If variable is integer, it will increment by 1 instead of a float value
    """
    dict = {"nvar":[],"value":[],"fit":[]};

    for i in range(len(individual)):
        current_value = individual[i]
        for new_value in np.arange(min_values[i], max_values[i], increment):
            new_individual = individual.copy()
            if isinstance(new_individual[i], int):
                new_value = int(new_value)
            new_individual[i] = new_value
            dict["nvar"].append(i)
            dict["value"].append(new_value)
            dict["fit"].append(fitness_fn(new_individual))
    return print(pd.DataFrame(dict))

def create_boxplot(history):
    """Create a boxplot for each variable in the population history        TA RUIM TEM Q VER"""
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


def create_plotfit(num_generations, bestfit, avgfit):
    """Plot the fit values over the number of generations"""
    fig = plt.figure()
    plt.plot(bestfit, label = "Best Fitness")
    plt.plot(avgfit, alpha = 0.3, linestyle = "--", label = "Average Fitness")
    plt.xlim(0, num_generations+1)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title('GA Convergence')
    plt.grid(True)
    plt.show()

def parallel_coordinates(history):
    """Create a parallel coordinates graph of the population history.        TA RUIM TEM Q VER"""
    history_df = pd.DataFrame(history)
    fig = px.parallel_coordinates(history_df, color='generation')
    fig.show()

def create_plotmetric(metrics):
    """Plot the metric values over the number of generations"""
    plt.plot(metrics)
    plt.xlabel('Generation')
    plt.ylabel('Diversity Metric')
    plt.title('GA Diversity Metric')
    plt.grid(True)
    plt.show()