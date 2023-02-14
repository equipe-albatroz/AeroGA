import time
import random
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from bisect import bisect_left
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = None

    def calculate_fitness(self, fitness_fn):
        self.fitness = fitness_fn(self.genes)

# #####################################################################################
# ###################################### Main #########################################
# #####################################################################################

def optimize(methods, param, fitness_fn):
    """Perform the genetic algorithm to find an optimal solution."""

    t_inicial = time.time()

    # Extracting variables
    min_values = param.lb
    max_values = param.ub
    num_variables = param.num_variables
    population_size = param.population_size
    num_generations = param.num_generations
    tournament_size = param.tournament_size
    eta = param.eta
    std_dev = param.std_dev
    elite_count = param.elite_count
    online_control = param.online_control
    mutation_rate = param.mutation_rate
    crossover_rate = param.crossover_rate

    # Definition of how many threads will be used to calculate the fitness function
    if methods.n_threads == -1:
        n_threads = multiprocessing.cpu_count()
    else:
        n_threads = methods.n_threads

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
        fitness_values = fitness(population, fitness_fn, n_threads)

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
        MUTPB_LIST, CXPB_LIST = online_parameter(online_control, num_generations, mutation_rate, crossover_rate)

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
            if methods.crossover == 'arithmetic':
                if random.uniform(0, 1) <= CXPB_LIST[generation]: 
                    offspring1, offspring2 = arithmetic_crossover(parent1, parent2, min_values, max_values, alpha = 0.05)
                    new_population.append(offspring1)
                    new_population.append(offspring2)
                else:
                    new_population.append(parent1)
                    new_population.append(parent2)
            elif methods.crossover == 'SBX':
                if random.uniform(0, 1) <= CXPB_LIST[generation]: 
                    offspring1, offspring2 = SBX_crossover(parent1, parent2, min_values, max_values, eta=0.5)
                    new_population.append(offspring1)
                    new_population.append(offspring2)
                else:
                    new_population.append(parent1)
                    new_population.append(parent2)
            elif methods.crossover == '1-point':
                if random.uniform(0, 1) <= CXPB_LIST[generation]: 
                    offspring1, offspring2 = crossover_1pt(parent1, parent2)
                    new_population.append(offspring1)
                    new_population.append(offspring2)
                else:
                    new_population.append(parent1)
                    new_population.append(parent2)
            elif methods.crossover == '2-point':
                if random.uniform(0, 1) <= CXPB_LIST[generation]: 
                    offspring1, offspring2 = crossover_2pt(parent1, parent2)
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

def arithmetic_crossover(parent1, parent2, min_values, max_values, alpha = 0.05):
    """Apply arithmetic crossover to produce two offspring."""
    offspring1 = []
    offspring2 = []
    for i in range(len(parent1)):
        offspring1.append(alpha*parent1[i] + (1-alpha)*parent2[i])
        offspring2.append(alpha*parent2[i] + (1-alpha)*parent1[i])
        if not isinstance(parent1[i], float):
            offspring1[i] = int(offspring1[i])
            offspring2[i] = int(offspring2[i])

        # Ensure the offspring stay within the bounds
        offspring1[i] = min(max(offspring1[i], min_values[i]), max_values[i])
        offspring2[i] = min(max(offspring2[i], min_values[i]), max_values[i])

    return offspring1, offspring2

def SBX_crossover(parent1, parent2, min_values, max_values, eta=0.5):
    """Apply SBX crossover to produce two offspring."""
    # Generate a random number for each variable
    u = random.uniform(0, 1)
    
    # Calculate the beta value for each variable
    beta = 0
    if u <= 0.5:
        beta = (2 * u) ** (1 / (eta + 1))
    else:
        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    
    # Calculate the offspring for each variable
    offspring1 = []
    offspring2 = []
    for i in range(len(parent1)):
        offspring1.append(0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i]))
        offspring2.append(0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i]))
        
        # Ensure the offspring stay within the bounds
        offspring1[i] = min(max(offspring1[i], min_values[i]), max_values[i])
        offspring2[i] = min(max(offspring2[i], min_values[i]), max_values[i])
        
        # Round integer values to the nearest integer
        if isinstance(parent1[i], int):
            offspring1[i] = round(offspring1[i])
            offspring2[i] = round(offspring2[i])
    
    return offspring1, offspring2

def crossover_1pt(parent1, parent2):
    """Apply 1 Point crossover to produce two offspring."""
    n = len(parent1)
    cxpoint = random.randint(1, n-1)
    offspring1 = parent1[:cxpoint] + parent2[cxpoint:]
    offspring2 = parent2[:cxpoint] + parent1[cxpoint:]
    return offspring1, offspring2

def crossover_2pt(parent1, parent2):
    """Apply 2 Point crossover to produce two offspring."""
    size = len(parent1)
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
    parent1[cxpoint1:cxpoint2], parent2[cxpoint1:cxpoint2] = parent2[cxpoint1:cxpoint2], parent1[cxpoint1:cxpoint2]
    return parent1, parent2


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

def online_parameter(online_control, num_generations, mutation_rate, crossover_rate):

    # MUTPB_LIST: Mutation Probability
    # CXPB_LIST: Crossover Probability

    if online_control == True:
        line_x = np.linspace(start=1, stop=50, num=num_generations)
        MUTPB_LIST = (-(np.log10(line_x) - np.log10(line_x[0]))/(np.log10(line_x[-1]) - np.log10(line_x[0])) + 1) * 0.2
        
        line_x = np.linspace(start=1, stop=5, num=num_generations)
        CXPB_LIST = (np.log10(line_x) - np.log10(line_x[0]))/(np.log10(line_x[-1]) - np.log10(line_x[0]))
    else:
        MUTPB_LIST = [mutation_rate]*num_generations
        CXPB_LIST = [crossover_rate]*num_generations

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

def create_boxplots(history):
    num_generations = len(history) - 1
    num_individuals = len(history[0])
    num_variables = len(history[0][0])
    fig, ax = plt.subplots(1, num_variables, figsize=(15, 5))
  
    for j in range(num_individuals):
        for i in range(num_variables):
            data = np.array([individual[j][i] for individual in history])
            ax[i].boxplot(data, vert=True)
            ax[i].set_title(f'Variable {i+1}')
    
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