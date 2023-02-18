from datetime import datetime
import time
import random
import numpy as np
import pandas as pd
import multiprocessing
from statistics import mean
from bisect import bisect_left
import matplotlib.pyplot as plt
import plotly.express as px
from functools import partial
from . import settings

class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = None

    def calculate_fitness(self, fitness_fn):
        self.fitness = fitness_fn(self.genes)

# #####################################################################################
# ###################################### Main #########################################
# #####################################################################################

def optimize(selection = "tournament", crossover = "1-point", mutation = "gaussian", n_threads = -1,
    min_values = list, max_values = list, num_variables = int, population_size = int, num_generations = int, elite_count = int,
    online_control = False, mutation_rate = 0.4, crossover_rate = 1, eta = 20, std_dev = 0.1,
    plotfit = True, plotbox = False, plotparallel = False, 
    fitness_fn = None
    ):
    """Perform the genetic algorithm to find an optimal solution."""
    
    t_inicial = time.time()

    # Definition of how many threads will be used to calculate the fitness function
    if n_threads == -1: n_threads = multiprocessing.cpu_count()

    # Generating initial population
    population = generate_population(population_size, num_variables, min_values, max_values)

    # Creating history, metrics and best/avg lists
    values_gen = {"best_fit":[],"avg_fit":[],"metrics":[]}
    history = {"ind":[],"gen":[],"fit":[],"score":[]}
    history_valid = {"ind":[],"gen":[],"fit":[],"score":[]}
    best_individual = {"ind":[],"fit":[]}

    # Initial value for the best fitness
    best_fitness = float('inf')

    # Initializing the main loop
    for generation in range(num_generations):

        t_gen = time.time()

        # Calculating the fitness values
        if n_threads != 0:
            fitness_values = parallel_fitness(population, fitness_fn, n_threads)
        else:
            fitness_values = fitness(population, fitness_fn)
        
        # Population sorted by the fitness value
        population = [x for _,x in sorted(zip(fitness_values,population))]
        fitness_values = sorted(fitness_values)

        # Add to history and valid fit history
        for i in range(len(population)):
            history["ind"].append(population[i])
            history["fit"].append(fitness_values[i])
            if fitness_values[i] != 0:
                history["score"].append(1/fitness_values[i])
            else:
                history["score"].append(float('inf'))
            history["gen"].append(generation)
            if fitness_values[i] < 1000:
                history_valid["ind"].append(population[i])
                history_valid["fit"].append(fitness_values[i])
                if fitness_values[i] != 0:
                    history_valid["score"].append(1/fitness_values[i])
                else:
                    history_valid["score"].append(float('inf'))
                history_valid["gen"].append(generation)

        # Best and average fitness and best individual at the generation
        best_individual["ind"].append(population[fitness_values.index(min(fitness_values))])
        best_individual["fit"].append(min(fitness_values))

        # Checking if the best fit is better than previus generations and the global value to plot the individual
        if best_individual["fit"][generation] < best_fitness:
            best_fitness = best_individual["fit"][generation]

        # Creating list of fitness values with individuals that returns valid score
        fitness_values_valid = []
        for i in range(population_size):
            if fitness_values[i] < 1000:
                fitness_values_valid.append(fitness_values[i]) 

        # Saving these values in lists
        values_gen["best_fit"].append(best_fitness)
        if isinstance(mean(fitness_values_valid), float):
            values_gen["avg_fit"].append(mean(fitness_values_valid))
        else:
            values_gen["avg_fit"].append(None)
        values_gen["metrics"].append(diversity_metric(population))
        
        # Applying the online parameter control
        MUTPB_LIST, CXPB_LIST = online_parameter(online_control, num_generations, mutation_rate, crossover_rate)

        if best_individual["fit"][generation] == 0:
            settings.log.info('Generation: {} | Time: {} | Best Fitness: {} -> Score: {} | Diversity Metric: {}'.format(generation+1, round(time.time() - t_gen, 2), best_individual["fit"][generation], float('inf'), round(values_gen["metrics"][generation],2)))
        else:    
            settings.log.info('Generation: {} | Time: {} | Best Fitness: {} -> Score: {} | Diversity Metric: {}'.format(generation+1, round(time.time() - t_gen, 2), best_individual["fit"][generation], 1/best_individual["fit"][generation], round(values_gen["metrics"][generation],2)))

        # Creating new population and aplying elitist concept  time.time() - t_inicial
        new_population = []
        if elite_count != 0:
            new_population = population[:elite_count]

        # Creating new population based on crossover methods
        for i in range(0, population_size - elite_count, 2):
            if selection == 'tournament':
                parent1 = tournament_selection(population, fitness_values, tournament_size=2)
                fitness_values.remove(fitness_values[population.index(parent1)])
                population.remove(parent1)

                parent2 = tournament_selection(population, fitness_values, tournament_size=2)
                fitness_values.remove(fitness_values[population.index(parent2)])
                population.remove(parent2)

            elif selection == 'rank':
                parent1 = rank_selection(population, fitness_values)
                fitness_values.remove(fitness_values[population.index(parent1)])
                population.remove(parent1)

                parent2 = rank_selection(population, fitness_values)
                fitness_values.remove(fitness_values[population.index(parent2)])
                population.remove(parent2)

            elif selection == 'roulette':
                parent1 = roulette_selection(population, fitness_values)
                fitness_values.remove(fitness_values[population.index(parent1)])
                population.remove(parent1)

                parent2 = roulette_selection(population, fitness_values)
                fitness_values.remove(fitness_values[population.index(parent2)])
                population.remove(parent2)

            # Applying crossover to the individuals
            if crossover == 'arithmetic':
                if random.uniform(0, 1) <= CXPB_LIST[generation]: 
                    offspring1, offspring2 = arithmetic_crossover(parent1, parent2, min_values, max_values, alpha = 0.05)
                    new_population.append(offspring1)
                    new_population.append(offspring2)
                else:
                    new_population.append(parent1)
                    new_population.append(parent2)
            elif crossover == 'SBX':
                if random.uniform(0, 1) <= CXPB_LIST[generation]: 
                    offspring1, offspring2 = SBX_crossover(parent1, parent2, min_values, max_values, eta=0.5)
                    new_population.append(offspring1)
                    new_population.append(offspring2)
                else:
                    new_population.append(parent1)
                    new_population.append(parent2)
            elif crossover == '1-point':
                if random.uniform(0, 1) <= CXPB_LIST[generation]: 
                    offspring1, offspring2 = crossover_1pt(parent1, parent2)
                    new_population.append(offspring1)
                    new_population.append(offspring2)
                else:
                    new_population.append(parent1)
                    new_population.append(parent2)
            elif crossover == '2-point':
                if random.uniform(0, 1) <= CXPB_LIST[generation]: 
                    offspring1, offspring2 = crossover_2pt(parent1, parent2)
                    new_population.append(offspring1)
                    new_population.append(offspring2)
                else:
                    new_population.append(parent1)
                    new_population.append(parent2)

        # Ensuring that the new population will have the correct size
        if len(new_population) != population_size:
            aux = len(new_population) - population_size
            new_population = new_population[ : -aux]
        
        # Applying mutation to the new population
        if mutation == 'polynomial':
            population = [polynomial_mutation(ind, min_values, max_values, eta) if random.uniform(0, 1) <= MUTPB_LIST[generation] else ind for ind in new_population]
        elif mutation == 'gaussian':
            population = [gaussian_mutation(ind, min_values, max_values, std_dev) if random.uniform(0, 1) <= MUTPB_LIST[generation] else ind for ind in new_population]

    
    # Printing global optimization results
    settings.log.warning("***************************** END ******************************")
    settings.log.warning('Best Global Individual: {}'.format(best_individual["ind"][best_individual["fit"].index(min(best_individual["fit"]))]))
    settings.log.warning('Best Global Fitness: {}'.format(min(best_individual["fit"])))
    settings.log.warning(f"Tempo de Execução: {time.time() - t_inicial}")

    # Listing outputs
    out = dict(history = history,
               history_valid = history_valid,
               best_individual = best_individual,
               values_gen = values_gen,
               )

    export_excell(out)

    if plotfit == True:
        create_plotfit(num_generations, values_gen)
    if plotbox == True:
        create_boxplots(out, num_generations, min_values, max_values)
    if plotparallel == True:
        parallel_coordinates(out)

    return out

# #####################################################################################
# ##################################### Fitness #######################################
# #####################################################################################

def parallel_fitness(population, fitness_fn, num_processes):
    with multiprocessing.Pool(num_processes) as pool:
        fitness_values = pool.map(fitness_fn, population)
    return fitness_values

# Fitness function without using multi threads
def fitness(population, fitness_fn):
    """Calculate the fitness of each individual in the population."""
    return [fitness_fn(ind) for ind in population]       

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
    pick = random.uniform(0, 1/total_fitness)

    current = 0
    for i, ind in enumerate(population):
        current += 1/fitness_values[i]
        if current > pick:
            parent = ind
            break
    
    return parent

def tournament_selection(population, fitness_values, tournament_size):
    """Select two parents using tournament selection."""
    if len(population) < tournament_size:
        tournament_pop = population
    else:
        tournament_pop = random.sample(population, tournament_size)
    tournament_fitness = [fitness_values[population.index(ind)] for ind in tournament_pop]
    parent = tournament_pop[tournament_fitness.index(min(tournament_fitness))]

    return parent

def rank_selection(population, fitness_values):
    """Select two parents using rank selection."""
    n = len(population)
    fitness_ranks = list(reversed(sorted(range(1, n+1), key=lambda x: fitness_values[x-1])))
    cumulative_prob = [sum(fitness_ranks[:i+1])/sum(fitness_ranks) for i in range(n)]
    parent = population[bisect_left(cumulative_prob, random.random())]

    return parent

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
                if max_values[i] != min_values[i]:
                    if ((individual[i] - min_values[i]) / (max_values[i] - min_values[i])) < 0.5:
                        delta = (2 * (individual[i] - min_values[i]) / (max_values[i] - min_values[i])) ** (1 + (eta + random.uniform(0,1))) - 1
                    else:
                        delta = 1 - (2 * (max_values[i] - individual[i]) / (max_values[i] - min_values[i])) ** (1 + (eta + random.uniform(0,1)))
                    mutated_gene = round(max(min_values[i], min(individual[i] + delta, max_values[i])))
                else:
                    if ((individual[i] - min_values[i])) < 0.5:
                        delta = 0
                    else:
                        delta = 0
                    mutated_gene = round(max(min_values[i], min(individual[i] + delta, max_values[i])))
            else:
                mutated_gene = individual[i]
        else:
            if random.uniform(0, 1) < 0.5:
                if max_values[i] != min_values[i]:
                    if ((individual[i] - min_values[i]) / (max_values[i] - min_values[i])) < 0.5:
                        delta = (2 * (individual[i] - min_values[i]) / (max_values[i] - min_values[i])) ** (1 + (eta + random.uniform(0,1))) - 1
                    else:
                        delta = 1 - (2 * (max_values[i] - individual[i]) / (max_values[i] - min_values[i])) ** (1 + (eta + random.uniform(0,1)))
                    mutated_gene = max(min_values[i], min(individual[i] + delta, max_values[i]))
                else:
                    if ((individual[i] - min_values[i])) < 0.5:
                        delta = 0
                    else:
                        delta = 0
                    mutated_gene = max(min_values[i], min(individual[i] + delta, max_values[i]))
            else:
                mutated_gene = individual[i]
        mutated_genes.append(mutated_gene)
    return mutated_genes


# #####################################################################################
# #################################### Metrics ########################################
# #####################################################################################

def diversity_metric(population):
    """Calculate the sum of euclidian distance for each generation whice represents the diversity of the current population."""

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
    """Calculate the probability for crossover and mutation each generation, the values respscts a exponencial function, that for mutation
       decreases each generation and increases for crossover. If online control is False than it is used the fixed parameters. 
    
        # MUTPB_LIST: Mutation Probability
        # CXPB_LIST: Crossover Probability
    """

    if online_control == True:
        line_x = np.linspace(start=1, stop=50, num=num_generations)
        MUTPB_LIST = (-(np.log10(line_x) - np.log10(line_x[0]))/(np.log10(line_x[-1]) - np.log10(line_x[0])) + 1) * 0.2
        
        # line_x = np.linspace(start=1, stop=5, num=num_generations)
        # CXPB_LIST = (np.log10(line_x) - np.log10(line_x[0]))/(np.log10(line_x[-1]) - np.log10(line_x[0]))
        CXPB_LIST = [crossover_rate]*num_generations
    else:
        MUTPB_LIST = [mutation_rate]*num_generations
        CXPB_LIST = [crossover_rate]*num_generations

    return MUTPB_LIST, CXPB_LIST


# #####################################################################################
# #################################### Graphs #########################################
# #####################################################################################

def sensibility(individual, fitness_fn, increment, min_values, max_values):
    """Calculate the fitness of an individual for each iteration, where one variable is incremented by a given value within the range of min and max values.
    If variable is integer, it will increment by 1 instead of a float value.
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

def create_plotfit(num_generations, values_gen):
    """Plot the fit and metrics values over the number of generations."""
   
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=0.5)

    ax1.plot(values_gen["best_fit"])
    ax1.set_xlim(0, num_generations - 1)
    ax1.set_title('BestFit x Iterations')
    ax1.set_ylabel('Best Fitness')
    ax1.grid(True)

    ax2.plot(values_gen["avg_fit"], alpha = 0.3, color = 'red',linestyle = "--")
    ax2.set_xlim(0, num_generations - 1)
    ax2.set_title('AvgFit x Iterations')
    ax2.set_ylabel('Average Fitness')
    ax2.grid(True)

    ax3.plot(values_gen["metrics"])
    ax3.set_xlim(0, num_generations - 1)
    ax3.set_title('Population Diversity x Iterations')
    ax3.set_ylabel('Diversity Metric')
    ax3.set_xlabel('Iterations')
    ax3.grid(True)

    plt.show()

def create_boxplots(out, num_generations, min_values, max_values):
    """Boxplot of all values used in the optimization for each variable."""

    history = out["history_valid"]    
    data = []; aux = []; aux2 = []
  
    for i in range(len(history["ind"][0])):
        for j in range(len(history["ind"])):
            aux.append(history["ind"][j][i])
        aux2.append(aux)
        aux = []

    data = list(map(lambda *x: list(x), *aux2))
    data_aux = data

    for i in range(len(data)):
        for j in range(len(min_values)):
            data_aux[i][j] = ((data[i][j] - min_values[j])/(max_values[j] - min_values[j]))

    df = pd.DataFrame(data_aux)
 
    plt.boxplot(df, vert=True)
    plt.title('Dispersion of values')
    plt.xlabel('Variables')
    plt.grid(True)
    
    plt.show()

def create_boxplots_import_xlsx(path):
    """Boxplot of all values used in the optimization for each variable."""

    df = pd.read_excel(path)
    del df["gen"]
    del df["fit"]
    del df["score"]
 
    plt.boxplot(df, vert=True)
    plt.title('Dispersion of values')
    plt.xlabel('Variables')
    plt.grid(True)
    
    plt.show()

def create_boxplots_por_gen_import_xlsx(path, n_gen, gen):       # TEM QUE NORMALIZAR OS DADOS
    """Boxplot of all values used in the generation for each variable."""

    df = pd.read_excel(path)
    del df["fit"]
    del df["score"]

    for i in range(n_gen):
        if df.iloc[i, len(df.iloc[0])-1] != gen:
            df.drop(df.index[[i]])
 
    del df["gen"]

    plt.boxplot(df, vert=True)
    plt.title('Dispersion of values')
    plt.xlabel('Variables')
    plt.grid(True)
    
    plt.show()

def parallel_coordinates(out):
    """Create a parallel coordinates graph of the population history."""
    
    history = out["history_valid"]
    lista = list(history["score"])

    num_ind = len(history["ind"])
    num_var = len(history["ind"][0])
    
    data = []; aux = []; aux2 = []
  
    for i in range(num_var):
        for j in range(num_ind):
            aux.append(history["ind"][j][i])
        aux2.append(aux)
        aux = []

    data = list(map(lambda *x: list(x), *aux2))
    df = pd.DataFrame(data)
    df['Score'] = lista
    
    fig = px.parallel_coordinates(df, color="score", dimensions=df.columns,
                              title="Parallel Coordinates Plot")
    fig.show()

def parallel_coordinates_import_xlsx(path):
    """Create a parallel coordinates graph of the population history."""
    
    df = pd.read_excel(path)
    del df["gen"]
   
    fig = px.parallel_coordinates(df, color="score", dimensions=df.columns,
                              title="Parallel Coordinates Plot")
    fig.show()


def export_excell(out):
    """Create a parallel coordinates graph of the population history.        TA RUIM TEM Q VER"""
    
    history = out["history"]
    lista = list(history["gen"])
    lista2 = list(history["fit"])
    lista3 = list(history["score"])

    num_ind = len(history["ind"])
    num_var = len(history["ind"][0])
    
    data = []; aux = []; aux2 = []
  
    for i in range(num_var):
        for j in range(num_ind):
            aux.append(history["ind"][j][i])
        aux2.append(aux)
        aux = []

    data = list(map(lambda *x: list(x), *aux2))
    df = pd.DataFrame(data)

    df['gen'] = lista
    df['fit'] = lista2
    df['score'] = lista3

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M")
    string = 'Resultados/Results_' + str(dt_string) + '.xlsx'

    df.to_excel(string, index=False)