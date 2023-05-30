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
from . import settings
import os
import copy

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
    min_values = list, max_values = list, num_variables = int, num_generations = int, elite_count = int, elite="local",
    plotfit = True, plotbox = False, plotparallel = False, TabuList = False, penalization_list = [550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
    fitness_fn = None, classe = "default"
    ):
    """Perform the genetic algorithm to find an optimal solution."""
    t_inicial = time.time()

    settings.log.warning("****************** INITIALIZING OPTIMIZATION *******************")

    # Definition of how many threads will be used to calculate the fitness function
    if n_threads == -1: n_threads = multiprocessing.cpu_count()
    settings.log.info(f"Optimization initialized with {n_threads} threads.")

    # Checking if num_varialbes matches the lb and ub sizes
    if len(max_values) != len(min_values) or num_variables != len(min_values):
        settings.log.critical("There is an inconsistency between the number of variables and the size of the bounds")
        return [0]
    
    # Creating normalized max and min values list
    max_values_norm = [1] * num_variables
    min_values_norm = [0] * num_variables

    # Defining population size
    population_size = 100
    population_size_old = population_size

    # Initial value for the best fitness
    best_fitness = float('inf')

    # Generating initial population
    population = generate_population_normalized(population_size, num_variables, min_values, max_values)

    # Creating history, metrics and best/avg lists
    values_gen = {"best_fit":[],"avg_fit":[],"metrics":[]}
    history = {"ind":[], "ind_norm":[],"gen":[],"fit":[],"score":[]}
    history_valid = {"ind":[], "ind_norm":[], "gen":[],"fit":[],"score":[]}
    best_individual = {"ind":[],"fit":[]}
    tabu_List = []

    # Initializing the main loop
    for generation in range(num_generations):
        t_gen = time.time()

        # Getting the fitness values from history if possible
        fit_pop_old = []; pop_old = []; pop_calc_fit = copy.deepcopy(population)
        for i in range(population_size_old):
            if population[i] in history["ind_norm"]:
                fit_pop_old.append(history["fit"][list(history["ind_norm"]).index(population[i])])
                pop_old.append(population[i])
                pop_calc_fit.remove(population[i])

        # Fitness calculation        
        if n_threads != 0:
            fit_values = parallel_fitness(denormalize_population(pop_calc_fit, min_values, max_values, classe), fitness_fn, n_threads)
        else:
            fit_values = fitness(denormalize_population(pop_calc_fit, min_values, max_values, classe), fitness_fn)
        
        fitness_values = fit_values + fit_pop_old
        population = pop_calc_fit + pop_old

        # Population sorted by the fitness value
        population = [x for _,x in sorted(zip(fitness_values,population))]
        fitness_values = sorted(fitness_values)
   
        # Add to history and valid fit history
        for i in range(len(population)):
            history["ind_norm"].append(population[i])
            history["fit"].append(fitness_values[i])
            if fitness_values[i] != 0:
                history["score"].append(1/fitness_values[i])
            else:
                history["score"].append(float('inf'))
            history["gen"].append(generation)
            if fitness_values[i] < min(penalization_list):
                history_valid["ind_norm"].append(population[i])
                history_valid["fit"].append(fitness_values[i])
                if fitness_values[i] != 0:
                    history_valid["score"].append(1/fitness_values[i])
                else:
                    history_valid["score"].append(float('inf'))
                history_valid["gen"].append(generation)
            else:
                tabu_List.append(population[i])            

        # Best and average fitness and best individual at the generation
        best_individual["ind"].append(denormalize_individual(population[fitness_values.index(min(fitness_values))], min_values, max_values, classe))
        best_individual["fit"].append(min(fitness_values))

        # Checking if the best fit is better than previus generations and the global value to plot the individual
        if best_individual["fit"][generation] < best_fitness:
            best_fitness = best_individual["fit"][generation]

        # Creating list of fitness values with individuals that returns valid score
        fitness_values_valid = []
        for i in range(population_size_old):
            if fitness_values[i] < min(penalization_list):
                fitness_values_valid.append(fitness_values[i]) 

        # Saving these values in lists
        values_gen["best_fit"].append(best_fitness)
        if mean(fitness_values) < min(penalization_list):
            values_gen["avg_fit"].append(mean(fitness_values_valid))
        else:
            values_gen["avg_fit"].append(None)
        values_gen["metrics"].append(diversity_metric(population))
        
        # Applying the online parameter control
        mutation_prob = random.uniform(0.05, 0.1) 
        MUTPB_LIST, CXPB_LIST = online_parameter(True, num_generations, mutation_prob, 1)

        # Printing logger informations
        if best_individual["fit"][generation] == 0:
            settings.log.info('Generation: {} | Time: {} | Population Size: {} | Best Fitness: {} -> Score: {} | Diversity Metric: {}'.format(generation+1, round(time.time() - t_gen, 2), population_size, best_individual["fit"][generation], float('inf'), round(values_gen["metrics"][generation],2)))
        else:    
            settings.log.info('Generation: {} | Time: {} | Population Size: {} | Best Fitness: {} -> Score: {} | Diversity Metric: {}'.format(generation+1, round(time.time() - t_gen, 2), population_size, best_individual["fit"][generation], round(1/best_individual["fit"][generation],2), round(values_gen["metrics"][generation],2)))

        # Creating new population and aplying elitist concept
        new_population = []
        elite_count_gen = 0
        if elite_count != 0 and mean(fitness_values) < min(penalization_list):
            if elite == "global":
                if generation == 0:
                    elite_pop = population[:1]
                    elite_fit = fitness_values[:1]
                    if elite_count > 1:
                        for i in range(1,population_size_old-1):
                            if len(elite_pop) < elite_count:
                                if population[i] not in elite_pop:
                                    elite_pop.append(population[i])
                                    elite_fit.append(fitness_values[population.index(population[i])])
                            else:
                                new_population = copy.deepcopy(elite_pop)
                                elite_pop_glob = elite_pop
                                elite_fit_glob = elite_fit
                                break
                else:
                    elite_pop = population[:1]
                    elite_fit = fitness_values[:1]
                    if elite_count > 1:
                        for i in range(1,population_size_old-1):
                            if len(elite_pop) < elite_count:
                                if population[i] not in elite_pop:
                                    elite_pop.append(population[i])
                                    elite_fit.append(fitness_values[population.index(population[i])])
                            else:
                                for j in range(elite_count):
                                    if elite_fit[j] < max(elite_fit_glob) and elite_fit[j] not in elite_fit_glob:
                                        elite_fit_glob[elite_fit_glob.index(max(elite_fit_glob))] = elite_fit[j]
                                        elite_pop_glob[elite_fit_glob.index(max(elite_fit_glob))] = elite_pop[j] 
                                new_population = copy.deepcopy(elite_pop_glob)
                                break

            elif elite == "local":
                for i in range(population_size_old): 
                    count_lst = list({x for x in fitness_values if fitness_values.count(x)})
                    for penality in penalization_list:
                        if penality in count_lst:
                            del count_lst[count_lst.index(penality)]
                        count = len(count_lst)

                if count < elite_count: 
                    elite_count_gen = count
                else:
                    elite_count_gen = elite_count
                
                new_population = population[:1]
                if elite_count_gen > 1:
                    for i in range(1,population_size_old-1):
                        if len(new_population) < elite_count_gen:
                            if population[i] not in new_population:
                                new_population.append(population[i])
                        else:
                            break

        # Creating new population based on crossover methods
        for i in range(0, population_size - len(new_population), 2):
            if selection == 'tournament':
                parent1 = tournament_selection(population, fitness_values, tournament_size=2)
                parent2 = tournament_selection(population, fitness_values, tournament_size=2)

            elif selection == 'rank':
                parent1 = rank_selection(population, fitness_values)
                parent2 = rank_selection(population, fitness_values)

            elif selection == 'roulette':
                parent1 = roulette_selection(population, fitness_values)
                parent2 = roulette_selection(population, fitness_values)

            # Applying crossover to the individuals
            if crossover == 'arithmetic':
                if random.uniform(0, 1) <= CXPB_LIST[generation]: 
                    offspring1, offspring2 = arithmetic_crossover(parent1, parent2, min_values_norm, max_values_norm, alpha = 0.05)
                    new_population.append(offspring1)
                    new_population.append(offspring2)
                else:
                    new_population.append(parent1)
                    new_population.append(parent2)
            elif crossover == 'SBX':
                if random.uniform(0, 1) <= CXPB_LIST[generation]: 
                    offspring1, offspring2 = SBX_crossover(parent1, parent2, min_values_norm, max_values_norm, eta=0.5)
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
        if len(new_population) > population_size:
            aux = len(new_population) - population_size
            new_population = new_population[ : -aux]
        population_size_old = population_size

        # Applying mutation to the new population
        std_dev = random.uniform(0.05, 0.3)
        eta = random.randint(10, 20)
        if mutation == 'polynomial':
            population = [polynomial_mutation(ind, min_values, max_values_norm, eta) if random.uniform(0, 1) <= MUTPB_LIST[generation] else ind for ind in new_population]
        elif mutation == 'gaussian':
            population = [gaussian_mutation(ind, min_values, max_values_norm, std_dev) if random.uniform(0, 1) <= MUTPB_LIST[generation] else ind for ind in new_population]

        # Checking if any inddividuals are in the Tabu list
        while TabuList is True:
            count = 0
            for i in range(population_size):
                if population[i] in tabu_List:
                    if mutation == 'polynomial':
                        polynomial_mutation(population[i], min_values, max_values_norm, eta)
                    elif mutation == 'gaussian':
                        gaussian_mutation(population[i], min_values, max_values_norm, std_dev)
                else:
                    count += 1
            if count == population_size:
                break

        # New population size       
        # population_size = population_size_old + random.randint(-1, 1)*int(population_size_old*random.betavariate(1,4))
        population_size = population_size_old + random.randint(-1, 1)*int(population_size_old*random.gauss(0,0.15))
        if population_size < population_size_old/2:
            population_size == population_size_old/2 + population_size_old*random.random()

    # Printing global optimization results
    settings.log.warning("***************************** END ******************************")
    settings.log.warning('Best Global Individual: {}'.format(best_individual["ind"][best_individual["fit"].index(min(best_individual["fit"]))]))
    settings.log.warning('Best Global Fitness: {}'.format(min(best_individual["fit"])))
    if min(best_individual["fit"]) != 0: settings.log.warning('Best Global Score: {}'.format(1/min(best_individual["fit"])))
    settings.log.warning(f"Tempo de Execução: {time.time() - t_inicial}")

    # Listing outputs
    history["ind"] = denormalize_population(history["ind_norm"], min_values, max_values, classe)
    history_valid["ind"] = denormalize_population(history_valid["ind_norm"], min_values, max_values, classe)
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

def parallel_fitness(population = list, fitness_fn = None, num_processes = int):
    """Calculate the fitness of each individual in the population."""
    with multiprocessing.Pool(num_processes) as pool:
        fitness_values = pool.map(fitness_fn, population)
    return fitness_values

# Fitness function without using multi threads
def fitness(population = list, fitness_fn = None):
    """Calculate the fitness of each individual in the population."""
    return [fitness_fn(ind) for ind in population]   

# #####################################################################################
# #################################### Init Pop #######################################
# #####################################################################################

# define the genetic algorithm functions
def generate_population(size = int, num_variables = int, min_values = list, max_values = list):
    """Generate a population of random genes."""
    population = [[round(random.uniform(min_values[i], max_values[i]),4) if isinstance(min_values[i],float) else random.randint(min_values[i], max_values[i]) for i in range(num_variables)] for _ in range(size)]
    return population

def generate_population_normalized(size = int, num_variables = int, min_values = list, max_values = list):
    """Generate a population of normalized random genes."""
    population = []
    for _ in range(size):
        individual = []
        for i in range(num_variables):
            if isinstance(min_values[i], float):
                gene = random.uniform(min_values[i], max_values[i])
            else:
                gene = random.randint(min_values[i], max_values[i])
            
            if min_values[i] == max_values[i]:
                normalized_gene = min_values[i]  # Atribui o valor mínimo diretamente
            else:
                normalized_gene = (gene - min_values[i]) / (max_values[i] - min_values[i])
            
            individual.append(round(normalized_gene, 4))
        
        population.append(individual)
    
    return population

# #####################################################################################
# ################################### Selection #######################################
# #####################################################################################

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

# #####################################################################################
# ################################### Crossover #######################################
# #####################################################################################

def arithmetic_crossover(parent1 = list, parent2 = list, min_values = list, max_values = list, alpha = 0.05):
    """Apply arithmetic crossover to produce two offspring."""
    offspring1 = []
    offspring2 = []
    for i in range(len(parent1)):
        offspring1.append(round(alpha*parent1[i] + (1-alpha)*parent2[i],4))
        offspring2.append(round(alpha*parent2[i] + (1-alpha)*parent1[i],4))
        if not isinstance(parent1[i], float):
            offspring1[i] = int(offspring1[i])
            offspring2[i] = int(offspring2[i])

        # Ensure the offspring stay within the bounds
        offspring1[i] = min(max(offspring1[i], min_values[i]), max_values[i])
        offspring2[i] = min(max(offspring2[i], min_values[i]), max_values[i])

    return offspring1, offspring2

def SBX_crossover(parent1 = list, parent2 = list, min_values = list, max_values = list, eta=0.5):
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
        offspring1.append(round(0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i]),4))
        offspring2.append(round(0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i]),4))
        
        # Ensure the offspring stay within the bounds
        offspring1[i] = min(max(offspring1[i], min_values[i]), max_values[i])
        offspring2[i] = min(max(offspring2[i], min_values[i]), max_values[i])
        
        # Round integer values to the nearest integer
        if isinstance(parent1[i], int):
            offspring1[i] = round(offspring1[i])
            offspring2[i] = round(offspring2[i])
    
    return offspring1, offspring2

def crossover_1pt(parent1 = list, parent2 = list):
    """Apply 1 Point crossover to produce two offspring."""
    n = len(parent1)
    cxpoint = random.randint(1, n-1)
    offspring1 = parent1[:cxpoint] + parent2[cxpoint:]
    offspring2 = parent2[:cxpoint] + parent1[cxpoint:]
    return offspring1, offspring2

def crossover_2pt(parent1 = list, parent2 = list):
    """Apply 2 Point crossover to produce two offspring."""
    size = len(parent1)
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
    parent1[cxpoint1:cxpoint2], parent2[cxpoint1:cxpoint2] = parent2[cxpoint1:cxpoint2], parent1[cxpoint1:cxpoint2]
    return parent1, parent2


# #####################################################################################
# #################################### Mutation #######################################
# #####################################################################################

def gaussian_mutation(individual = list, min_values = list, max_values = list, std_dev = float):
    """Perform gaussian mutation on an individual."""
    mutated_genes = []
    for i in range(len(individual)):
        if isinstance(individual[i], int):
            mutated_gene = min(max(round(random.gauss(individual[i], std_dev)),min_values[i]), max_values[i])
        else:
            mutated_gene = round(min(max(random.gauss(individual[i], std_dev),min_values[i]), max_values[i]),4)
        mutated_genes.append(mutated_gene)
    return mutated_genes

def polynomial_mutation(individual = list, min_values = list, max_values = list, eta = int):
    """Perform polynomial mutation on an individual."""
    mutated_genes = []
    for i in range(len(individual)):
        if isinstance(individual[i], int):
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
            if max_values[i] != min_values[i]:
                if ((individual[i] - min_values[i]) / (max_values[i] - min_values[i])) < 0.5:
                    delta = (2 * (individual[i] - min_values[i]) / (max_values[i] - min_values[i])) ** (1 + (eta + random.uniform(0,1))) - 1
                else:
                    delta = 1 - (2 * (max_values[i] - individual[i]) / (max_values[i] - min_values[i])) ** (1 + (eta + random.uniform(0,1)))
                mutated_gene = round(max(min_values[i], min(individual[i] + delta, max_values[i])),4)
            else:
                if ((individual[i] - min_values[i])) < 0.5:
                    delta = 0
                else:
                    delta = 0
                mutated_gene = round(max(min_values[i], min(individual[i] + delta, max_values[i])),4)
        mutated_genes.append(mutated_gene)
    return mutated_genes

# #####################################################################################
# ################################## Normalization ####################################
# #####################################################################################

def denormalize_individual(individual = list, min_values = list, max_values = list, classe = str):
    """Denormalize the genes of an individual."""
    denormalized_individual = []
    for i in range(len(individual)):
        gene = individual[i]
        min_val = min_values[i]
        max_val = max_values[i]

        if classe == "default" or "micro":
            rounding_value = 4
        elif classe == "regular":
            rounding_value = 6

        if isinstance(min_val, int) and isinstance(max_val, int):
            denormalized_gene = round((gene * (max_val - min_val)) + min_val)
        else:
            denormalized_gene = round((gene * (max_val - min_val)) + min_val, rounding_value)

        denormalized_individual.append(denormalized_gene)

    return denormalized_individual


def denormalize_population(population = list, min_values = list, max_values = list, classe = str):
    """Denormalize the entire population."""
    denormalized_population = []
    for individual in population:
        denormalized_individual = denormalize_individual(individual, min_values, max_values, classe)
        denormalized_population.append(denormalized_individual)
    
    return denormalized_population

# #####################################################################################
# #################################### Metrics ########################################
# #####################################################################################

def diversity_metric(population = list):
    """Calculate the sum of euclidian distance for each generation whice represents the diversity of the current population."""

    diversity = 0
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            ind1 = Individual(population[i])
            ind2 = Individual(population[j])
            diversity += sum((ind1.genes[k] - ind2.genes[k])**2 for k in range(len(ind1.genes)))
    return round(diversity,4)


# #####################################################################################
# ################################ Online Parameters ##################################
# #####################################################################################

def online_parameter(online_control = bool, num_generations = int, mutation_prob = float, crossover_prob = float):
    """Calculate the probability for crossover and mutation each generation, the values respscts a exponencial function, that for mutation
       decreases each generation and increases for crossover. If online control is False than it is used the fixed parameters. 
    
        # MUTPB_LIST: Mutation Probability
        # CXPB_LIST: Crossover Probability
    """

    if online_control == True:
        line_x = np.linspace(start=1, stop=50, num=num_generations)
        MUTPB_LIST = (-(np.log10(line_x) - np.log10(line_x[0]))/(np.log10(line_x[-1]) - np.log10(line_x[0])) + 1) * mutation_prob
        
        # line_x = np.linspace(start=1, stop=5, num=num_generations)
        # CXPB_LIST = (np.log10(line_x) - np.log10(line_x[0]))/(np.log10(line_x[-1]) - np.log10(line_x[0]))
        CXPB_LIST = [crossover_prob]*num_generations
    else:
        MUTPB_LIST = [mutation_prob]*num_generations
        CXPB_LIST = [crossover_prob]*num_generations

    return MUTPB_LIST, CXPB_LIST


# #####################################################################################
# #################################### Graphs #########################################
# #####################################################################################

def sensibility(individual = list, fitness_fn = None, increment = None, min_values = list, max_values = list):
    """Calculate the fitness of an individual for each iteration, where one variable is incremented by a given value within the range of min and max values.
    If variable is integer, it will increment by 1 instead of a float value.
    """
    dict = {"nvar":[],"value":[],"fit":[]}

    if isinstance(increment, float): step = [ increment for _ in range(len(min_values))]
    elif isinstance(increment, list): step = increment

    for i in range(len(individual)):
        settings.log.info('Iteração: {} de {}'.format(i, len(individual)))
        for new_value in np.arange(min_values[i], max_values[i], step[i]):
            new_individual = individual.copy()
            if isinstance(new_individual[i], int):
                new_value = int(new_value)
            new_individual[i] = new_value
            dict["nvar"].append(i)
            dict["value"].append(new_value)
            dict["fit"].append(fitness_fn(new_individual))

    df = pd.DataFrame(dict)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M")
    string = 'Resultados/Sensibility_' + str(dt_string) + '.xlsx'
    df.to_excel(string, index=False)

    return print(pd.DataFrame(dict))

def create_plotfit(num_generations = int, values_gen = None):
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

def create_boxplots(out = None, min_values = list, max_values = list):
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

def create_boxplots_import_xlsx(path = None):
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

def create_boxplots_por_gen_import_xlsx(path = None, min_values = list, max_values = list, generation = int):
    """Boxplot of all values used in the generation for each variable."""

    df = pd.read_excel(path)
    del df["fit"]
    del df["score"]
    
    filter = df['gen'] == generation
    df_aux = df[filter]
    del df_aux["gen"]

    lista = df_aux.values.tolist()
    aux = [ [ 0 for _ in range(len(lista[0]))] for _ in range(len(lista)) ]
    for i in range(len(lista)):
        for j in range(len(lista[0])):
            aux[i][j] = (lista[i][j]-min_values[j])/(max_values[j]-min_values[j])

    df = pd.DataFrame(aux)

    plt.boxplot(df, vert=True)
    plt.title('Value dispersion in generation ' + str(generation))
    plt.xlabel('Variables')
    plt.ylabel('Normalized data')
    plt.grid(True)
    
    plt.show()

def parallel_coordinates(out = None):
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

def parallel_coordinates_import_xlsx(path = None, classe = None):
    """Create a parallel coordinates graph of the population history."""
    
    df = pd.read_excel(path)
    del df["gen"]
    del df["fit"]

    micro = ['c1', 'chord_ratio2','b1','span_ratio2','iw','nperfilw1','nperfilw2','zwGround','xCG','vh', 'ih','nperfilh','motorindex']
    regular = ['b1', 'span_ratio_2', 'span_ratio_b3', 'c1', 'chord_ratio_c2', 'chord_ratio_c3', 'nperfilw1', 'nperfilw2', 'nperfilw3', 'iw', 'zwground', 'xCG', 'Vh', 'ARh', 'nperfilh', 'lt', 'it', 'xTDP', 'AtivaProfundor', 'motorIndex']

    if classe != None:
        if isinstance(classe,list):
            for i in range(df.shape[1]-1):
                df = df.rename({df.columns[i]: classe[i]}, axis='columns')
        else:
            if classe == "micro":
                nomes = micro
            else:
                nomes = regular
            for i in range(df.shape[1]-1):
                df = df.rename({df.columns[i]: nomes[i]}, axis='columns')
   
    fig = px.parallel_coordinates(df, color="score", dimensions=df.columns,
                              title="Parallel Coordinates Plot")
    fig.show()

def parallel_coordinates_per_gen_import_xlsx(path = None, classe = None, generation = int):
    """Create a parallel coordinates graph of the population history."""
    
    df_aux = pd.read_excel(path)
   
    filter = df_aux['gen'] == generation
    df = df_aux[filter]

    del df["gen"]
    del df["fit"]

    micro = ['c1', 'chord_ratio2','b1','span_ratio2','iw','nperfilw1','nperfilw2','zwGround','xCG','vh', 'ih','nperfilh','motorindex']
    regular = ['b1', 'span_ratio_2', 'span_ratio_b3', 'c1', 'chord_ratio_c2', 'chord_ratio_c3', 'nperfilw1', 'nperfilw2', 'nperfilw3', 'iw', 'zwground', 'xCG', 'Vh', 'ARh', 'nperfilh', 'lt', 'it', 'xTDP', 'AtivaProfundor', 'motorIndex']

    if classe != None:
        if isinstance(classe,list):
            for i in range(df.shape[1]-1):
                df = df.rename({df.columns[i]: classe[i]}, axis='columns')
        else:
            if classe == "micro":
                nomes = micro
            else:
                nomes = regular
            for i in range(df.shape[1]-1):
                df = df.rename({df.columns[i]: nomes[i]}, axis='columns')
   
    fig = px.parallel_coordinates(df, color="score", dimensions=df.columns,
                              title="Parallel Coordinates Plot")
    fig.show()


def export_excell(out):
    """Create a parallel coordinates graph of the population history."""
    
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

    # Check whether the specified path exists or not
    path = "Resultados"
    pathExist = os.path.exists(path)
    if not pathExist:
        os.makedirs(path)   

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M")
    string = 'Resultados/Results_' + str(dt_string) + '.xlsx'

    df.to_excel(string, index=False)