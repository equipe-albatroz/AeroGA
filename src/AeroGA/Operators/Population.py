"""
Functions dedicated to creating and manipulating the population.
"""

from statistics import mean
from .. import settings
import random
import copy

# #####################################################################################
# #################################### Init Pop #######################################
# #####################################################################################

# define the genetic algorithm functions
def generate_population(size = int, num_variables = int, min_values = list, max_values = list) -> list[list]:
    """Generate a population of random genes."""

    population = [[round(random.uniform(min_values[i], max_values[i]),4) if isinstance(min_values[i],float) else random.randint(min_values[i], max_values[i]) for i in range(num_variables)] for _ in range(size)]
    return population

def generate_population_normalized(size = int, num_variables = int, min_values = list, max_values = list)-> list[list]:
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
# ################################## New Population ###################################
# #####################################################################################

def create_new_population(population: list, fitness_values: list, elite_count: int, penalization_list: list, elite: str, elite_fit_glob: list, elite_pop_glob: list, generation: int, population_size: int):
    new_population = []
    elite_count_gen = 0

    population_sorted, fitness_values_sorted = zip(*sorted(zip(population, fitness_values), key=lambda x: x[1]))

    if elite_count != 0 and fitness_values_sorted and penalization_list and mean(fitness_values_sorted) < min(penalization_list):
        if elite == "global":
            elite_pop, elite_fit = [], []
            
            for ind, fit in zip(population_sorted, fitness_values_sorted):
                if len(elite_pop) < elite_count and ind not in elite_pop:
                    elite_pop.append(ind)
                    elite_fit.append(fit)

            if generation == 0:
                new_population = elite_pop[:]
                elite_pop_glob = elite_pop[:]
                elite_fit_glob = elite_fit[:]
            else:
                for j in range(len(elite_fit)):
                    if elite_fit[j] < max(elite_fit_glob) and elite_fit[j] not in elite_fit_glob:
                        idx = elite_fit_glob.index(max(elite_fit_glob))
                        elite_fit_glob[idx] = elite_fit[j]
                        elite_pop_glob[idx] = elite_pop[j]
                new_population = elite_pop_glob[:]

            # print("Elite_glob_pop: ", new_population)

        elif elite == "local":
            elite_pop, elite_fit = [], []
            
            count_lst = list({x for x in fitness_values_sorted if fitness_values_sorted.count(x)})
            for penality in penalization_list:
                if penality in count_lst:
                    del count_lst[count_lst.index(penality)]
            count = len(count_lst)

            if count < elite_count: 
                elite_count_gen = count
            else:
                elite_count_gen = elite_count
            
            for ind, fit in zip(population_sorted, fitness_values_sorted):
                if len(elite_pop) < elite_count_gen and ind not in elite_pop:
                    elite_pop.append(ind)
                    elite_fit.append(fit)

            new_population = elite_pop[:]
    
    return new_population, elite_fit_glob, elite_pop_glob

def check_population_validity(population: list):
    """ Check if pop is correct. """
    for individuo in population:
        if not all(isinstance(valor, (int, float)) for valor in individuo):
            settings.log.error("There is an invalid type within the population.")
            return False
    return True


# #####################################################################################
# ################################## Normalization ####################################
# #####################################################################################

def denormalize_individual(individual = list, min_values = list, max_values = list, classe = str) -> list:
    """Denormalize the genes of an individual."""

    denormalized_individual = []

    for i in range(len(individual)):

        if classe == "default" or "Micro":
            rounding_value = 4
        elif classe == "Regular":
            rounding_value = 6
        if isinstance(min_values[i], int):
            denormalized_gene = int(round((individual[i] * (max_values[i] - min_values[i])) + min_values[i]))
        else:
            denormalized_gene = round((individual[i] * (max_values[i] - min_values[i])) + min_values[i], rounding_value)

        denormalized_individual.append(denormalized_gene)
    return denormalized_individual


def denormalize_population(population = list, min_values = list, max_values = list, classe = str) -> list[list]:
    """Denormalize the entire population."""

    denormalized_population = []

    for individual in population:
        denormalized_individual = denormalize_individual(individual, min_values, max_values, classe)
        denormalized_population.append(denormalized_individual)
    
    return denormalized_population
    

# #####################################################################################
# ############################### Populating history ##################################
# #####################################################################################

def init_history():
    values_gen = {"best_fit":[],"avg_fit":[],"metrics":[]}
    history = {"ind":[], "ind_norm":[],"gen":[],"fit":[],"score":[]}
    history_valid = {"ind":[], "ind_norm":[], "gen":[],"fit":[],"score":[]}
    best_individual = {"ind":[],"fit":[]}
    old_mut_param = {"std_dev":[],"eta":[]}
    tabu_List = []
    return values_gen, history, history_valid, best_individual, old_mut_param, tabu_List
