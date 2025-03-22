"""
Functions dedicated to creating and manipulating the population.
"""

from AeroGA.Classes.Error import ErrorType
from statistics import mean
import random
import copy

# #####################################################################################
# #################################### Init Pop #######################################
# #####################################################################################

# define the genetic algorithm functions
def generate_population(size = int, num_variables = int, min_values = list, max_values = list):
    """Generate a population of random genes."""
    try:
        population = [[round(random.uniform(min_values[i], max_values[i]),4) if isinstance(min_values[i],float) else random.randint(min_values[i], max_values[i]) for i in range(num_variables)] for _ in range(size)]
        return population
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'generate_population')
        return error.message

def generate_population_normalized(size = int, num_variables = int, min_values = list, max_values = list):
    """Generate a population of normalized random genes."""

    try:
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
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'generate_population_normalized')
        return error.message
    

def create_new_population(population = list, fitness_values = list, elite_count = int, penalization_list = list, elite = str, generation = int, population_size = int):
    new_population = []
    elite_count_gen = 0
    if elite_count != 0 and mean(fitness_values) < min(penalization_list):
        if elite == "global":
            if generation == 0:
                elite_pop = population[:1]
                elite_fit = fitness_values[:1]
                if elite_count > 1:
                    for i in range(1,population_size-1):
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
                    for i in range(1,population_size-1):
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
            for i in range(population_size): 
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
                for i in range(1,population_size-1):
                    if len(new_population) < elite_count_gen:
                        if population[i] not in new_population:
                            new_population.append(population[i])
                    else:
                        break
    
    return new_population



# #####################################################################################
# ################################## Normalization ####################################
# #####################################################################################

def denormalize_individual(individual = list, min_values = list, max_values = list, classe = str):
    """Denormalize the genes of an individual."""

    try:
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
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'denormalize_individual')
        return error.message


def denormalize_population(population = list, min_values = list, max_values = list, classe = str):
    """Denormalize the entire population."""

    try:
        denormalized_population = []

        for individual in population:
            denormalized_individual = denormalize_individual(individual, min_values, max_values, classe)
            denormalized_population.append(denormalized_individual)
        
        return denormalized_population
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'denormalize_population')
        return error.message