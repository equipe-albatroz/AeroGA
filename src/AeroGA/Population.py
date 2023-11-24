import random

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
# ################################## Normalization ####################################
# #####################################################################################

def denormalize_individual(individual = list, min_values = list, max_values = list, classe = str):
    """Denormalize the genes of an individual."""
    denormalized_individual = []
    for i in range(len(individual)):
        gene = individual[i]
        min_val = min_values[i]
        max_val = max_values[i]

        if classe == "default" or "Micro":
            rounding_value = 4
        elif classe == "Regular":
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