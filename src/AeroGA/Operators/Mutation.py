"""
Functions dedicated to the genetic algorithm's mutation operators.
"""

import numpy as np
import random
import math
from AeroGA.Classes.Error import ErrorType

def gaussian_mutation(individual = list, min_values = list, max_values = list, std_dev = float):
    """Perform gaussian mutation on an individual."""

    try:
        mutated_genes = []

        for i in range(len(individual)):
            if isinstance(individual[i], int):
                mutated_gene = min(max(round(random.gauss(individual[i], std_dev)),min_values[i]), max_values[i])
            else:
                mutated_gene = round(min(max(random.gauss(individual[i], std_dev),min_values[i]), max_values[i]),4)
            mutated_genes.append(mutated_gene)

        return mutated_genes
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'gaussian_mutation')
        return error.message

def polynomial_mutation(individual, min_values, max_values, eta):
    """Perform polynomial mutation on an individual with relative adjustments."""
    try:
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
    except Exception as e:
        return f"Error in polynomial_mutation: {str(e)}"

def online_parameter(online_control = bool, num_generations = int):
    """Calculate the probability for crossover and mutation each generation, the values respscts a exponencial function, that for mutation
       decreases each generation and increases for crossover. If online control is False than it is used the fixed parameters. 
    
        # MUTPB_LIST: Mutation Probability
        # CXPB_LIST: Crossover Probability
    """

    mutation_prob = 0.2
    crossover_prob = 1

    try:
        if online_control:
            line_x = np.linspace(start=1, stop=50, num=num_generations)
            MUTPB_LIST = (-(np.log10(line_x) - np.log10(line_x[0]))/(np.log10(line_x[-1]) - np.log10(line_x[0])) + 1) * mutation_prob
            
            # line_x = np.linspace(start=1, stop=5, num=num_generations)
            # CXPB_LIST = (np.log10(line_x) - np.log10(line_x[0]))/(np.log10(line_x[-1]) - np.log10(line_x[0]))
            CXPB_LIST = [crossover_prob]*num_generations
        else:
            MUTPB_LIST = [mutation_prob]*num_generations
            CXPB_LIST = [crossover_prob]*num_generations
        
        return MUTPB_LIST, CXPB_LIST
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'online_parameter')
        return error.message

def online_expected_diversity(current_generation = int, max_generations = int, current_diversity = float, old_mut_param = dict, func = str):
    """Model the relationship between the number of generations and diversity."""
    
    try:
        generation = max(0, min(current_generation, max_generations))
        midpoint = max_generations / 2.0 # Adjusts the parameters of the sigmoidal function
        steepness = 0.1  # You can adjust this value to control the steepness of the curve
        
        # functions to control diversity
        if func == 'sigmoidal':
            expected_diversity = 1 / (1 + math.exp(-steepness * (generation - midpoint)))
        elif func == 'inverse_sigmoidal':
            expected_diversity = 1 - (1 / (1 + math.exp(-steepness * (generation - midpoint))))
        elif func == 'linear':
            expected_diversity = 1 + (-1 / max_generations) * generation
        
        if current_generation == 0:
            std_dev = random.uniform(0.1, 0.3)
            eta = random.randint(10, 20)
        else:
            if current_diversity < expected_diversity:
                std_dev = abs(old_mut_param["std_dev"][current_generation-1] + old_mut_param["std_dev"][current_generation-1]*(current_diversity/expected_diversity))
                eta = abs(old_mut_param["eta"][current_generation-1] + old_mut_param["eta"][current_generation-1]*(current_diversity/expected_diversity))
            elif current_diversity > expected_diversity:
                std_dev = abs(old_mut_param["std_dev"][current_generation-1] - old_mut_param["std_dev"][current_generation-1]*(expected_diversity/current_diversity))
                eta = abs(old_mut_param["eta"][current_generation-1] - old_mut_param["eta"][current_generation-1]*(expected_diversity/current_diversity))
            elif current_diversity == expected_diversity:
                std_dev = old_mut_param["std_dev"][current_generation-1]
                eta = old_mut_param["eta"][current_generation-1]

        old_mut_param["std_dev"].append(std_dev); old_mut_param["eta"].append(eta)
        return std_dev, eta, old_mut_param
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'online_expected_diversity')
        return error.message
    


















#     """
# Functions dedicated to the genetic algorithm's mutation operators.
# """

# import numpy as np
# import random
# import math
# from AeroGA.Classes.Error import ErrorType

# def gaussian_mutation(individual = list, min_values = list, max_values = list, std_dev = float):
#     """Perform gaussian mutation on an individual."""

#     try:
#         mutated_genes = []

#         for i in range(len(individual)):
#             # mutated_gene = round(min(max(random.gauss(individual[i], std_dev),min_values[i]), max_values[i]),4)
#             mutated_gene = individual[i] + random.gauss(0, std_dev) * (max_values[i] - min_values[i])
#             mutated_genes.append(mutated_gene)

#         return mutated_genes
#     except Exception as e:
#         error = ErrorType("ValueError", str(e), 'gaussian_mutation')
#         return error.message

# def polynomial_mutation(individual, min_values, max_values, eta):
#     """Perform polynomial mutation on an individual with relative adjustments."""
#     try:
#         mutated_genes = []

#         for i in range(len(individual)):
#             if max_values[i] != min_values[i]:  # Garante que há uma faixa válida para mutação
                
#                 if individual[i] < 0.5:
#                     delta = (2 * individual[i]) ** (1 + (eta + random.uniform(0,1))) - 1
#                 else:
#                     delta = 1 - (2 * (1 - individual[i])) ** (1 + (eta + random.uniform(0,1)))

#                 # O novo gene já está no intervalo normalizado [0,1]
#                 mutated_gene = individual[i] + delta

#                 # Garante que está dentro dos limites de normalização
#                 mutated_gene = max(0, min(mutated_gene, 1))

#                 # Arredondamento adequado
#                 mutated_gene = round(mutated_gene, 4)
#             else:
#                 mutated_gene = individual[i]  # Sem mutação se min == max

#             mutated_genes.append(mutated_gene)
        
#         return mutated_genes
#     except Exception as e:
#         return f"Error in polynomial_mutation: {str(e)}"

# def online_parameter(online_control = bool, num_generations = int):
#     """Calculate the probability for crossover and mutation each generation, the values respscts a exponencial function, that for mutation
#        decreases each generation and increases for crossover. If online control is False than it is used the fixed parameters. 
    
#         # MUTPB_LIST: Mutation Probability
#         # CXPB_LIST: Crossover Probability
#     """

#     mutation_prob = 0.3
#     crossover_prob = 1

#     try:
#         if online_control:
#             line_x = np.linspace(start=1, stop=50, num=num_generations)
#             MUTPB_LIST = (-(np.log10(line_x) - np.log10(line_x[0]))/(np.log10(line_x[-1]) - np.log10(line_x[0])) + 1) * mutation_prob
            
#             # line_x = np.linspace(start=1, stop=5, num=num_generations)
#             # CXPB_LIST = (np.log10(line_x) - np.log10(line_x[0]))/(np.log10(line_x[-1]) - np.log10(line_x[0]))
#             CXPB_LIST = [crossover_prob]*num_generations
#         else:
#             MUTPB_LIST = [mutation_prob]*num_generations
#             CXPB_LIST = [crossover_prob]*num_generations
        
#         return MUTPB_LIST, CXPB_LIST
#     except Exception as e:
#         error = ErrorType("ValueError", str(e), 'online_parameter')
#         return error.message

# def online_expected_diversity(current_generation = int, max_generations = int, current_diversity = float, old_mut_param = dict, func = str):
#     """Model the relationship between the number of generations and diversity."""
    
#     try:
#         generation = max(0, min(current_generation, max_generations))
#         midpoint = max_generations / 2.0 # Adjusts the parameters of the sigmoidal function
#         steepness = 0.1  # You can adjust this value to control the steepness of the curve
        
#         # functions to control diversity
#         if func == 'sigmoidal':
#             expected_diversity = 1 / (1 + math.exp(-steepness * (generation - midpoint)))
#         elif func == 'inverse_sigmoidal':
#             expected_diversity = 1 - (1 / (1 + math.exp(-steepness * (generation - midpoint))))
#         elif func == 'linear':
#             expected_diversity = 1 + (-1 / max_generations) * generation
        
#         if current_generation == 0:
#             std_dev = random.uniform(0.1, 0.3)
#             eta = random.randint(10, 20)
#         else:
#             if current_diversity < expected_diversity:
#                 std_dev = abs(old_mut_param["std_dev"][current_generation-1] + old_mut_param["std_dev"][current_generation-1]*(current_diversity/expected_diversity))
#                 eta = abs(old_mut_param["eta"][current_generation-1] + old_mut_param["eta"][current_generation-1]*(current_diversity/expected_diversity))
#             elif current_diversity > expected_diversity:
#                 std_dev = abs(old_mut_param["std_dev"][current_generation-1] - old_mut_param["std_dev"][current_generation-1]*(expected_diversity/current_diversity))
#                 eta = abs(old_mut_param["eta"][current_generation-1] - old_mut_param["eta"][current_generation-1]*(expected_diversity/current_diversity))
#             elif current_diversity == expected_diversity:
#                 std_dev = old_mut_param["std_dev"][current_generation-1]
#                 eta = old_mut_param["eta"][current_generation-1]

#         old_mut_param["std_dev"].append(std_dev); old_mut_param["eta"].append(eta)
#         return std_dev, eta, old_mut_param
#     except Exception as e:
#         error = ErrorType("ValueError", str(e), 'online_expected_diversity')
#         return error.message