"""
Functions dedicated to the genetic algorithm's crossover operators.
"""

import random
from AeroGA.Classes.Error import ErrorType

def arithmetic_crossover(parent1 = list, parent2 = list, min_values = list, max_values = list, alpha = 0.05):
    """Apply arithmetic crossover to produce two offspring."""

    try:
        offspring1 = []
        offspring2 = []

        for i in range(len(parent1)):
            offspring1.append(round(alpha*parent1[i] + (1-alpha)*parent2[i],4))
            offspring2.append(round(alpha*parent2[i] + (1-alpha)*parent1[i],4))

            # Ensure the offspring stay within the bounds
            offspring1[i] = min(max(offspring1[i], min_values[i]), max_values[i])
            offspring2[i] = min(max(offspring2[i], min_values[i]), max_values[i])

        return offspring1, offspring2
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'arithmetic_crossover')
        return error.message

def SBX_crossover(parent1 = list, parent2 = list, min_values = list, max_values = list, eta=20):
    """Apply SBX crossover to produce two offspring."""

    try:
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
            child1 = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
            child2 = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])

            # Ensure values remain in [0,1]
            child1 = max(0, min(child1, 1))
            child2 = max(0, min(child2, 1))

            # Round to 4 decimal places
            offspring1.append(round(child1, 4))
            offspring2.append(round(child2, 4))
            
        return offspring1, offspring2
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'SBX_crossover')
        return error.message

def crossover_1pt(parent1 = list, parent2 = list):
    """Apply 1 Point crossover to produce two offspring."""
    try:
        n = len(parent1)
        cxpoint = random.randint(1, n-1)
        offspring1 = parent1[:cxpoint] + parent2[cxpoint:]
        offspring2 = parent2[:cxpoint] + parent1[cxpoint:]
        return offspring1, offspring2
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'crossover_1pt')
        return error.message

def crossover_2pt(parent1 = list, parent2 = list):
    """Apply 2 Point crossover to produce two offspring."""
    try:
        size = len(parent1)
        cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
        parent1[cxpoint1:cxpoint2], parent2[cxpoint1:cxpoint2] = parent2[cxpoint1:cxpoint2], parent1[cxpoint1:cxpoint2]
        return parent1, parent2
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'crossover_2pt')
        return error.message
