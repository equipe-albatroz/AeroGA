"""
Functions dedicated to calculating the performance metrics of the ga population.
"""

from AeroGA.Classes.Error import ErrorType, Log
from AeroGA.Classes.Individual import Individual

# Setting error log file
ErrorLog = Log("error.log", 'Metrics')

def diversity_metric(population = list):
    """Calculate the normalized average distance between individuals in the population."""
    
    try:

        if not population or len(population) < 2:
            return 0.0

        total_distance = 0
        max_possible_distance = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                ind1 = Individual(population[i])
                ind2 = Individual(population[j])
                total_distance += sum(abs(ind1.genes[k] - ind2.genes[k]) for k in range(len(ind1.genes)))
                max_possible_distance += max(abs(ind1.genes[k] - ind2.genes[k]) for k in range(len(ind1.genes)))

        # Calculate normalized average distance
        if max_possible_distance == 0:
            normalized_average_distance = 0
        else:
            normalized_average_distance = total_distance / ((len(population[0])/2.45)*max_possible_distance)

        return round(normalized_average_distance, 4)
    except Exception as e:
        ErrorLog.error(str(e))
        return ErrorType("danger", str(e), 'diversity_metric')