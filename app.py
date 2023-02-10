################
## Versão 2.0 ##
################

import matplotlib.pyplot as plt
from AeroGA2 import *
from ypstruct import structure
from Benchmarks import *
# from MDOJunior import MDOJunior

num_variables = 6
min_values = [0, 0, 0.0, 0.0, 0.0, 0.0]  
max_values = [5, 4, 0.4, 1.5, 0.5, 1.0] 
population_size = 50
num_generations = 100
tournament_size = 50
mutation_rate = 0.2
crossover_rate = 1
alpha = 0.05
eta = 20
std_dev = 1.8
elite_count = 1
fitness_fn = Rastrigin 

# GA Methods
methods = structure()
methods.selection = "tournament"                             # Available methods: "roulette", "rank", "tournament" -> Read README.md for detailed info
methods.crossover = "arithmetic"                       # Available methods: "arithmetic", "1-point", "2-point" -> Read README.md for detailed info
methods.mutation = "gaussian"                          # Available methods: "gaussian", "polynomial" -> Read README.md for detailed info
methods.n_threads = 4                                  # Number of threads used for the fitness calculation

# Run the genetic algorithm
out = genetic_algorithm(methods, num_variables, min_values, max_values, population_size, mutation_rate, eta, std_dev, num_generations, crossover_rate, alpha, tournament_size, fitness_fn, elite_count) 

# genes, history, best_individual, best_fit, avg_fit, metrics

# parallel_coordinates(history)
create_plotfit(num_generations, out["best_fit"], out["avg_fit"])

# increment=0.01
# sensibility(best_individual, fitness_fn, increment, min_values, max_values)