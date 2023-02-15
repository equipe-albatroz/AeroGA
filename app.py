import matplotlib.pyplot as plt
from AeroGA.AeroGA import *
from ypstruct import structure
from Benchmarks import *
# from MDOJunior import MDOJunior


# GA Parameters
param = structure()
param.lb = [0, 0, 0.0, 0.0, 0.0, 0.0]
param.ub = [5, 4, 0.4, 1.5, 0.5, 1.0]
param.num_variables = 6
param.population_size = 50
param.num_generations = 100
param.eta = 5
param.std_dev = 0.1
param.elite_count = 1
param.online_control = False
param.mutation_rate = 0.4
param.crossover_rate = 1

# Fitness function
fitness_fn = Rastrigin

# GA Methods
methods = structure()
methods.selection = "tournament"                       # Available methods: "roulette", "rank", "tournament" -> Read README.md for detailed info
methods.crossover = "1-point"                       # Available methods: "arithmetic", "SBX" ,"1-point", "2-point" -> Read README.md for detailed info
methods.mutation = "gaussian"                          # Available methods: "gaussian", "polynomial" -> Read README.md for detailed info
methods.n_threads = 1                                  # Number of threads used for the fitness calculation

# Plots and Auxiliary functions
# AuxFn = structure()
# AuxFn.fit = True
# AuxFn.box = False
# AuxFn.parallel = False
# AuxFn.sensitivity = False
# AuxFn.sensitivity_incr = 0.01

# Run the genetic algorithm
out = optimize(methods, param, fitness_fn) # out = [população, history, best_individual, values_gen]

# create_boxplots(out["history"])
# parallel_coordinates(out["history"])
# sensibility(best_individual, fitness_fn, increment=0.01, min_values, max_values)