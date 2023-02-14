import matplotlib.pyplot as plt
from AeroGA import *
from ypstruct import structure
from Benchmarks import *
import sys
import os


# GA Parameters
param = structure()
param.lb = [0, 0, 0.0, 0.0, 0.0, 0.0]
param.ub = [5, 4, 0.4, 1.5, 0.5, 1.0]
param.num_variables = 6
param.population_size = 50
param.num_generations = 30
param.eta = 20
param.std_dev = 1.8
param.elite_count = 2
param.online_control = True
param.mutation_rate = 0.2
param.crossover_rate = 1

# Fitness function
fitness_fn = Rastrigin

# GA Methods
methods = structure()
methods.selection = "rank"                       # Available methods: "roulette", "rank", "tournament" -> Read README.md for detailed info
methods.crossover = "SBX"                       # Available methods: "arithmetic", "SBX" ,"1-point", "2-point" -> Read README.md for detailed info
methods.mutation = "gaussian"                          # Available methods: "gaussian", "polynomial" -> Read README.md for detailed info
methods.n_threads = -1                                  # Number of threads used for the fitness calculation

# Run the genetic algorithm
out = optimize(methods, param, fitness_fn) # out = [população, history, best_individual, best_fit, avg_fit, metrics]

create_plotfit(param.num_generations, out["best_fit"], out["avg_fit"], out["metrics"])
# create_boxplots(out["history"])
# parallel_coordinates(out["history"])

# increment=0.01
# sensibility(best_individual, fitness_fn, increment, min_values, max_values)