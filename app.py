import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ypstruct import structure
import AeroGA
# import AeroGA_parallel
# import AeroGA_p

# Sphere Test Function
def sphere(x):
    return sum(x**2)

# Graphs
plot_iterations = 1
plot_box = 0

# Problem Definition
problem = structure()                                  # Creating the Problem Structure
problem.fitness = sphere                               # Fitness Function
problem.nvar = 5                                       # Variables number
problem.lb = [-10, -10, -10, -5, -5]                   # Lower Bounds
problem.ub = [ 10, 10, 10,  5, 5]                      # Upper Bounds
problem.integer = [1,2]                                # Indice de números inteiros

# GA Parameters
params = structure()                                   # Creating the Parameters Structure
params.max_iterations = 100                            # Number of Max Iterations
params.npop = 50                                       # Number of the Population
params.pc = 1                                          # Proportion of children compared to the fathers pop
params.mu = [0.5, 0.5, 0.5, 0.5, 0.5]                  # Mutation Rate -> It can be used only one parameter for the whole chromossome or a value for each one
params.sigma = 0.1                                     # Standart deviation of the Mutation
params.sigma_int = 0.7                                 # Standart deviation of the Mutation (integer number)
params.gamma = 0.1                                     # Arithmetic crossover amplitude

# Parallel Parameters
parallel = structure()                                 # Creating the Parallel processing Structure
parallel.Multiprocessing = True
parallel.threads = 8

# GA Methods
methods = structure()
methods.selection = "rank"                             # Available methods: "roulette", "rank", "tournament", "elitism" -> Read README.md for detailed info
methods.crossover = "arithmetic"                       # Available methods: "arithmetic", "1-point", "2-point" -> Read README.md for detailed info
methods.mutation = "gaussian"                          # Available methods: "gaussian", "default" -> Read README.md for detailed info


# Run GA
out = AeroGA.optimize(problem, params, methods)        # Running the Simulation
# out = AeroGA_parallel.optimize(problem, params, methods)        # Running the Simulation 
# out = AeroGA_p.optimize(problem, params, methods,parallel)        # Running the Simulation 

# print(out.archive)

# Gráficos - Linear das iterações
if plot_iterations == 1:
    fig1 = out.plots[0]
    plt.show()

# Gráficos - Box plot
if plot_box == 1:
    fig2 = out.plots[1]
    plt.show()