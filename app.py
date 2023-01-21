################
## Versão 2.0 ##
################

import matplotlib.pyplot as plt
from AeroGA2 import *
from Benchmarks import Rastrigin
from ypstruct import structure

num_variables = 5
min_values = [0, 0.0, 0.0, 0, 0]
max_values = [5, 0.8, 0.9, 5, 4]
population_size = 100
mutation_rate = 0.1
num_generations = 100
crossover_rate = 1
alpha = 0.02
tournament_size = 50
eta = 20
std_dev = 0.1
fitness_fn = Rastrigin

# GA Methods
methods = structure()
methods.selection = "rank"                             # Available methods: "roulette", "rank", "tournament" -> Read README.md for detailed info
methods.crossover = "arithmetic"                       # Available methods: "arithmetic", "1-point", "2-point" -> Read README.md for detailed info
methods.mutation = "gaussian"                          # Available methods: "gaussian", "default", "polynomial" -> Read README.md for detailed info
methods.n_threads = 4                                  # Number of threads used for the fitness calculation

# Run the genetic algorithm
genes, history, best_individual = genetic_algorithm(methods, num_variables, min_values, max_values, population_size, mutation_rate, eta, std_dev, num_generations, crossover_rate, alpha, tournament_size, fitness_fn) 

# parallel_coordinates(history)
# create_plotfit(history)



################
## Versão 1.0 ##
################


# import matplotlib.pyplot as plt
# from ypstruct import structure
# import AeroGA
# from Benchmarks import Rastrigin

# # Problem Definition
# problem = structure()                                  # Creating the Problem Structure
# problem.fitness = Rastrigin                            
# problem.nvar = 15                                   
# problem.lb = [-0.2, -10, -10, 0, -5, 0, 0, 0, 0, 0,-0.2, -10, -10, 0, -5]    
# problem.ub = [0.4 , 10, 10,  5, 5, 10, 10, 10, 10, 10, 0.4 , 10, 10,  5, 5]  
# problem.integer = [1,2]                               

# # GA Parameters
# params = structure()                                   # Creating the Parameters Structure
# params.max_iterations = 100                            # Number of Max Iterations
# params.npop = 50                                       # Number of the Population
# params.pc = 2                                          # Proportion of children compared to the fathers pop
# params.sigma = 0.5                                     # Standart deviation of the Mutation
# params.sigma_int = 0.2                                 # Standart deviation of the Mutation (integer number)
# params.eta = 18
# params.elitism = 0.05                                   # Elitism rate

# # GA Methods
# methods = structure()
# methods.selection = "rank"                             # Available methods: "roulette", "rank", "tournament" -> Read README.md for detailed info
# methods.crossover = "arithmetic"                       # Available methods: "arithmetic", "1-point", "2-point" -> Read README.md for detailed info
# methods.mutation = "gaussian"                          # Available methods: "gaussian", "default", "polynomial" -> Read README.md for detailed info

# # Run GA
# out = AeroGA.optimize(problem, params, methods)        # Running the Simulation

# # Display convergence graph
# # AeroGA.plot_convergence(params,out.bestfit,out.avgfit)
# # AeroGA.plot_searchspace(problem, out.searchspace)
# fig = out.plots[1]
# plt.show()

# # Run Sensitivity Analysis
# # df_sensibility = AeroGA.sensibility(problem, out.bestsol)
# # fig = AeroGA.statistical_analysis(problem, params, methods,100)
# # plt.show()
# # fig = AeroGA.plot_pop(params, out.archive, 2)
# # plt.show()