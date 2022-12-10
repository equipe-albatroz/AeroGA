import matplotlib.pyplot as plt
from ypstruct import structure
import AeroGA
from Benchmarks import Rastrigin

# Problem Definition
problem = structure()                                  # Creating the Problem Structure
problem.fitness = Rastrigin                            # Fitness Function
problem.nvar = 5                                       # Variables number
problem.lb = [0.2, -10, -10, -5, -5]                   # Lower Bounds
problem.ub = [0.4 , 10, 10,  5, 5]                     # Upper Bounds
problem.integer = [1,2]                                # Indice de números inteiros

# GA Parameters
params = structure()                                   # Creating the Parameters Structure
params.max_iterations = 100                            # Number of Max Iterations
params.npop = 50                                       # Number of the Population
params.pc = 1                                          # Proportion of children compared to the fathers pop
params.sigma = 0.05                                    # Standart deviation of the Mutation
params.sigma_int = 0.2                                 # Standart deviation of the Mutation (integer number)
params.elitism = 0.1                                   # Elitism rate

# GA Methods
methods = structure()
methods.selection = "roulette"                       # Available methods: "roulette", "rank", "tournament", "elitism" -> Read README.md for detailed info
methods.crossover = "2-point"                          # Available methods: "arithmetic", "1-point", "2-point" -> Read README.md for detailed info
methods.mutation = "gaussian"                          # Available methods: "gaussian", "default" -> Read README.md for detailed info

# Run GA
out = AeroGA.optimize(problem, params, methods)        # Running the Simulation

# Display convergence graph
AeroGA.plot_convergence(params,out.bestfit,out.avgfit)
plt.show()

# Best solution found
print(out.bestsol)

# Run Sensitivity Analysis
df_sensibility = AeroGA.sensibility(problem, out.bestsol)
