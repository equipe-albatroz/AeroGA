import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import AeroGA
import AeroGA_parallel

# Sphere Test Function
def sphere(x):
    return sum(x**2)

# Problem Definition
problem = structure()                                  # Creating the Problem Structure
problem.fitness = sphere                               # Fitness Function
problem.nvar = 5                                       # Variables number
problem.lb = [-10, 0.2,  0, -5, 0.06]                  # Lower Bounds
problem.ub = [ 10, 0.4,  1,  5, 0.45]                  # Upper Bounds

# GA Parameters
params = structure()                                   # Creating the Parameters Structure
params.max_iterations = 100                            # Number of Max Iterations
params.npop = 100                                      # Number of the Population
params.pc = 1                                          # Proporção da população de filhos em relação aos pais
params.gamma = 0.1                                     # Amplitude do crossover        
params.mu = 0.1                                        # Mutation Rate
params.sigma = 0.1                                     # Step of the Mutation

# GA Methods
methods = structure()
methods.selection = "rank"                             # Available methods: "roulette", "rank", "tournament", "elitism" -> Read README.md for detailed info
methods.crossover = "arithmetic"                       # Available methods: "arithmetic", "1-point", "2-point" -> Read README.md for detailed info
methods.mutation = "default"                           # Available methods: "gaussian", "default" -> Read README.md for detailed info


# Run GA
out = AeroGA.optimize(problem, params, methods)        # Running the Simulation
# out = AeroGA_parallel.optimize(problem, params, methods)        # Running the Simulation with parallel processing (not working yet)


# Results
plt.plot(out.bestfit)
plt.xlim(0, params.max_iterations+1)
plt.xlabel('Iterations')
plt.ylabel('Best Fit')
plt.title('Fitness x Iterations')
plt.grid(True)
plt.show()

