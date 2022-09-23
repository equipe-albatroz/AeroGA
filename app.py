import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import AeroGA
import test

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
params.npop = 50                                       # Number of the Population
params.beta = 1 
params.pc = 1
params.gamma = 0.1
params.mu = 0.1                                        # Mutation Rate
params.sigma = 0.1                                     # Step of the Mutation

# Run GA
out = AeroGA.optimize(problem, params)                     # Running the Simulation

# Results
plt.plot(out.bestfit)
plt.semilogy(out.bestfit)
plt.xlim(0, params.max_iterations)
plt.xlabel('Iterations')
plt.ylabel('Best Fit')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()

