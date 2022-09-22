import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga
import test

# Sphere Test Function
def sphere(x):
    return sum(x**2)

# Problem Definition
problem = structure()
problem.fitness = sphere
problem.nvar = 5
problem.lb = [-10, 0.2,  0, -5, 0.06]
problem.ub = [ 10, 0.4,  1,  5, 0.45]

# GA Parameters
params = structure()
params.max_iterations = 100
params.npop = 50
params.beta = 1
params.pc = 1
params.gamma = 0.1
params.mu = 0.1  # mutation rate
params.sigma = 0.1 # step of the mutation

# Run GA
out = ga.optimize(problem, params)

# Results
plt.plot(out.bestfit)
plt.semilogy(out.bestfit)
plt.xlim(0, params.max_iterations)
plt.xlabel('Iterations')
plt.ylabel('Best Fit')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()

