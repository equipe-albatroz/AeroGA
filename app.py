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
problem.lb = [0.2, -10, -10, -5, -5]                   # Lower Bounds
problem.ub = [0.4 , 10, 10,  5, 5]                     # Upper Bounds
problem.integer = [1,2]                                # Indice de números inteiros
problem.var_names = ['var1', 'var2', 'var3', 'var4', 'var5']

# GA Parameters
params = structure()                                   # Creating the Parameters Structure
params.max_iterations = 100                            # Number of Max Iterations
params.npop = 50                                       # Number of the Population
params.pc = 1                                          # Proportion of children compared to the fathers pop
params.mu = [0.5, 0.5, 0.5, 0.5, 0.5]                  # Mutation Rate -> It can be used only one parameter for the whole chromossome or a value for each one
params.sigma = 0.05                                     # Standart deviation of the Mutation
params.sigma_int = 0.2                                 # Standart deviation of the Mutation (integer number)
params.gamma = 0.1                                     # Arithmetic crossover amplitude
params.elitism = 0.1                                   # Elitism rate

# GA Methods
methods = structure()
methods.selection = "tournament"                       # Available methods: "roulette", "rank", "tournament", "elitism" -> Read README.md for detailed info
methods.crossover = "2-point"                          # Available methods: "arithmetic", "1-point", "2-point" -> Read README.md for detailed info
methods.mutation = "gaussian"                          # Available methods: "gaussian", "default" -> Read README.md for detailed info

# Parallel Parameters
parallel = structure()                                 # Creating the Parallel processing Structure
parallel.Multiprocessing = True
parallel.threads = 8

# Run GA
out = AeroGA.optimize(problem, params, methods)        # Running the Simulation
# out = AeroGA_parallel.optimize(problem, params, methods)        # Running the Simulation 

# Run Sensitivity Analysis
df_sensibility = AeroGA.sensibility(problem, out.bestsol)

# Graphs
fig1 = out.plots[0]
fig1.show()

# # Gráficos - Box plot
# fig2 = out.plots[2]
# plt.show()