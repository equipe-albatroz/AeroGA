import numpy as np
import pandas as pd
from ypstruct import structure
import matplotlib.pyplot as plt
from ypstruct import structure


# Sphere Test Function
def sphere(x):
    return sum(x**2)
fitness = sphere                               # Fitness Function

nvar = 5                                       # Variables number
lb = [-10, 0.2,  0, -5, 0.06]                  # Lower Bounds
ub = [ 10, 0.4,  1,  5, 0.45]                  # Upper Bounds

# Parameters
maxit = 10
npop = 10
mu=0.5
sigma=0.1
beta = 1

# Empty Individual Template
empty_individual = structure()
empty_individual.position = None
empty_individual.fit = None

# Best Solution Ever Found
bestsol = empty_individual.deepcopy()
bestsol.fit = np.inf

archive = []

# Initialize Population
pop = empty_individual.repeat(npop)
for i in range(npop):
    pop[i].position = np.random.uniform(lb, ub, nvar)
    pop[i].fit = fitness(pop[i].position)
    archive.append(pop[i].position)
    if pop[i].fit < bestsol.fit:
        bestsol = pop[i].deepcopy()



arc=structure().repeat(nvar)
for i in range(0,len(pop)):
    for j in range(0,nvar):
        arc[j][i] = archive[i][j]

arc2=[]
for i in range(0,nvar): arc2.append(dict(arc[i]))

df_archive = pd.DataFrame(arc2).transpose()

 
fig = plt.figure(figsize =(10, 7))
plt.boxplot(df_archive)
plt.show()
