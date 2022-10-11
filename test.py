import numpy as np
import pandas as pd
from ypstruct import structure
import matplotlib.pyplot as plt
from ypstruct import structure
import copy


# Sphere Test Function
def sphere(x):
    return sum(x**2)
fitness = sphere                               # Fitness Function

nvar = 5                                       # Variables number
lb = [-10, 0.2,  0, -5, 0.06]                  # Lower Bounds
ub = [ 10, 0.4,  1,  5, 0.45]                  # Upper Bounds
integer = [0,3]

def remove_index(lista,remove):
    aux_lista = copy.deepcopy(lista)
    k=0
    for i in range(len(remove)):
        aux_lista.pop(remove[i]-k)
        k+=1
    return aux_lista

# Parameters
maxit = 10
npop = 10
mu=0.5
sigma=0.1
beta = 1

# Empty Individual Template
empty_individual = structure()
empty_individual.chromossome = None
empty_individual.fit = None

# Best Solution Ever Found
bestsol = empty_individual.deepcopy()
bestsol.fit = np.inf

archive = []

continuous = list(range(0,nvar))
cont = remove_index(continuous, integer)

pop = empty_individual.repeat(npop)
for i in range(npop):
    pop[i].chromossome = np.random.uniform(lb,ub,nvar)
    print(pop[i].chromossome)
    pop[i].chromossome[cont] = np.random.uniform(remove_index(lb,integer),remove_index(ub,integer),len(cont))
    pop[i].chromossome[integer] = np.random.randint(remove_index(lb,cont),remove_index(ub,cont),len(integer))
    # pop[i].chromossome[j] = np.random.uniform(-5,5 , 1)
    pop[i].fit = fitness(pop[i].chromossome)
    print(pop[i].chromossome)
    archive.append(pop[i].chromossome)
    if pop[i].fit < bestsol.fit:
        bestsol = pop[i].deepcopy()


# for i in range(5):
#     pop[i].chromossome[cont] = np.random.uniform(remove_index(lb,integer),remove_index(ub,integer),len(cont))
#     pop[i].chromossome[integer] = np.random.randint(remove_index(lb,cont),remove_index(ub,cont),len(integer))

#     # print(pop[2].chromossome[cont])
#     # print(pop[2].chromossome[integer])
#     print(pop[i].chromossome)


print(0)

# arc=structure().repeat(nvar)
# for i in range(0,len(pop)):
#     for j in range(0,nvar):
#         arc[j][i] = archive[i][j]

# arc2=[]
# for i in range(0,nvar): arc2.append(dict(arc[i]))

# df_archive = pd.DataFrame(arc2).transpose()

 
# fig = plt.figure(figsize =(10, 7))
# plt.boxplot(df_archive)
# plt.show()
