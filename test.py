import numpy as np
from ypstruct import structure
import numpy as np
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

# Empty Individual Template
empty_individual = structure()
empty_individual.position = None
empty_individual.fit = None

# Best Solution Ever Found
bestsol = empty_individual.deepcopy()
bestsol.fit = np.inf

mu=0.5
sigma=0.1
beta = 1

# Initialize Population
pop = empty_individual.repeat(npop)
for i in range(npop):
    pop[i].position = np.random.uniform(lb, ub, nvar)
    pop[i].fit = fitness(pop[i].position)
    if pop[i].fit < bestsol.fit:
        bestsol = pop[i].deepcopy()

aux = sorted(pop, key=lambda x: x.fit)
fits = np.array([x.fit for x in aux])                                # Lista todos os valores de fitness

#aux = list(range(0,len(fits)))

print(pop)
print(fits)
# print(pop[2])
# print(pop[2].position)
# print(pop[2].position[1])

# for i in range(0, nvar): print(pop[2].position[i])


#print(pop[2].position.shape)
#print(pop[2].position.shape)
#print(*pop[2].position.shape)
#print(np.random.uniform(-0.1, 1+0.1, 5))
print(0)

# fits = np.array([x.fit for x in pop])
# avg_fit = np.mean(fits)
# if avg_fit != 0:
#     fits = fits/avg_fit
# probs = np.exp(-beta*fits)

# c = np.cumsum(probs)                           # Retorna a soma cumulativa
# r = sum(probs)*np.random.rand()
# print(0)


# p1 = pop[2].deepcopy()

# y = p1.deepcopy()
# flag = np.random.rand(*p1.position.shape) <= mu # array de True e False indicando onde a mutação vai ocorrer
# ind = np.argwhere(flag)  # indica quais indices vao ser mutados
# y.position[ind] += sigma*np.random.randn(*ind.shape) #aplica a mutação no indices


# print(np.random.rand())
