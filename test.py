import numpy as np
from ypstruct import structure



nvar = 5
varmin = 0
varmax = 10

# Parameters
maxit = 10
npop = 10

# Empty Individual Template
empty_individual = structure()
empty_individual.position = None
empty_individual.cost = None

# Best Solution Ever Found
bestsol = empty_individual.deepcopy()
bestsol.cost = np.inf

# Initialize Population
pop = empty_individual.repeat(npop)
for i in range(npop):
    pop[i].position = np.random.uniform(varmin, varmax, nvar)

p1 = pop[2].deepcopy()

mu=0.5
sigma=0.1

y = p1.deepcopy()
flag = np.random.rand(*p1.position.shape) <= mu # array de True e False indicando onde a mutação vai ocorrer
ind = np.argwhere(flag)  # indica quais indices vao ser mutados
y.position[ind] += sigma*np.random.randn(*ind.shape) #aplica a mutação no indices


print(np.random.rand())



print(0)
