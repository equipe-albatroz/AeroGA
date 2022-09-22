import numpy as np
from ypstruct import structure

def optimize(problem, params):
    
    # Problem Information
    fitness = problem.fitness
    nvar = problem.nvar
    lb = problem.lb
    ub = problem.ub

    # Parameters
    max_iterations = params.max_iterations
    npop = params.npop
    beta = params.beta
    pc = params.pc
    nc = int(np.round(pc*npop/2)*2)
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma

    # Empty Individual Template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.fit = None

    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.fit = np.inf

    # Initialize Population
    pop = empty_individual.repeat(npop)
    for i in range(npop):
        pop[i].position = np.random.uniform(lb, ub, nvar)
        pop[i].fit = fitness(pop[i].position)
        if pop[i].fit < bestsol.fit:
            bestsol = pop[i].deepcopy()

    # Best Fit of Iterations
    bestfit = np.empty(max_iterations)
    
    # Main Loop
    for iterations in range(max_iterations):

        fits = np.array([x.fit for x in pop])
        avg_fit = np.mean(fits)
        if avg_fit != 0:
            fits = fits/avg_fit
        probs = np.exp(-beta*fits)

        popc = []
        for _ in range(nc//2):

            # Perform Roulette Wheel Selection
            parent1 = pop[roulette_wheel_selection(probs)]
            parent2 = pop[roulette_wheel_selection(probs)]
            
            # Perform Crossover
            child1, child2 = crossover(parent1, parent2, gamma)

            # Perform Mutation
            child1 = mutate(child1, mu, sigma)
            child2 = mutate(child2, mu, sigma)

            # Apply Bounds
            apply_bound(child1, lb, ub)
            apply_bound(child2, lb, ub)

            # Evaluate First Offspring
            child1.fit = fitness(child1.position)
            if child1.fit < bestsol.fit:
                bestsol = child1.deepcopy()

            # Evaluate Second Offspring
            child2.fit = fitness(child2.position)
            if child2.fit < bestsol.fit:
                bestsol = child2.deepcopy()

            # Add Offsprings to popc
            popc.append(child1)
            popc.append(child2)
        

        # Merge, Sort and Select
        pop += popc
        pop = sorted(pop, key=lambda x: x.fit)
        pop = pop[0:npop]

        # Store Best Fit
        bestfit[iterations] = bestsol.fit

        # Show Iteration Information
        print("Iteration {}: Best Fit = {}".format(iterations, bestfit[iterations]))

    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestfit = bestfit
    return out

def crossover(parent1, parent2, gamma=0.1):
    child1 = parent1.deepcopy()
    child2 = parent1.deepcopy()
    alpha = np.random.uniform(-gamma, 1+gamma, *child1.position.shape)
    child1.position = alpha*parent1.position + (1-alpha)*parent2.position
    child2.position = alpha*parent2.position + (1-alpha)*parent1.position
    return child1, child2

def mutate(x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.position.shape) <= mu # array de True e False indicando onde a mutação vai ocorrer
    ind = np.argwhere(flag)  # indica quais indices vao ser mutados
    y.position[ind] += sigma*np.random.randn(*ind.shape) # aplica a mutação no indices
    return y

def apply_bound(x, lb, ub):
    x.position = np.maximum(x.position, lb)
    x.position = np.minimum(x.position, ub)

def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]
