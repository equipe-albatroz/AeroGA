'''
Albatroz AeroDesign Genetic Algorithm (AeroGA)

Algoritmo Metaheurístico para Otimização do MDO da Equipe Albatroz.

Mais detalhes sobre a construção do algoritmo podem ser encontradas no arquivo README.md

Author: Krigor Rosa
Email: krigorsilva13@gmail.com
'''

import numpy as np
from ypstruct import structure

def optimize(problem, params, methods):

    # Methods Information
    selection_method = methods.selection
    crossover_method = methods.crossover
    mutation_method = methods.mutation
    
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
    empty_individual.chromosome = None
    empty_individual.fit = None

    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.fit = np.inf

    # Initialize Population
    pop = empty_individual.repeat(npop)
    for i in range(npop):
        pop[i].chromosome = np.random.uniform(lb, ub, nvar)
        pop[i].fit = fitness(pop[i].chromosome)
        if pop[i].fit < bestsol.fit:
            bestsol = pop[i].deepcopy()

    # Best Fit of Iterations
    bestfit = np.empty(max_iterations)
    
    # Main Loop
    for iterations in range(max_iterations):

        if selection_method == "roulette":
            fits = np.array([x.fit for x in pop])                                # Lista todos os valores de fitness
            avg_fit = np.mean(fits)                                              # Média dos valores de fitness
            if avg_fit != 0:
                fits = fits/avg_fit
            probs = np.exp(-beta*fits)
        elif selection_method == "rank":   
            aux = sorted(pop, key=lambda x: x.fit)
            fits = np.array([x.fit for x in aux])                                # Lista todos os valores de fitness
            inds = list(range(0,len(fits)))
            avg_ind = np.mean(inds) 
            if avg_ind != 0:
                inds = inds/avg_ind
            probs = np.exp(-beta*inds)                                          # Calcula a probabilidade de cada indivíduo com base em fç exponencial

        popc = []
        for _ in range(nc//2):

            # Perform Roulette Wheel Selection
            if selection_method == "roulette":
                parent1 = pop[roulette_wheel_selection(probs)]
                parent2 = pop[roulette_wheel_selection(probs)]
            elif selection_method == "rank":  
                parent1 = aux[rank_selection(probs)]
                parent2 = aux[rank_selection(probs)]

            # Perform Crossover
            child1, child2 = crossover(parent1, parent2, gamma)

            # Perform Mutation
            child1 = mutate(child1, mu, sigma)
            child2 = mutate(child2, mu, sigma)

            # Apply Bounds
            apply_bound(child1, lb, ub)
            apply_bound(child2, lb, ub)

            # Evaluate First Offspring
            child1.fit = fitness(child1.chromosome)
            if child1.fit < bestsol.fit:
                bestsol = child1.deepcopy()

            # Evaluate Second Offspring
            child2.fit = fitness(child2.chromosome)
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

def crossover(parent1, parent2, gamma):
    child1 = parent1.deepcopy()
    child2 = parent2.deepcopy()
    alpha = np.random.uniform(-gamma, 1+gamma, *child1.chromosome.shape)
    child1.chromosome = alpha*parent1.chromosome + (1-alpha)*parent2.chromosome
    child2.chromosome = alpha*parent2.chromosome + (1-alpha)*parent1.chromosome
    return child1, child2

def mutate(x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.chromosome.shape) <= mu # array de True e False indicando onde a mutação vai ocorrer
    ind = np.argwhere(flag)  # indica quais indices vao ser mutados
    y.chromosome[ind] += mu + sigma*np.random.randn(*ind.shape) # aplica a mutação no indices
    return y

def apply_bound(x, lb, ub):
    x.chromosome = np.maximum(x.chromosome, lb)
    x.chromosome = np.minimum(x.chromosome, ub)

def roulette_wheel_selection(p):
    c = np.cumsum(p)                           # Retorna a soma cumulativa
    r = sum(p)*np.random.rand()                # Gera valor aleatório para simular o giro da roleta
    ind = np.argwhere(r <= c)                  # Retorna uma lista de índices dos valores que podem ser selecionados na roleta
    return ind[0][0]                           # Retorna o primeiro valor dos que podem ser selecionados pela roleta

def rank_selection(p):
    c = np.cumsum(p)                           # Retorna a soma cumulativa
    r = sum(p)*np.random.rand()                # Gera valor aleatório para simular o giro da roleta
    ind = np.argwhere(r <= c)                  # Retorna uma lista de índices dos valores que podem ser selecionados na roleta
    return ind[0][0]                     # Retorna o primeiro valor dos que podem ser selecionados pela roleta


def tournament_selection(p):

    return 1

def elitism_selection(p):

    return 1
