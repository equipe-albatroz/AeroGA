'''
Albatroz AeroDesign Genetic Algorithm (AeroGA)

Algoritmo Metaheurístico para Otimização do MDO da Equipe Albatroz.

Mais detalhes sobre a construção do algoritmo podem ser encontradas no arquivo README.md

Author: Krigor Rosa
Email: krigorsilva13@gmail.com
'''

from audioop import cross
import numpy as np
from ypstruct import structure
import matplotlib.pyplot as plt

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

        popc = []
        for _ in range(nc//2):

            # Perform Roulette Wheel Selection
            if selection_method == "roulette":
                parent1 = pop[roulette_wheel_selection(pop)]
                parent2 = pop[roulette_wheel_selection(pop)]
            elif selection_method == "rank": 
                parent1 = sorted(pop, key=lambda x: x.fit)[rank_selection(pop)]
                parent2 = sorted(pop, key=lambda x: x.fit)[rank_selection(pop)]
            elif selection_method == "tournament":
                parent1 = pop[tournament_selection(pop)]
                parent2 = pop[tournament_selection(pop)]
            elif selection_method == "elitism":
                parent1 = pop[elitism_selection(pop)]
                parent2 = pop[elitism_selection(pop)]

            # Perform Crossover
            if crossover_method == "normal":
                child1, child2 = crossover(parent1, parent2, gamma)

            # Perform Mutation
            if mutation_method == "gaussian":
                child1 = gaussian_mutation(child1, mu, sigma)
                child2 = gaussian_mutation(child2, mu, sigma)
            elif mutation_method == "default":
                child1 = default_mutation(child1, mu)
                child2 = default_mutation(child2, mu)

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
        print("Iteration {}: Best Fit = {}".format(iterations+1, bestfit[iterations]))

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

def gaussian_mutation(x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.chromosome.shape) <= mu          # Lista Booleana indicando em quais posições a mutação vai ocorrer
    ind = np.argwhere(flag)                                   # Lista das posições a serem mutadas
    y.chromosome[ind] += sigma*np.random.randn(*ind.shape)    # Aplicação da mutação nos alelos
    return y

def default_mutation(x, mu):
    y = x.deepcopy()
    flag = np.random.rand(*x.chromosome.shape) <= mu          # Lista Booleana indicando em quais posições a mutação vai ocorrer
    ind = np.argwhere(flag)                                   # Lista das posições a serem mutadas
    y.chromosome[ind] += np.random.randn(*ind.shape)          # Aplicação da mutação nos alelos
    return y


def apply_bound(x, lb, ub):
    x.chromosome = np.maximum(x.chromosome, lb)               # Aplica a restrição de bounds superior caso necessária em algum alelo
    x.chromosome = np.minimum(x.chromosome, ub)               # Aplica a restrição de bounds inferior caso necessária em algum alelo

def roulette_wheel_selection(pop):
    fits = sum([x.fit for x in pop])                          # Realiza a soma de todos os valores de Fitness da População
    probs = list(reversed([x.fit/fits for x in pop]))         # Cria lista de probabilidades em relação ao Fitness (lista invertida -> otimiz de minimização)
    indice = np.random.choice(len(pop), p=probs)              # Escolha aleaória com base nas probabilidades
    return indice                                        

def rank_selection(pop):
    aux = list(reversed(range(1,len(pop)+1)))                 # Criação do rankeamento da população
    probs = list(0 for i in range(0,len(pop)))                 
    for i in range(0,len(pop)): probs[i] = aux[i]/sum(aux)    # Criação de lista com as probabilidados do ranking
    indice = np.random.choice(len(pop), p=probs)              # Escolha aleaória com base nas probabilidades
    return indice


def tournament_selection(pop):

    return 1

def elitism_selection(pop):

    return 1
