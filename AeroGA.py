'''
Albatroz AeroDesign Genetic Algorithm (AeroGA)

Algoritmo Metaheurístico para Otimização do MDO da Equipe Albatroz.

Mais detalhes sobre a construção do algoritmo podem ser encontradas no arquivo README.md

Author: Krigor Rosa
Email: krigorsilva13@gmail.com
'''

import time
import numpy as np
import pandas as pd
from ypstruct import structure
import matplotlib.pyplot as plt

def optimize(problem, params, methods):

    t_inicial = time.time()

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
    empty_individual.chromossome = None
    empty_individual.fit = None

    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.fit = np.inf

    # Archive for all population created
    archive = []

    # Initialize Population
    pop = empty_individual.repeat(npop)
    for i in range(npop):
        pop[i].chromossome = np.random.uniform(lb, ub, nvar)
        pop[i].fit = fitness(pop[i].chromossome)
        archive.append(pop[i].chromossome)
        if pop[i].fit < bestsol.fit:
            bestsol = pop[i].deepcopy()

    # Best Fit of Iterations
    bestfit = np.empty(max_iterations)

    # Error for each iteration
    error = np.empty(max_iterations)
    error[0] = 100
    
    # Main Loop
    for iterations in range(max_iterations):

        popc = []
        for _ in range(nc//2):

            pop = sorted(pop, key=lambda x: x.fit)
            # Perform Roulette Wheel Selection
            if selection_method == "roulette":
                parent1 = pop[roulette_wheel_selection(pop)]
                parent2 = pop[roulette_wheel_selection(pop)]
            elif selection_method == "rank": 
                parent1 = pop[rank_selection(pop)]
                parent2 = pop[rank_selection(pop)]
            elif selection_method == "tournament":
                parent1 = pop[tournament_selection(pop)]
                parent2 = pop[tournament_selection(pop)]
            elif selection_method == "elitism":
                parent1 = pop[elitism_selection(pop)]
                parent2 = pop[elitism_selection(pop)]

            # Perform Crossover
            if crossover_method == "arithmetic":
                child1, child2 = arithmetic_crossover(parent1, parent2, gamma)
            elif crossover_method == "1-point":
                child1, child2 = onepoint_crossover(parent1, parent2, gamma)
            elif crossover_method == "2-point":
                child1, child2 = twopoint_crossover(parent1, parent2, gamma)

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
            child1.fit = fitness(child1.chromossome)
            if child1.fit < bestsol.fit:
                bestsol = child1.deepcopy()

            # Evaluate Second Offspring
            child2.fit = fitness(child2.chromossome)
            if child2.fit < bestsol.fit:
                bestsol = child2.deepcopy()

            # Add Offsprings to popc
            popc.append(child1)
            popc.append(child2)

            # Saving children data to archive
            archive.append(child1.chromossome)
            archive.append(child2.chromossome)
        
        # Merge, Sort and Select
        pop += popc
        pop = sorted(pop, key=lambda x: x.fit)
        pop = pop[0:npop]

        # Store Best Fit
        bestfit[iterations] = bestsol.fit

        # Calculating the error
        if iterations >= 1: error[iterations] = round(((bestfit[iterations-1]-bestfit[iterations])/bestfit[iterations])*100,4)

        # Show Iteration Information
        print("Iteration {}: Best Fit = {}: Erro(%) = {}".format(iterations+1, bestfit[iterations], error[iterations]))

    print(f"Tempo de Execução: {time.time() - t_inicial}")

    # Normalização dos dados da população
    archive = np.array(list(map(list,archive))).T.tolist()
    archive_scaled = normalize_data(archive,ub,lb)

    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestfit = bestfit
    out.error = error
    out.archive = archive
    out.archive_scaled = archive_scaled
    return out


# Crossover methods
def arithmetic_crossover(parent1, parent2, gamma):
    child1 = parent1.deepcopy()
    child2 = parent2.deepcopy()
    alpha = np.random.uniform(-gamma, 1+gamma, *child1.chromossome.shape)
    child1.chromossome = alpha*parent1.chromossome + (1-alpha)*parent2.chromossome
    child2.chromossome = alpha*parent2.chromossome + (1-alpha)*parent1.chromossome
    return child1, child2

def onepoint_crossover(parent1, parent2, gamma):
    child1 = parent1.deepcopy()
    child2 = parent2.deepcopy()
    alpha = np.random.uniform(-gamma, 1+gamma, *child1.chromossome.shape)
    child1.chromossome = alpha*parent1.chromossome + (1-alpha)*parent2.chromossome
    child2.chromossome = alpha*parent2.chromossome + (1-alpha)*parent1.chromossome
    return child1, child2

def twopoint_crossover(parent1, parent2, gamma):
    child1 = parent1.deepcopy()
    child2 = parent2.deepcopy()
    alpha = np.random.uniform(-gamma, 1+gamma, *child1.chromossome.shape)
    child1.chromossome = alpha*parent1.chromossome + (1-alpha)*parent2.chromossome
    child2.chromossome = alpha*parent2.chromossome + (1-alpha)*parent1.chromossome
    return child1, child2

# Mutation methods
def gaussian_mutation(x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.chromossome.shape) <= mu          # Lista Booleana indicando em quais posições a mutação vai ocorrer
    ind = np.argwhere(flag)                                   # Lista das posições a serem mutadas
    y.chromossome[ind] += sigma*np.random.randn(*ind.shape)    # Aplicação da mutação nos alelos
    return y

def default_mutation(x, mu):
    y = x.deepcopy()
    flag = np.random.rand(*x.chromossome.shape) <= mu          # Lista Booleana indicando em quais posições a mutação vai ocorrer
    ind = np.argwhere(flag)                                   # Lista das posições a serem mutadas
    y.chromossome[ind] += np.random.randn(*ind.shape)          # Aplicação da mutação nos alelos
    return y

# Selection methods
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

# To guarantee bounds limits
def apply_bound(x, lb, ub):
    x.chromossome = np.maximum(x.chromossome, lb)               # Aplica a restrição de bounds superior caso necessária em algum alelo
    x.chromossome = np.minimum(x.chromossome, ub)               # Aplica a restrição de bounds inferior caso necessária em algum alelo


def normalize_data(lista,ub,lb):
    lista_aux = [[None for _ in range(len(lista[1]))] for _ in range(len(ub))]
    for i in range(0,5):
        for j in range(len(lista[1])):
            if lista[i][j] < 0:
                alpha = -1
            else:
                alpha = 1
            lista_aux[i][j] = alpha*((lista[i][j]-lb[i])/(ub[i]-lb[i]))
    
    return lista_aux
