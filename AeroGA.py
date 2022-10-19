'''
Albatroz AeroDesign Genetic Algorithm (AeroGA)

Mais detalhes sobre a construção do algoritmo podem ser encontradas no arquivo README.md
'''

from multiprocessing import pool
import time
import numpy as np
import pandas as pd
from ypstruct import structure
import matplotlib.pyplot as plt
import copy


######################################################################
#################### Main Optimization Function ######################
######################################################################

def optimize(problem, params, methods):

    t_inicial = time.time()

    # Separeting continuous variables indices from the integers
    cont = remove_index(list(range(0,problem.nvar)), problem.integer)
   
    # Initialize Population 
    pop, archive, bestsol = initialize_population(problem, params, methods, cont)
    
    # Main Loop
    pop, bestfit, bestsol, archive, error = main_loop(problem, params, methods, cont, archive, pop, bestsol)

    # Normalização dos dados da população
    archive = np.array(list(map(list,archive))).T.tolist()
    archive_scaled = normalize_data(archive, problem)

    metrics=[]

    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestfit = bestfit
    out.error = error
    out.archive = archive
    out.archive_scaled = archive_scaled
    out.plots = plots_bestfit(params, bestfit, archive_scaled)
    out.metrics = metrics

    print(f"Tempo de Execução: {time.time() - t_inicial}")

    return out


######################################################################
################## Initialize Population Function ####################
######################################################################

def initialize_population(problem, params, methods, cont):

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
    pop = empty_individual.repeat(params.npop)
    for i in range(params.npop):
        pop[i].chromossome = np.random.uniform(problem.lb,problem.ub,problem.nvar)
        pop[i].chromossome[cont] = np.random.uniform(remove_index(problem.lb,problem.integer),remove_index(problem.ub,problem.integer),len(cont))
        pop[i].chromossome[problem.integer] = np.random.randint(remove_index(problem.lb,cont),remove_index(problem.ub,cont),len(problem.integer))
        pop[i].fit = problem.fitness(pop[i].chromossome)
        archive.append(pop[i].chromossome)
        if pop[i].fit < bestsol.fit:
            bestsol = pop[i].deepcopy()

    return pop, archive, bestsol



######################################################################
######################## Main Loop Function ##########################
######################################################################

def main_loop(problem, params, methods, cont, archive, pop, bestsol):

    # Error for each iteration
    error = np.empty(params.max_iterations); error[0] = 100

    # Best Fit of Iterations
    bestfit = np.empty(params.max_iterations)
    
    # Number of children to be generated 
    nc = int(np.round(params.pc*params.npop/2)*2)

    for iterations in range(params.max_iterations):

        popc = []
        for _ in range(nc//2):

            pop = sorted(pop, key=lambda x: x.fit)
            # Perform Roulette Wheel Selection
            if methods.selection == "roulette":
                parent1 = pop[roulette_wheel_selection(pop)]
                parent2 = pop[roulette_wheel_selection(pop)]
            elif methods.selection == "rank": 
                parent1 = pop[rank_selection(pop)]
                parent2 = pop[rank_selection(pop)]
            elif methods.selection == "tournament":
                parent1 = pop[tournament_selection(pop)]
                parent2 = pop[tournament_selection(pop)]
            elif methods.selection == "elitism":
                parent1 = pop[elitism_selection(pop)]
                parent2 = pop[elitism_selection(pop)]

            # Perform Crossover
            if methods.crossover == "arithmetic":
                child1, child2 = arithmetic_crossover(problem, params, parent1, parent2, cont)
            elif methods.crossover == "1-point":
                child1, child2 = onepoint_crossover(problem, params, parent1, parent2, cont)
            elif methods.crossover == "2-point":
                child1, child2 = twopoint_crossover(problem, params, parent1, parent2, cont)

            # Perform Mutation
            if methods.mutation == "gaussian":
                child1 = gaussian_mutation(child1, params)
                child2 = gaussian_mutation(child2, params)
            elif methods.mutation == "default":
                child1 = default_mutation(child1, params)
                child2 = default_mutation(child2, params)

            # Apply Bounds
            apply_bound(child1, problem)
            apply_bound(child2, problem)

            # Evaluate First Offspring
            child1.fit = problem.fitness(child1.chromossome)
            if child1.fit < bestsol.fit:
                bestsol = child1.deepcopy()

            # Evaluate Second Offspring
            child2.fit = problem.fitness(child2.chromossome)
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
        pop = pop[0:params.npop]

        # Store Best Fit
        bestfit[iterations] = bestsol.fit

        # Calculating the error
        if iterations >= 1: error[iterations] = round(((bestfit[iterations-1]-bestfit[iterations])/bestfit[iterations])*100,4)

        # Show Iteration Information
        print("Iteration {}: Best Fit = {}".format(iterations+1, bestfit[iterations]))
    
    return pop, bestfit, bestsol, archive, error


######################################################################
######################## Crossover Functions #########################
######################################################################

# Crossover methods
def arithmetic_crossover(problem, params, parent1, parent2, cont):
    child1 = parent1.deepcopy()
    child2 = parent2.deepcopy()
    alpha = np.random.uniform(-params.gamma, 1+params.gamma, *child1.chromossome.shape)
    child1.chromossome[cont] = alpha[cont]*parent1.chromossome[cont] + (1-alpha[cont])*parent2.chromossome[cont]
    child2.chromossome[cont] = alpha[cont]*parent2.chromossome[cont] + (1-alpha[cont])*parent1.chromossome[cont]
    child1.chromossome[problem.integer] = alpha[problem.integer]*parent1.chromossome[problem.integer] + (1-alpha[problem.integer])*parent2.chromossome[problem.integer]
    child2.chromossome[problem.integer] = alpha[problem.integer]*parent2.chromossome[problem.integer] + (1-alpha[problem.integer])*parent1.chromossome[problem.integer]
    for i in problem.integer:
        child1.chromossome[i] = round(child1.chromossome[i])
        child2.chromossome[i] = round(child2.chromossome[i])
    return child1, child2

def onepoint_crossover(problem, params, parent1, parent2, cont):
    child1 = parent1.deepcopy()
    child2 = parent2.deepcopy()
    alpha = np.random.uniform(-params.gamma, 1+params.gamma, *child1.chromossome.shape)
    child1.chromossome[cont] = alpha[cont]*parent1.chromossome[cont] + (1-alpha[cont])*parent2.chromossome[cont]
    child2.chromossome[cont] = alpha[cont]*parent2.chromossome[cont] + (1-alpha[cont])*parent1.chromossome[cont]
    child1.chromossome[problem.integer] = alpha[problem.integer]*parent1.chromossome[problem.integer] + (1-alpha[problem.integer])*parent2.chromossome[problem.integer]
    child2.chromossome[problem.integer] = alpha[problem.integer]*parent2.chromossome[problem.integer] + (1-alpha[problem.integer])*parent1.chromossome[problem.integer]
    for i in problem.integer:
        child1.chromossome[i] = round(child1.chromossome[i])
        child2.chromossome[i] = round(child2.chromossome[i])
    return child1, child2

def twopoint_crossover(problem, params, parent1, parent2, cont):
    child1 = parent1.deepcopy()
    child2 = parent2.deepcopy()
    alpha = np.random.uniform(-params.gamma, 1+params.gamma, *child1.chromossome.shape)
    child1.chromossome[cont] = alpha[cont]*parent1.chromossome[cont] + (1-alpha[cont])*parent2.chromossome[cont]
    child2.chromossome[cont] = alpha[cont]*parent2.chromossome[cont] + (1-alpha[cont])*parent1.chromossome[cont]
    child1.chromossome[problem.integer] = alpha[problem.integer]*parent1.chromossome[problem.integer] + (1-alpha[problem.integer])*parent2.chromossome[problem.integer]
    child2.chromossome[problem.integer] = alpha[problem.integer]*parent2.chromossome[problem.integer] + (1-alpha[problem.integer])*parent1.chromossome[problem.integer]
    for i in problem.integer:
        child1.chromossome[i] = round(child1.chromossome[i])
        child2.chromossome[i] = round(child2.chromossome[i])
    return child1, child2


######################################################################
######################### Mutation Functions #########################
######################################################################

# Mutation methods
def gaussian_mutation(x, params):
    y = x.deepcopy()
    flag = np.random.rand(*x.chromossome.shape) <= params.mu          # Lista Booleana indicando em quais posições a mutação vai ocorrer
    ind = np.argwhere(flag)                                    # Lista das posições a serem mutadas
    for i in ind:
        if isinstance(y.chromossome[ind], int) == False:
            y.chromossome[i] += params.sigma*np.random.randn()        # Aplicação da mutação nos alelos
        else:
            y.chromossome[i] += round(params.sigma_int*np.random.randn())    # Aplicação da mutação nos alelos
    
    return y

def default_mutation(x, params):
    y = x.deepcopy()
    flag = np.random.rand(*x.chromossome.shape) <= params.mu          # Lista Booleana indicando em quais posições a mutação vai ocorrer
    ind = np.argwhere(flag)                                   # Lista das posições a serem mutadas
    for i in ind:
        if isinstance(y.chromossome[ind], int) == False:
            y.chromossome[i] += params.sigma*np.random.randn()        # Aplicação da mutação nos alelos
        else:
            y.chromossome[i] += round(params.sigma_int*np.random.randn())    # Aplicação da mutação nos alelos

    # y.chromossome[ind] += np.random.randn(*ind.shape)          # Aplicação da mutação nos alelos
    return y

######################################################################
######################### Selection Functions ########################
######################################################################

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


######################################################################
######################## Auxiliary Functions #########################
######################################################################

# To guarantee bounds limits
def apply_bound(x, problem):
    x.chromossome = np.maximum(x.chromossome, problem.lb)               # Aplica a restrição de bounds superior caso necessária em algum alelo
    x.chromossome = np.minimum(x.chromossome, problem.ub)               # Aplica a restrição de bounds inferior caso necessária em algum alelo


def normalize_data(lista,problem):
    lista_aux = [[None for _ in range(len(lista[1]))] for _ in range(len(problem.ub))]
    for i in range(0,5):
        for j in range(len(lista[1])):
            if lista[i][j] < 0:
                alpha = -1
            else:
                alpha = 1
            lista_aux[i][j] = alpha*((lista[i][j]-problem.lb[i])/(problem.ub[i]-problem.lb[i]))
    
    return lista_aux

def remove_index(lista,remove):
    aux_lista = copy.deepcopy(lista)
    k=0
    for i in range(len(remove)):
        aux_lista.pop(remove[i]-k)
        k+=1
    return aux_lista

######################################################################
########################### Plots Functions ##########################
######################################################################

def plots_bestfit(params, bestfit, archive_scaled):
    fig1 = plt.figure()
    plt.plot(bestfit)
    plt.xlim(0, params.max_iterations+1)
    plt.xlabel('Iterations')
    plt.ylabel('Best Fit')
    plt.title('Fitness x Iterations')
    plt.grid(True)
    return fig1

def plots_searchspace(params, bestfit, archive_scaled):
    fig2 = plt.figure()
    plt.boxplot(archive_scaled)
    plt.xlabel('Variáveis')
    plt.ylabel('Valores do GA')
    plt.title('Dispersão das Variáveis')
    plt.grid(True)
    return fig2