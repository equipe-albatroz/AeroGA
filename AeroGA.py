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
    pop, bestfit, bestsol, archive, metrics, error = main_loop(problem, params, methods, cont, archive, pop, bestsol)

    # Normalização dos dados da população
    dispersion_scaled = normalize_data(np.array(list(map(list,list(archive["chromossome"])))).T.tolist(), problem)

    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestfit = bestfit
    out.error = error
    out.archive = archive
    out.plots = [plots_bestfit(params, bestfit, dispersion_scaled)]
    # out.plots = [plots_bestfit(params, bestfit, archive_scaled), plots_searchspace(params, bestfit, archive_scaled)]       
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
    archive = {"chromossome":[],"fit":[]};

    # Initialize Population
    pop = empty_individual.repeat(params.npop)
    for i in range(params.npop):
        pop[i].chromossome = np.random.uniform(problem.lb,problem.ub,problem.nvar)
        pop[i].chromossome[cont] = np.random.uniform(remove_index(problem.lb,problem.integer),remove_index(problem.ub,problem.integer),len(cont))
        pop[i].chromossome[problem.integer] = np.random.randint(remove_index(problem.lb,cont),remove_index(problem.ub,cont),len(problem.integer))
        pop[i].fit = problem.fitness(pop[i].chromossome)
        archive["chromossome"].append(pop[i].chromossome)
        archive["fit"].append(pop[i].fit)
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

        # Population sorted by the fitness value
        pop = sorted(pop, key=lambda x: x.fit)

        # Elitist population
        pope = elitist_population(params,pop)

        for _ in range(nc - round(len(pop)*params.elitism)):

            # Perform Roulette Wheel Selection
            if methods.selection == "roulette":
                aux1 = roulette_wheel_selection(pop)
                parent1 = pop[aux1]; del pop[aux1]
                aux2 = roulette_wheel_selection(pop)
                parent2 = pop[aux2]; del pop[aux2]
            elif methods.selection == "rank": 
                aux1 = rank_selection(pop)
                parent1 = pop[aux1]; del pop[aux1]
                aux2 = rank_selection(pop)
                parent2 = pop[aux2]; del pop[aux2]
            elif methods.selection == "tournament":
                aux1 = tournament_selection(pop)
                parent1 = pop[aux1]
                aux2 = tournament_selection(pop)
                parent2 = pop[aux2]

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
            archive["chromossome"].append(child1.chromossome)
            archive["fit"].append(child2.chromossome)
        
        # Merge, Sort and Select
        del pop
        pop = popc; pop += pope
        pop = sorted(pop, key=lambda x: x.fit)
        pop = pop[0:params.npop]

        # Store Best Fit
        bestfit[iterations] = bestsol.fit

        # Quality metrics calculation
        metrics=quality_metrics(params,bestfit)

        # Calculating the error
        if iterations >= 1: error[iterations] = round(((bestfit[iterations-1]-bestfit[iterations])/bestfit[iterations])*100,4)

        # Show Iteration Information
        print("Iteration {}: Best Fit = {}".format(iterations+1, bestfit[iterations]))
    
    return pop, bestfit, bestsol, archive, metrics, error


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
    aux1 = parent1.deepcopy(); child1 = parent1.deepcopy()
    aux2 = parent2.deepcopy(); child2 = parent2.deepcopy()

    cut_point = np.random.choice(problem.nvar-1)

    child1.chromossome[cut_point+1:problem.nvar] = aux2.chromossome[cut_point+1:problem.nvar]
    child1.chromossome[0:cut_point+1] = aux1.chromossome[0:cut_point+1]

    child2.chromossome[0:cut_point+1] = aux2.chromossome[0:cut_point+1]
    child2.chromossome[cut_point+1:problem.nvar] = aux1.chromossome[cut_point+1:problem.nvar]

    return child1, child2

def twopoint_crossover(problem, params, parent1, parent2, cont):
    aux1 = parent1.deepcopy(); child1 = parent1.deepcopy()
    aux2 = parent2.deepcopy(); child2 = parent2.deepcopy()

    cut_point1 = np.random.choice(problem.nvar-1)
    cut_point2 = np.random.choice(problem.nvar-1)
    if cut_point1 == cut_point2:
        while cut_point1 == cut_point2:
            cut_point2 = np.random.choice(problem.nvar-1)

    if cut_point2 < cut_point1:
        cut_point1, cut_point2 = cut_point2, cut_point1

    child1.chromossome[0:cut_point1+1] = aux1.chromossome[0:cut_point1+1]
    child1.chromossome[cut_point1+1:cut_point2+1] = aux2.chromossome[cut_point1+1:cut_point2+1]
    child1.chromossome[cut_point2+1:problem.nvar] = aux1.chromossome[cut_point2+1:problem.nvar]

    child2.chromossome[0:cut_point1+1] = aux2.chromossome[0:cut_point1+1]
    child2.chromossome[cut_point1+1:cut_point2+1] = aux1.chromossome[cut_point1+1:cut_point2+1]
    child2.chromossome[cut_point2+1:problem.nvar] = aux2.chromossome[cut_point2+1:problem.nvar]

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
    individual1 = np.random.choice(len(pop))
    individual2 = np.random.choice(len(pop))
    if individual1 == individual2: 
        while individual1 == individual2: 
            individual2 = np.random.choice(len(pop))
    if pop[individual1].fit >= pop[individual2].fit:
        return individual1
    else:
        return individual2


######################################################################
########################## Metrics Function ##########################
######################################################################

def quality_metrics(params, bestfit):

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


def elitist_population(params,pop):
    lst = []
    for i in range(round(len(pop)*params.elitism)):
        lst.append(pop[i])

    return lst


######################################################################
######################## Sensibility Analisys ########################
######################################################################

def sensibility(problem, bestsol):

    dict = {"nvar":[],"value":[],"fit":[]};

    for j in range(problem.nvar):

        if isinstance(problem.lb[j], int) == False:
            increment = 0.01
            lst=[0]*round(abs(((problem.ub[j]-problem.lb[j])/increment)+1)); lst[0]=problem.lb[j]
            for i in range(round(((problem.ub[j]-problem.lb[j])/increment)+1)):
                if i >= 1: lst[i]=round(lst[i-1]+increment,2)
        else:
            increment = 1
            lst = list(range(problem.lb[j],problem.ub[j]+1,increment))
        
        for value in lst:
            bestsol.chromossome[j] = value
            dict["nvar"].append(j)
            dict["value"].append(value)
            dict["fit"].append(problem.fitness(bestsol.chromossome))

    return pd.DataFrame(dict)

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