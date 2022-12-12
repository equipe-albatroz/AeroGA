'''
Albatroz AeroDesign Genetic Algorithm (AeroGA)

Mais detalhes sobre a construção do algoritmo podem ser encontradas no arquivo README.md
'''

import time
import numpy as np
import pandas as pd
from ypstruct import structure
import matplotlib.pyplot as plt
import copy
import random 
import math


######################################################################
#################### Main Optimization Function ######################
######################################################################

def optimize(problem, params, methods):

    t_inicial = time.time()

    # Separeting continuous variables indices from the integers
    cont = remove_index(list(range(0,problem.nvar)), problem.integer)
   
    # Initialize Population 
    pop, archive, bestsol = initialize_population(problem, params, cont)
    
    # Main Loop
    pop, bestfit, avgfit, bestsol, archive, metrics = main_loop(problem, params, methods, cont, archive, pop, bestsol)

    # Population data normalization
    dispersion_scaled = normalize_data(np.array(list(map(list,list(archive["chromossome"])))).T.tolist(), problem)
   
    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestfit = bestfit
    out.avgfit = avgfit
    out.archive = archive
    out.searchspace = dispersion_scaled
    out.plots = [plot_convergence(params, bestfit, avgfit), plot_searchspace(problem, dispersion_scaled), plot_metrics(params, metrics)]
    out.metrics = metrics

    print(f"Tempo de Execução: {time.time() - t_inicial}")

    return out


######################################################################
################## Initialize Population Function ####################
######################################################################

def initialize_population(problem, params, cont):

    # Empty Individual Template
    empty_individual = structure()
    empty_individual.chromossome = None
    empty_individual.fit = None

    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.fit = np.inf

    # Archive for all population created
    archive = {"chromossome":[],"fit":[],"iteration":[]};

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

    # Best Fit of Iterations
    bestfit = np.empty(params.max_iterations)
    # Average Fit of Iterations
    avgfit = np.empty(params.max_iterations)

    # Number of children to be generated 
    nc = int(np.round(params.pc*params.npop/2)*2)

    # Quality metrics calculation
    metrics = {"popdiv":[]}

    for iterations in range(params.max_iterations):

        popc = []

        # Population sorted by the fitness value
        pop = sorted(pop, key=lambda x: x.fit)

        # Quality metrics calculation
        metrics["popdiv"].append(quality_metrics(problem, params, pop))

        # Elitist population
        pope = elitist_population(params,pop)

        # Online parameters control list
        MUTPB_LIST, CXPB_LIST = online_parameter(True, params)

        for _ in range(nc - round(len(pop)*params.elitism)):

            # Perform Roulette Wheel Selection
            if methods.selection == "roulette":
                aux1 = roulette_wheel_selection(pop)
                parent1 = pop[aux1]
                aux2 = roulette_wheel_selection(pop)
                parent2 = pop[aux2]
            elif methods.selection == "rank": 
                aux1 = rank_selection(pop)
                parent1 = pop[aux1]
                aux2 = rank_selection(pop)
                parent2 = pop[aux2]
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
                child1 = gaussian_mutation(params, child1, MUTPB_LIST[iterations])
                child2 = gaussian_mutation(params, child2, MUTPB_LIST[iterations])
            elif methods.mutation == "polynomial":
                child1 = polynomial_mutation(problem, child1, MUTPB_LIST[iterations], params.eta)
                child2 = polynomial_mutation(problem, child2, MUTPB_LIST[iterations], params.eta)

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
            archive["iteration"].append(iterations)
        
        # Merge, Sort and Select
        del pop
        pop = popc; pop += pope
        pop = sorted(pop, key=lambda x: x.fit)
        pop = pop[0:params.npop]

        # Store Best Fit
        bestfit[iterations] = bestsol.fit
        avgfit[iterations] = fitsum_pop(params, pop)/params.npop

        # Show Iteration Information
        print("Iteration {}: Best Fit = {}: Average Fitness = {}".format(iterations+1, bestfit[iterations], avgfit[iterations]))
    
    print("Best Solution = {}".format(bestsol))

    return pop, bestfit, avgfit, bestsol, archive, metrics


######################################################################
######################### Selection Functions ########################
######################################################################

# Selection methods
def roulette_wheel_selection(pop):

    """Roulette wheel selection operator.

    :param pop: population for the current generation
    """
    
    fits = sum([x.fit for x in pop])                          # Realiza a soma de todos os valores de Fitness da População
    probs = list(reversed([x.fit/fits for x in pop]))         # Cria lista de probabilidades em relação ao Fitness (lista invertida -> otimiz de minimização)
    indice = np.random.choice(len(pop), p=probs)              # Escolha aleaória com base nas probabilidades
    return indice                                        

def rank_selection(pop):

    """Roulette wheel selection operator.

    :param pop: population for the current generation
    """

    aux = list(reversed(range(1,len(pop)+1)))                 # Criação do rankeamento da população
    probs = list(0 for i in range(0,len(pop)))                 
    for i in range(0,len(pop)): probs[i] = aux[i]/sum(aux)    # Criação de lista com as probabilidados do ranking
    indice = np.random.choice(len(pop), p=probs)              # Escolha aleaória com base nas probabilidades
    return indice

def tournament_selection(pop):

    """Roulette wheel selection operator.

    :param pop: population for the current generation
    """

    individual1 = np.random.choice(len(pop))
    individual2 = np.random.choice(len(pop))
    if individual1 == individual2: 
        while individual1 == individual2: 
            individual2 = np.random.choice(len(pop))
    if pop[individual1].fit <= pop[individual2].fit:
        return individual1
    else:
        return individual2

######################################################################
######################## Crossover Functions #########################
######################################################################

# Crossover methods
def arithmetic_crossover(problem, params, parent1, parent2, cont):

    """Arithmetical crossover operator.

    :param problem: indices with integer values
    :param params: gamma value
    :param parent1: parent 1 selected for the process
    :param parent2: parent 2 selected for the process
    :param CXPB: Independent probability for each attribute
    :param cont: indices with continuous values
    """

    gamma = 0.1                                                      # Arithmetic crossover amplitude
    child1 = parent1.deepcopy()
    child2 = parent2.deepcopy()
    alpha = np.random.uniform(-gamma, 1+gamma, *child1.chromossome.shape)
    child1.chromossome[cont] = alpha[cont]*parent1.chromossome[cont] + (1-alpha[cont])*parent2.chromossome[cont]
    child2.chromossome[cont] = alpha[cont]*parent2.chromossome[cont] + (1-alpha[cont])*parent1.chromossome[cont]
    child1.chromossome[problem.integer] = alpha[problem.integer]*parent1.chromossome[problem.integer] + (1-alpha[problem.integer])*parent2.chromossome[problem.integer]
    child2.chromossome[problem.integer] = alpha[problem.integer]*parent2.chromossome[problem.integer] + (1-alpha[problem.integer])*parent1.chromossome[problem.integer]
    for i in problem.integer:
        child1.chromossome[i] = round(child1.chromossome[i])
        child2.chromossome[i] = round(child2.chromossome[i])
    return child1, child2

def onepoint_crossover(problem, params, parent1, parent2, cont):

    """1-point crossover operator.

    :param problem: indices with integer values
    :param params: gamma value
    :param parent1: parent 1 selected for the process
    :param parent2: parent 2 selected for the process
    :param CXPB: Independent probability for each attribute
    :param cont: indices with continuous values
    """

    aux1 = parent1.deepcopy(); child1 = parent1.deepcopy()
    aux2 = parent2.deepcopy(); child2 = parent2.deepcopy()

    cut_point = np.random.choice(problem.nvar-1)

    child1.chromossome[cut_point+1:problem.nvar] = aux2.chromossome[cut_point+1:problem.nvar]
    child1.chromossome[0:cut_point+1] = aux1.chromossome[0:cut_point+1]

    child2.chromossome[0:cut_point+1] = aux2.chromossome[0:cut_point+1]
    child2.chromossome[cut_point+1:problem.nvar] = aux1.chromossome[cut_point+1:problem.nvar]

    return child1, child2

def twopoint_crossover(problem, params, parent1, parent2, cont):

    """2-point crossover operator.

    :param problem: indices with integer values
    :param params: gamma value
    :param parent1: parent 1 selected for the process
    :param parent2: parent 2 selected for the process
    :param CXPB: Independent probability for each attribute
    :param cont: indices with continuous values
    """

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
def gaussian_mutation(params, x, MUTPB):

    """This function applies a gaussian mutation of mean *mu* and standard
    deviation *sigma* on the input individual. This mutation expects a
    :term:`sequence` individual composed of real valued attributes.
    
    :param params.sigma: Standard deviation or :term:`python:sequence` of
                  standard deviations for the gaussian addition mutation.
    :param x: Individual to be mutated.
    :param MUTPB: Independent probability for each attribute to be mutated.
    """

    y = x.deepcopy()
    flag = np.random.rand(*x.chromossome.shape) <= MUTPB                # Lista Booleana indicando em quais posições a mutação vai ocorrer
    ind = np.argwhere(flag)                                                # Lista das posições a serem mutadas

    for i in ind:
        if isinstance(y.chromossome[i], int) == False:
            y.chromossome[i] += random.gauss(0, params.sigma)                    # Aplicação da mutação nos alelos
        else:
            y.chromossome[i] += round(random.gauss(0, params.sigma))    # Aplicação da mutação nos alelos
    
    return y

def polynomial_mutation(problem, x, MUTPB, eta):

    """Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb.

    :param params: lower and upper bounds will be used.
    :param x: individual to be mutated.
    :param MUTPB: Independent probability for each attribute
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    """

    y = x.deepcopy()
    flag = np.random.rand(*x.chromossome.shape) <= MUTPB                 # Lista Booleana indicando em quais posições a mutação vai ocorrer
    ind = np.argwhere(flag)                                                # Lista das posições a serem mutadas

    rand = random.random()

    for i in range(problem.nvar):
        delta1 = (x.chromossome[i] - problem.lb[i])/(problem.ub[i] - problem.lb[i])
        delta2 = (problem.ub[i] - x.chromossome[i])/(problem.ub[i] - problem.lb[i])

        if i in ind:
            delta = (2*rand + (1 - 2*rand)*(1 - delta1)**(eta+1))**(1/(eta+1))-1
        else:
            delta = 1 - (2*(1 - rand) + 2*(rand - 0.5)*(1 - delta2)**(eta+1))**(1/(eta+1))
    
        y.chromossome[i] = x.chromossome[i] + delta*(problem.ub[i] - problem.lb[i])

    return y



######################################################################
########################## Online Parameter ##########################
######################################################################

def online_parameter(Use, params):

    # MUTPB_LIST: Mutation Probability
    # CXPB_LIST: Crossover Probability

    if Use == True:
        line_x = np.linspace(start=1, stop=50, num=params.max_iterations)
        MUTPB_LIST = (-(np.log10(line_x) - np.log10(line_x[0]))/(np.log10(line_x[-1]) - np.log10(line_x[0])) + 1) * 0.2
        
        line_x = np.linspace(start=1, stop=5, num=params.max_iterations)
        CXPB_LIST = (np.log10(line_x) - np.log10(line_x[0]))/(np.log10(line_x[-1]) - np.log10(line_x[0]))
    else:
        MUTPB_LIST = [0.2]*params.max_iterations
        CXPB_LIST = [1.0]*params.max_iterations

    return MUTPB_LIST, CXPB_LIST

######################################################################
########################## Metrics Function ##########################
######################################################################

def quality_metrics(problem,params,pop):
    Gn = []; SPDi = []

    for j in range(problem.nvar):
        aux = []
        for i in range(params.npop):

            if pop[i].chromossome[j] < 0:
                alpha = -1
            else:
                alpha = 1

            aux.append(alpha*((pop[i].chromossome[j]-problem.lb[j])/(problem.ub[j]-problem.lb[j])))

            if i == (params.npop - 1):
                soma = sum(aux)
                Gn.append(soma/params.npop)
    
    for j in range(problem.nvar):
        aux = []
        for i in range(params.npop):
            
            if pop[i].chromossome[j] < 0:
                alpha = -1
            else:
                alpha = 1

            norm = alpha*((pop[i].chromossome[j]-problem.lb[j])/(problem.ub[j]-problem.lb[j]))

            aux.append(((norm-Gn[j])**2))
            if i == (params.npop - 1): 
                SPDi.append(sum(aux)/params.npop)

    SPD = []
    for j in range(problem.nvar):
        SPD.append(SPDi[j]/Gn[j])


    return round(sum(SPD)/problem.nvar,2)

######################################################################
######################## Auxiliary Functions #########################
######################################################################

# To guarantee bounds limits
def apply_bound(x, problem):
    x.chromossome = np.maximum(x.chromossome, problem.lb)               # Aplica a restrição de bounds superior caso necessária em algum alelo
    x.chromossome = np.minimum(x.chromossome, problem.ub)               # Aplica a restrição de bounds inferior caso necessária em algum alelo
    
    for i in problem.integer:
        x.chromossome[i] = math.floor(x.chromossome[i])


def normalize_data(lista,problem):
    lista_aux = [[None for _ in range(len(lista[1]))] for _ in range(len(problem.ub))]
    for i in range(0,problem.nvar):
        for j in range(len(lista[1])):
            if lista[i][j] < 0:
                lista_aux[i][j] = -1*abs(lista[i][j]/problem.lb[i])
            elif lista[i][j] == 0:
                lista_aux[i][j] = 0
            elif lista[i][j] > 0:
                lista_aux[i][j] = lista[i][j]/problem.ub[i]
    
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

def fitsum_pop(params, pop):
    soma = 0
    for i in range(params.npop):
        soma += pop[i].fit
    return soma


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

def plot_convergence(params, bestfit, avgfit):
    fig = plt.figure()
    plt.plot(bestfit, label = "Best Fitness")
    plt.plot(avgfit, alpha = 0.3, linestyle = "--", label = "Average Fitness")
    plt.xlim(0, params.max_iterations+1)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title('GA Convergence')
    plt.grid(True)
    return fig

def plot_searchspace(problem, dispersion_scaled):
    fig = plt.figure()

    for i in problem.lb:
        if i < 0: a = -1; break
        else: a = 0
    
    index = []; label = []
    for i in range(problem.nvar):
        index.append(len(dispersion_scaled[i])*[i])
        label.append("Var"+str(i+1))
    
    for i in range(problem.nvar):
        plt.scatter(index[i], dispersion_scaled[i], s=1, alpha=0.2, color='black', marker='o')

    plt.xticks(range(problem.nvar), label, rotation = 90)
    plt.yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], ["-100%", "-75%", "-50%", "-25%", "0%", "25%", "50%", "75%", "100%"])
    plt.ylim(a,1)
    plt.ylabel('Values used')
    plt.title('Search Space')

    return fig

def plot_metrics(params, metrics):
    fig = plt.figure()
    plt.plot(metrics["popdiv"])
    plt.xlim(0, params.max_iterations+1)
    plt.xlabel('Iterations')
    plt.ylabel('Metrics')
    plt.grid(True)
    return fig

def plot_pop(params, archive, iteration):
    fig = plt.figure()
    individual = []; fit = []; iter = []

    for i in range(len(archive["chromossome"])):
        individual.append(np.sqrt(sum(archive["chromossome"][i]**2)))
        fit.append(archive["fit"][i])
        iter.append(archive["iteration"][i])

    for i in range(params.max_iterations):
        if iter[i] == iteration:
            plt.scatter(individual[i], fit[i])

    return fig


def statistical_analysis(problem, params, methods, nruns):
    
    fitness = []; bestsol = []

    for i in range(nruns):
        out = optimize(problem, params, methods)
        fitness.append(out.bestfit)
        bestsol.append(out.bestsol)

    fig = plt.figure()
    plt.boxplot(fitness)
    plt.xticks([0],["uuu"])
    plt.ylabel('Fitness')
    plt.title('Dispersion of variables')
    plt.grid(True)

    return fig, bestsol