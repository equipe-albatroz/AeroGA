import copy
import time
import random
import multiprocessing
from statistics import mean
from . import settings
from .Classes import Individual, ErrorType, Log
from .Operators_Selection import roulette_selection, tournament_selection, rank_selection
from .Operators_Crossover import arithmetic_crossover, SBX_crossover, crossover_1pt, crossover_2pt
from .Operators_Mutation import gaussian_mutation, polynomial_mutation, online_expected_diversity, online_parameter
from .Population import denormalize_population, denormalize_individual, generate_population_normalized
from .Evaluators import parallel_fitness, fitness
from .Performance_Metrics import diversity_metric
from .Utilities import export_excell, create_plotfit, create_boxplots, parallel_coordinates

# Setting error log file
ErrorLog = Log("error.log")

# #####################################################################################
# ###################################### Main #########################################
# #####################################################################################

def optimize(selection = "tournament", crossover = "1-point", mutation = "gaussian", n_threads = -1,
    min_values = list, max_values = list, num_variables = int, num_generations = int, elite_count = int, elite="local",
    plotfit = True, plotbox = False, plotparallel = False, TabuList = False, penalization_list = [1000],
    fitness_fn = None, classe = "default"):

    """Perform the genetic algorithm to find an optimal solution."""

    try:
        t_inicial = time.time()

        settings.log.warning("****************** INITIALIZING OPTIMIZATION *******************")

        # Definition of how many threads will be used to calculate the fitness function
        if n_threads == -1: n_threads = multiprocessing.cpu_count()
        settings.log.info(f"Optimization initialized with {n_threads} threads.")

        # Checking if num_varialbes matches the lb and ub sizes
        if len(max_values) != len(min_values) or num_variables != len(min_values):
            settings.log.critical("There is an inconsistency between the number of variables and the size of the bounds")
            return [0]
        
        # Creating normalized max and min values list
        max_values_norm = [1] * num_variables
        min_values_norm = [0] * num_variables

        # Defining population size
        population_size = 100
        population_size_old = population_size

        # Initial value for the best fitness
        best_fitness = float('inf')

        # Generating initial population
        population = generate_population_normalized(population_size, num_variables, min_values, max_values)

        # Creating history, metrics and best/avg lists
        values_gen = {"best_fit":[],"avg_fit":[],"metrics":[]}
        history = {"ind":[], "ind_norm":[],"gen":[],"fit":[],"score":[]}
        history_valid = {"ind":[], "ind_norm":[], "gen":[],"fit":[],"score":[]}
        best_individual = {"ind":[],"fit":[]}
        old_mut_param = {"std_dev":[],"eta":[]}
        tabu_List = []

        # Initializing the main loop
        for generation in range(num_generations):
            t_gen = time.time()

            # Getting the fitness values from history if possible
            fit_pop_old = []; pop_old = []; pop_calc_fit = copy.deepcopy(population)
            for i in range(population_size_old):
                if population[i] in history["ind_norm"]:
                    fit_pop_old.append(history["fit"][list(history["ind_norm"]).index(population[i])])
                    pop_old.append(population[i])
                    pop_calc_fit.remove(population[i])

            # Fitness calculation        
            if n_threads != 0:
                fit_values = parallel_fitness(denormalize_population(pop_calc_fit, min_values, max_values, classe), fitness_fn, n_threads)
            else:
                fit_values = fitness(denormalize_population(pop_calc_fit, min_values, max_values, classe), fitness_fn)
            
            fitness_values = fit_values + fit_pop_old
            population = pop_calc_fit + pop_old

            # Population sorted by the fitness value
            population = [x for _,x in sorted(zip(fitness_values,population))]
            fitness_values = sorted(fitness_values)
    
            # Add to history and valid fit history
            for i in range(len(population)):
                history["ind_norm"].append(population[i])
                history["fit"].append(fitness_values[i])
                if fitness_values[i] != 0:
                    history["score"].append(1/fitness_values[i])
                else:
                    history["score"].append(float('inf'))
                history["gen"].append(generation)
                if fitness_values[i] < min(penalization_list):
                    history_valid["ind_norm"].append(population[i])
                    history_valid["fit"].append(fitness_values[i])
                    if fitness_values[i] != 0:
                        history_valid["score"].append(1/fitness_values[i])
                    else:
                        history_valid["score"].append(float('inf'))
                    history_valid["gen"].append(generation)
                else:
                    tabu_List.append(population[i])            

            # Best and average fitness and best individual at the generation
            best_individual["ind"].append(denormalize_individual(population[fitness_values.index(min(fitness_values))], min_values, max_values, classe))
            best_individual["fit"].append(min(fitness_values))

            # Checking if the best fit is better than previus generations and the global value to plot the individual
            if best_individual["fit"][generation] < best_fitness:
                best_fitness = best_individual["fit"][generation]

            # Creating list of fitness values with individuals that returns valid score
            fitness_values_valid = []
            for i in range(population_size_old):
                if fitness_values[i] < min(penalization_list):
                    fitness_values_valid.append(fitness_values[i]) 

            # Saving these values in lists
            values_gen["best_fit"].append(best_fitness)
            if mean(fitness_values) < min(penalization_list):
                values_gen["avg_fit"].append(mean(fitness_values_valid))
            else:
                values_gen["avg_fit"].append(None)
            values_gen["metrics"].append(diversity_metric(population))

            # Applying the online parameter control
            mutation_prob = random.uniform(0.05, 0.15) 
            MUTPB_LIST, CXPB_LIST = online_parameter(True, num_generations, mutation_prob, 1)

            # Printing logger informations
            if best_individual["fit"][generation] == 0:
                settings.log.info('Generation: {} | Time: {} | Population Size: {} | Best Fitness: {} -> Score: {} | Diversity Metric: {}'.format(generation+1, round(time.time() - t_gen, 2), population_size, best_individual["fit"][generation], float('inf'), round(values_gen["metrics"][generation],2)))
            else:    
                settings.log.info('Generation: {} | Time: {} | Population Size: {} | Best Fitness: {} -> Score: {} | Diversity Metric: {}'.format(generation+1, round(time.time() - t_gen, 2), population_size, best_individual["fit"][generation], round(1/best_individual["fit"][generation],2), round(values_gen["metrics"][generation],2)))

            # Creating new population and aplying elitist concept
            new_population = []
            elite_count_gen = 0
            if elite_count != 0 and mean(fitness_values) < min(penalization_list):
                if elite == "global":
                    if generation == 0:
                        elite_pop = population[:1]
                        elite_fit = fitness_values[:1]
                        if elite_count > 1:
                            for i in range(1,population_size_old-1):
                                if len(elite_pop) < elite_count:
                                    if population[i] not in elite_pop:
                                        elite_pop.append(population[i])
                                        elite_fit.append(fitness_values[population.index(population[i])])
                                else:
                                    new_population = copy.deepcopy(elite_pop)
                                    elite_pop_glob = elite_pop
                                    elite_fit_glob = elite_fit
                                    break
                    else:
                        elite_pop = population[:1]
                        elite_fit = fitness_values[:1]
                        if elite_count > 1:
                            for i in range(1,population_size_old-1):
                                if len(elite_pop) < elite_count:
                                    if population[i] not in elite_pop:
                                        elite_pop.append(population[i])
                                        elite_fit.append(fitness_values[population.index(population[i])])
                                else:
                                    for j in range(elite_count):
                                        if elite_fit[j] < max(elite_fit_glob) and elite_fit[j] not in elite_fit_glob:
                                            elite_fit_glob[elite_fit_glob.index(max(elite_fit_glob))] = elite_fit[j]
                                            elite_pop_glob[elite_fit_glob.index(max(elite_fit_glob))] = elite_pop[j] 
                                    new_population = copy.deepcopy(elite_pop_glob)
                                    break

                elif elite == "local":
                    for i in range(population_size_old): 
                        count_lst = list({x for x in fitness_values if fitness_values.count(x)})
                        for penality in penalization_list:
                            if penality in count_lst:
                                del count_lst[count_lst.index(penality)]
                            count = len(count_lst)

                    if count < elite_count: 
                        elite_count_gen = count
                    else:
                        elite_count_gen = elite_count
                    
                    new_population = population[:1]
                    if elite_count_gen > 1:
                        for i in range(1,population_size_old-1):
                            if len(new_population) < elite_count_gen:
                                if population[i] not in new_population:
                                    new_population.append(population[i])
                            else:
                                break

            # Creating new population based on crossover methods
            for i in range(0, population_size - len(new_population), 2):
                if selection == 'tournament':
                    parent1 = tournament_selection(population, fitness_values, tournament_size=2)
                    parent2 = tournament_selection(population, fitness_values, tournament_size=2)

                elif selection == 'rank':
                    parent1 = rank_selection(population, fitness_values)
                    parent2 = rank_selection(population, fitness_values)

                elif selection == 'roulette':
                    parent1 = roulette_selection(population, fitness_values)
                    parent2 = roulette_selection(population, fitness_values)

                # Applying crossover to the individuals
                if crossover == 'arithmetic':
                    if random.uniform(0, 1) <= CXPB_LIST[generation]: 
                        offspring1, offspring2 = arithmetic_crossover(parent1, parent2, min_values_norm, max_values_norm, alpha = 0.05)
                        new_population.append(offspring1)
                        new_population.append(offspring2)
                    else:
                        new_population.append(parent1)
                        new_population.append(parent2)
                elif crossover == 'SBX':
                    if random.uniform(0, 1) <= CXPB_LIST[generation]: 
                        offspring1, offspring2 = SBX_crossover(parent1, parent2, min_values_norm, max_values_norm, eta=0.5)
                        new_population.append(offspring1)
                        new_population.append(offspring2)
                    else:
                        new_population.append(parent1)
                        new_population.append(parent2)
                elif crossover == '1-point':
                    if random.uniform(0, 1) <= CXPB_LIST[generation]: 
                        offspring1, offspring2 = crossover_1pt(parent1, parent2)
                        new_population.append(offspring1)
                        new_population.append(offspring2)
                    else:
                        new_population.append(parent1)
                        new_population.append(parent2)
                elif crossover == '2-point':
                    if random.uniform(0, 1) <= CXPB_LIST[generation]: 
                        offspring1, offspring2 = crossover_2pt(parent1, parent2)
                        new_population.append(offspring1)
                        new_population.append(offspring2)
                    else:
                        new_population.append(parent1)
                        new_population.append(parent2)

            # Ensuring that the new population will have the correct size
            if len(new_population) > population_size:
                aux = len(new_population) - population_size
                new_population = new_population[ : -aux]
            population_size_old = population_size

            # Applying mutation to the new population 
            std_dev, eta, old_mut_param = online_expected_diversity(generation, num_generations, values_gen["metrics"][generation], old_mut_param)
            if mutation == 'polynomial':
                population = [polynomial_mutation(ind, min_values, max_values_norm, eta) if random.uniform(0, 1) <= MUTPB_LIST[generation] else ind for ind in new_population]
            elif mutation == 'gaussian':
                population = [gaussian_mutation(ind, min_values, max_values_norm, std_dev) if random.uniform(0, 1) <= MUTPB_LIST[generation] else ind for ind in new_population]

            # Checking if any inddividuals are in the Tabu list
            while TabuList is True:
                count = 0
                for i in range(population_size):
                    if population[i] in tabu_List:
                        if mutation == 'polynomial':
                            polynomial_mutation(population[i], min_values, max_values_norm, eta)
                        elif mutation == 'gaussian':
                            gaussian_mutation(population[i], min_values, max_values_norm, std_dev)
                    else:
                        count += 1
                if count == population_size:
                    break

            # New population size       
            population_size = population_size_old + random.randint(-1, 1)*int(population_size_old*random.gauss(0,0.1))
            if population_size < population_size_old/2:
                population_size += population_size_old*random.random()
            elif population_size > 2*population_size_old:
                population_size -= population_size_old*random.random()

        # Printing global optimization results
        settings.log.warning("***************************** END ******************************")
        settings.log.warning('Best Global Individual: {}'.format(best_individual["ind"][best_individual["fit"].index(min(best_individual["fit"]))]))
        settings.log.warning('Best Global Fitness: {}'.format(min(best_individual["fit"])))
        if min(best_individual["fit"]) != 0: settings.log.warning('Best Global Score: {}'.format(1/min(best_individual["fit"])))
        settings.log.warning(f"Tempo de Execução: {round(time.time() - t_inicial, 2)}")

        # Listing outputs
        history["ind"] = denormalize_population(history["ind_norm"], min_values, max_values, classe)
        history_valid["ind"] = denormalize_population(history_valid["ind_norm"], min_values, max_values, classe)
        out = dict(history = history,
                history_valid = history_valid,
                best_individual = best_individual,
                values_gen = values_gen,
                )

        export_excell(out)

        if plotfit == True:
            create_plotfit(num_generations, values_gen)
        if plotbox == True:
            create_boxplots(out, num_generations, min_values, max_values)
        if plotparallel == True:
            parallel_coordinates(out)

    except Exception as e:
        ErrorLog.erro(str(e))
        return ErrorType("danger", str(e))

    return out