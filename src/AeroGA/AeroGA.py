"""Main module of the AeroGA package."""

from AeroGA.Operators.Mutation import gaussian_mutation, polynomial_mutation, online_expected_diversity, online_parameter
from AeroGA.Operators.Population import denormalize_population, denormalize_individual, generate_population_normalized, create_new_population
from AeroGA.Operators.Crossover import arithmetic_crossover, SBX_crossover, crossover_1pt, crossover_2pt
from AeroGA.Operators.Selection import roulette_selection, tournament_selection, rank_selection
from AeroGA.Utilities.Plots import create_plotfit, create_boxplots, parallel_coordinates
from AeroGA.Utilities.generate_report import create_report
from AeroGA.Utilities.PostProcessing import export_excell
from AeroGA.Operators.Metrics import diversity_metric
from AeroGA.Utilities.Validity import check_bounds
from AeroGA.Operators.Evaluators import evaluate
from AeroGA.Classes.Individual import Individual
from statistics import mean
import multiprocessing
from . import settings
import random
import time

# Setting error log file
# ErrorLog = Log("error.log", 'AeroGA')

# #####################################################################################
# ###################################### Main #########################################
# #####################################################################################

def optimize(selection: str = "tournament", crossover: str = "1-point", mutation: str = "polynomial", n_threads: int = -1,
            min_values: list = None, max_values: list = None, population_size: int = 100, num_generations: int = 100, 
            var_names: list | None = None, classe: str = "default", elite_count: int = 5, elite: str = "local",
            control_func: str = 'inverse_sigmoidal', TabuList: bool = False, penalization_list: list = [1000],
            plotfit: bool = True, plotbox: bool = False, plotparallel: bool = False, fitness_fn: any = None, report: bool = False) -> dict:

    """Perform the genetic algorithm to find an optimal solution."""

    t_inicial = time.time()
    if n_threads == -1: n_threads = multiprocessing.cpu_count()

    # Creating history, metrics and best/avg lists
    values_gen = {"best_fit":[],"avg_fit":[],"metrics":[]}
    history = {"ind":[], "ind_norm":[],"gen":[],"fit":[],"score":[]}
    history_valid = {"ind":[], "ind_norm":[], "gen":[],"fit":[],"score":[]}
    best_individual = {"ind":[],"fit":[]}
    old_mut_param = {"std_dev":[],"eta":[]}
    tabu_List = []

    settings.log.warning("****************** INITIALIZING OPTIMIZATION *******************")
    settings.log.info(f"Optimization initialized with {n_threads} threads.")

    # Checking if num_varialbes matches the lb and ub sizes
    check_bounds(max_values, min_values)

    # Initial value for the best fitness
    best_fitness = float('inf')

    #### Generating initial random population
    num_variables = len(max_values)
    population = generate_population_normalized(population_size, num_variables, min_values, max_values)

    #### Initial Fitness calculation for rand pop
    fitness_values = evaluate(denormalize_population(population, min_values, max_values, classe), history, fitness_fn, n_threads) 

    # Initial Diversity calculation for rand pop
    values_gen["metrics"].append(diversity_metric(population))

    #### Initializing the main loop
    for generation in range(num_generations):
        t_gen = time.time()

        #### Creating new population and aplying elitist concept
        new_population = create_new_population(population, fitness_values, elite_count, penalization_list, elite, generation, population_size)

        # Getting the online parameter control values
        MUTPB_LIST, CXPB_LIST = online_parameter(True, num_generations)

        #### Applying sselection and crossover methods
        for i in range(0, population_size - len(new_population), 2):
            # Selection
            if selection == 'tournament':
                parent1 = tournament_selection(population, fitness_values, tournament_size=2)
                parent2 = tournament_selection(population, fitness_values, tournament_size=2)

            elif selection == 'rank':
                parent1 = rank_selection(population, fitness_values)
                parent2 = rank_selection(population, fitness_values)

            elif selection == 'roulette':
                parent1 = roulette_selection(population, fitness_values)
                parent2 = roulette_selection(population, fitness_values)

            # Crossover
            if crossover == 'arithmetic':
                if random.uniform(0, 1) <= CXPB_LIST[generation]: 
                    offspring1, offspring2 = arithmetic_crossover(parent1, parent2, [0] * num_generations, [1] * num_generations, alpha = 0.05)
                    new_population.append(offspring1)
                    new_population.append(offspring2)
                else:
                    new_population.append(parent1)
                    new_population.append(parent2)
            elif crossover == 'SBX':
                if random.uniform(0, 1) <= CXPB_LIST[generation]: 
                    offspring1, offspring2 = SBX_crossover(parent1, parent2, [0] * num_generations, [1] * num_generations, eta=0.5)
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

        #### Applying mutation to the new population
        std_dev, eta, old_mut_param = online_expected_diversity(generation, num_generations, values_gen["metrics"][generation], old_mut_param, control_func)
        if mutation == 'polynomial':
            population = [polynomial_mutation(ind, min_values, [1] * num_generations, eta) if random.uniform(0, 1) <= MUTPB_LIST[generation] else ind for ind in new_population]
        elif mutation == 'gaussian':
            population = [gaussian_mutation(ind, min_values, [1] * num_generations, std_dev) if random.uniform(0, 1) <= MUTPB_LIST[generation] else ind for ind in new_population]

        # Checking if any inddividuals are in the Tabu list
        while TabuList is True:
            count = 0
            for i in range(population_size):
                if population[i] in tabu_List:
                    if mutation == 'polynomial':
                        polynomial_mutation(population[i], min_values, [1] * num_generations, eta)
                    elif mutation == 'gaussian':
                        gaussian_mutation(population[i], min_values, [1] * num_generations, std_dev)
                else:
                    count += 1
                if count == population_size:
                    break 

        #### Fitness calculation   
        fitness_values = evaluate(denormalize_population(population, min_values, max_values, classe), history, fitness_fn, n_threads) 

        #### Add to history and valid fit history
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
        for i in range(population_size):
            if fitness_values[i] < min(penalization_list):
                fitness_values_valid.append(fitness_values[i]) 

        # Saving these values in lists
        values_gen["best_fit"].append(best_fitness)
        if mean(fitness_values) < min(penalization_list):
            values_gen["avg_fit"].append(mean(fitness_values_valid))
        else:
            values_gen["avg_fit"].append(None)
        values_gen["metrics"].append(diversity_metric(population))

        # Printing logger informations
        if best_individual["fit"][generation] == 0:
            settings.log.info('Generation: {} | Time: {} | Population Size: {} | Best Fitness: {} -> Score: {} | Diversity Metric: {}'.format(generation+1, round(time.time() - t_gen, 2), population_size, best_individual["fit"][generation], float('inf'), round(values_gen["metrics"][generation],2)))
        else:    
            settings.log.info('Generation: {} | Time: {} | Population Size: {} | Best Fitness: {} -> Score: {} | Diversity Metric: {}'.format(generation+1, round(time.time() - t_gen, 2), population_size, best_individual["fit"][generation], round(1/best_individual["fit"][generation],2), round(values_gen["metrics"][generation],2)))


    #### Printing global optimization results
    settings.log.warning("***************************** END ******************************")
    settings.log.warning('Best Global Individual: {}'.format(best_individual["ind"][best_individual["fit"].index(min(best_individual["fit"]))]))
    settings.log.warning('Best Global Fitness: {}'.format(min(best_individual["fit"])))
    if min(best_individual["fit"]) != 0: settings.log.warning('Best Global Score: {}'.format(1/min(best_individual["fit"])))
    settings.log.warning(f"Tempo de Execução: {round(time.time() - t_inicial, 2)}")

    # Listing outputs
    history["ind"] = denormalize_population(history["ind_norm"], min_values, max_values, classe)
    history_valid["ind"] = denormalize_population(history_valid["ind_norm"], min_values, max_values, classe)

    # Returning the results
    out = dict(history = history, history_valid = history_valid, best_individual = best_individual, values_gen = values_gen)

    # Exporting results to Excell
    export_excell(out)

    # Plotting
    if plotfit: create_plotfit(num_generations, values_gen, False, '#FFFFFF')
    if plotbox: create_boxplots(out, min_values, max_values, False, '#FFFFFF')
    if plotparallel: parallel_coordinates(out, min_values, max_values, False, '#FFFFFF')
    if report: create_report('AeroGA Report', num_variables, var_names, out, min_values, max_values, best_individual, num_generations, values_gen)
    
    return out