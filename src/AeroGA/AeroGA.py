"""Main module of the AeroGA package."""

from AeroGA.Operators.Population import denormalize_population, denormalize_individual, generate_population_normalized, create_new_population, init_history, check_population_validity
from AeroGA.Operators.Mutation import gaussian_mutation, polynomial_mutation, online_expected_diversity, online_parameter
from AeroGA.Utilities.Plots import create_plotfit, create_boxplots, parallel_coordinates
from AeroGA.Operators.Crossover import apply_crossover_operator
from AeroGA.Operators.Selection import apply_selection_operator
from AeroGA.Utilities.generate_report import create_report
from AeroGA.Utilities.PostProcessing import export_excell
from AeroGA.Operators.Metrics import diversity_metric
from AeroGA.Utilities.Validity import check_bounds
from AeroGA.Operators.Evaluators import evaluate
from AeroGA.Classes.Error import ErrorType
from statistics import mean
import multiprocessing
from . import settings
import traceback
import random
import time
import os

# Setting error log folder
os.makedirs('./Logs', exist_ok=True)

# #####################################################################################
# ###################################### Main #########################################
# #####################################################################################

def optimize(selection: str = "tournament", crossover: str = "1-point", mutation: str = "polynomial", n_threads: int = -1, 
            min_values: list = None, max_values: list = None, population_size: int = 100, num_generations: int = 100, 
            var_names: list[str] | None = None, classe: str = "default", 
            elite_count: int = 5, elite: str = "local", Online_control: bool = True, control_func: str = 'inverse_sigmoidal',
            mutation_prob: float = 0.35, std_dev_offline: float = 0.2, eta_offline: int = 15,
            penalization_list: list = [1000],
            plotfit: bool = True, plotbox: bool = False, plotparallel: bool = False, report: bool = False, export_xlsx: bool = True,
            fitness_fn: callable = None
    ) -> dict:

    """Perform the genetic algorithm to find an optimal solution."""
    try:

        t_inicial = time.time()
        if n_threads == -1: n_threads = multiprocessing.cpu_count()
        num_variables = len(max_values)
        elite_fit_glob = None
        elite_pop_glob = None

        settings.log.warning("****************** INITIALIZING OPTIMIZATION *******************")
        settings.log.info(f"Optimization initialized with {n_threads} threads.")

        # Creating history, metrics and best/avg lists
        values_gen, history, history_valid, best_individual, old_mut_param, tabu_List = init_history()

        # Getting the online parameter control lists
        MUTPB_LIST, CXPB_LIST = online_parameter(Online_control, num_generations, mutation_prob)

        # Checking if num_varialbes matches the lb and ub sizes
        check_bounds(max_values, min_values)

        #### Generating initial random population
        population = generate_population_normalized(population_size, num_variables, min_values, max_values)
        if check_population_validity(population): pass

        #### Initial Fitness calculation for rand pop
        fitness_values = evaluate(denormalize_population(population, min_values, max_values, classe), history, fitness_fn, n_threads)

        # Initial Diversity calculation for rand pop
        values_gen["metrics"].append(diversity_metric(population))

        #### Initializing the main loop
        for generation in range(num_generations):
            t_gen = time.time()

            #### Creating new population and aplying elitist concept
            new_pop, elite_fit_glob, elite_pop_glob = create_new_population(population, fitness_values, elite_count, penalization_list, elite, elite_fit_glob, elite_pop_glob, generation, population_size)
            try: len_new_pop = len(new_pop)
            except: len_new_pop = 0

            #### Applying selection and crossover methods
            for i in range(0, population_size - len_new_pop, 2):
                ## Selection
                parent1, parent2 = apply_selection_operator(selection, population, fitness_values)

                ## Crossover
                new_population = apply_crossover_operator(crossover, new_pop, parent1, parent2, num_generations, generation, CXPB_LIST)

            # Ensuring that the new population will have the correct size
            if len(new_population) > population_size:
                aux = len(new_population) - population_size
                new_population = new_population[ : -aux]

            #### Mutation
            if check_population_validity(new_population): pass
            std_dev, eta, old_mut_param = online_expected_diversity(Online_control, generation, num_generations, values_gen["metrics"][generation], old_mut_param, control_func, std_dev_offline, eta_offline)
            if mutation == 'polynomial':
                population = [polynomial_mutation(ind, [0] * num_variables, [1] * num_variables, eta) if random.uniform(0, 1) <= MUTPB_LIST[generation] else ind for ind in new_population]
            elif mutation == 'gaussian':
                population = [gaussian_mutation(ind, [0] * num_variables, [1] * num_variables, std_dev) if random.uniform(0, 1) <= MUTPB_LIST[generation] else ind for ind in new_population]

            # Checking if any individuals are in the Tabu list, if yes then the individual undergoes a mutation
            if tabu_List:
                if mutation == 'polynomial':
                    population = [polynomial_mutation(ind, [0] * num_variables, [1] * num_variables, eta) if ind in tabu_List else ind for ind in population]
                elif mutation == 'gaussian':
                    population = [gaussian_mutation(ind, [0] * num_variables, [1] * num_variables, std_dev) if ind in tabu_List else ind for ind in population]

            #### Fitness calculation
            fitness_values = evaluate(denormalize_population(population, min_values, max_values, classe), history, fitness_fn, n_threads)

            #### Add to history and valid fit history
            for i in range(len(population)):
                score = float('inf') if fitness_values[i] == 0 else 1/fitness_values[i]
                history["ind_norm"].append(population[i])
                history["ind"].append(denormalize_individual(population[i], min_values, max_values, classe))
                history["fit"].append(fitness_values[i])
                history["gen"].append(generation)           
                history["score"].append(score)

                if fitness_values[i] < min(penalization_list):
                    history_valid["ind_norm"].append(population[i])
                    history_valid["fit"].append(fitness_values[i])
                    history_valid["score"].append(score)
                    history_valid["gen"].append(generation)
                else:
                    tabu_List.append(population[i])            

            
            # Best and average fitness and best individual at the generation
            best_individual["ind"].append(denormalize_individual(population[fitness_values.index(min(fitness_values))], min_values, max_values, classe))
            best_individual["fit"].append(min(fitness_values))

            # Checking if the best fit is better than previus generations and the global value to plot the individual
            best_fitness = best_individual["fit"][generation] if generation == 0 else min(best_fitness, best_individual["fit"][generation])
            
            #### Getting best individuals and printing log
            values_gen = printing_generation_info(generation, best_individual, values_gen, best_fitness, population, fitness_values, min_values, max_values, classe, penalization_list, t_gen)


        #### Printing global optimization results
        settings.log.warning("***************************** END ******************************")
        settings.log.warning('Best Global Individual: {}'.format(best_individual["ind"][best_individual["fit"].index(min(best_individual["fit"]))]))
        settings.log.warning('Best Global Fitness: {}'.format(min(best_individual["fit"])))
        settings.log.warning('Best Global Score: {}'.format(1 / min(best_individual["fit"]))) if min(best_individual["fit"]) != 0 else None
        settings.log.warning(f"Tempo de Execução: {round(time.time() - t_inicial, 2)}")

        # Listing outputs
        history_valid["ind"] = denormalize_population(history_valid["ind_norm"], min_values, max_values, classe)
        history["ind"] = denormalize_population(history["ind_norm"], min_values, max_values, classe)
        out = dict(history = history, history_valid = history_valid, best_individual = best_individual, values_gen = values_gen)

        # Exporting results to Excell
        if export_xlsx:
            export_excell(out)

        # Plotting
        if out["history_valid"]["ind"]:
            if plotfit: create_plotfit(num_generations, values_gen, False, '#FFFFFF')
            if plotbox: create_boxplots(out, min_values, max_values, False, '#FFFFFF')
            if plotparallel: parallel_coordinates(out, min_values, max_values, False, '#FFFFFF')
            if report: create_report('AeroGA Report', num_variables, var_names, out, min_values, max_values, best_individual, num_generations, values_gen)
        else:
            settings.log.warning("Report and/or plots could not be created: history has no valid result. (len(history) = 0)")
        
        return out
    
    except Exception as e:
        traceback_info = traceback.extract_tb(e.__traceback__)  # Stack trace real
        error = ErrorType(type(e).__name__, str(e), 'Optimize', traceback_info)
        settings.log.error(error.message)
        return None
    


def printing_generation_info(generation: int, best_individual: dict, values_gen: dict, best_fitness: float, population: list, fitness_values: list, min_values: list, max_values: list, classe: str, penalization_list: list, t_gen: float):

    # Creating list of fitness values with individuals that returns valid score
    fitness_values_valid = []
    for i in range(len(population)):
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
    score = float('inf') if best_individual["fit"][generation] == 0 else round(1 / best_individual["fit"][generation], 2)
    settings.log.info('Generation: {} | Time: {} | Best Fitness: {} -> Score: {} | Diversity Metric: {}'.format(
        generation + 1, round(time.time() - t_gen, 2), best_individual["fit"][generation], score, round(values_gen["metrics"][generation], 2)))
    
    return values_gen