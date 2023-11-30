"""
Dedicated functions for plotting graphs or exporting/importing results.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from AeroGA import settings
from AeroGA.Classes.Error import ErrorType, Log

# Setting error log file
ErrorLog = Log("error.log", 'Postprocessing')

def sensibility(individual = list, fitness_fn = None, increment = None, min_values = list, max_values = list):
    """Calculate the fitness of an individual for each iteration, where one variable is incremented by a given value within the range of min and max values.
    If variable is integer, it will increment by 1 instead of a float value.
    """

    try:
        dict = {"nvar":[],"value":[],"fit":[]}

        if isinstance(increment, float): step = [ increment for _ in range(len(min_values))]
        elif isinstance(increment, list): step = increment

        for i in range(len(individual)):
            settings.log.info('Iteração: {} de {}'.format(i, len(individual)))
            for new_value in np.arange(min_values[i], max_values[i], step[i]):
                new_individual = individual.copy()
                if isinstance(new_individual[i], int):
                    new_value = int(new_value)
                new_individual[i] = new_value
                dict["nvar"].append(i)
                dict["value"].append(new_value)
                dict["fit"].append(fitness_fn(new_individual))

        df = pd.DataFrame(dict)
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M")
        string = 'Resultados/Sensibility_' + str(dt_string) + '.xlsx'
        df.to_excel(string, index=False)

        return print(pd.DataFrame(dict))
    except Exception as e:
        ErrorLog.error(str(e))
        return ErrorType("danger", str(e), 'sensibility')


def export_excell(out):
    """Create a parallel coordinates graph of the population history."""
    
    try:
        history = out["history"]
        lista = list(history["gen"])
        lista2 = list(history["fit"])
        lista3 = list(history["score"])

        num_ind = len(history["ind"])
        num_var = len(history["ind"][0])
        
        data = []; aux = []; aux2 = []
    
        for i in range(num_var):
            for j in range(num_ind):
                aux.append(history["ind"][j][i])
            aux2.append(aux)
            aux = []

        data = list(map(lambda *x: list(x), *aux2))
        df = pd.DataFrame(data)

        df['gen'] = lista
        df['fit'] = lista2
        df['Score'] = lista3

        # Check whether the specified path exists or not
        path = "Resultados"
        pathExist = os.path.exists(path)
        if not pathExist:
            os.makedirs(path)   

        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M")
        string = 'Resultados/Results_' + str(dt_string) + '.xlsx'

        df.to_excel(string, index=False)
    except Exception as e:
        ErrorLog.error(str(e))
        return ErrorType("danger", str(e), 'export_excell')

# #####################################################################################
# ##################################### Tests #########################################
# #####################################################################################

# def run_n_times(selection1, crossover1, mutation1, n_threads1,
#     min_values1, max_values1, num_variables1, num_generations1, elite_count1,
#     fitness_fn1, classe1, n):

#     fit = []
#     best_fit = []
    
#     for _ in range(0,n):

#         out = optimize(selection = selection1, crossover = crossover1, mutation = mutation1, n_threads = n_threads1,
#         min_values = min_values1, max_values = max_values1, num_variables = num_variables1, num_generations = num_generations1, elite_count = elite_count1,
#         fitness_fn = fitness_fn1, classe = classe1, plotfit=False)

#         fit.append(out["best_individual"]["fit"])
#         best_fit.append(min(out["best_individual"]["fit"]))
    
#     aux = list(map(lambda *x: list(x), *fit))
#     df = pd.DataFrame(aux)
#     df2 = pd.DataFrame(best_fit)

#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.subplots_adjust(hspace=0.5)

#     ax1.boxplot(df, vert=True)
#     ax1.set_title('Fitness for '+ str(n) +' Runs')
#     ax1.set_ylabel('Fitness')
#     ax1.grid(True)

#     ax2.boxplot(df2, vert=True)
#     ax2.set_title('Best Fitness for All Runs')
#     ax2.grid(True)

#     plt.show()