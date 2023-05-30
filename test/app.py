import matplotlib.pyplot as plt
from AeroGA.AeroGA import *
from Benchmarks import *

if __name__ == '__main__':
    """ 
    **********  Dicas de utilização  ***********

    # Métodos seleção: "roulette", "rank", "tournament" -> Read README.md for detailed info
    # Métodos crossover: "arithmetic", "SBX" ,"1-point", "2-point" -> Read README.md for detailed info
    # Métodos mutação: "gaussian", "polynomial" -> Read README.md for detailed info

    *VALORES DEFAULT*
    out = optimize(selection = "tournament", crossover = "1-point", mutation = "gaussian", n_threads = -1,
    min_values = [], max_values = [], num_variables = [], population_size = [], num_generations = [], elite_count = [],
    online_control = False, mutation_rate = 0.4, crossover_rate = 1, eta = 5, std_dev = 0.1,
    plotfit = True, plotbox = False, plotparallel = False, 
    fitness_fn = function

    # mutação gaussiana -> std_dev                      Ou seja não é neceessário definir eta se estiver usando
    # mutação polinomial -> eta                         mutação gaussiana, para o polinomial serve o mesmo
       
    # mutation_rate e crossover_rate só são ativados caso online_control = False

    *VALORES RETORNADOS PELO OUT -> out = [população, history, best_individual, values_gen]

    # Função de sensibilidade -> sensibility(out["best_individual"], fitness_fn, increment=0.01, lb, ub)
    """
    
    lb = [0, 0, 0.0, 0.0, 0.0, 0.0]
    ub = [5, 0, 0.4, 1.5, 0.5, 1.0]
    
    # Run the genetic algorithm
    out = optimize(selection = "tournament", crossover = "1-point", mutation = "polynomial", n_threads=-1,
    min_values = lb, max_values = ub, num_variables = 6, num_generations = 50, elite_count = 0,
    fitness_fn = Rastrigin
    )

# lb = [0, 0, 0.0, 0.0, 0.0, 0.0]
# ub = [5, 0, 0.4, 1.5, 0.5, 1.0]
# path = "Resultados\Results_03-03-2023_17-50.xlsx"
# create_boxplots_por_gen_import_xlsx(path, lb, ub, n_gen = 10, gen = 5)
# parallel_coordinates_import_xlsx(path,nvar = 13, classe = "micro")
# create_boxplots_import_xlsx(path)
# create_boxplots_por_gen_import_xlsx(path, 100, 80)