from AeroGA.AeroGA import optimize
from AeroGA.Utilities.Plots import *
from AeroGA.Utilities.PostProcessing import *
from Benchmarks import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    lb = [0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ub = [5, 0, 0.4, 1.5, 0.5, 1.0, 5, 0, 0.4, 1.5, 0.5, 1.0, 1.0]
    
    # Run the genetic algorithm
    out = optimize(selection = "tournament", crossover = "1-point", mutation = "polynomial", n_threads=-1,
        min_values = lb, max_values = ub, num_generations = 15, elite_count = 2, plotfit=False,
        fitness_fn = Rastrigin, report=True)

# path = "Resultados\Results_03-03-2023_17-50.xlsx"
# create_boxplots_por_gen_import_xlsx(path, lb, ub, n_gen = 10, gen = 5)
# parallel_coordinates_import_xlsx(path,nvar = 13, classe = "micro")
# create_boxplots_import_xlsx(path)
# create_boxplots_por_gen_import_xlsx(path, 100, 80)