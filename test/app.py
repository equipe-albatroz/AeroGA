from AeroGA.AeroGA import optimize
from AeroGA.Utilities.Plots import *
from AeroGA.Utilities.PostProcessing import *
from AeroGA.Utilities.generate_report import *
from Benchmarks import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    micro = ['c1', 'chord_ratio2','b1','span_ratio2','iw','nperfilw1','nperfilw2','zwGround','xCG','vh', 'ih','nperfilh','motorindex']
    regular = ['b1', 'span_ratio_2', 'span_ratio_b3', 'c1', 'chord_ratio_c2', 'chord_ratio_c3', 'nperfilw1', 'nperfilw2', 'nperfilw3', 'iw', 'zwground', 'xCG', 'Vh', 'ARh', 'nperfilh', 'lt', 'it', 'xTDP', 'AtivaProfundor', 'motorIndex']


    lb = [0, -3, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ub = [5, 3, 0.4, 1.5, 0.5, 1.0, 5, 0, 0.4, 1.5, 0.5, 1.0, 1.0]
    
    # Run the genetic algorithm
    out = optimize(selection = "tournament", crossover = "1-point", mutation = "polynomial", n_threads=-1, elite="local",
        min_values = lb, max_values = ub, num_generations = 10, elite_count = 5, plotfit=False, Online_control=False, mutation_prob=0.35,
        fitness_fn = Rastrigin, report=True)


# sensibility(individual = [3, 3, 3.0, 3.0, 3.0, 3.0, 3, 3, 3.0, 3.0, 3.0, 3.0, 3.0], 
#             fitness_fn = Rastrigin, 
#             increment  = [1, 1, 0.1, 0.1, 0.1, 0.1, 1, 1, 0.1, 0.0, 0.1, 0.1, 0.1], 
#             min_values = [0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0], 
#             max_values = [5, 5, 5.0, 5.0, 5.0, 5.0, 5, 0, 5.0, 5.0, 5.0, 5.0, 5.0]
#             )

# path = "Resultados\Results_22-03-2025_02-56.xlsx"
# lb = [0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0]
# ub = [5, 0, 0.4, 1.5, 0.5, 1.0, 5, 0, 0.4, 1.5, 0.5, 1.0, 1.0]
# var_names = [f'Var_{i+1}' for i in range(len(lb))]
# parallel_coordinates_import_xlsx(path, lb, ub, var_names)
# parallel_coordinates_per_gen_import_xlsx(path, lb, ub, generation=1, var_names=var_names)
# create_boxplots_import_xlsx(path)
# create_boxplots_por_gen_import_xlsx(path, lb, ub, generation=2)

# open_report("Report_29-03-2025_03-24.html")