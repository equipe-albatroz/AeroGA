"""
Dedicated functions for plotting graphs or exporting/importing results.
"""

import base64
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
from AeroGA.Classes.Error import ErrorType

def create_plotfit(num_generations = int, values_gen = None, report = bool, color = str):
    """Plot the fit and metrics values over the number of generations."""
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        fig.subplots_adjust(hspace=0.5)

        ax1.plot(values_gen["best_fit"])
        ax1.set_xlim(0, num_generations - 1)
        ax1.set_title('BestFit x Iterations')
        ax1.set_ylabel('Best Fitness')
        ax1.grid(True)

        ax2.plot(values_gen["avg_fit"], alpha = 0.3, color = 'red',linestyle = "--")
        ax2.set_xlim(0, num_generations - 1)
        ax2.set_title('AvgFit x Iterations')
        ax2.set_ylabel('Average Fitness')
        ax2.grid(True)

        ax3.plot(values_gen["metrics"])
        ax3.set_xlim(0, num_generations - 1)
        ax3.set_title('Population Diversity x Iterations')
        ax3.set_ylabel('Diversity Metric')
        ax3.set_xlabel('Iterations')
        ax3.grid(True)

        if report:
            ax1.set_facecolor(color)
            ax2.set_facecolor(color)
            ax3.set_facecolor(color)
            fig.set_facecolor(color)

        # Salvar o gráfico em um BytesIO para posterior inclusão no HTML
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Codificar a imagem em base64
        fitplot_data_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')

        if not report: plt.show()
        else: plt.close(fig); return fitplot_data_uri
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'create_plotfit')
        return error.message

def create_boxplots(out = None, min_values = list, max_values = list, report = bool, color = str):
    """Boxplot of all values used in the optimization for each variable."""
    
    try:
        history = out["history_valid"]    
        data = []; aux = []; aux2 = []

        for i in range(len(history["ind"][0])):
            for j in range(len(history["ind"])):
                aux.append(history["ind"][j][i])
            aux2.append(aux)
            aux = []

        data = list(map(lambda *x: list(x), *aux2))
        data_aux = data

        for i in range(len(data)):
            for j in range(len(min_values)):
                if max_values[j] - min_values[j] != 0:
                    data_aux[i][j] = (data[i][j] - min_values[j]) / (max_values[j] - min_values[j])
                else:
                    data_aux[i][j] = 0

        df = pd.DataFrame(data_aux)
    
        fig, ax = plt.subplots()
        ax.boxplot(df, vert=True)
        ax.set_title('Dispersion of values')
        ax.set_xlabel('Variables')
        ax.grid(True)

        if report:
            ax.set_facecolor(color)
            fig.set_facecolor(color)

        # Salvar o gráfico em um BytesIO para posterior inclusão no HTML
        buffer2 = BytesIO()
        plt.savefig(buffer2, format='png')
        buffer2.seek(0)

        # Codificar a imagem em base64
        boxplot_data_uri = "data:image_box/png;base64," + base64.b64encode(buffer2.getvalue()).decode('utf-8')

        if not report: plt.show()
        else: return boxplot_data_uri
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'create_boxplots')
        return error.message


def create_boxplots_import_xlsx(path = None):
    """Boxplot of all values used in the optimization for each variable."""

    try:
        df = pd.read_excel(path)
        del df["gen"]
        del df["fit"]
        del df["score"]
    
        plt.boxplot(df, vert=True)
        plt.title('Dispersion of values')
        plt.xlabel('Variables')
        plt.grid(True)

        plt.show()
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'create_boxplots_import_xlsx')
        return error.message

def create_boxplots_por_gen_import_xlsx(path = None, min_values = list, max_values = list, generation = int):
    """Boxplot of all values used in the generation for each variable."""

    try:
        df = pd.read_excel(path)
        del df["fit"]
        del df["score"]
        
        filter = df['gen'] == generation
        df_aux = df[filter]
        del df_aux["gen"]

        lista = df_aux.values.tolist()
        aux = [ [ 0 for _ in range(len(lista[0]))] for _ in range(len(lista)) ]
        for i in range(len(lista)):
            for j in range(len(lista[0])):
                if (max_values[j]-min_values[j]) == 0:
                    aux[i][j] = 0
                else:
                    aux[i][j] = (lista[i][j]-min_values[j])/(max_values[j]-min_values[j])

        df = pd.DataFrame(aux)

        plt.boxplot(df, vert=True)
        plt.title('Value dispersion in generation ' + str(generation))
        plt.xlabel('Variables')
        plt.ylabel('Normalized data')
        plt.grid(True)

        plt.show()
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'create_boxplots_por_gen_import_xlsx')
        return error.message


def parallel_coordinates(out = None, lb = list, ub = list, report = bool, color = str):
    """Create a parallel coordinates graph of the population history."""
    
    try:
        history = out["history_valid"]
        lista = list(history["score"])

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
        df['Score'] = lista
        
        fig = px.parallel_coordinates(df, color="Score", dimensions=df.columns,title="Parallel Coordinates Plot")

        # Ajustar os limites dos eixos
        for i, dim in enumerate(df.columns[:-1]):
            fig.update_layout(yaxis_range=[lb[i], ub[i]])

        if not report: 
            fig.show()
        else: 
            fig.update_layout(plot_bgcolor=color,paper_bgcolor=color)
            file_name = "parallel_plot.html"
            fig.write_html(file_name)
            return file_name
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'parallel_coordinates')
        return error.message

def parallel_coordinates_import_xlsx(path = None, lb = list, ub = list, var_names = list):
    """Create a parallel coordinates graph of the population history."""
    
    try:
        df = pd.read_excel(path)
        del df["gen"]
        del df["fit"]

        if isinstance(var_names,list):
            for i in range(df.shape[1]):
                df = df.rename({df.columns[i]: var_names[i]}, axis='columns')
    
        fig = px.parallel_coordinates(df, color="Score", dimensions=df.columns,title="Parallel Coordinates Plot")

        # Ajustar os limites dos eixos
        for i, dim in enumerate(df.columns[:-1]):
            fig.update_layout(yaxis_range=[lb[i], ub[i]])
        
        fig.show()
    except Exception as e:
       error = ErrorType("ValueError", str(e), 'parallel_coordinates_import_xlsx')
       return error.message

def parallel_coordinates_per_gen_import_xlsx(path = None, lb = list, ub = list, generation = int, var_names = list):
    """Create a parallel coordinates graph of the population history."""
    
    try:
        df_aux = pd.read_excel(path)
    
        filter = df_aux['gen'] == generation
        df = df_aux[filter]

        del df["gen"]
        del df["fit"]

        if isinstance(var_names,list):
            for i in range(df.shape[1]):
                df = df.rename({df.columns[i]: var_names[i]}, axis='columns')
    
        fig = px.parallel_coordinates(df, color="Score", dimensions=df.columns,title="Parallel Coordinates Plot")

        # Ajustar os limites dos eixos
        for i, dim in enumerate(df.columns[:-1]):
            fig.update_layout(yaxis_range=[lb[i], ub[i]])

        fig.show()
    except Exception as e:
        error = ErrorType("ValueError", str(e), 'parallel_coordinates_per_gen_import_xlsx')
        return error.message


# #####################################################################################
# ##################################### Tests #########################################
# #####################################################################################

if __name__ == '__main__':

    # Teste de create_plotfit
    num_generations = 15
    values_gen = {"best_fit": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  "avg_fit": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  "metrics": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    create_plotfit(num_generations, values_gen, report=False, color='white')

    # Teste de create_boxplots
    out = {"history_valid": {"ind": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                             "score": [1, 2, 3, 4]}}
    min_values = [0, 0, 0]
    max_values = [10, 10, 10]
    create_boxplots(out, min_values, max_values, report=False, color='white')

    # Teste de parallel_coordinates
    out = {"history_valid": {"ind": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                             "score": [1, 2, 3, 4]}}
    min_values = [0, 0, 0]
    max_values = [10, 10, 10]
    parallel_coordinates(out, min_values, max_values, report=False, color='white')

    # Teste de parallel_coordinates_import_xlsx
    path = "Resultados\Results_08-07-2023_19-48.xlsx"
    lb = [0, 0, 0.0, 0.0, 0.0, 0.0]
    ub = [5, 0, 0.4, 1.5, 0.5, 1.0]
    var_names = [f'Var_{i+1}' for i in range(6)]; var_names.append('Score')

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