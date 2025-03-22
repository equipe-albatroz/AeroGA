from AeroGA.Utilities.Plots import create_plotfit, create_boxplots, parallel_coordinates
from datetime import datetime
import pandas as pd
import webbrowser
import jinja2
import os

def create_report(titulo_pagina: str = 'AeroGA Report', num_variables: int = None, var_names: list = None, out: dict = None, min_values: list = None, max_values: list = None, best_individual: dict = None, num_generations: int = None, values_gen: dict = None, report: bool = True):

    plotfit = create_plotfit(num_generations, values_gen, report, '#FFFFFF') # '#FFFFFF' -> white
    boxplot = create_boxplots(out, min_values, max_values, report, '#FFFFFF')
    parallel = parallel_coordinates(out, min_values, max_values, report, '#FFFFFF')

    # Defining variables names in case none is given
    if var_names is None: var_names = [f'Var_{i+1}' for i in range(num_variables)]

    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M")
    page_title = titulo_pagina + ' - ' + str(dt_string)
    lst_html = best_individual["ind"][best_individual["fit"].index(min(best_individual["fit"]))]
    df_html = pd.DataFrame(lst_html).transpose()
    for i in range(df_html.shape[1]): df_html = df_html.rename({df_html.columns[i]: var_names[i]}, axis='columns')
    try: df_html['Score'] = 1/min(best_individual["fit"])
    except: df_html['Score'] = float('inf')
    table_html = df_html.to_html(index=False)

    # Renderizar o HTML usando o Jinja2
    template_loader = jinja2.FileSystemLoader(searchpath="./")
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template('report_template.html')

    # Ler o conteúdo do arquivo HTML do gráfico paralelo
    with open(parallel, 'r', encoding='utf-8') as parallel_file:
        parallel_content = parallel_file.read()

    # Incluir o gráfico no HTML
    html_output = template.render(titulo_pagina=page_title, grafico_plot_fit=plotfit, grafico_box_plot=boxplot, 
                                  grafico_parallel_plot=parallel_content, tabela=table_html)

    # Salvar o HTML em um arquivo
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M")
    report_name = 'Report_' + str(dt_string) + '.html'
    report_path = os.path.abspath(os.path.join('./Resultados/', report_name))
    with open(report_path, 'w', encoding='utf-8') as html_file:
        html_file.write(html_output)

    # Abrir o arquivo HTML no navegador padrão
    webbrowser.open(report_path)

def open_report(report_name):
    # Abrir o arquivo HTML no navegador padrão
    webbrowser.open(os.path.abspath(os.path.join('./Resultados/', report_name)))