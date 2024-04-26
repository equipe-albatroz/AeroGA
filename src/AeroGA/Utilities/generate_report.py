import os
import jinja2
import webbrowser
from datetime import datetime

def create_report(titulo_pagina, table_html, plotfit, boxplot, parallel):
    # Renderizar o HTML usando o Jinja2
    template_loader = jinja2.FileSystemLoader(searchpath="./")
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template('report_template.html')

    # Ler o conteúdo do arquivo HTML do gráfico paralelo
    with open(parallel, 'r', encoding='utf-8') as parallel_file:
        parallel_content = parallel_file.read()

    # Incluir o gráfico no HTML
    html_output = template.render(titulo_pagina=titulo_pagina, grafico_plot_fit=plotfit, grafico_box_plot=boxplot, 
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