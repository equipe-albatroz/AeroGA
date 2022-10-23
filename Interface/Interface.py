import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dash_table
from dash_table.Format import Format, Group, Scheme, Symbol
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

dados2s = pd.read_csv('mdo2s2.csv')
dados3s = pd.read_csv('mdo3s2.csv')
label2s = list(dados2s.columns)
label3s = list(dados3s.columns)

app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP, dbc.themes.GRID,
        ],
)

BACKGROUND_COLOR = "#eceae5"

info_icon = html.I(className="bi bi-info-circle-fill",
                    style={"color": "#0d6efd", "font-size": "1.5em"})

info_1 = html.Button(info_icon, id='info-1', n_clicks=0, style={
                                "width": "10px", "height": "10px", "padding": "10px", "border": "10px", "float": "right", "margin-right": "10px"})
info_2 = html.Button(info_icon, id='info-2', n_clicks=0, style={
                                "width": "10px", "height": "10px", "padding": "10px", "border": "10px", "float": "right", "margin-right": "10px"})

card_main1 = dbc.Card(
    [
        dbc.CardHeader(
            dbc.Row([
                #dbc.Col(html.H2("Histogram Plot"),className="ms-4 m-auto"),
                dbc.Col(info_1, className="ms-4"),
            ])
        ),

        dbc.CardBody(
            [
                html.H4("Seleção de dados", className="card-title"),
                html.P(
                    "Selecione o número de células da bateria para análise dos resultados do MDO.",
                    className="card-text",
                ),
                dcc.Dropdown(id='dropdown', options=[{'label': 'Motor_2S', 'value': 'MDO_2S'},
                                                         {'label': 'Motor_3S', 'value': 'MDO_3S'}],
                             value='MDO_2S', clearable=False, style={"color": "#000000"}),
                # dbc.Button("Montar Gráfico", color="primary",id='button'),
            ]
        ),
    ],
    # color="dark",   # https://bootswatch.com/default/ for more card colors
    # inverse=True,   # change color of text (black or white)
    # outline=False,  # True = remove the block colors from the background and header
    style={"margin-top": "40px"},
)

card_main2 = dbc.Card([
        dbc.CardHeader(
            dbc.Row([
                dbc.Col(html.H2("Análise de Convergência"), className="ms-4"),
                dbc.Col(info_2, className="ms-4"),
            ]),
        ),
        dcc.Loading([
            dbc.CardBody([

               dcc.Graph(figure={}, id='my-graph')

            ], id="cardbody-tabs", style = {'width': '100%', 'align-items': 'center', 'justify-content': 'center', 'height': '100%'}),
        ], type="circle"),
    ]
    )


app.layout = html.Div([

    html.Div([  
        html.Div([
                html.Div([html.H3("TEST DATASET", style={'display': 'inline-block', 'align-self': 'flex-end'})],style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'height': '100%'}),
            ], style={"width": "90%", "margin": "0 auto", "margin-top": "20px"}),

            html.Div(card_main1, style={"margin": "40px auto"}),
            html.Div(card_main2, style={"margin": "40px auto"}),
            
    ], style={"width":"90%", "background-color": BACKGROUND_COLOR, "margin": "0 auto"}),

],style={"background-color": BACKGROUND_COLOR})


@app.callback(
    Output(component_id='my-graph', component_property='figure'),
    Input(component_id='dropdown', component_property='value'),
)
def update_parallel(nCel):
    if nCel == 'MDO_2S':
        fig = px.parallel_coordinates(dados2s, color="species_id",
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             color_continuous_midpoint=2)
    elif nCel == 'MDO_3S':
        fig = px.parallel_coordinates(dados3s, color="species_id",
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             color_continuous_midpoint=2)
    else:
        print('Deu ruim')   
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)