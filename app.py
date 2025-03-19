import numpy as np
import pandas as pd
import yfinance as yf
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
from VaR_models import GARCHModel, VaR_traditionnelle, SkewStudentVaR

# Télécharger les données du CAC 40
data = yf.download("^FCHI")
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data = data.dropna()

# Dates par défaut
default_train_start = '2008-10-15'
default_train_end = '2022-07-26'
default_test_start = '2022-07-27'
default_test_end = '2024-06-11'

# Interface Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container([
    html.H1("Analyse de la VaR avec GARCH", className="text-center"),
    html.Br(),

    dbc.Row([
        dbc.Col(dcc.DatePickerRange(
            id='train-range',
            start_date=default_train_start,
            end_date=default_train_end,
            display_format='YYYY-MM-DD'
        ), width=4),
        dbc.Col(dcc.DatePickerRange(
            id='test-range',
            start_date=default_test_start,
            end_date=default_test_end,
            display_format='YYYY-MM-DD'
        ), width=4),
        dbc.Col(dcc.Dropdown(
            id='var-type',
            options=[
                {'label': 'Historique', 'value': 'historical'},
                {'label': 'Gaussien', 'value': 'gaussian'},
                {'label': 'GARCH', 'value': 'garch'},
                {'label': 'Skew Student', 'value': 'skewstudent'}
            ],
            value='garch',
            clearable=False
        ), width=4)
    ], className="mb-4"),

    dcc.Graph(id='returns-plot'),
    dcc.Graph(id='var-plot'),
    
    html.H4("Dynamique de μ_t et σ_t", className="text-center"),
    dcc.Graph(id='mu-plot'),
    dcc.Graph(id='sigma-plot'),

    html.H4("Tests de diagnostic"),
    dbc.Table(id='diagnostic-table', bordered=True, hover=True, responsive=True, striped=True),

    html.H4("Paramètres estimés"),
    dbc.Table(id='param-table', bordered=True, hover=True, responsive=True, striped=True)
])


@app.callback(
    [Output('returns-plot', 'figure'),
     Output('var-plot', 'figure'),
     Output('mu-plot', 'figure'),
     Output('sigma-plot', 'figure'),
     Output('diagnostic-table', 'children'),
     Output('param-table', 'children')],
    [Input('train-range', 'start_date'),
     Input('train-range', 'end_date'),
     Input('test-range', 'start_date'),
     Input('test-range', 'end_date'),
     Input('var-type', 'value')]
)
def update_graphs(train_start, train_end, test_start, test_end, var_type):
    # Filtrage des données en fonction des dates sélectionnées
    train = data[['log_return', "Close"]][train_start:train_end]
    data_train = train['log_return']

    test = data[['log_return', "Close"]][test_start:test_end]
    data_test = test['log_return']

    # Fit du modèle GARCH
    garch_model = GARCHModel(data_train, data_test)
    garch_model.fit_garch()

    # Tests de diagnostic
    lb_test_resid, lb_test_resid_sq, jb_test = garch_model.diagnostic_tests()

    # Calcul de la VaR historique et gaussienne
    alpha = 0.99
    res_std = garch_model.std_resid
    var_model_res = VaR_traditionnelle(res_std, data_test, alpha)
    historical_var = var_model_res.historical_var()

    # Calcul de la VaR GARCH
    garch_model.dynamique_var()
    var_garch, nb_exp = garch_model.garch_var(historical_var)

    var_model = VaR_traditionnelle(data_train, data_test, alpha)
    # Sélection de la VaR selon la méthode choisie
    if var_type == 'historical':
        var = var_model.historical_var()
        var_series = pd.Series(var, index=data_test.index)
        params = {"Méthode": "Historique", "VaR": var}
    elif var_type == 'gaussian':
        var = var_model.gaussian_var()
        var_series = pd.Series(var, index=data_test.index)
        params = {"Méthode": "Gaussien", "VaR": var}
    elif var_type == 'garch':
        var = var_garch
        var_series = pd.Series(var, index=data_test.index)
        params = {"Méthode": "GARCH"} | garch_model.params
    elif var_type == 'skewstudent':
        skew_model = SkewStudentVaR(data_train)
        skew_model.fit()
        var = skew_model.sstd_var()
        var_series = pd.Series(var, index=data_test.index)
        params = {"Méthode": "Skew Student", "VaR": var} | skew_model.params_sstd

    # Graphique des rendements
    returns_fig = go.Figure()
    returns_fig.add_trace(go.Scatter(x=data_train.index, y=data_train, mode='lines', name="Train"))
    returns_fig.add_trace(go.Scatter(x=data_test.index, y=data_test, mode='lines', name="Test", line=dict(dash='dash')))
    returns_fig.update_layout(title="Série des rendements", xaxis_title="Date", yaxis_title="Rendement")

    # Graphique de la VaR
    var_fig = go.Figure()
    var_fig.add_trace(go.Scatter(x=data_test.index, y=data_test, mode='lines', name="Rendements"))
    var_fig.add_trace(go.Scatter(x=var_series.index, y=var_series, mode='lines', name=f"VaR {var_type.upper()}", line=dict(color='red')))
    var_fig.update_layout(title="VaR Dynamique", xaxis_title="Date", yaxis_title="Valeur")

    # Graphique de la dynamique de μ_t
    mu_fig = go.Figure()
    mu_fig.add_trace(go.Scatter(x=data.index, y=garch_model.mu_t, mode='lines', name="μ_t (Moyenne conditionnelle)", line=dict(color='green')))
    mu_fig.update_layout(title="Dynamique de μ_t", xaxis_title="Date", yaxis_title="Valeur")

    # Graphique de la dynamique de σ_t
    sigma_fig = go.Figure()
    sigma_fig.add_trace(go.Scatter(x=data.index, y=np.sqrt(garch_model.sigma2), mode='lines', name="σ_t (Volatilité)", line=dict(color='red')))
    sigma_fig.update_layout(title="Dynamique de σ_t", xaxis_title="Date", yaxis_title="Valeur")

    # Tableau des tests de diagnostic
    diagnostic_table = [html.Tr([html.Td("Jarque-Bera"), html.Td(jb_test[0]), html.Td(jb_test[1])]),
                         html.Tr([html.Td("Ljung-Box (résidus)"), html.Td(lb_test_resid.iloc[-1, 0]), html.Td(lb_test_resid.iloc[-1, 1])]),
                         html.Tr([html.Td("Ljung-Box (résidus²)"), html.Td(lb_test_resid_sq.iloc[-1, 0]), html.Td(lb_test_resid_sq.iloc[-1, 1])])]

    # Tableau des paramètres estimés
    param_table = [html.Tr([html.Td(k), html.Td(v)]) for k, v in params.items()]

    return returns_fig, var_fig, mu_fig, sigma_fig, diagnostic_table, param_table


if __name__ == "__main__":
    app.run_server(debug=True)
