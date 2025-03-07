import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from TP_SRE import *

# Initialisation de l'application Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Projet de Statistiques des Risques Extrêmes"

# Liste des tickers
tickers = ['^FCHI', 'AAPL', 'GOOG', 'MSFT', 'AMZN']

# Mise en page avec onglets
app.layout = html.Div(style={'fontFamily': 'Arial', 'padding': '20px'}, children=[
    # Titre et noms au-dessus des onglets
    html.H1("Projet de Statistiques des Risques Extrêmes", style={'textAlign': 'center', 'color': '#333'}),
    html.Div([html.H4("Mathis Bouillon, Cheryl Kouadio", style={'textAlign': 'center', 'color': '#555'})], style={'marginBottom': '30px'}),
    
    # Onglets
    dcc.Tabs(id="tabs", value='tab1', children=[
        dcc.Tab(label='VaR et ES', value='tab1'),
        dcc.Tab(label='EVT', value='tab2'),
        dcc.Tab(label='GARCH', value='tab3')
    ]),
    html.Div(id='tabs-content')
])

# Callback pour afficher le contenu des onglets
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def update_tab(tab_name):
    if tab_name == 'tab1':
        return html.Div(children=[
            # Sélection du ticker
            html.Div([
                html.Label("Sélectionnez un Ticker :", style={'fontSize': '18px'}),
                dcc.Dropdown(id='ticker-dropdown', options=[{'label': t, 'value': t} for t in tickers], value='^FCHI', style={'width': '50%'})
            ], style={'marginBottom': '20px'}),
            
            # Sélection des dates
            html.Div([
                html.Label("Sélectionnez les dates de train et test :", style={'fontSize': '18px'}),
                dcc.DatePickerRange(id='train-date-picker', start_date='2008-10-15', end_date='2022-07-26', display_format='YYYY-MM-DD'),
                dcc.DatePickerRange(id='test-date-picker', start_date='2022-07-27', end_date='2024-06-11', display_format='YYYY-MM-DD')
            ], style={'marginBottom': '20px'}),
            
            # Graphique des prix et rendements
            dcc.Graph(id='price-graph', style={'borderRadius': '10px', 'boxShadow': '2px 2px 10px #aaa'}),
            
            # Sélection du niveau de confiance
            html.Div([
                html.Label("Niveau de confiance :", style={'fontSize': '18px'}),
                dcc.Slider(id='confidence-level-slider', min=90, max=99, step=1, marks={i: f"{i}%" for i in range(90, 100)}, value=99)
            ], style={'marginTop': '20px'}),
            
            # Section VaR
            html.Div(style={'marginTop': '40px'}, children=[
                html.H3("Value at Risk (VaR)", style={'textAlign': 'center', 'color': '#333'}),
                html.P(
                    "La Value at Risk (VaR) est une mesure statistique utilisée pour évaluer le risque de perte d'un actif ou d'un portefeuille. "
                    "Elle permet d'estimer la perte maximale attendue sur une période donnée, avec un certain niveau de confiance.",
                    style={'fontSize': '16px', 'color': '#555', 'textAlign': 'center', 'marginBottom': '20px'}
                ),
                html.Div(style={'display': 'flex', 'justifyContent': 'center', 'gap': '30px', 'marginTop': '20px'}, children=[
                    html.Div(id='var-historical-box', style={
                        'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa',
                        'textAlign': 'center', 'backgroundColor': '#f8f9fa', 'width': '30%'
                    }),
                    html.Div(style={
                        'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa',
                        'textAlign': 'center', 'backgroundColor': '#f1f3f4', 'width': '30%'
                    }, children=[
                        html.Div(id='var-bootstrap-box'),
                        html.Label("Intervalle de confiance (α_IC) :", style={'fontSize': '16px', 'marginTop': '10px'}),
                        dcc.Slider(
                            id='alpha-ic-slider',
                            min=80,
                            max=99,
                            step=1,
                            marks={i: f"{i}%" for i in range(80, 100, 5)},
                            value=90
                        ),
                    ]),
                    html.Div(id='var-gaussian-box', style={
                        'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa',
                        'textAlign': 'center', 'backgroundColor': '#e3f2fd', 'width': '30%'
                    }),
                    html.Div(id='var-student-box', style={
                        'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa',
                        'textAlign': 'center', 'backgroundColor': '#f7c2c2', 'width': '30%'
                    }),
                ]),
            ]),
            
            # Section Expected Shortfall
            html.Div(style={'marginTop': '40px'}, children=[
                html.H3("Expected Shortfall (ES)", style={'textAlign': 'center', 'color': '#333'}),
                html.P(
                    "L'Expected Shortfall (ES), ou perte attendue, est la moyenne des pertes qui dépassent la VaR. "
                    "Contrairement à la VaR qui donne un seuil de perte pour un certain niveau de confiance, "
                    "l'ES donne l'ampleur moyenne de la perte au-delà de ce seuil.",
                    style={'fontSize': '16px', 'color': '#555', 'textAlign': 'center', 'marginBottom': '20px'}
                ),
                html.Div(style={'display': 'flex', 'justifyContent': 'center', 'gap': '30px'}, children=[
                    html.Div(id='es-gaussian-box', style={
                        'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa',
                        'textAlign': 'center', 'backgroundColor': '#ffeb3b', 'width': '30%'
                    }),
                    html.Div(id='es-historical-box', style={
                        'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa',
                        'textAlign': 'center', 'backgroundColor': '#8bc34a', 'width': '30%'
                    }),
                    html.Div(id='es-student-box', style={
                        'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa',
                        'textAlign': 'center', 'backgroundColor': '#f44336', 'width': '30%'
                    }),
                ]),
            ]),
            
            # Section Expected Shortfall Théorique
            html.Div(style={'marginTop': '40px'}, children=[
                html.H3("Expected Shortfall (ES) Théorique", style={'textAlign': 'center', 'color': '#333'}),
                html.P("L'ES théorique est défini comme :", style={'fontSize': '16px', 'color': '#555', 'textAlign': 'center'}),
                dcc.Markdown(
                    r"""
                    $$
                    ES_\alpha = \frac{1}{1 - \alpha} 
                    \int_{-\infty}^{VaR_\alpha} x f(x) \,dx 
                    $$
                    """,
                    mathjax=True,
                    style={'textAlign': 'center', 'fontSize': '18px'}
                ),
                html.Div(style={'display': 'flex', 'justifyContent': 'center', 'gap': '30px'}, children=[
                    html.Div(id='es-theo-gaussian-box', style={
                        'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa',
                        'textAlign': 'center', 'backgroundColor': '#ffeb3b', 'width': '30%'
                    }),
                    html.Div(id='es-theo-student-box', style={
                        'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa',
                        'textAlign': 'center', 'backgroundColor': '#f44336', 'width': '30%'
                    }),
                ]),
            ]),
        ])
    elif tab_name == 'tab2':
        return html.Div(children=[
            html.H3("Analyse TVE (Threshold Exceedances)", style={'textAlign': 'center', 'color': '#333'}),
        
            # Sélection du ticker
            html.Div([
                html.Label("Sélectionnez un Ticker :", style={'fontSize': '18px'}),
                dcc.Dropdown(id='ticker-dropdown-tve', options=[{'label': t, 'value': t} for t in tickers], value='^FCHI', style={'width': '50%'})
            ], style={'marginBottom': '20px'}),
        
            # Sélection des dates
            html.Div([
                html.Label("Sélectionnez les dates de train et test :", style={'fontSize': '18px'}),
                dcc.DatePickerRange(id='train-date-picker-tve', start_date='2008-10-15', end_date='2022-07-26', display_format='YYYY-MM-DD'),
                dcc.DatePickerRange(id='test-date-picker-tve', start_date='2022-07-27', end_date='2024-06-11', display_format='YYYY-MM-DD')
            ], style={'marginBottom': '20px'}),
        
            # Section Block Maxima
            html.Div([
                html.H4("Partie Block Maxima", style={'textAlign': 'center', 'color': '#333'}),
                html.P(
                    "La méthode Block Maxima consiste à diviser les données en blocs (par exemple, mensuels ou annuels) "
                    "et à extraire le maximum de chaque bloc pour modéliser les extrêmes.",
                    style={'fontSize': '16px', 'color': '#555', 'textAlign': 'center', 'marginBottom': '20px'}
                ),
            ]),
        
            # Sélection de la fréquence des blocs
            html.Div([
                html.Label("Fréquence des blocs :", style={'fontSize': '18px'}),
                dcc.Dropdown(id='freq-dropdown-tve', options=[
                    {'label': 'Mensuel', 'value': 'M'},
                    {'label': 'Trimestriel', 'value': 'Q'},
                    {'label': 'Annuel', 'value': 'Y'}
                ], value='M', style={'width': '50%'})
            ], style={'marginBottom': '20px'}),
        
            # Graphique des Block Maxima
            dcc.Graph(id='block-maxima-graph', style={'borderRadius': '10px', 'boxShadow': '2px 2px 10px #aaa'}),

            # Sélection du niveau de confiance
            html.Div([
                html.Label("Niveau de confiance :", style={'fontSize': '18px'}),
                dcc.Slider(id='confidence-level-slider_tve', min=90, max=99, step=1, marks={i: f"{i}%" for i in range(90, 100)}, value=99)
            ], style={'marginTop': '20px'}),

            html.Div(id='var-tve-box', style={
                    'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa',
                    'textAlign': 'center', 'backgroundColor': '#e3f2fd', 'width': '30%'
            }),
            
            # Section PoT (Peaks over Threshold)
            html.Div([
                html.H4("Partie Peaks over Threshold (PoT)", style={'textAlign': 'center', 'color': '#333'}),
                html.P(
                    "La méthode PoT consiste à modéliser les excès au-dessus d'un seuil donné en utilisant "
                    "la distribution de Pareto généralisée (GPD).",
                    style={'fontSize': '16px', 'color': '#555', 'textAlign': 'center', 'marginBottom': '20px'}
                ),
            ]),
        
            # Sélection du quantile pour le Mean Excess Plot
            html.Div([
                html.Label("Quantile pour le Mean Excess Plot :", style={'fontSize': '18px'}),
                dcc.Slider(id='quantile-slider', min=0.90, max=1, step=0.01, marks={i: f"{int(i*100)}%" for i in np.arange(0.90, 1.01, 0.01)}, value=0.95)
            ], style={'marginBottom': '20px'}),
        
            # Graphique du Mean Excess Plot
            dcc.Graph(id='mean-excess-plot', style={'borderRadius': '10px', 'boxShadow': '2px 2px 10px #aaa'}),
        
            # Dans la section PoT (Peaks over Threshold), remplacer le slider par un dcc.Input
            html.Div([
                html.Label("Seuil pour ajuster la GPD :", style={'fontSize': '18px'}),
                dcc.Input(
                    id='threshold-input',
                    type='number',
                    value=0.012,  # Valeur par défaut
                    min=0.001,    # Valeur minimale
                    max=0.050,    # Valeur maximale
                    step=0.001,   # Pas d'incrémentation
                    style={'width': '50%', 'marginBottom': '20px'}
                ),
            ], style={'marginBottom': '20px'}),
        
            # Affichage de la VaR PoT
            html.Div(id='var-pot-box', style={
                'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa',
                'textAlign': 'center', 'backgroundColor': '#e3f2fd', 'width': '30%', 'marginTop': '20px'
            }),
        ])
    elif tab_name == 'tab3':
        return html.Div([html.H3("to be continued mdrr")])
    return html.Div()


# Callback pour mettre à jour les valeurs et le graphique
@app.callback(
    [Output('price-graph', 'figure'), Output('var-historical-box', 'children'), Output('var-bootstrap-box', 'children'),
     Output('var-gaussian-box', 'children'), Output('var-student-box', 'children'), Output('es-gaussian-box', 'children'),
     Output('es-historical-box', 'children'), Output('es-student-box', 'children'), Output('es-theo-gaussian-box','children'),
     Output('es-theo-student-box','children')],
    [Input('ticker-dropdown', 'value'), Input('confidence-level-slider', 'value'), Input('alpha-ic-slider', 'value'),
     Input('train-date-picker', 'start_date'), Input('train-date-picker', 'end_date'), Input('test-date-picker', 'start_date'),
     Input('test-date-picker', 'end_date')]
)
def update_graphs(ticker, confidence_level, alpha_ic, train_start_date, train_end_date, test_start_date, test_end_date):

    df = telecharger_donnees(ticker, "2000-01-01")

    df_train, df_test = split_train_test(df, train_start_date, train_end_date, test_start_date, test_end_date)

    price_fig = plot_dual_axis_graph(df, ticker, train_start_date, train_end_date, test_start_date, test_end_date)
    
    # Calcul des VaR
    var_hist = VaR_Hist(df_train['log_returns'], confidence_level / 100)
    var_bootstrap, lower, upper = VaR_Hist_Bootstrap(df_train['log_returns'], confidence_level / 100, B=1000, alpha_IC=alpha_ic / 100)
    var_gauss = VaR_Gauss(df_train['log_returns'], confidence_level / 100)
    var_student = calculate_var_student(df_train['log_returns'], confidence_level / 100)
    
    # Calcul des ES empiriques
    es_emp_gauss = ES_empirique(df_train['log_returns'], var_gauss)
    es_emp_hist = ES_empirique(df_train['log_returns'], var_hist)
    es_emp_student = ES_empirique(df_train['log_returns'], var_student)
    
    # Calcul des ES théoriques
    mu_hat, sigma_hat, gamma_hat, nu_hat = estimate_skew_student_params(df_train['log_returns'].dropna())
    
    def pdf_gauss(x):
        return st.norm.pdf(x, loc=np.mean(df_train['log_returns']), scale=np.std(df_train['log_returns']))
    
    def pdf_skew_student(x):
        return skew_student_pdf(x, mu_hat, sigma_hat, gamma_hat, nu_hat)
    
    es_theo_gauss = ES_theorique(confidence_level / 100, var_gauss, pdf_gauss)
    es_theo_student = ES_theorique(confidence_level / 100, var_student, pdf_skew_student)
    
    # Mise en forme des résultats
    var_historical_content = html.Div([html.H3(f"VaR Historique ({confidence_level}%)"), html.P(f"{var_hist:.4f}")])
    var_bootstrap_content = html.Div([html.H3(f"VaR Bootstrap ({confidence_level}%)"), html.P(f"{var_bootstrap:.4f}"), html.P(f"IC {alpha_ic}% : [{lower:.4f}, {upper:.4f}]")])
    var_gaussian_content = html.Div([html.H3(f"VaR Gaussienne ({confidence_level}%)"), html.P(f"{var_gauss:.4f}")])
    var_student_content = html.Div([html.H3(f"VaR Student ({confidence_level}%)"), html.P(f"{var_student:.4f}")])
    es_gaussian_content = html.Div([html.H3(f"ES Gaussien ({confidence_level}%)"), html.P(f"{es_emp_gauss:.4f}")])
    es_historical_content = html.Div([html.H3(f"ES Historique ({confidence_level}%)"), html.P(f"{es_emp_hist:.4f}")])
    es_student_content = html.Div([html.H3(f"ES Student ({confidence_level}%)"), html.P(f"{es_emp_student:.4f}")])
    es_theo_gaussian_content = html.Div([html.H3(f"ES Gaussien Théorique ({confidence_level}%)"), html.P(f"{es_theo_gauss:.4f}")])
    es_theo_student_content = html.Div([html.H3(f"ES Skew-Student Théorique ({confidence_level}%)"), html.P(f"{es_theo_student:.4f}")])
    
    return price_fig, var_historical_content, var_bootstrap_content, var_gaussian_content, var_student_content, es_gaussian_content, es_historical_content, es_student_content, es_theo_gaussian_content, es_theo_student_content

@app.callback(
    [Output('block-maxima-graph', 'figure'),
     Output('var-tve-box', 'children')],
    [Input('ticker-dropdown-tve', 'value'),
     Input('train-date-picker-tve', 'start_date'),
     Input('train-date-picker-tve', 'end_date'),
     Input('test-date-picker-tve', 'start_date'),
     Input('test-date-picker-tve', 'end_date'),
     Input('freq-dropdown-tve', 'value'),
     Input('confidence-level-slider_tve', 'value')]
)
def update_tve(ticker, train_start_date, train_end_date, test_start_date, 
               test_end_date, freq, confidence_level_tve):
    # Charger les données
    df = telecharger_donnees(ticker, "2000-01-01")
    
    # Découpage en train/test
    df_train, df_test = split_train_test(df, train_start_date, train_end_date, test_start_date, test_end_date)
    
    # Calculer les maxima par bloc
    block_maxima = compute_block_maxima(df_train, freq=freq)
    
    fig_maxima = plot_block_maxima(df_train, block_maxima)

    # Ajuster la loi GEV
    c, loc, scale = fit_gev(block_maxima)
    
    var_tve = compute_var_tve(confidence_level_tve /100, c, loc, scale)
    
    print(f"La VaR TVE à {confidence_level_tve}% de confiance est : {var_tve:.4f}")

    # Retourner la figure dans une liste
    return [fig_maxima , var_tve]

@app.callback(
    [Output('mean-excess-plot', 'figure'),
     Output('var-pot-box', 'children')],
    [Input('ticker-dropdown-tve', 'value'),
     Input('train-date-picker-tve', 'start_date'),
     Input('train-date-picker-tve', 'end_date'),
     Input('test-date-picker-tve', 'start_date'),
     Input('test-date-picker-tve', 'end_date'),
     Input('quantile-slider', 'value'),
     Input('threshold-input', 'value'),  # Utiliser threshold-input au lieu de threshold-slider
     Input('confidence-level-slider_tve', 'value')]
)
def update_pot(ticker, train_start_date, train_end_date, test_start_date, 
               test_end_date, quantile, threshold, confidence_level_tve):
    # Charger les données
    df = telecharger_donnees(ticker, "2000-01-01")
    
    # Découpage en train/test
    df_train, df_test = split_train_test(df, train_start_date, train_end_date, test_start_date, test_end_date)
    
    # Créer une copie du DataFrame et inverser les rendements négatifs
    df_copy = df_train.copy()
    df_copy['neg_log_returns'] = -df_copy['log_returns']
    
    # Calculer les excès moyens pour le Mean Excess Plot
    quantile_percent = df_copy['neg_log_returns'].quantile(quantile)
    threshold_values = np.linspace(max(df_copy['neg_log_returns'].min(), 0), min(quantile_percent, df_copy['neg_log_returns'].max()), 150)
    mean_excesses, thresholds = mean_excess_plot(df_copy['neg_log_returns'], threshold_values)
    
    # Tracer le Mean Excess Plot avec Plotly
    fig_mean_excess = go.Figure()
    fig_mean_excess.add_trace(go.Scatter(
        x=thresholds, y=mean_excesses, mode='lines+markers', name='Mean Excess',
        line=dict(color='blue', width=2), marker=dict(size=8)
    ))
    fig_mean_excess.update_layout(
        title="Mean Excess Plot",
        xaxis_title="Threshold",
        yaxis_title="Mean Excess",
        showlegend=True,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    
    # Ajuster la GPD sur les excès
    exceedances = df_copy['neg_log_returns'][df_copy['neg_log_returns'] > threshold] - threshold
    c, loc, scale = fit_gpd_to_exceedances(df_copy, threshold)
    
    # Calculer alpha_pot
    alpha_pot = calculate_alpha_pot(df_copy, exceedances, confidence_level_tve / 100)
    
    # Calculer la VaR PoT
    var_pot = calculate_var_tve_pot(alpha_pot, c, loc, scale, threshold)
    
    # Afficher la VaR PoT
    var_pot_content = html.Div([
        html.H3(f"VaR PoT ({confidence_level_tve}%)"),
        html.P(f"{var_pot:.4f}")
    ])
    
    return fig_mean_excess, var_pot_content


if __name__ == '__main__':
    app.run_server(debug=True)