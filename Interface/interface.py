import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from TP_SRE import *
import dash_bootstrap_components as dbc
from VaR_models import GARCHModel, VaR_traditionnelle, SkewStudentVaR


# Initialisation de l'application Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Projet de Statistiques des Risques Extrêmes"

# Liste des tickers
tickers = [
    '^FCHI',  # CAC 40
    '^DJI',   # Dow Jones Industrial Average
    '^GSPC',  # S&P 500
    '^IXIC',  # NASDAQ Composite
    'AAPL',   # Apple
    'GOOGL',  # Google
    'MSFT',   # Microsoft
    'AMZN',   # Amazon
    'NVDA',   # NVIDIA
]

# Mise en page avec onglets
app.layout = html.Div(style={'fontFamily': 'Arial', 'padding': '20px'}, children=[
    # Titre et noms au-dessus des onglets
    html.H1("Projet de Statistiques des Risques Extrêmes", style={'textAlign': 'center', 'color': '#333'}),
    html.Div([html.H4("Mathis Bouillon, Cheryl Kouadio", style={'textAlign': 'center', 'color': '#555'})], style={'marginBottom': '30px'}),
    
    # Paragraphe de présentation
    html.Div([
        html.P(
            "Cette interface permet d'explorer différentes méthodes de calcul de la Value-at-Risk (VaR) et de l'Expected Shortfall (ES). "
            "Elle est divisée en trois onglets :",
            style={'fontSize': '16px', 'color': '#555', 'textAlign': 'left', 'marginBottom': '20px'}
        ),
        html.Ul([
            html.Li(
                "I. Value-at-Risk : Fondements et modélisation - Cet onglet présente des méthodes paramétriques (Gaussienne, Student) "
                "et non paramétriques (historique, bootstrap) pour calculer la VaR, ainsi que l'introduction de l'Expected Shortfall.",
                style={'fontSize': '16px', 'color': '#555', 'marginBottom': '10px'}
            ),
            html.Li(
                "II. Value-at-Risk par modélisation des extrêmes - Cet onglet se concentre sur la théorie des valeurs extrêmes (EVT), "
                "avec des méthodes comme Block Maxima et Peaks Over Threshold (PoT).",
                style={'fontSize': '16px', 'color': '#555', 'marginBottom': '10px'}
            ),
            html.Li(
                "III. Value-at-Risk dynamique - Cet onglet explore la VaR dynamique, notamment à l'aide d'un modèle GARCH.",
                style={'fontSize': '16px', 'color': '#555', 'marginBottom': '10px'}
            ),
        ]),
        html.P(
            "Vous pouvez naviguer à votre gré dans les différents onglets ; tout s'actualise au fil de vos choix. "
            "Par exemple, il faudra notamment spécifier le niveau de confiance alpha pour le calcul des VaR ou de l'ES, "
            "ou encore un seuil pour la méthode PoT (voir onglet 2).",
            style={'fontSize': '16px', 'color': '#555', 'textAlign': 'left', 'marginBottom': '20px'}
        ),
    ], style={'marginBottom': '30px'}),
    
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
                    style={'fontSize': '16px', 'color': '#555', 'textAlign': 'left', 'marginBottom': '20px'}
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
                        'textAlign': 'center', 'backgroundColor': '#f8f9fa', 'width': '30%'
                    }),
                    html.Div(id='var-student-box', style={
                        'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa',
                        'textAlign': 'center', 'backgroundColor': '#f1f3f4', 'width': '30%'
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
                    style={'fontSize': '16px', 'color': '#555', 'textAlign': 'left', 'marginBottom': '20px'}
                ),
                html.Div(style={'display': 'flex', 'justifyContent': 'center', 'gap': '30px'}, children=[
                    html.Div(id='es-gaussian-box', style={
                        'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa',
                        'textAlign': 'center', 'backgroundColor': '#f8f9fa', 'width': '30%'
                    }),
                    html.Div(id='es-historical-box', style={
                        'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa',
                        'textAlign': 'center', 'backgroundColor': '#f1f3f4', 'width': '30%'
                    }),
                    html.Div(id='es-student-box', style={
                        'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa',
                        'textAlign': 'center', 'backgroundColor': '#f8f9fa', 'width': '30%'
                    }),
                ]),
            ]),
            
            # Section Expected Shortfall Théorique
            html.Div(style={'marginTop': '40px'}, children=[
                html.H3("Expected Shortfall (ES) Théorique", style={'textAlign': 'center', 'color': '#333'}),
                html.P("L'ES théorique est définie comme :", style={'fontSize': '16px', 'color': '#555', 'textAlign': 'left'}),
                dcc.Markdown(
                    r"""
                    $
                    ES_\alpha = \frac{1}{1 - \alpha} 
                    \int_{-\infty}^{VaR_\alpha} x f(x) \,dx 
                    $
                    """,
                    mathjax=True,
                    style={'textAlign': 'left', 'fontSize': '18px'}
                ),
                html.Div(style={'display': 'flex', 'justifyContent': 'center', 'gap': '30px'}, children=[
                    html.Div(id='es-theo-gaussian-box', style={
                        'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa',
                        'textAlign': 'center', 'backgroundColor': '#f1f3f4', 'width': '30%'
                    }),
                    html.Div(id='es-theo-student-box', style={
                        'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa',
                        'textAlign': 'center', 'backgroundColor': '#f8f9fa', 'width': '30%'
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
                html.H4("Block Maxima", style={'textAlign': 'center', 'color': '#333'}),
                html.P(
                    "La méthode Block Maxima consiste à diviser les données en blocs (par exemple, mensuels ou annuels) "
                    "et à extraire le maximum de chaque bloc pour modéliser les extrêmes.",
                    "Vous pouvez choisir la fréquence des blocs à l'aide de la liste déroulante ci-dessous, "
                    "et cela sera visible sur le graphique interactif.",
                    style={'fontSize': '16px', 'color': '#555', 'textAlign': 'left', 'marginBottom': '20px'}
                ),
            ]),
        
            # Sélection de la fréquence des blocs
            html.Div([
                html.Label("Fréquence des blocs :", style={'fontSize': '18px'}),
                dcc.Dropdown(id='freq-dropdown-tve', options=[
                    {'label': 'Mensuelle', 'value': 'M'},
                    {'label': 'Trimestrielle', 'value': 'Q'},
                    {'label': 'Annuelle', 'value': 'Y'}
                ], value='M', style={'width': '50%'})
            ], style={'marginBottom': '20px'}),
        
            # Graphique des Block Maxima
            dcc.Graph(id='block-maxima-graph', style={'borderRadius': '10px', 'boxShadow': '2px 2px 10px #aaa'}),

            # Sélection du niveau de confiance
            html.Div([
                html.Label("Niveau de confiance :", style={'fontSize': '18px'}),
                dcc.Slider(id='confidence-level-slider_tve', min=90, max=99, step=1, marks={i: f"{i}%" for i in range(90, 100)}, value=99)
            ], style={'marginTop': '20px'}),

            html.Div(id='var-tve-box', style={'marginTop': '20px'}),
            
            # Section PoT (Peaks over Threshold)
            html.Div([
                html.H4("Peaks over Threshold (PoT)", style={'textAlign': 'center', 'color': '#333'}),
                html.P(
                    "La méthode PoT consiste à modéliser les excès au-dessus d'un seuil donné en utilisant "
                    "la distribution de Pareto généralisée (GPD). "
                    "Le graphique ci-dessous (Mean Excess Plot) permet de régler l'affichage pour s'affranchir "
                    "du comportement erratique des queues de distribution. Cela permet d'identifier le seuil "
                    "à partir duquel on observe une tendance linéaire. Vous pouvez ensuite choisir la valeur "
                    "de ce seuil à l'aide du champ de saisie.",
                    style={'fontSize': '16px', 'color': '#555', 'textAlign': 'left', 'marginBottom': '20px'}
                ),
            ]),
        
            # Sélection du quantile pour le Mean Excess Plot
            html.Div([
                html.Label("Quantile pour l'affichage du Mean Excess Plot :", style={'fontSize': '18px'}),
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
            html.Div(id='var-pot-box', style={'marginTop': '20px'}),
        ])
    elif tab_name == 'tab3':
        return dbc.Container([
            html.H1("Analyse de la VaR avec GARCH", className="text-center"),
            html.P(
                "Dans cet onglet, nous allons explorer la Value-at-Risk (VaR) avec un modèle GARCH. "
                "Le modèle GARCH (Generalized Autoregressive Conditional Heteroskedasticity) est une "
                "extension du modèle ARCH qui permet de modéliser la volatilité conditionnelle des rendements.",
                className="text-center"
            ),
    
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Label("Période d'entraînement", className="fw-bold mb-2"),
                    dcc.DatePickerRange(
                        id='train-range',
                        start_date=default_train_start,
                        end_date=default_train_end,
                        display_format='YYYY-MM-DD'
                    )
                ], width=4),
                dbc.Col([
                    html.Label("Période de test", className="fw-bold mb-2"),
                    dcc.DatePickerRange(
                        id='test-range',
                        start_date=default_test_start,
                        end_date=default_test_end,
                        display_format='YYYY-MM-DD'
                    )
                ], width=4)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='returns-plot'),
                    dcc.Graph(id='var-plot'),
                    html.H4("Dynamique de μ_t et σ_t", className="text-center"),
                    dcc.Graph(id='mu-plot'),
                    dcc.Graph(id='sigma-plot')
                ], width=8),

                dbc.Col([
                    html.H4("Tests de diagnostic"),
                    dbc.Table(id='diagnostic-table', bordered=True, hover=True, responsive=True, striped=True,
                            style={'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'boxShadow': '2px 2px 10px #aaa'}),
                    html.Br(),
                    html.H4("Paramètres estimés"),
                    dbc.Table(id='param-table', bordered=True, hover=True, responsive=True, striped=True,
                            style={'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'boxShadow': '2px 2px 10px #aaa'})
                ], width=4)
            ])
        ])


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
    var_bootstrap, lower, upper = VaR_Hist_Bootstrap(df_train['log_returns'], confidence_level / 100, B=10000, alpha_IC=alpha_ic / 100)
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
    var_historical_content = html.Div([
        html.H3(f"VaR Historique ({confidence_level}%)"),
        html.P(f"{var_hist*100:.4f}%"),
        html.P(f"Avec un niveau de confiance de {confidence_level}%, une VaR historique de {var_hist*100:.4f}% signifie qu'il y a une probabilité de {100 - confidence_level}% que la perte journalière dépasse cette valeur.")
    ])
    
    var_bootstrap_content = html.Div([
        html.H3(f"VaR Bootstrap ({confidence_level}%)"),
        html.P(f"{var_bootstrap*100:.4f}%"),
        html.P(f"IC {alpha_ic}% : [{lower*100:.4f}%, {upper*100:.4f}%]")
    ])
    
    var_gaussian_content = html.Div([
        html.H3(f"VaR Gaussienne ({confidence_level}%)"),
        html.P(f"{var_gauss*100:.4f}%")
    ])
    
    var_student_content = html.Div([
        html.H3(f"VaR Student ({confidence_level}%)"),
        html.P(f"{var_student*100:.4f}%")
    ])
    
    es_gaussian_content = html.Div([
        html.H3(f"ES Gaussien ({confidence_level}%)"),
        html.P(f"{es_emp_gauss*100:.4f}%"),
        html.P(f"L'Expected Shortfall (ES) Gaussien de {es_emp_gauss*100:.4f}% signifie que, en cas de dépassement de la VaR, la perte moyenne attendue est de {es_emp_gauss*100:.4f}%.")
    ], style={'backgroundColor': '#f8f9fa'})
    
    es_historical_content = html.Div([
        html.H3(f"ES Historique ({confidence_level}%)"),
        html.P(f"{es_emp_hist*100:.4f}%"),
        html.P(f"L'Expected Shortfall (ES) Historique de {es_emp_hist*100:.4f}% signifie que, en cas de dépassement de la VaR, la perte moyenne attendue est de {es_emp_hist*100:.4f}%.")
    ], style={'backgroundColor': '#f1f3f4'})
    
    es_student_content = html.Div([
        html.H3(f"ES Student ({confidence_level}%)"),
        html.P(f"{es_emp_student*100:.4f}%"),
        html.P(f"L'Expected Shortfall (ES) Student de {es_emp_student*100:.4f}% signifie que, en cas de dépassement de la VaR, la perte moyenne attendue est de {es_emp_student*100:.4f}%.")
    ], style={'backgroundColor': '#f8f9fa'})
    
    es_theo_gaussian_content = html.Div([
        html.H3(f"ES Gaussien Théorique ({confidence_level}%)"),
        html.P(f"{es_theo_gauss*100:.4f}%")
    ], style={'backgroundColor': '#f1f3f4'})
    
    es_theo_student_content = html.Div([
        html.H3(f"ES Skew-Student Théorique ({confidence_level}%)"),
        html.P(f"{es_theo_student*100:.4f}%")
    ], style={'backgroundColor': '#f8f9fa'})
    
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
    
    # Mapper les fréquences à des libellés explicites
    freq_labels = {
        'M': 'Mensuelle',
        'Q': 'Trimestrielle',
        'Y': 'Annuelle'
    }
    freq_label = freq_labels.get(freq, freq)  # Utiliser la fréquence telle quelle si non trouvée
    
    # Mise en forme des résultats pour l'onglet 2
    var_tve_content = html.Div([
        html.P(
            f"Pour une fréquence {freq_label.lower()} à un niveau de confiance de {confidence_level_tve}%, la VaR avec la méthode Block Maxima est de : ",
            style={'fontSize': '16px', 'color': '#555', 'textAlign': 'left', 'marginTop': '20px'}
        ),
        html.P(
            f"**{var_tve*100:.4f}%**",
            style={'fontSize': '16px', 'color': '#333', 'fontWeight': 'bold', 'textDecoration': 'underline','textAlign': 'center'}
        )
    ])
    
    return fig_maxima, var_tve_content

@app.callback(
    [Output('mean-excess-plot', 'figure'),
     Output('var-pot-box', 'children')],
    [Input('ticker-dropdown-tve', 'value'),
     Input('train-date-picker-tve', 'start_date'),
     Input('train-date-picker-tve', 'end_date'),
     Input('test-date-picker-tve', 'start_date'),
     Input('test-date-picker-tve', 'end_date'),
     Input('quantile-slider', 'value'),
     Input('threshold-input', 'value'),
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
    
    var_pot_content = html.Div([
        html.P(
            f"Pour un seuil de {threshold} et un niveau de confiance de {confidence_level_tve}%, la VaR avec la méthode PoT est de : ",
            style={'fontSize': '16px', 'color': '#555', 'textAlign': 'left', 'marginTop': '20px'}
        ),
        html.P(
            f"**{var_pot*100:.4f}%**",
            style={'fontSize': '16px', 'color': '#333', 'fontWeight': 'bold', 'textDecoration': 'underline','textAlign': 'center'}
        )
    ])
    
    return fig_mean_excess, var_pot_content




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
     Input('test-range', 'end_date')]
)
def update_graphs(train_start, train_end, test_start, test_end):
    # Filtrage des données en fonction des dates sélectionnées

    # Télécharger les données du CAC 40
    data = yf.download("^FCHI")
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna()

    # Dates par défaut
    default_train_start = '2008-10-15'
    default_train_end = '2022-07-26'
    default_test_start = '2022-07-27'
    default_test_end = '2024-06-11'
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

    var = var_garch
    var_series = pd.Series(var, index=data_test.index)
    params = {"Méthode": "GARCH"} | garch_model.params

    # Graphique des rendements
    returns_fig = go.Figure()
    returns_fig.add_trace(go.Scatter(x=data_train.index, y=data_train, mode='lines', name="Train"))
    returns_fig.add_trace(go.Scatter(x=data_test.index, y=data_test, mode='lines', name="Test", line=dict(dash='dash')))
    returns_fig.update_layout(title="Série des rendements", xaxis_title="Date", yaxis_title="Rendement")

    # Graphique de la VaR
    var_fig = go.Figure()
    var_fig.add_trace(go.Scatter(x=data_test.index, y=data_test, mode='lines', name="Rendements"))
    var_fig.add_trace(go.Scatter(x=var_series.index, y=var_series, mode='lines', line=dict(color='red')))
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



if __name__ == '__main__':
    app.run_server(debug=True)