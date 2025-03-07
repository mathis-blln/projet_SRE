# Importations
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from TP_SRE import *

# Initialisation de l'application Dash
app = dash.Dash(__name__)

# Liste des tickers
tickers = ['^FCHI', 'AAPL', 'GOOG', 'MSFT', 'AMZN']

# Mise en page de l'application
app.layout = html.Div(style={'fontFamily': 'Arial', 'padding': '20px'}, children=[ 
    html.H1("Projet de Statistiques des Risques Extrêmes", style={'textAlign': 'center', 'color': '#333'}),

    # Ajout des noms
    html.Div([
        html.H4("Mathis Bouillon, Cheryl Kouadio", style={'textAlign': 'center', 'color': '#555'}),
    ], style={'marginBottom': '30px'}),


    # Sélection du ticker
    html.Div([ 
        html.Label("Sélectionnez un Ticker :", style={'fontSize': '18px'}), 
        dcc.Dropdown( 
            id='ticker-dropdown', 
            options=[{'label': ticker, 'value': ticker} for ticker in tickers], 
            value='^FCHI', 
            style={'width': '50%'} 
        ), 
    ], style={'marginBottom': '20px'}), 

    # Sélection des dates pour train et test
    html.Div([ 
        html.Label("Sélectionnez les dates de train et test :", style={'fontSize': '18px'}), 
        html.Div([ 
            html.Label("Ensemble d'apprentissage (train) :"), 
            dcc.DatePickerRange( 
                id='train-date-picker', 
                start_date='2008-10-15', 
                end_date='2022-07-26', 
                display_format='YYYY-MM-DD', 
                style={'width': '45%', 'marginRight': '5%'} 
            ), 
            html.Label("Ensemble de test :"), 
            dcc.DatePickerRange( 
                id='test-date-picker', 
                start_date='2022-07-27', 
                end_date='2024-06-11', 
                display_format='YYYY-MM-DD', 
                style={'width': '45%'} 
            ), 
        ], style={'marginBottom': '20px'}), 
    ]), 

    # Graphique des prix de clôture et log-rendements avec 2 axes Y
    dcc.Graph(id='price-graph', style={'borderRadius': '10px', 'boxShadow': '2px 2px 10px #aaa'}), 

  
    # Section VaR avec explication
    html.Div(style={'marginTop': '40px'}, children=[
        html.H3("Value at Risk (VaR)", style={'textAlign': 'center', 'color': '#333'}),
        html.P(
            "La Value at Risk (VaR) est une mesure statistique utilisée pour évaluer le risque de perte d'un actif ou d'un portefeuille. "
            "Elle permet d'estimer la perte maximale attendue sur une période donnée, avec un certain niveau de confiance.",
            style={'fontSize': '16px', 'color': '#555', 'textAlign': 'center', 'marginBottom': '20px'}
        ),
        html.P(
            "Mathématiquement, la VaR est définie comme le quantile inférieur des rendements d'un actif ou d'un portefeuille. "
            "Autrement dit, c'est la valeur de perte qui ne sera pas dépassée avec une probabilité donnée (par exemple 99%).",
            style={'fontSize': '16px', 'color': '#555', 'textAlign': 'center', 'marginBottom': '20px'}
        ),
        html.P(
            "La VaR est utilisée pour évaluer la perte potentielle dans un scénario défavorable et pour déterminer les besoins en capital de couverture pour une entreprise ou un investisseur.",
            style={'fontSize': '16px', 'color': '#555', 'textAlign': 'center', 'marginBottom': '20px'}
        ),
    ]),

    # Sélection du niveau de confiance (α)
    html.Div([ 
        html.Label("Niveau de confiance :", style={'fontSize': '18px'}), 
        dcc.Slider( 
            id='confidence-level-slider', 
            min=90, 
            max=99, 
            step=1, 
            marks={i: f"{i}%" for i in range(90, 100)}, 
            value=99 
        ), 
    ], style={'marginTop': '20px'}), 

    # Résultats des calculs de VaR sous forme de cartes
    html.Div(style={'display': 'flex', 'justifyContent': 'center', 'gap': '30px', 'marginTop': '20px'}, children=[ 
        # Carte VaR Historique
        html.Div(id='var-historical-box', style={ 
            'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa', 
            'textAlign': 'center', 'backgroundColor': '#f8f9fa', 'width': '30%' 
        }), 

        # Carte VaR Bootstrap
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
    ]), 

    # Nouvelle ligne pour VaR Gaussienne
    html.Div(style={'display': 'flex', 'justifyContent': 'center', 'gap': '30px', 'marginTop': '20px'}, children=[ 
        # Carte VaR Gaussienne
        html.Div(id='var-gaussian-box', style={ 
            'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa', 
            'textAlign': 'center', 'backgroundColor': '#e3f2fd', 'width': '30%' 
        }), 

        # Nouvelle carte pour VaR Student
        html.Div(id='var-student-box', style={ 
            'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa', 
            'textAlign': 'center', 'backgroundColor': '#f7c2c2', 'width': '30%' 
        }), 
    ]),

    # Section Expected Shortfall
    html.Div(style={'marginTop': '40px'}, children=[
        html.H3("Calcul des Expected Shortfall (ES)", style={'textAlign': 'center', 'color': '#333'}),
        html.P(
            "L'Expected Shortfall (ES), ou perte attendue, est la moyenne des pertes qui dépassent la VaR. "
            "Contrairement à la VaR qui donne un seuil de perte pour un certain niveau de confiance, "
            "l'ES donne l'ampleur moyenne de la perte au-delà de ce seuil.",
            style={'fontSize': '16px', 'color': '#555', 'textAlign': 'center', 'marginBottom': '20px'}
        ),
        html.H3("Calcul des ES empiriques", style={'textAlign': 'left', 'color': '#333'}),
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
        html.H3("Calcul des Expected Shortfall (ES) Théoriques", 
                style={'textAlign': 'left', 'color': '#333'}),
    
        html.P("L'ES théorique est défini comme :", 
                style={'fontSize': '16px', 'color': '#555', 'textAlign': 'center'}),

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
                'textAlign': 'center', 'backgroundColor': '#ffeb3b', 'width': '30%'}),
        
            html.Div(id='es-theo-student-box', style={
                'padding': '20px', 'borderRadius': '15px', 'boxShadow': '2px 2px 10px #aaa', 
                'textAlign': 'center', 'backgroundColor': '#f44336', 'width': '30%'})
        ]),
    ]),
])

# Callback pour mettre à jour les valeurs et le graphique
@app.callback( 
    [Output('price-graph', 'figure'), 
     Output('var-historical-box', 'children'), 
     Output('var-bootstrap-box', 'children'), 
     Output('var-gaussian-box', 'children'), 
     Output('var-student-box', 'children'),
     Output('es-gaussian-box', 'children'),
     Output('es-historical-box', 'children'),
     Output('es-student-box', 'children'),
     Output('es-theo-gaussian-box','children'),
     Output('es-theo-student-box','children')], 
    [Input('ticker-dropdown', 'value'), 
     Input('confidence-level-slider', 'value'), 
     Input('alpha-ic-slider', 'value'), 
     Input('train-date-picker', 'start_date'), 
     Input('train-date-picker', 'end_date'), 
     Input('test-date-picker', 'start_date'), 
     Input('test-date-picker', 'end_date')] 
) 


def update_graphs(ticker, confidence_level, alpha_ic, train_start_date, train_end_date, test_start_date, test_end_date): 
    # Télécharger les données
    df = telecharger_donnees(ticker, "2000-01-01")

    # Diviser les données en train/test
    df_train, df_test = split_train_test(df, train_start_date, train_end_date, test_start_date, test_end_date)

    # Graphique à deux axes : log-rendements et prix de clôture
    price_fig = plot_dual_axis_graph(df, ticker, train_start_date, train_end_date, test_start_date, test_end_date)

    # Calcul de la VaR historique
    var_hist = VaR_Hist(df_train['log_returns'], confidence_level / 100)

    # Calcul de la VaR bootstrap avec IC ajustable
    var_bootstrap, lower, upper = VaR_Hist_Bootstrap(df_train['log_returns'], confidence_level / 100, B=1000, alpha_IC=alpha_ic / 100)

    # Calcul de la VaR Gaussienne
    var_gauss = VaR_Gauss(df_train['log_returns'], confidence_level / 100)

    # Calcul de la VaR Student
    var_student = calculate_var_student(df_train['log_returns'], confidence_level / 100)

    # Calcul des ES
    es_emp_gauss = ES_empirique(df_train['log_returns'], var_gauss)
    es_emp_hist = ES_empirique(df_train['log_returns'], var_hist)
    es_emp_student = ES_empirique(df_train['log_returns'], var_student)

    # Estimation des paramètres Skew-Student pour l'ES théorique
    mu_hat, sigma_hat, gamma_hat, nu_hat = estimate_skew_student_params(df_train['log_returns'].dropna())

    def pdf_gauss(x):
        return st.norm.pdf(x, loc=np.mean(df_train['log_returns']), scale=np.std(df_train['log_returns']))

    def pdf_skew_student(x):
        return skew_student_pdf(x, mu_hat, sigma_hat, gamma_hat, nu_hat)

    # Calcul des ES théoriques
    es_theo_gauss = ES_theorique(confidence_level /100, var_gauss, pdf_gauss)
    es_theo_student = ES_theorique(confidence_level/100, var_student, pdf_skew_student)

    # Mise en page des résultats sous forme de cartes pour la VaR
    var_historical_content = [
        html.H3(f"VaR Historique ({confidence_level}%)", style={'color': '#333'}),
        html.P(f"{var_hist:.4f}", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#dc3545'})
    ]

    var_bootstrap_content = [
        html.H3(f"VaR Bootstrap ({confidence_level}%)", style={'color': '#333'}),
        html.P(f"{var_bootstrap:.4f}", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#007bff'}),
        html.P(f"IC {alpha_ic}% : [{lower:.4f}, {upper:.4f}]", style={'fontSize': '16px', 'color': '#555'})
    ]

    var_gaussian_content = [
        html.H3(f"VaR Gaussienne ({confidence_level}%)", style={'color': '#333'}),
        html.P(f"{var_gauss:.4f}", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#28a745'})
    ]

    var_student_content = [
        html.H3(f"VaR Student ({confidence_level}%)", style={'color': '#333'}),
        html.P(f"{var_student:.4f}", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#f44336'})
    ]

    # Mise en page des résultats des ES sous forme de cartes
    es_gaussian_content = [
        html.H3(f"ES Gaussien ({confidence_level}%)", style={'color': '#333'}),
        html.P(f"{es_emp_gauss:.4f}", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#007bff'})
    ]

    es_historical_content = [
        html.H3(f"ES Historique ({confidence_level}%)", style={'color': '#333'}),
        html.P(f"{es_emp_hist:.4f}", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#007bff'})
    ]

    es_student_content = [
        html.H3(f"ES Student ({confidence_level}%)", style={'color': '#333'}),
        html.P(f"{es_emp_student:.4f}", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#007bff'})
    ]

    es_gaussian_theo_content = [
        html.H3(f"ES Gaussien Théorique ({confidence_level}%)", style={'color': '#333'}),
        html.P(f"{es_theo_gauss:.4f}", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#dc3545'})
    ]

    es_student_theo_content = [
        html.H3(f"ES Skew-Student Théorique ({confidence_level}%)", style = {'color' : '#333'}),
        html.P(f"{es_theo_student:.4f}", style = {'fontSize':'24px', 'fontWeight': 'bold', 'color':'#dc3545'})
    ]

    return price_fig, var_historical_content, var_bootstrap_content, \
        var_gaussian_content, var_student_content, es_gaussian_content, es_historical_content, es_student_content, \
        es_gaussian_theo_content, es_student_theo_content



# Lancer l'application Dash
if __name__ == '__main__':
    app.run_server(debug=True)
