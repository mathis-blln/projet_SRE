import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.stats import jarque_bera, norm, genpareto, genextreme as gev, t
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import integrate
import numpy as np
from scipy.stats import t
from scipy.optimize import minimize, minimize_scalar
from scipy import integrate
from statsmodels.tsa.arima.model import ARIMA


class VaR_traditionnelle :
    def __init__(self, data_train,data_test, alpha):
        self.data_train = data_train
        self.data_test = data_test
        self.alpha = alpha

    def historical_var(self):
        """ Compute historical VaR """
        return np.percentile(self.data_train, 100 * (1 - self.alpha))

    def gaussian_var(self):
        """ Compute Gaussian VaR """
        mu = np.mean(self.data_train)
        sigma = np.std(self.data_train)
        return (mu + sigma * norm.ppf(1 - self.alpha))

class SkewStudentVaR:
    def __init__(self, data):
        """
        Initialize the class with a dataset of standardized residuals.
        param data: Series of standardized residuals (or returns).
        """
        self.data = data
        self.params_sstd = None  # Store estimated Skew Student parameters

    @staticmethod
    def skew_student_pdf(x, mu, sigma, gamma, nu):
        """
        Compute the Skew Student-t probability density function (PDF).
        """
        t_x = ((x - mu) * gamma / sigma) * np.sqrt((nu + 1) / (nu + ((x - mu) / sigma) ** 2))
        pdf_t = t.pdf(x, df=nu, loc=mu, scale=sigma)
        cdf_t = t.cdf(t_x, df=nu + 1)
        return 2 * pdf_t * cdf_t

    def log_likelihood(self, params):
        """
        Compute the negative log-likelihood for the Skew Student distribution.
        """
        mu, sigma, gamma, nu = params
        density = self.skew_student_pdf(self.data, mu, sigma, gamma, nu)
        return -np.sum(np.log(density))

    def fit(self):
        """
        Estimate the parameters of the Skew Student distribution using MLE.
        """
        x0 = [np.mean(self.data), np.std(self.data), 1, 4]  # Initial guess
        bounds = [(None, None), (0, None), (None, None), (None, None)]  # Constraints

        res = minimize(self.log_likelihood, x0, bounds=bounds)

        if res.success:
            self.params_sstd = {
                "mu": res.x[0],
                "sigma": res.x[1],
                "gamma": res.x[2],
                "nu": res.x[3]
            }
            print("=" * 80)
            print("Les paramètres estimés de la loi de Skew Student sont :")
            print("-" * 15)
            print(f"Mu : {self.params_sstd['mu']:.6f}")
            print(f"Sigma : {self.params_sstd['sigma']:.6f}")
            print(f"Gamma : {self.params_sstd['gamma']:.6f}")
            print(f"Nu : {self.params_sstd['nu']:.6f}")
            print("=" * 80)
        else:
            raise ValueError("L'optimisation de la log-vraisemblance a échoué.")

    def integrale_SkewStudent(self, x):
        """
        Compute the integral of the Skew Student PDF from -∞ to x.
        """
        if self.params_sstd is None:
            raise ValueError("Veuillez d'abord ajuster la distribution avec `fit()`.")

        borne_inf = -np.inf
        resultat_integration, _ = integrate.quad(
            lambda x: self.skew_student_pdf(x, **self.params_sstd), borne_inf, x
        )
        return resultat_integration

    def fonc_minimize(self, x, alpha):
        """
        Function to minimize in order to find the quantile of the Skew Student distribution.
        """
        return abs(self.integrale_SkewStudent(x) - alpha)

    def skew_student_quantile(self, alpha):
        """
        Compute the quantile of the Skew Student distribution.
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("Veuillez entrer un niveau alpha entre 0 et 1.")

        if self.params_sstd is None:
            raise ValueError("Veuillez d'abord ajuster la distribution avec `fit()`.")

        resultat_minimisation = minimize_scalar(lambda x: self.fonc_minimize(x, alpha))
        return resultat_minimisation.x

    def sstd_var(self, alpha=0.99):
        """
        Compute the Skew Student-t Value at Risk (VaR).
        :param alpha: Confidence level (e.g., 0.99 for 99% confidence).
        :return: Estimated Value at Risk.
        """
        return self.skew_student_quantile(1 - alpha)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox

class GARCHModel:
    def __init__(self, data_train, data_test):
        self.data_train = data_train
        self.data_test = data_test
        self.r = pd.concat([data_train, data_test], axis=0)
        self.AR1 = None
        self.model = None
        self.result = None
        self.cond_vol = None
        self.resid = None
        self.std_resid = None
        self.mu_t = None
        self.sigma2 = None
        self.params = None

    def fit_garch(self):
        """ Fit a GARCH(1,1) model with AR(1) mean process """
        
        self.AR1 = ARIMA(self.data_train, order=(1, 0, 0))
        AR1_resid = self.AR1.fit().resid
        self.model = arch_model(AR1_resid, vol='Garch', p=1, q=1, mean='zero', dist='normal')
        self.result = self.model.fit(disp="off")

        # Extract conditional volatility and residuals
        self.cond_vol = self.result.conditional_volatility
        self.resid = self.result.resid
        self.std_resid = self.result.std_resid  # Standardized residuals

        # Store estimated parameters

        mu = self.AR1.fit().params[0]
        phi = self.AR1.fit().params[1]
        omega = self.result.params[0]
        a = self.result.params[1]
        b = self.result.params[2]

        self.params = {
            "mu": mu,
            "phi": phi,
            "omega": omega,
            "alpha": a,
            "beta": b
        }

    def diagnostic_tests(self):
        """ Perform diagnostic tests on residuals """
        # Jarque-Bera test for normality
        jb_test = jarque_bera(self.std_resid)

        # Ljung-Box test for autocorrelation
        lb_test_resid = acorr_ljungbox(self.std_resid, lags=[i for i in range(1, 13)], return_df=True)
        lb_test_resid_sq = acorr_ljungbox(self.std_resid**2, lags=[i for i in range(1, 13)], return_df=True)

        return lb_test_resid, lb_test_resid_sq, jb_test

    def dynamique_var(self):
        """ Compute the dynamic variance using estimated GARCH parameters """
        T_train = len(self.data_train)
        T_test = len(self.data_test)
        T = T_train + T_test

        # Extract GARCH parameters
        mu = self.params["mu"]
        omega = self.params["omega"]
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        phi = self.params["phi"]

        # Initialize arrays
        self.mu_t = np.zeros(T)    
        self.sigma2 = np.zeros(T)  

        # Initial conditions
        self.mu_t[0] = mu
        self.sigma2[0] = omega / (1 - alpha - beta)  # Long-run variance

        # Compute variance and mean recursively
        for t in range(1, T):
            self.mu_t[t] = mu + phi * self.r.iloc[t-1]  # Mean equation
            self.sigma2[t] = omega + alpha * (self.r.iloc[t-1] - self.mu_t[t-1])**2 + beta * self.sigma2[t-1]
        
        return self.mu_t, self.sigma2


    def garch_var(self, var):
        """Compute GARCH-based Value-at-Risk (VaR)"""
        T_test = len(self.data_test)
        T_train = len(self.data_train)
        var_t = np.zeros(T_test)
        nb_exp = 0

        var_t = np.zeros(T_test)    # Composante moyenne
        nb_exp = 0
        for t in range(T_test):
            var_t[t] = (self.mu_t[t+T_train] + np.sqrt(self.sigma2[t+T_train])*var)
            nb_exp += (self.r[t+T_train] < var_t[t]).astype(int)
            
        var_t = pd.Series(var_t, index=self.data_test.index)

        return var_t, nb_exp