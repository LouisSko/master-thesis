import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from scipy.optimize import minimize
from src.core.base import AbstractPostprocessor
from src.core.timeseries import PredictionLeadTimes, PredictionLeadTime
import torch
from tqdm import tqdm
from typing import Tuple


class PostprocessorMLE(AbstractPostprocessor):
    def __init__(self) -> None:
        super().__init__()
        self.params = {}  # Store {lead_time: {item_id: (a, b, c, d)}}
        self.epsilon = 100_000

    def fit(self, data: PredictionLeadTimes) -> None:
        self.params = {}
        ignore_first_n = 500

        lead_times = data.results.keys()

        for lt in tqdm(lead_times):
            self.params[lt] = {}
            for item_id in data.results[lt].data.item_ids:
                df = data.results[lt].to_dataframe(item_ids=[item_id]).iloc[ignore_first_n:].dropna()
                quantiles = data.results[lt].quantiles

                log_df = np.log(df[quantiles + ["target"]] + self.epsilon)
                log_df["std_target"] = log_df["target"].rolling(20, min_periods=20, center=True).std()
                log_df = log_df.dropna()

                M = log_df[0.5].values
                IQR = (log_df[0.9] - log_df[0.1]).values
                y_mu = log_df["target"].values
                y_sigma = log_df["std_target"].values

                init_params = self._estimate_init_params(M, IQR, y_mu, y_sigma)
                result = minimize(self._neg_log_likelihood, args=(M, IQR, y_mu), x0=init_params, method="Nelder-Mead")

                if not result.success:
                    raise ValueError(f"MLE failed for lt={lt}, item={item_id}")

                self.params[lt][item_id] = result.x

    def postprocess(self, data: PredictionLeadTimes) -> PredictionLeadTimes:

        results = {}
        lead_times = data.results.keys()

        for lt in tqdm(lead_times):
            quantiles = data.results[lt].quantiles
            freq = data.results[lt].freq
            all_preds = []

            for item_id in data.results[lt].data.item_ids:
                df = data.results[lt].to_dataframe(item_ids=[item_id])

                log_df = np.log(df[quantiles] + self.epsilon)

                M = log_df[0.5].values
                IQR = (log_df[0.9] - log_df[0.1]).values
                a, b, c, d = self.params[lt][item_id]

                mu = a + b * M
                sigma = c + d * IQR
                log_predictions = stats.norm.ppf(np.array(quantiles).reshape(-1, 1), loc=mu, scale=sigma).T
                all_preds.append(np.exp(log_predictions) - self.epsilon)

            preds_np = np.vstack(all_preds)
            results[lt] = PredictionLeadTime(lead_time=lt, predictions=torch.tensor(preds_np), quantiles=quantiles, freq=freq, data=data.results[lt].data)

        return PredictionLeadTimes(results=results)

    def _neg_log_likelihood(self, params: list, M: np.ndarray, IQR: np.ndarray, y: np.ndarray):
        """Computes the negative log-likelihood for a normal distribution, parameterized
        by median (M) and interquartile range (IQR), used for maximum likelihood estimation.

        Parameters
        ----------
        params : list
            List of parameters [a, b, c, d] where:
            - mu = a + b * M
            - sigma = c + d * IQR
        M : np.ndarray
            Median predictions.
        IQR : np.ndarray
            Interquartile ranges of predictions (e.g., 0.9 quantile - 0.1 quantile).
        y : np.ndarray
            Observed target values.

        Returns
        -------
        float
            Negative log-likelihood value.
        """

        a, b, c, d = params
        mu = a + b * M
        sigma = c + d * IQR

        # penalize negative standard deviation
        if any(sigma <= 0):
            return np.inf

        nll = -np.sum(stats.norm.logpdf(y, loc=mu, scale=sigma))

        return nll

    def _estimate_init_params(self, m: np.ndarray, iqr: np.ndarray, y_mu: np.ndarray, y_sigma: np.ndarray) -> Tuple[float, float, float, float]:
        """Estimates initial parameters [a, b, c, d] using linear regression for mean and std.

        Parameters
        ----------
        m : np.ndarray
            Median values.
        iqr : np.ndarray
            Interquartile ranges.
        y_mu : np.ndarray
            Observed means (log target).
        y_sigma : np.ndarray
            Observed standard deviations of log target.

        Returns
        -------
        Tuple[float, float, float, float]
            Initial parameter estimates [a, b, c, d] for mu and sigma formulas.
        """

        # mean = a + b * Median
        x_mu = sm.add_constant(m)
        model_mu = sm.OLS(y_mu, x_mu).fit()
        a_init, b_init = model_mu.params

        # std = c + d * IQR
        x_sigma = sm.add_constant(iqr)
        model_sigma = sm.OLS(y_sigma, x_sigma).fit()
        c_init, d_init = model_sigma.params

        return a_init, b_init, c_init, d_init
