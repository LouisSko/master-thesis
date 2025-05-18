import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from scipy.optimize import minimize
from src.core.base import AbstractPostprocessor
from src.core.timeseries_evaluation import ForecastCollection, TimeSeriesForecast, HorizonForecast
import torch
from tqdm import tqdm
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")


class PostprocessorMLE(AbstractPostprocessor):
    def __init__(self) -> None:
        """
        Postprocessor that adjusts quantile regression outputs using Maximum Likelihood Estimation (MLE).

        Learns a simple parametric relationship between predicted medians (M) and interquartile ranges (IQR)
        to true target values by fitting a normal distribution to the log transformed predictions.
        """
        super().__init__()
        self.params = {}  # Store {lead_time: {item_id: (a, b, c, d)}}
        self.epsilon = 100_000

    def fit(self, data: ForecastCollection) -> None:
        """
        Fits the MLE parameters to the provided prediction data.

        Parameters
        ----------
        data : PredictionLeadTimes
            The prediction results for different lead times and item IDs,
            including quantile predictions and true target values.
        """
        self.params = {}

        for item_id in tqdm(data.get_item_ids(), desc="Fitting MLE Postprocessor for each time series (item)"):
            self.params[item_id] = {}
            item = data.get_time_series_forecast(item_id)
            for lead_time in item.get_lead_times():

                df = item.to_dataframe(lead_time).iloc[self.ignore_first_n_train_entries :].dropna()
                log_df = np.log(df[item.quantiles + ["target"]] + self.epsilon)
                log_df["std_target"] = log_df["target"].rolling(20, min_periods=20, center=True).std()
                log_df = log_df.dropna()

                M = log_df[0.5].values
                IQR = (log_df[0.9] - log_df[0.1]).values
                y_mu = log_df["target"].values
                y_sigma = log_df["std_target"].values

                init_params = self._estimate_init_params(M, IQR, y_mu, y_sigma)
                result = minimize(self._neg_log_likelihood, args=(M, IQR, y_mu), x0=init_params, method="Nelder-Mead")

                if result.success:
                    self.params[item_id][lead_time] = result.x
                else:
                    logging.warning("MLE failed for forecast horizon=%s, item=%s. Predictions won't get postprocessed for this one.", lead_time, item_id)
                    self.params[item_id][lead_time] = None

    def postprocess(self, data: ForecastCollection) -> ForecastCollection:
        """
        Postprocesses the prediction data using the fitted MLE parameters.

        Parameters
        ----------
        data : ForecastCollection
            The prediction results to postprocess using the estimated parameters.

        Returns
        -------
        ForecastCollection
            The postprocessed prediction results with adjusted quantiles based on the MLE calibration.
        """
        results_item_ids = {}

        for item_id in tqdm(data.get_item_ids(), desc="Updating Forecasts using MLE Postprocessor."):
            results_lt = {}
            item = data.get_time_series_forecast(item_id)
            for lead_time in item.get_lead_times():

                #### specific code #####
                df = item.to_dataframe(lead_time)
                log_df = np.log(df[item.quantiles] + self.epsilon)
                M = log_df[0.5].values
                IQR = (log_df[0.9] - log_df[0.1]).values
                params = self.params[item_id][lead_time]
                if params is None:
                    # keeping original predictions
                    logging.info("No params available for forecast horizon=%s, item=%s. Keeping original predictions.", lead_time, item_id)
                    predictions = df[item.quantiles].to_numpy()
                else:
                    a, b, c, d = params
                    mu = a + b * M
                    sigma = c + d * IQR
                    log_predictions = stats.norm.ppf(np.array(item.quantiles).reshape(-1, 1), loc=mu, scale=sigma).T
                    predictions = np.exp(log_predictions) - self.epsilon
                #### specific code #####

                results_lt[lead_time] = HorizonForecast(lead_time=lead_time, predictions=torch.tensor(predictions))

            results_item_ids[item_id] = TimeSeriesForecast(item_id=item_id, lead_time_forecasts=results_lt, data=item.data, freq=item.freq, quantiles=item.quantiles)

        return ForecastCollection(item_ids=results_item_ids)

    def _neg_log_likelihood(self, params: list, M: np.ndarray, IQR: np.ndarray, y: np.ndarray):
        """
        Computes the negative log-likelihood for a normal distribution, parameterized
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
        """
        Estimates initial parameters [a, b, c, d] using linear regression for mean and std.

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
        x_mu = sm.add_constant(m, has_constant="add")
        model_mu = sm.OLS(y_mu, x_mu).fit()
        a_init, b_init = model_mu.params

        # std = c + d * IQR
        x_sigma = sm.add_constant(iqr, has_constant="add")
        model_sigma = sm.OLS(y_sigma, x_sigma).fit()
        c_init, d_init = model_sigma.params

        return a_init, b_init, c_init, d_init
