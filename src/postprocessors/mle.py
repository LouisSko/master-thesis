import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from scipy.optimize import minimize
from src.core.base import AbstractPostprocessor
from src.core.timeseries_evaluation import TimeSeriesForecast, HorizonForecast
from src.data.transformer import DataTransformer
import torch
from typing import Tuple, Dict, Union, Optional, Literal
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")


class PostprocessorMLE(AbstractPostprocessor):
    """
    Postprocessor that adjusts quantile regression outputs using Maximum Likelihood Estimation (MLE).

    Learns a simple parametric relationship between predicted medians (M) and interquartile ranges (IQR)
    to true target values by fitting a normal distribution to the log transformed predictions.
    """

    def __init__(self, output_dir: Optional[Path] = None, name: Optional[str] = None, transformer: Optional[Literal["yeo-johnson", "box-cox", "log", "arcsinh"]] = None) -> None:
        super().__init__(output_dir, name)
        self.transformer = transformer

    def extract_m_iqr(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract median and inter quartile range from df"""
        M = df[0.5].values
        IQR = (df[0.9] - df[0.1]).values

        return M, IQR

    def _fit(self, data: TimeSeriesForecast) -> Dict[int, Union[Tuple[float, float, float, float], None]]:
        """
        Fits the MLE parameters to the provided prediction data.

        Parameters
        ----------
        data : TimeSeriesForecast
            The quantile predictions for a time series.

        Returns
        -------
        Dict[int, Union[Tuple[float, float, float, float], None]]
            A dict of the fitted parameters {lead_time: (a, b, c, d)}. None if MLE failed.
        """
        params = {}
        transformer = DataTransformer(self.transformer)
        transformer.fit(data.data["target"])  # or use a separate transformer for each lead time/item id
        for lead_time in data.get_lead_times():
            df = data.to_dataframe(lead_time).iloc[self.ignore_first_n_train_entries :].copy().dropna()
            df["target"] = transformer.transform(df[["target"]])
            df[data.quantiles] = transformer.transform(df[data.quantiles])
            df["std_target"] = df["target"].rolling(20, min_periods=20, center=True).std()
            df = df.dropna()

            if len(df) == 0:
                logging.info("No calibration data available for item_id: %s, lead time: %s.", data.item_id, lead_time)
                params[lead_time] = None
                continue

            M, IQR = self.extract_m_iqr(df)
            y_mu = df["target"].values
            y_sigma = df["std_target"].values

            init_params = self._estimate_init_params(M, IQR, y_mu, y_sigma)
            result = minimize(self._neg_log_likelihood, args=(M, IQR, y_mu), x0=init_params, method="Nelder-Mead")

            if result.success:
                params[lead_time] = result.x
            else:
                logging.warning("MLE failed for forecast horizon=%s, item=%s. Predictions won't get postprocessed for this one.", lead_time, data.item_id)
                params[lead_time] = None

        params["transformer"] = transformer

        return params

    def _postprocess(self, data: TimeSeriesForecast, params: Dict[int, Union[Tuple[float, float, float, float], None]]) -> TimeSeriesForecast:
        """
        Applies MLE-based calibration to quantile predictions for each lead time.

        For each lead time, this method uses the fitted MLE parameters to adjust the
        predicted quantiles. If no parameters are available for a lead time, the original
        predictions are retained.

        Parameters
        ----------
        data : TimeSeriesForecast
            The forecast data containing quantile predictions for a single time series item.

        params : Dict[int, Union[Tuple[float, float, float, float], None]]
            A dictionary mapping each lead time to a tuple of MLE parameters (a, b, c, d),
            where:
                - mu = a + b * M
                - sigma = c + d * IQR
            If the parameters are None for a lead time, the predictions are left unchanged.

        Returns
        -------
        TimeSeriesForecast
            The forecast object with postprocessed quantile predictions, adjusted using
            the MLE calibration parameters.
        """
        results_lt = {}

        transformer: DataTransformer = params["transformer"]

        for lead_time in data.get_lead_times():
            params_lt = params[lead_time]
            if params_lt is None:
                logging.info("No params available for item: %s, lead time: %s. Keeping original predictions.", data.item_id, lead_time)
                predictions = df[data.quantiles].to_numpy()
                continue

            df = data.to_dataframe(lead_time).copy()
            df = transformer.transform(df[data.quantiles])
            M, IQR = self.extract_m_iqr(df)
            a, b, c, d = params_lt
            mu = a + b * M
            sigma = c + d * IQR
            log_predictions = stats.norm.ppf(np.array(data.quantiles).reshape(-1, 1), loc=mu, scale=sigma).T
            predictions = transformer.inverse_transform(log_predictions)

            results_lt[lead_time] = HorizonForecast(lead_time=lead_time, predictions=torch.tensor(predictions))

        return TimeSeriesForecast(item_id=data.item_id, lead_time_forecasts=results_lt, data=data.data, freq=data.freq, quantiles=data.quantiles)

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
