import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from scipy.optimize import minimize
from src.core.base import AbstractPostprocessor
from src.core.utils import set_global_seed
from src.core.timeseries_evaluation import TimeSeriesForecast
from src.data.transformer import DataTransformer
import torch
from typing import Tuple, Dict, Union, Optional, Literal
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")
set_global_seed()


class PostprocessorMLE(AbstractPostprocessor):
    """
    Postprocessor that adjusts quantile regression outputs using Maximum Likelihood Estimation (MLE).

    Learns a simple parametric relationship between predicted medians (M) and interquartile ranges (IQR)
    to true target values by fitting a normal distribution to the log transformed predictions.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        name: Optional[str] = None,
        transformer: Optional[Literal["yeo-johnson", "box-cox", "log", "arcsinh"]] = None,
        epsilon: Optional[float] = 1e-8,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(output_dir, name, n_jobs)
        self.transformer = transformer
        self.epsilon = epsilon  # relevant for log

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

        y_pred, y_true = data.get_aligned_predictions_and_targets()
        y_pred = y_pred[self.ignore_first_n_train_entries :]
        y_true = y_true[self.ignore_first_n_train_entries :]
        T, H, Q = y_pred.shape
        params_array = np.full((H, 4), np.nan)

        # 2.  Prepare transformer and containers
        transformer = DataTransformer(self.transformer, self.epsilon)
        y_true_series = transformer.fit_transform(data.data["target"].values)
        nan_mask = ~np.isnan(y_true_series)
        mean = np.mean(y_true_series[nan_mask])
        std = np.std(y_true_series[nan_mask])
        y_true = (y_true - mean) / std
        y_pred = (y_pred - mean) / std

        q_idx = {q: i for i, q in enumerate(data.quantiles)}
        init_params = (0, 1, 0, 1)
        # Step 3: Get M and IQR → shape (T, H)
        M = y_pred[:, :, q_idx[0.5]]  # (T, H)
        IQR = y_pred[:, :, q_idx[0.9]] - y_pred[:, :, q_idx[0.1]]  # (T, H)

        for h in range(H):  # h = 0..H‑1, corresponds to lead_time = h+1

            M_h = M[:, h]
            IQR_h = IQR[:, h]
            y_h = y_true[:, h]
            valid = (~np.isnan(M_h)) & (~np.isnan(IQR_h)) & (~np.isnan(y_h))

            if valid.sum() == 0:
                logging.info("No calibration data available for item_id: %s, lead time: %s.", data.item_id, h + 1)
                params[h + 1] = None
                continue

            M_h, IQR_h, y_h = M_h[valid], IQR_h[valid], y_h[valid]
            result = minimize(self._neg_log_likelihood, args=(M_h, IQR_h, y_h), x0=init_params, method="Nelder-Mead")

            params_array[h] = result.x
            params[h + 1] = result.x

            if not result.success:
                logging.warning("success=false for forecast horizon=%s, item=%s.", h, data.item_id)
                logging.warning(result.message)
                logging.info(f"Init params: {init_params}")
                logging.info(f"found params: {result.x}")

        params["params"] = params_array
        params["transformer"] = transformer
        params["mean"] = mean
        params["std"] = std
        return params

    def _postprocess(self, data: TimeSeriesForecast, params: Dict) -> TimeSeriesForecast:
        """
        Vectorized MLE-based calibration to quantile predictions for each lead time.
        """
        transformer: DataTransformer = params["transformer"]
        mean = params["mean"]
        std = params["std"]
        params_array = params["params"]  # (H, 4)

        q_idx = {q: i for i, q in enumerate(data.quantiles)}
        quantiles = np.array(data.quantiles)
        T = len(next(iter(data.lead_time_forecasts.values())).predictions)

        # Step 1: Collect predictions → shape (T, H, Q)
        y_pred = np.stack([fc.predictions for fc in data.lead_time_forecasts.values()]).swapaxes(0, 1)

        # Step 2: Apply transform and standardization
        y_pred = transformer.transform(y_pred)
        y_pred = (y_pred - mean) / std  # shape (T, H, Q)

        # Step 3: Get M and IQR → shape (T, H)
        M = y_pred[:, :, q_idx[0.5]]  # (T, H)
        IQR = y_pred[:, :, q_idx[0.9]] - y_pred[:, :, q_idx[0.1]]  # (T, H)

        # Step 4: Get (a, b, c, d) per horizon
        a = params_array[:, 0]  # (H,)
        b = params_array[:, 1]
        c = params_array[:, 2]
        d = params_array[:, 3]

        # Step 5: Compute mu and sigma → shape (T, H)
        mu = a[None, :] + b[None, :] * M
        sigma = c[None, :] + d[None, :] * IQR

        # Step 6: Compute adjusted quantiles using norm.ppf
        # Shape: (Q, T, H) → then transpose to (T, H, Q)
        q_probs = quantiles[:, None, None]  # (Q, 1, 1)
        mu_exp = mu[None, :, :]  # (1, T, H)
        sigma_exp = sigma[None, :, :]  # (1, T, H)

        log_preds = stats.norm.ppf(q_probs, loc=mu_exp, scale=sigma_exp)  # (Q, T, H)
        log_preds = log_preds.transpose(1, 2, 0)  # (T, H, Q)

        # Step 7: Inverse standardization + inverse transform
        log_preds = log_preds * std + mean
        adj_preds = transformer.inverse_transform(log_preds)

        # Step 8: Where MLE failed (NaNs in a/b/c/d), use original predictions
        failed_mask = np.isnan(params_array[:, 0])  # (H,)
        if np.any(failed_mask):
            original_preds = np.stack([fc.predictions for fc in data.lead_time_forecasts.values()]).swapaxes(0, 1)
            adj_preds[:, failed_mask, :] = original_preds[:, failed_mask, :]

        # Step 9: Write back
        ts_fc = data.model_copy(deep=True)
        for h, fc in ts_fc.lead_time_forecasts.items():
            fc.predictions = torch.tensor(adj_preds[:, h - 1, :])  # h-1 because lead_time=1-based

        return ts_fc

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

        if c <= 0 or d <= 0:
            return np.inf

        nll = -np.sum(stats.norm.logpdf(y, loc=mu, scale=sigma))

        return nll

    def _estimate_init_params(self, m: np.ndarray, iqr: np.ndarray, y_mu: np.ndarray, y_sigma: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Estimates initial parameters [a, b, c, d] using linear regression for mean and std.
        Clips values and adds fallback defaults for robustness.

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
        try:
            x_mu = sm.add_constant(m, has_constant="add")
            model_mu = sm.OLS(y_mu, x_mu).fit()
            a_init, b_init = model_mu.params
        except Exception:
            logging.warning("OLS fit for mu failed, using fallback.")
            a_init, b_init = 0.0, 1.0

        # std = c + d * IQR
        try:
            x_sigma = sm.add_constant(iqr, has_constant="add")
            model_sigma = sm.OLS(y_sigma, x_sigma).fit()
            c_init, d_init = model_sigma.params

            # Enforce positive std estimates
            c_init = max(c_init, 1e-4)
            d_init = max(d_init, 1e-4)
        except Exception:
            logging.warning("OLS fit for sigma failed, using fallback.")
            c_init, d_init = 1e-2, 1.0

        return a_init, b_init, c_init, d_init
