import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from scipy.optimize import minimize
from src.core.base import AbstractPostprocessor, AbstractPytorchCalibrator, ModelOutput
from src.core.utils import set_global_seed
from src.core.timeseries_evaluation import TimeSeriesForecast
from src.data.transformer import DataTransformer
import torch
from typing import Tuple, Dict, Union, Optional, Literal, Any
import logging
from pathlib import Path
import pandas as pd
from torch import nn

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

        if self.transformer is not None:
            raise ValueError("Only transformer=None is supported in the current implementation.")
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

            # if not result.success:
            #     logging.warning("success=false for forecast horizon=%s, item=%s.", h, data.item_id)
            #     logging.warning(result.message)
            #     logging.info(f"Init params: {init_params}")
            #     logging.info(f"found params: {result.x}")

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


class NormalMLECalibrator(AbstractPytorchCalibrator):
    """
    Horizon-wise MLE calibrator using:
      mu = a[h] + b[h] * M
      sigma = c[h] + d[h] * IQR

    Applies calibration per horizon and outputs adjusted quantile predictions.
    """

    def __init__(self, H: int, init_vals: Optional[Tuple[float, float, float, float]] = None, dtype=torch.float32):
        super().__init__()
        if init_vals is None:
            init_vals = (0.0, 1.0, 0.0, 1.0)
        a, b, c, d = init_vals

        self.a = nn.Parameter(torch.full((H,), a, dtype=dtype))
        self.b = nn.Parameter(torch.full((H,), b, dtype=dtype))
        self.c = nn.Parameter(torch.full((H,), c, dtype=dtype))
        self.d = nn.Parameter(torch.full((H,), d, dtype=dtype))
        self.dtype = dtype
        self.loc = None
        self.scale = None

    def forward(
        self,
        x: torch.Tensor,  # (T, H, Q)
        quantiles: torch.Tensor,  # (Q,)
        target: Optional[torch.Tensor] = None,  # (T, H)
        mask: Optional[torch.Tensor] = None,  # (T, H)
    ) -> "ModelOutput":

        if target is not None:
            self.scale = torch.std(target, dim=0, keepdim=True)  # (1, H)
            self.loc = torch.mean(target, dim=0, keepdim=True)  # (1, H)
            target = (target - self.loc) / self.scale

        x = (x - self.loc.unsqueeze(-1)) / self.scale.unsqueeze(-1)

        # TODO: make this more robust
        idx_05 = 4
        idx_09 = 8
        idx_01 = 0

        # Extract median and IQR
        M = x[:, :, idx_05]  # (T, H)
        IQR = x[:, :, idx_09] - x[:, :, idx_01]  # (T, H)

        mu = self.a.unsqueeze(0) + self.b.unsqueeze(0) * M
        sigma = torch.clamp(self.c.unsqueeze(0) + self.d.unsqueeze(0) * IQR, min=1e-4)

        # Adjust quantile predictions
        mu_exp = mu.unsqueeze(-1)  # (T, H, 1)
        sigma_exp = sigma.unsqueeze(-1)  # (T, H, 1)
        q = quantiles.view(1, 1, -1)  # (1, 1, Q)

        dist = torch.distributions.Normal(mu_exp, sigma_exp)
        quantile_preds = dist.icdf(q)  # (T, H, Q)

        loss = mle_nll_loss(target, mu, sigma, mask) if target is not None else None

        quantile_preds = quantile_preds * self.scale.unsqueeze(-1) + self.loc.unsqueeze(-1)

        return ModelOutput(loss=loss, quantile_preds=quantile_preds)


def mle_nll_loss(
    target: torch.Tensor,  # (T, H)
    mu: torch.Tensor,  # (T, H)
    sigma: torch.Tensor,  # (T, H), positive std
    mask: Optional[torch.Tensor] = None,  # (T, H) bool
) -> torch.Tensor:
    """
    Computes the negative log-likelihood loss for Normal(mu, sigma).

    Parameters:
        target: actual values
        mu: predicted means
        sigma: predicted stddevs
        mask: optional boolean mask

    Returns:
        scalar loss
    """
    if mask is None:
        mask = ~torch.isnan(target)
    target = torch.where(mask, target, torch.zeros_like(target))  # (T, H)

    dist = torch.distributions.Normal(loc=mu, scale=sigma)
    log_prob = dist.log_prob(target)  # (T, H)

    return -log_prob[mask].mean()


class PostprocessorFastMLE(AbstractPostprocessor):
    """
    Vectorized MLE using PyTorch.

    This version calibrates all quantiles and horizons jointly by training a
    batched linear model with the quantile (pinball) loss using PyTorch. It is
    designed for efficient calibration on large forecast matrices.

    Parameters
    ----------
    output_dir : pathlib.Path, optional
        Directory for saving outputs or artifacts, if any.
    name : str, optional
        Optional identifier for the post-processor.
    transformer : {'yeo-johnson', 'box-cox', 'log', 'arcsinh'}, optional
        Transformation applied to both targets and predictions before training.
        If None, no transformation is applied.
    device : {'mps', 'cuda', 'cpu'}, optional
        Device used for PyTorch training and inference. If None, selected automatically.
    n_jobs : int, default=1
        Number of parallel jobs during fitting.

    Returns
    -------
    dict
        A dictionary containing:
        - "model": BatchedQuantileCalibrator
          Trained PyTorch model with learned calibration parameters.
        - "invalid_h": torch.BoolTensor of shape (H,)
          Mask indicating which horizons had no valid training targets.
        - "transformer": DataTransformer
          The transformer instance used during fitting.

    Notes
    -----
    - Calibrates all (horizon, quantile) pairs jointly using a single model.
    - Falls back to original predictions for horizons without training data.
    - Does not enforce monotonicity between quantiles unless regularization is added.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        name: Optional[str] = None,
        transformer: Optional[Literal["yeo-johnson", "box-cox", "log", "arcsinh"]] = None,
        device: Optional[Literal["mps", "cuda", "cpu"]] = None,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(output_dir, name, n_jobs)
        self.transformer = transformer
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

    def _fit(self, data: TimeSeriesForecast):
        """
        Returns dict with trained torch model, transformer, and invalid horizon mask.
        """
        n_steps = 5000
        lr = 0.001

        lambda_noncross = 0.0
        verbose = False
        # --- 1) Build y_pred (T,H,Q) and y_true (T,H) exactly as you already do ---
        y_pred = np.stack([fc.predictions for fc in data.lead_time_forecasts.values()]).swapaxes(0, 1)
        T, H, Q = y_pred.shape

        y_true_series = data.data["target"].values
        y_true_series = np.roll(y_true_series, -1)
        y_true_series[-1] = np.nan
        pad = np.full(H - 1, np.nan)
        y_true_padded = np.concatenate([y_true_series, pad])
        y_true = np.lib.stride_tricks.sliding_window_view(y_true_padded, window_shape=H)  # (T, H)

        y_true = y_true[data.forecast_mask][self.ignore_first_n_train_entries :]
        y_pred = y_pred[self.ignore_first_n_train_entries :]

        # --- 2) Transform both predictors and targets once ---
        transformer = DataTransformer(self.transformer)
        transformer.fit(data.data)

        x_trans = transformer.transform(y_pred)  # (T,H,Q)
        y_true_trans = transformer.transform(y_true)  # (T,H)

        # --- 3) Torch tensors ---
        x_t = torch.from_numpy(x_trans)  # (T,H,Q)
        y_t = torch.from_numpy(y_true_trans)  # (T,H)
        q_t = torch.from_numpy(np.array(data.quantiles))  # (Q,)

        # --- 4) Train batched calibrator ---
        model = NormalMLECalibrator(H, init_vals=[0, 1, 0, 1], dtype=torch.float32).to(self.device)

        invalid_h = model.fit(
            y_true=y_t,
            x=x_t,
            quantiles=q_t,
            num_steps=n_steps,
            lr=lr,
            lambda_noncross=lambda_noncross,
            verbose=verbose,
            device=self.device,
        )

        # params_array = pack_params_for_old_postprocess(model, invalid_h)
        self.params = {"model": model, "invalid_h": invalid_h, "transformer": transformer}

        return self.params

    def _postprocess(self, data: TimeSeriesForecast, params: Any) -> TimeSeriesForecast:
        model: NormalMLECalibrator = params["model"].to(device=self.device)
        transformer: DataTransformer = params["transformer"]
        invalid_h: torch.Tensor = params["invalid_h"]

        # Load raw predictions and transform
        y_pred = np.stack([fc.predictions for fc in data.lead_time_forecasts.values()]).swapaxes(0, 1)
        x = transformer.transform(y_pred)  # (T, H, Q)

        quantiles = torch.tensor(data.quantiles, dtype=model.dtype, device=self.device)
        # Run calibrator
        x_t = torch.from_numpy(x).to(dtype=model.dtype, device=self.device)
        with torch.no_grad():
            adj_trans = model(x_t, quantiles).quantile_preds.cpu().numpy()  # (T, H, Q)

        # Inverse transform
        adj = transformer.inverse_transform(adj_trans)

        # Fallback for horizons with no training data
        adj = adj.copy()
        invalid_h_np = invalid_h.cpu().numpy()
        adj[:, invalid_h_np, :] = y_pred[:, invalid_h_np, :]

        # Write predictions back
        ts_fc = data.model_copy(deep=True)
        for h, fc in ts_fc.lead_time_forecasts.items():
            fc.predictions = torch.tensor(adj[:, h - 1, :])
        return ts_fc
