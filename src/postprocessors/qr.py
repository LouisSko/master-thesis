import numpy as np
import statsmodels.api as sm
import torch
from src.core.base import AbstractPostprocessor, AbstractPytorchCalibrator, ModelOutput
from src.core.timeseries_evaluation import TimeSeriesForecast
from src.data.transformer import DataTransformer
from src.core.utils import set_global_seed
from pathlib import Path
import logging
from typing import Any, Optional, Literal
import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning, ConvergenceWarning
import torch
from typing import Optional
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")
set_global_seed()


class PostprocessorQR(AbstractPostprocessor):
    """
    Quantile regression post-processor using statsmodels.

    This post-processor calibrates predicted quantiles independently for each
    horizon and quantile using linear quantile regression. The calibration is
    performed in a transformed space (e.g., log or Yeo-Johnson), and the results
    are transformed back to the original space after adjustment.

    Parameters
    ----------
    output_dir : pathlib.Path, optional
        Directory for saving outputs or artifacts, if any.
    name : str, optional
        Optional identifier for the post-processor. Defaults to the class name.
    transformer : {'yeo-johnson', 'box-cox', 'log', 'arcsinh'}, optional
        Transformation applied to both targets and predictions before regression.
        If None, no transformation is applied.
    n_jobs : int, default=1
        Number of parallel jobs during fitting.

    Returns
    -------
    dict
        A dictionary containing:
        - "params": np.ndarray of shape (H, Q, 2), regression coefficients
          [intercept, slope] for each horizon and quantile.
        - "transformer": DataTransformer
          The transformer instance used during fitting.

    Notes
    -----
    - Each quantile and horizon pair is calibrated independently.
    - Forecasts for horizons with insufficient data fall back to the original predictions.
    - Quantile crossing is possible since each quantile is fit separately.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        name: Optional[str] = None,
        transformer: Optional[Literal["yeo-johnson", "box-cox", "log", "arcsinh"]] = None,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(output_dir, name, n_jobs)
        self.transformer = transformer

    def _fit(self, data: TimeSeriesForecast) -> Any:

        warnings.simplefilter("ignore", IterationLimitWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)

        # 1.  Pre‑compute matrices shared by all horizons
        y_pred = np.stack([fc.predictions for fc in data.lead_time_forecasts.values()]).swapaxes(0, 1)
        T, H, Q = y_pred.shape
        params_array = np.full((H, Q, 2), np.nan)

        # targets aligned for all horizons: (T, H)
        y_true_series = data.data["target"].values
        y_true_series = np.roll(y_true_series, -1)
        y_true_series[-1] = np.nan  # last obs has no 1‑step‑ahead truth

        pad = np.full(H - 1, np.nan)
        y_true_padded = np.concatenate([y_true_series, pad])
        y_true = np.lib.stride_tricks.sliding_window_view(y_true_padded, window_shape=H)  # (T, H)

        # apply global forecast mask + burn‑in once
        y_true = y_true[data.forecast_mask][self.ignore_first_n_train_entries :]
        y_pred = y_pred[self.ignore_first_n_train_entries :]

        # 2.  Prepare transformer and containers
        transformer = DataTransformer(self.transformer)
        transformer.fit(data.data)

        for h in range(H):  # h = 0..H‑1, corresponds to lead_time = h+1
            # column‑select once per horizon
            y_target_h = y_true[:, h]
            y_preds_h = y_pred[:, h, :]  # (T_valid, Q)

            # rows that are still NaN after masking (can only happen for
            # extremely short series) are dropped here
            valid = ~np.isnan(y_target_h)
            if valid.sum() == 0:
                params_array[h,] = np.full(len(data.quantiles), np.nan)
                continue

            y_train = transformer.transform(y_target_h[valid])

            for i, q in enumerate(data.quantiles):
                x_raw = transformer.transform(y_preds_h[valid, i])
                x_train = sm.add_constant(x_raw.reshape(-1, 1), has_constant="add")
                model = sm.QuantReg(y_train, x_train)
                params_array[h, i] = model.fit(q=q).params

        return {
            "params": params_array,
            "transformer": transformer,
        }

    def _postprocess(self, data: TimeSeriesForecast, params: Any) -> TimeSeriesForecast:

        transformer: DataTransformer = params["transformer"]
        params_array: np.ndarray = params["params"]  # (H, Q, 2)
        y_pred = np.stack([fc.predictions for fc in data.lead_time_forecasts.values()]).swapaxes(0, 1)
        x = transformer.transform(y_pred)  # (T, H, Q)

        # Add constant
        x_full = np.ones((x.shape[0], x.shape[1], x.shape[2], 2))  # (T, H, Q, 2)
        x_full[..., 1] = x  # constant term in position 0

        # Compute adjusted predictions using einsum
        adj = np.einsum("thqk,hqk->thq", x_full, params_array)

        # Inverse transform and mask NaNs
        adj = transformer.inverse_transform(adj)
        adj = np.where(np.isnan(params_array[..., 0])[np.newaxis, :, :], y_pred, adj)

        # Write back to copy
        ts_fc = data.model_copy(deep=True)
        for h, fc in ts_fc.lead_time_forecasts.items():
            fc.predictions = torch.tensor(adj[:, h - 1, :])
        return ts_fc


class LinearQRCalibrator(AbstractPytorchCalibrator):
    """
    Linear calibrator: ŷ = a[h,q] + b[h,q] * x[t,h,q]
    - a: (H, Q)
    - b: (H, Q)
    Forward expects x of shape (T, H, Q).
    """

    def __init__(self, H: int, Q: int, init_a: float = 0.0, init_b: float = 1.0, dtype=torch.float32):
        super().__init__()
        self.a = nn.Parameter(torch.full((H, Q), init_a, dtype=dtype))
        self.b = nn.Parameter(torch.full((H, Q), init_b, dtype=dtype))

    def forward(
        self,
        x: torch.Tensor,
        quantiles: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        x_transformed = torch.arcsinh(x)  # safe for all real values
        y = self.a.unsqueeze(0) + self.b.unsqueeze(0) * x_transformed
        x_adj = torch.sinh(y)

        # x_adj = self.a.unsqueeze(0) + self.b.unsqueeze(0) * x
        loss = smoothed_pinball_loss(target, x_adj, quantiles, mask) if target is not None else None

        return ModelOutput(loss=loss, quantile_preds=x_adj)


def smoothed_pinball_loss(
    y_true: torch.Tensor,  # (T, H)
    y_pred: torch.Tensor,  # (T, H, Q)
    quantiles: torch.Tensor,  # (Q,)
    mask: torch.Tensor = None,  # (T, H) boolean, True where valid
    kappa: float = 1e-2,  # smoothing radius; smaller -> closer to pinball
) -> torch.Tensor:
    """
    Quantile Huber (smoothed pinball) loss.

    References:
      - Dabney et al., "Implicit Quantile Networks for Distributional Reinforcement Learning", ICML 2018.
        (Quantile Huber loss; pinball recovered as kappa -> 0)

    Shapes:
      y_true: (T, H)
      y_pred: (T, H, Q)
      quantiles: (Q,)
      mask: (T, H) boolean (optional)

    Returns:
      scalar loss averaged over valid (T, H) and all Q.
    """
    # ---- shape checks to catch silent broadcasting bugs ----
    if y_pred.dim() != 3:
        raise ValueError(f"y_pred must be (T,H,Q), got {y_pred.shape}")
    if y_true.dim() != 2:
        raise ValueError(f"y_true must be (T,H), got {y_true.shape}")
    if quantiles.dim() != 1:
        raise ValueError(f"quantiles must be (Q,), got {quantiles.shape}")
    T, H, Q = y_pred.shape
    if y_true.shape != (T, H):
        raise ValueError(f"y_true shape {y_true.shape} must match (T,H)=({T},{H}) from y_pred")
    if quantiles.shape[0] != Q:
        raise ValueError(f"quantiles length {quantiles.shape[0]} must match Q={Q}")

    # default mask: all valid
    if mask is None:
        mask = torch.ones((T, H), dtype=torch.bool, device=y_true.device)
    else:
        if mask.shape != (T, H):
            raise ValueError(f"mask shape {mask.shape} must be (T,H)=({T},{H})")

    # residuals and broadcast
    # e = y_true - y_pred
    e = y_true.unsqueeze(-1) - y_pred  # (T, H, Q)
    q = quantiles.view(1, 1, -1)  # (1, 1, Q)

    # Asymmetry weights |tau - 1(e < 0)|
    # = tau when e >= 0, and (1 - tau) when e < 0
    w = torch.where(e < 0, 1.0 - q, q)  # (T, H, Q)

    # Huber on residual (symmetric), smooths the kink near 0
    abs_e = e.abs()
    if kappa <= 0:
        # fall back to unsmoothed pinball: w * |e|
        huber = abs_e
    else:
        huber = torch.where(abs_e <= kappa, 0.5 * (e**2) / kappa, abs_e - 0.5 * kappa)  # quadratic region  # linear tails

    loss = w * huber  # (T, H, Q)

    # Apply mask across all quantiles
    loss = loss * mask.unsqueeze(-1)  # (T, H, Q)

    # Normalize by number of valid (T,H) positions * Q
    denom = mask.sum() * Q
    # Avoid divide-by-zero if everything is masked
    denom = torch.clamp(denom, min=1.0)
    return loss.sum() / denom


def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, quantiles: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    y_true: (T, H)
    y_pred: (T, H, Q)
    quantiles: (Q,) in (0,1)
    mask: (T, H) boolean -> True where y_true is valid
    """
    # Broadcast to (T, H, Q)
    y_true_3d = y_true.unsqueeze(-1)
    q = quantiles.view(1, 1, -1)

    e = y_true_3d - y_pred  # residuals
    loss_per = torch.maximum(q * e, (q - 1.0) * e)  # pinball
    return loss_per[mask].mean()


class PostprocessorFastQR(AbstractPostprocessor):
    """
    Vectorized quantile regression post-processor using PyTorch.

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
        model = LinearQRCalibrator(H, Q, 0, 1, dtype=torch.float32).to(self.device)

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
        model: LinearQRCalibrator = params["model"].to(device=self.device)
        transformer: DataTransformer = params["transformer"]
        invalid_h: torch.Tensor = params["invalid_h"]

        # Load raw predictions and transform
        y_pred = np.stack([fc.predictions for fc in data.lead_time_forecasts.values()]).swapaxes(0, 1)
        x = transformer.transform(y_pred)  # (T, H, Q)

        # Run calibrator
        x_t = torch.from_numpy(x).to(dtype=torch.float32, device=next(model.parameters()).device)
        with torch.no_grad():
            adj_trans = model(x_t).quantile_preds.cpu().numpy()  # (T, H, Q)

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
