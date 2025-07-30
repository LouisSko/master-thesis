import numpy as np
import statsmodels.api as sm
import torch
from src.core.base import AbstractPostprocessor
from src.core.timeseries_evaluation import TimeSeriesForecast
from src.data.transformer import DataTransformer
from src.core.utils import set_global_seed
from pathlib import Path
import logging
from typing import Any, Optional, Literal, Tuple
import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning, ConvergenceWarning
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


class BatchedQuantileCalibrator(nn.Module):
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
        # self.lambda_ = nn.Parameter(torch.full((1,), 1.0, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_transformed = torch.arcsinh(x * self.lambda_)  # safe for all real values
        # y = self.a.unsqueeze(0) + self.b.unsqueeze(0) * x_transformed
        # return torch.sinh(y) / self.lambda_

        return self.a.unsqueeze(0) + self.b.unsqueeze(0) * x


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


@torch.no_grad()
def horizons_with_no_data(mask: torch.Tensor) -> torch.Tensor:
    """Return boolean mask (H,) True where a horizon has no valid targets."""
    # mask: (T, H)
    return mask.sum(dim=0) == 0


def train_calibrator(
    y_true: torch.Tensor,  # (T,H), may contain NaNs
    x: torch.Tensor,  # (T,H,Q) predictors (typically transformed y_pred)
    quantiles: torch.Tensor,  # (Q,)
    num_steps: int = 1000,
    lr: float = 0.05,
    weight_decay: float = 0.0,  # L2 on parameters (optional)
    lambda_noncross: float = 0.0,  # penalty to discourage quantile crossings
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Tuple[BatchedQuantileCalibrator, torch.Tensor]:
    """
    Trains a batched linear calibrator on all horizons/quantiles at once.
    Returns:
      model, invalid_horizons_mask (H,)
    """
    assert x.dim() == 3 and y_true.dim() == 2
    T, H, Q = x.shape
    assert y_true.shape == (T, H)
    assert quantiles.shape == (Q,)

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    dtype = torch.float32
    # Use float32 for stability (matches statsmodels); convert and move to device
    y_true = y_true.to(device=device, dtype=dtype)
    x = x.to(device=device, dtype=dtype)
    quantiles = quantiles.to(device=device, dtype=dtype)

    # Build mask from NaNs in y_true (True=valid)
    mask = ~torch.isnan(y_true)
    # Replace NaNs in y_true with zeros to avoid propagating NaNs; they are masked out anyway
    y_true = torch.where(mask, y_true, torch.zeros_like(y_true))

    model = BatchedQuantileCalibrator(H, Q, 0, 1, dtype=dtype).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    patience, bad = 10, 0  # simple early stopping
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for step in range(num_steps):
        opt.zero_grad()
        y_hat = model(x)

        loss = pinball_loss(y_true, y_hat, quantiles, mask)

        # Optional non-crossing penalty: enforce ŷ[..., q] <= ŷ[..., q+1]
        if lambda_noncross > 0:
            diff = y_hat[..., 1:] - y_hat[..., :-1]  # (T,H,Q-1)
            viol = torch.relu(-diff)  # only negative diffs
            loss = loss + lambda_noncross * (viol.pow(2).mean())

        loss.backward()
        opt.step()

        if verbose and (step % 50 == 0 or step == num_steps - 1):
            print(f"step {step:4d}  loss {loss.item():.6f}")

        # early stopping
        if loss.item() < best_loss - 1e-3:
            best_loss = loss.item()
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    # load best model
    model.load_state_dict(best_state)

    invalid_h = horizons_with_no_data(mask)  # (H,)
    return model, invalid_h


@torch.no_grad()
def apply_calibrator(
    model: BatchedQuantileCalibrator,
    x: torch.Tensor,  # (T,H,Q) same transform as during training
    y_pred_orig: torch.Tensor,  # (T,H,Q) original (untransformed) model outputs
    transformer: DataTransformer,  # your DataTransformer instance
    invalid_h: torch.Tensor,  # (H,) horizons with no training data
):
    """
    Returns adjusted predictions in original space, with fallback to y_pred_orig
    for horizons that had no training data.
    """
    device = next(model.parameters()).device
    x = x.to(device=device, dtype=torch.float32)

    # Forward in transformed space
    adj_trans = model(x)  # (T,H,Q)

    # Inverse transform back to original space (expects numpy -> convert)
    adj = transformer.inverse_transform(adj_trans.cpu().numpy())

    # Fallback for invalid horizons (use original predictions)
    adj = adj.copy()
    invalid_h_np = invalid_h.cpu().numpy()
    adj[:, invalid_h_np, :] = y_pred_orig[:, invalid_h_np, :]

    return adj  # numpy array (T,H,Q)


def pack_params_for_old_postprocess(model: BatchedQuantileCalibrator, invalid_h):
    """
    model: BatchedQuantileCalibrator with attributes a (H,Q), b (H,Q)
    invalid_h: torch.BoolTensor of shape (H,), True where no training data
    Returns: params_array (H, Q, 2) with [intercept, slope]
    """
    # pull to CPU numpy
    a = model.a.detach().cpu().numpy()  # (H, Q)
    b = model.b.detach().cpu().numpy()  # (H, Q)

    params_array = np.empty((a.shape[0], a.shape[1], 2), dtype=np.float32)
    params_array[..., 0] = a  # intercept in slot 0
    params_array[..., 1] = b  # slope in slot 1

    # mark horizons with no data as NaN so your postprocess falls back to y_pred
    invalid_h_np = invalid_h.detach().cpu().numpy()
    params_array[invalid_h_np, :, :] = np.nan
    return params_array


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

        y_true = y_true[data.forecast_mask]
        # if you use burn-in:
        y_true = y_true[self.ignore_first_n_train_entries :]
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
        model, invalid_h = train_calibrator(
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
        model: BatchedQuantileCalibrator = params["model"].to(device=self.device)
        transformer: DataTransformer = params["transformer"]
        invalid_h: torch.Tensor = params["invalid_h"]

        # Load raw predictions and transform
        y_pred = np.stack([fc.predictions for fc in data.lead_time_forecasts.values()]).swapaxes(0, 1)
        x = transformer.transform(y_pred)  # (T, H, Q)

        # Run calibrator
        x_t = torch.from_numpy(x).to(dtype=torch.float32, device=next(model.parameters()).device)
        with torch.no_grad():
            adj_trans = model(x_t).cpu().numpy()  # (T, H, Q)

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
