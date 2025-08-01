import numpy as np
from src.core.base import AbstractPostprocessor
from src.core.timeseries_evaluation import TimeSeriesForecast
import torch
from pathlib import Path
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")


class PostprocessorEQC(AbstractPostprocessor):
    """
    Empirical quantile calibrator for time series forecasts.

    This postprocessor corrects forecast quantiles by estimating and
    applying empirical offsets between true and predicted values.
    For each forecast horizon *h* and quantile level *q*, the offset is:

        offset(h, q) = Q_q[ y_true(t) − y_pred_q(t, h) ]  over *t*

    where *Q_q[⋅]* is the empirical *q*-th quantile.

    The computation leverages vectorized operations to estimate the
    entire `(H × Q)` offset matrix in one call.

    Parameters
    ----------
    ignore_first_n_train_entries : int, optional (default=0)
        Number of initial training entries to skip when constructing
        the residual sample.
    output_dir : Path or None, optional
        Directory to which postprocessor outputs are saved.
        Passed through to :class:`AbstractPostprocessor`.
    name : str or None, optional
        Name of this postprocessor instance.
        Passed through to :class:`AbstractPostprocessor`.
    n_jobs : int, optional (default=1)
        Number of jobs for parallelism when called from a higher-level
        loop. The class itself is single-threaded.

    Examples
    --------
    >>> pp = EmpiricalQuantileCalibrator(ignore_first_n_train_entries=24)
    >>> pp.fit(train_forecast)            # calibrate offsets
    >>> adjusted = pp.postprocess(test)   # apply to new data
    """

    def __init__(
        self,
        *,
        ignore_first_n_train_entries: int = 0,
        output_dir: Optional[Path] = None,
        name: Optional[str] = None,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(output_dir=output_dir, name=name, n_jobs=n_jobs)
        self.ignore_first_n_train_entries = ignore_first_n_train_entries

    def _fit(self, data: TimeSeriesForecast) -> np.ndarray:
        """
        Fit the calibrator by computing empirical quantile offsets.

        This method computes residuals between the ground truth series
        and forecasted quantiles for each horizon. It then estimates the
        `(H × Q)` offset matrix by taking the empirical quantile of
        residuals across time.

        Parameters
        ----------
        data : TimeSeriesForecast
            Training forecast data containing predictions and the true
            target values.

        Returns
        -------
        np.ndarray
            A 2D array of shape (H, Q), where `H` is the number of
            forecast horizons and `Q` is the number of quantile levels.
            Each entry `offset[h, q]` represents the empirical offset
            for quantile `q` at horizon `h`.
        """
        y_pred, y_true = data.get_aligned_predictions_and_targets()
        y_pred = y_pred[self.ignore_first_n_train_entries :]
        y_true = y_true[self.ignore_first_n_train_entries :]

        T, H, Q = y_pred.shape  # grab sizes
        residuals = y_true[:, :, None] - y_pred  # T, H, Q

        # describes the offset for each quantile and forecast
        offset_matrix = np.empty((H, Q), dtype=residuals.dtype)  # (H, Q)

        for i, q in enumerate(data.quantiles):
            offset_matrix[:, i] = np.nanquantile(residuals[:, :, i], q=q, axis=0)

        # fallback. replace nans with 0 -> no change
        offset_matrix = np.nan_to_num(offset_matrix, nan=0)

        return offset_matrix

    def _postprocess(
        self,
        data: TimeSeriesForecast,
        params: np.ndarray,
    ) -> TimeSeriesForecast:
        """
        Apply previously learned offsets to forecast predictions.

        Parameters
        ----------
        data : TimeSeriesForecast
            The forecast data whose predictions will be adjusted.
        params : np.ndarray
            A 2D array of shape (H, Q) containing the previously
            computed empirical offsets for each horizon and quantile.

        Returns
        -------
        TimeSeriesForecast
            A deep copy of the input forecast with adjusted predictions.
        """
        # 1. Retrieve predictions tensor.
        y_pred = np.stack([fc.predictions for fc in data.lead_time_forecasts.values()]).swapaxes(0, 1)

        # 3. Broadcast addition – NumPy handles (H, Q) vs (T, H, Q).
        y_adj = torch.tensor(y_pred + params)  # (T, H, Q)

        # 4. Wrap adjusted array back into container.
        ts_fc = data.model_copy(deep=True)
        for h, fc in ts_fc.lead_time_forecasts.items():
            fc.predictions = y_adj[:, h - 1, :]

        return ts_fc
