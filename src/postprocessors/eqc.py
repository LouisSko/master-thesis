import numpy as np
from src.core.base import AbstractPostprocessor
from src.core.timeseries_evaluation import TimeSeriesForecast, HorizonForecast, TARGET
import torch
from pathlib import Path
import logging
from typing import Dict, Optional

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
        # get the preds as (T, H, Q)
        y_pred = np.stack([fc.predictions for fc in data.lead_time_forecasts.values()]).swapaxes(0, 1)

        # get the label
        y_true = data.data["target"].values
        y_true = np.roll(y_true, -1)  # move everything one step left
        y_true[-1] = np.nan

        T, H, Q = y_pred.shape  # grab sizes
        pad = np.full(H - 1, np.nan)  # so late rows can be NaN-padded
        y_true_padded = np.concatenate([y_true, pad])
        y_true = np.lib.stride_tricks.sliding_window_view(y_true_padded, window_shape=H)

        y_true = y_true[data.forecast_mask][-self.ignore_first_n_train_entries :]

        residuals = y_true[:, :, None] - y_pred  # T, H, Q

        # describes the offset for each quantile and forecast
        offset_matrix = np.empty((H, Q), dtype=residuals.dtype)  # (H, Q)

        for i, q in enumerate(data.quantiles):
            offset_matrix[:, i] = np.nanquantile(residuals[:, :, i], q=q, axis=0)

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


class PostprocessorEQC_old(AbstractPostprocessor):
    """
    EmpiricalQuantileCalibrator.

    Slow implementation.

    This postprocessor adjusts quantile regression outputs by computing empirical offsets to improve quantile coverage.
    """

    def __init__(self, output_dir: Optional[Path] = None, name: Optional[str] = None, n_jobs: int = 1) -> None:
        super().__init__(output_dir, name, n_jobs)

    def _fit(self, data: TimeSeriesForecast) -> Dict[int, Dict[float, float]]:
        """
        Calibrates predicted quantiles by computing empirical offsets for each lead time.

        This method estimates how much each predicted quantile should be shifted so that
        the resulting quantile forecasts achieve the correct empirical coverage on the
        calibration set. For each lead time and quantile, it calculates the empirical
        error between the predicted quantile and the true target and stores the
        corresponding offset.

        Parameters
        ----------
        data : TimeSeriesForecast
            Forecast data for a single item, including predicted quantiles and targets,
            used for calibration.

        Returns
        -------
        Dict[int, Dict[float, float]]
            A nested dictionary of empirical quantile offsets structured as:
            {
                lead_time_1: {
                    quantile_1: offset,
                    quantile_2: offset,
                    ...
                },
                ...
            }
            where each offset can be used to shift the corresponding quantile prediction.
        """
        conf_thresholds = {}
        for lead_time in data.get_lead_times():
            conf_thresholds[lead_time] = {}
            # TODO: could be made more efficient by accessing the predictions directly
            df = data.to_dataframe(lead_time).iloc[self.ignore_first_n_train_entries :].dropna().copy()

            if len(df) == 0:
                logging.info("No calibration data available for item_id: %s, lead time: %s.", data.item_id, lead_time)
                for q in data.quantiles:
                    conf_thresholds[lead_time][q] = None
                continue

            for q in data.quantiles:
                scores = df[TARGET] - df[q]
                conf_thresholds[lead_time][q] = np.quantile(scores, q=q)

        return conf_thresholds

    def _postprocess(self, data: TimeSeriesForecast, params: Dict[int, Dict[float, float]]) -> TimeSeriesForecast:
        """
        Applies the empirical quantile offsets to adjust predictions.

        Parameters
        -----------
        data : TimeSeriesForecast
            Prediction data to be postprocessed, containing raw quantile predictions.

        params : Dict[int, Dict[float, float]]
            A nested dictionary of empirical quantile offsets

        Returns
        --------
        TimeSeriesForecast
            A new `TimeSeriesForecast` object with calibrated quantile predictions.
        """
        results_lt = {}
        for lead_time in data.get_lead_times():
            df = data.to_dataframe(lead_time)  # TODO: could be made more efficient by accessing the predictions directly
            adjusted_predictions = []
            for quantile in data.quantiles:
                offset = params[lead_time][quantile]
                if offset is None:
                    logging.info("No params available for item: %s, lead time: %s, quantile: %s. Keeping original predictions.", data.item_id, lead_time, quantile)
                    conformalized_predictions = np.array(df[quantile])
                else:
                    conformalized_predictions = np.array(df[quantile] + offset)
                adjusted_predictions.append(conformalized_predictions)
            adjusted_predictions = np.column_stack(adjusted_predictions)

            results_lt[lead_time] = HorizonForecast(lead_time=lead_time, predictions=torch.tensor(adjusted_predictions))

        return TimeSeriesForecast(
            item_id=data.item_id,
            lead_time_forecasts=results_lt,
            data=data.data,
            freq=data.freq,
            quantiles=data.quantiles,
            forecast_mask=data.forecast_mask,
        )
