from autogluon.timeseries import TimeSeriesDataFrame
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional, Union
import torch
from tqdm import tqdm
from src.core.timeseries_evaluation import PredictionLeadTimes, PredictionLeadTime
from src.core.base import AbstractPredictor
import logging
from pydantic import Field
from pathlib import Path
from pandas.tseries.frequencies import to_offset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")


class RollingSeasonalQuantilePredictor(AbstractPredictor):
    """
    Rolling Seasonal Quantile Predictor based on time-dependent bucketing.

    This predictor estimates future quantiles by grouping past target values into
    time-based "buckets" (e.g., same hour of day, same weekday, etc.), depending on
    the frequency of the time series. For each forecast timestamp, the most relevant
    historical bucket is identified, and empirical quantiles are computed from the
    most recent observations in that bucket.

    Parameters
    ----------
    quantiles : List[float], optional
        List of quantiles to predict (e.g., [0.1, 0.5, 0.9]).
    lead_times : List[int], optional
        List of lead times (in time steps) for which forecasts should be produced.
    freq : Union[str, pd.Timedelta, pd.DateOffset], optional
        Frequency of the time series data; can be a pandas-parsable string
        (e.g., "1h", "1D"), a Timedelta, or a DateOffset.
    last_n_samples : int, optional
        Number of most recent samples per bucket to use for quantile estimation.
    output_dir : Optional[Union[str, Path]], optional
        Directory to store model outputs or logs.
    """

    def __init__(
        self,
        quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        lead_times: List[int] = Field(default_factory=lambda: [1, 2, 3]),
        freq: Union[pd.Timedelta, pd.DateOffset] = pd.Timedelta("1h"),
        last_n_samples: Optional[int] = 10,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        # Normalize freq into a pandas DateOffset
        self.offset = to_offset(freq)
        super().__init__(lead_times=lead_times, freq=freq, output_dir=output_dir)
        self.quantiles = quantiles
        self.last_n_samples = last_n_samples

        # Prepare bucket keys and key‐making function based on freq
        self._setup_buckets()

    def _setup_buckets(self) -> None:
        """
        Set up bucket keys and the function to map timestamps to bucket keys.

        The bucketing scheme depends on the frequency of the time series:
        - Daily or business-day: by weekday.
        - Hourly: by weekday and hour.
        - Minute-level: by weekday, hour, and time slot.
        """
        fstr = self.offset.freqstr  # e.g. "1H","B","15T","D"
        code = fstr[-1]

        if code.upper() in ("D", "B"):
            # daily or business‐day: bucket by weekday only
            self._make_key = lambda ts: str(ts.weekday())
            self.bucket_keys = [str(d) for d in range(7)]

        elif code.upper() == "H":
            # hourly: bucket by weekday_hour
            self._make_key = lambda ts: f"{ts.weekday()}_{ts.hour}"
            self.bucket_keys = [f"{d}_{h}" for d in range(7) for h in range(24)]

        elif code.upper() == "T":
            # minute frequency, e.g. 15T, 5T, etc.
            n = self.offset.n  # number of minutes

            def make_minute_key(ts: pd.Timestamp) -> str:
                slot = ts.minute // n
                return f"{ts.weekday()}_{ts.hour}_{slot}"

            self._make_key = make_minute_key

            slots_per_hour = 60 // n
            self.bucket_keys = [f"{d}_{h}_{slot}" for d in range(7) for h in range(24) for slot in range(slots_per_hour)]

        else:
            raise ValueError(f"Unsupported frequency '{fstr}' for NNPredictor")

    def _initialize_history(self, item_ids: List[Any]) -> Dict[Any, Dict[str, np.ndarray]]:
        """
        Initialize empty history for each item ID and each possible bucket.

        Parameters
        ----------
        item_ids : List[Any]
            List of item IDs in the dataset.

        Returns
        -------
        Dict[Any, Dict[str, np.ndarray]]
            A nested dictionary with item IDs and bucket keys as keys,
            mapping to arrays of past observed target values.
        """
        return {item_id: {key: np.array([], dtype=float) for key in self.bucket_keys} for item_id in item_ids}

    def _build_history_from_context(self, context_data: TimeSeriesDataFrame) -> Dict[Any, Dict[str, np.ndarray]]:
        """
        Build history from past context data by assigning each observation
        to a time-based bucket and filtering out NaN values.

        Parameters
        ----------
        context_data : TimeSeriesDataFrame
            Historical time series data containing target values.

        Returns
        -------
        Dict[Any, Dict[str, np.ndarray]]
            Dictionary containing historical target values per item and bucket.
        """
        df = context_data.reset_index()
        df["bucket"] = df["timestamp"].apply(self._make_key)
        grouped = df.groupby(["item_id", "bucket"])["target"]

        history = self._initialize_history(context_data.item_ids)
        for (item_id, bucket), vals in grouped:
            vals = vals.to_numpy()
            clean_vals = vals[~np.isnan(vals)]
            history[item_id][bucket] = clean_vals[-self.last_n_samples :] if self.last_n_samples else clean_vals

        return history

    def fit(
        self,
        data_train: TimeSeriesDataFrame,
        data_val: Optional[TimeSeriesDataFrame] = None,
    ) -> None:
        """
        No fitting required. This predictor uses only historical patterns at predict time.
        """
        logging.info("NNPredictor: no fit step; predict() will build or update history.")

    def predict(
        self,
        data: TimeSeriesDataFrame,
        previous_context_data: Optional[TimeSeriesDataFrame] = None,
        predict_only_last_timestep: bool = False,
    ) -> PredictionLeadTimes:
        """
        Generate quantile forecasts by matching each forecasted timestamp to its
        nearest historical "bucket" and computing empirical quantiles.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            Time series data used to update history and for which forecasts are required.
        previous_context_data : Optional[TimeSeriesDataFrame], optional
            Contextual data used to pre-fill history before forecasting.
        predict_only_last_timestep : bool, optional
            Not used in this implementation.

        Returns
        -------
        PredictionLeadTimes
            Quantile forecasts for each lead time and item id.
        """
        if previous_context_data is not None:
            logging.info("Building history from provided context_data.")
            history = self._build_history_from_context(previous_context_data)
        else:
            logging.info("Initializing empty history.")
            history = self._initialize_history(data.item_ids)

        forecasts, updated_history = self._forecast_from_history(data, history)
        return forecasts

    def _forecast_from_history(
        self,
        data: TimeSeriesDataFrame,
        history: Dict[Any, Dict[str, np.ndarray]],
    ) -> Tuple[PredictionLeadTimes, Dict[Any, Dict[str, np.ndarray]]]:
        """
        Update historical buckets with new target values and compute forecasts.

        For each (item_id, timestamp), the method:
        - Updates the appropriate bucket with the latest observation.
        - Computes quantile forecasts for each lead time based on the future timestamp's bucket.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            New data to incorporate into history and forecast.
        history : Dict[Any, Dict[str, np.ndarray]]
            Current history of past values per item and bucket.

        Returns
        -------
        Tuple[PredictionLeadTimes, Dict[Any, Dict[str, np.ndarray]]]
            - The predicted quantiles wrapped in a PredictionLeadTimes object.
            - The updated history with new observations.
        """
        forecasts: Dict[int, List[np.ndarray]] = {lt: [] for lt in self.lead_times}
        percentiles = (np.array(self.quantiles) * 100).astype(int)

        for (item_id, ts), row in tqdm(data.iterrows(), total=len(data), desc="Predicting using Nearest Neighbour"):
            # update history now
            if not np.isnan(row["target"].item()):
                key_now = self._make_key(ts)
                updated = np.append(history[item_id][key_now], row["target"].item())
                history[item_id][key_now] = updated[-self.last_n_samples :] if self.last_n_samples else updated

            # forecast for each lead time
            for lt in self.lead_times:
                pred_ts = ts + self.offset * lt
                key_pred = self._make_key(pred_ts)
                arr = history[item_id].get(key_pred, np.array([], dtype=float))
                if arr.size == 0:
                    q_vals = np.full(len(self.quantiles), np.nan)
                else:
                    recent = arr[-self.last_n_samples :] if self.last_n_samples else arr
                    q_vals = np.percentile(recent, percentiles)
                forecasts[lt].append(q_vals)

        # wrap into PredictionLeadTimes
        results: Dict[int, PredictionLeadTime] = {}
        for lt, fc_list in forecasts.items():
            results[lt] = PredictionLeadTime(
                lead_time=lt,
                predictions=torch.tensor(np.vstack(fc_list)),
                freq=self.offset,
                data=data,
            )

        return PredictionLeadTimes(results=results), history


class RollingQuantilePredictor(AbstractPredictor):
    """
    Rolling Window Predictor that generates empirical quantile forecasts based on
    the most recent observed target values.

    This predictor does not learn parameters from training data. Instead, it uses
    a rolling window of the last `n` observed values to estimate the empirical
    distribution for each forecasted timestamp.

    Parameters
    ----------
    quantiles : List[float], optional
        List of quantiles to predict (e.g., [0.1, 0.5, 0.9]).
    lead_times : List[int], optional
        List of lead times (in time steps) for which forecasts should be produced.
    freq : Union[str, pd.Timedelta, pd.DateOffset], optional
        Frequency of the time series data; can be a pandas-parsable string
        (e.g., "1h", "1D"), a Timedelta, or a DateOffset.
    last_n_samples : int, optional
        Number of most recent samples to use for quantile estimation.
    output_dir : Optional[Union[str, Path]], optional
        Directory to store model outputs or logs.
    """

    def __init__(
        self,
        quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        lead_times: List[int] = Field(default_factory=lambda: [1, 2, 3]),
        freq: Union[pd.Timedelta, pd.DateOffset] = pd.Timedelta("1h"),
        last_n_samples: Optional[int] = 100,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.offset = to_offset(freq)
        super().__init__(lead_times=lead_times, freq=freq, output_dir=output_dir)
        self.quantiles = quantiles
        self.last_n_samples = last_n_samples

    def _initialize_history(self, item_ids: List[Any]) -> Dict[Any, Dict[str, np.ndarray]]:
        """
        Initialize an empty history dictionary for each item_id.

        Parameters
        ----------
        item_ids : List[Any]
            List of item IDs to initialize history for.

        Returns
        -------
        Dict[Any, np.ndarray]
            Dictionary mapping item IDs to empty arrays.
        """
        return {item_id: np.array([], dtype=float) for item_id in item_ids}

    def _build_history_from_context(self, context_data: TimeSeriesDataFrame) -> Dict[Union[int, str], np.ndarray]:
        """
        Build history from past context data by collecting non-NaN target values.

        Parameters
        ----------
        context_data : TimeSeriesDataFrame
            Historical time series data containing target values.

        Returns
        -------
        Dict[Union[int, str], np.ndarray]
            Dictionary mapping item IDs to arrays of recent target values.
        """

        history = self._initialize_history(context_data.item_ids)
        df = context_data.reset_index()

        for (item_id,), vals in df.groupby(["item_id"])["target"]:
            vals = vals.to_numpy()
            clean_vals = vals[~np.isnan(vals)]
            history[item_id] = clean_vals[-self.last_n_samples :] if self.last_n_samples else clean_vals

        return history

    def fit(
        self,
        data_train: TimeSeriesDataFrame,
        data_val: Optional[TimeSeriesDataFrame] = None,
    ) -> None:
        """
        No fitting required. This predictor uses only historical patterns at predict time.
        """
        logging.info("RollingQuantilePredictor: no fit step; predict() will build or update history.")

    def predict(
        self,
        data: TimeSeriesDataFrame,
        previous_context_data: Optional[TimeSeriesDataFrame] = None,
        predict_only_last_timestep: bool = False,
    ) -> PredictionLeadTimes:
        """
        Generate forecasts using rolling quantiles over past target values.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            Time series data used to update history and for which forecasts are required.
        previous_context_data : Optional[TimeSeriesDataFrame], optional
            Contextual data used to pre-fill history before forecasting.
        predict_only_last_timestep : bool, optional
            Not used in this implementation.

        Returns
        -------
        PredictionLeadTimes
            Forecasted quantiles for each item and lead time.
        """
        if previous_context_data is not None:
            logging.info("Building history from provided context_data.")
            history = self._build_history_from_context(previous_context_data)
        else:
            logging.info("Initializing empty history.")
            history = self._initialize_history(data.item_ids)

        forecasts, updated_history = self._forecast_from_history(data, history)
        return forecasts

    def _forecast_from_history(
        self,
        data: TimeSeriesDataFrame,
        history: Dict[Any, Dict[str, np.ndarray]],
    ) -> Tuple[PredictionLeadTimes, Dict[Any, Dict[str, np.ndarray]]]:
        """
        Forecast quantiles for each lead time using rolling history of target values.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            New data to incorporate into history and forecast.
        history : Dict[Any, Dict[str, np.ndarray]]
            Current history of past values per item and bucket.

        Returns
        -------
        Tuple[PredictionLeadTimes, Dict[Any, Dict[str, np.ndarray]]]
            - The predicted quantiles wrapped in a PredictionLeadTimes object.
            - The updated history with new observations.
        """
        forecasts: Dict[int, List[np.ndarray]] = {lt: [] for lt in self.lead_times}
        percentiles = (np.array(self.quantiles) * 100).astype(int)

        for (item_id, ts), row in tqdm(data.iterrows(), total=len(data), desc="Predicting using Rolling Window Benchmark"):
            if not np.isnan(row["target"].item()):
                updated = np.append(history[item_id], row["target"].item())
                history[item_id] = updated[-self.last_n_samples :] if self.last_n_samples else updated

            arr = history.get(item_id, np.array([], dtype=float))
            if arr.size == 0:
                q_vals = np.full(len(self.quantiles), np.nan)
            else:
                q_vals = np.percentile(arr, percentiles)

            for lt in self.lead_times:
                forecasts[lt].append(q_vals)

        results: Dict[int, PredictionLeadTime] = {}
        for lt, fc_list in forecasts.items():
            results[lt] = PredictionLeadTime(
                lead_time=lt,
                predictions=torch.tensor(np.vstack(fc_list)),
                freq=self.offset,
                data=data,
            )

        return PredictionLeadTimes(results=results), history
