from autogluon.timeseries import TimeSeriesDataFrame
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Deque
import torch
from tqdm.auto import tqdm
from src.core.base import AbstractPredictor
from src.core.timeseries_evaluation import ForecastCollection, TimeSeriesForecast, HorizonForecast, TARGET
import logging
from pydantic import Field
from pathlib import Path
from pandas.tseries.frequencies import to_offset
from collections import deque
from scipy import stats
import logging

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
    freq : Union[str, pd.DateOffset], Optional
        Frequency of the time series data; can be a pandas-parsable string
        (e.g., "1h", "1D"), or a DateOffset.
    last_n_samples : int, optional
        Number of most recent samples per bucket to use for quantile estimation.
    output_dir : Optional[Union[str, Path]], optional
        Directory to store model outputs or logs.
    """

    def __init__(
        self,
        quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        lead_times: List[int] = Field(default_factory=lambda: [1, 2, 3]),
        freq: Union[str, pd.DateOffset] = "1h",
        last_n_samples: Optional[int] = 10,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        # Normalize freq into a pandas DateOffset
        self.offset = to_offset(freq)
        super().__init__(lead_times=lead_times, output_dir=output_dir)
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
        fstr = self.offset.rule_code  # e.g. "1H","B","15T","D"

        if fstr.upper() in ("D", "B"):
            # daily or business‐day: bucket by weekday only
            self._make_key = lambda ts: str(ts.weekday())
            self.bucket_keys = [str(d) for d in range(7)]

        elif fstr.upper() == "H":
            # hourly: bucket by weekday_hour
            self._make_key = lambda ts: f"{ts.weekday()}_{ts.hour}"
            self.bucket_keys = [f"{d}_{h}" for d in range(7) for h in range(24)]

        elif (fstr.upper() == "T") or (fstr.upper() == "MIN"):
            # minute frequency, e.g. 15T, 5T, etc.
            n = self.offset.n  # number of minutes

            def make_minute_key(ts: pd.Timestamp) -> str:
                slot = ts.minute // n
                return f"{ts.weekday()}_{ts.hour}_{slot}"

            self._make_key = make_minute_key

            slots_per_hour = 60 // n
            self.bucket_keys = [f"{d}_{h}_{slot}" for d in range(7) for h in range(24) for slot in range(slots_per_hour)]

        else:
            raise ValueError(f"Unsupported frequency '{fstr}' for RollingSeasonalQuantilePredictor")

    def _initialize_history(self, item_ids: List[Any]) -> Dict[int, Dict[int, Deque[float]]]:
        """
        Initialize empty history for each item ID and each possible bucket.

        Parameters
        ----------
        item_ids : List[Any]
            List of item IDs in the dataset.

        Returns
        -------
        Dict[int, Dict[int, Deque[float]]]
            A nested dictionary with item IDs and bucket keys as keys,
            mapping to arrays of past observed target values.
        """
        return {item_id: {key: deque(maxlen=self.last_n_samples) for key in self.bucket_keys} for item_id in item_ids}

    def _build_history_from_context(self, context_data: TimeSeriesDataFrame) -> Dict[int, Dict[int, Deque[float]]]:
        """
        Build history from past context data by assigning each observation
        to a time-based bucket and filtering out NaN values.

        Parameters
        ----------
        context_data : TimeSeriesDataFrame
            Historical time series data containing target values.

        Returns
        -------
        Dict[int, Dict[int, Deque[float]]]
            Dictionary containing historical target values per item and bucket.
        """
        df = context_data.reset_index()
        df["bucket"] = df["timestamp"].apply(self._make_key)
        grouped = df.groupby(["item_id", "bucket"])["target"]

        history = self._initialize_history(context_data.item_ids)
        for (item_id, bucket), vals in grouped:
            clean_vals = vals.dropna().values[-self.last_n_samples :] if self.last_n_samples else vals.dropna().values
            history[item_id][bucket].extend(clean_vals.tolist())  # extend with list of floats

        return history

    def _fit(
        self,
        data_train: TimeSeriesDataFrame,
        data_val: Optional[TimeSeriesDataFrame] = None,
    ) -> None:
        """
        No fitting required. This predictor uses only historical patterns at predict time.
        """
        logging.info("RollingSeasonalQuantilePredictor: No fit step; predict() will build or update history.")

    def predict(
        self,
        data: TimeSeriesDataFrame,
        previous_context_data: Optional[TimeSeriesDataFrame] = None,
        rolling: bool = False,
        window_step: int = 1,
    ) -> ForecastCollection:
        """
        Generates forecasts for each time series.

        This method can perform either:
        - *single-shot prediction* (predicting from the most recent context window), or
        - *rolling backtesting* (sliding a window across the time series to predict at each time point).

        Parameters
        ----------
        data : TimeSeriesDataFrame
            The time series data to forecast.
        previous_context_data : Optional[TimeSeriesDataFrame], default=None
            Optional preceding time series data for extending the context window.
        rolling : bool, default=False
            If True, performs rolling evaluation across all available time steps.
            If False, predicts only from the latest observation.
        window_step : int, default=1
            The number of time steps to move the sliding (rolling) prediction window forward between each prediction.
            This controls how densely forecasts are generated across time. A smaller value creates more overlapping
            forecasts, while a larger value skips more observations between windows.
            The rolling procedure is applied independently to each time series in the dataset.

        Returns
        -------
        ForecastCollection
            A nested dictionary mapping each item_id to lead time forecasts.
        """
        if window_step < 1:
            raise ValueError("window_step must be a positive integer (≥1)")

        # 1. Initialise / pre-seed bucket history
        if previous_context_data is not None:
            logging.info("Building history from provided context data.")
            history = self._build_history_from_context(previous_context_data)
        else:
            logging.info("Initialising empty history.")
            history = self._initialize_history(data.item_ids)

        freq = pd.tseries.frequencies.to_offset(data.freq)

        percentiles = (np.array(self.quantiles) * 100).astype(int)
        ts_forecast: Dict[int, TimeSeriesForecast] = {}

        # 2. Per-item loop
        for item_id in tqdm(data.item_ids, desc="RollingQuantilePredictor"):
            item_df = data.loc[[item_id]]
            item_hist = history[item_id]  # shortcut
            timestamps = item_df.index.get_level_values("timestamp")
            target_vals = item_df["target"].values

            # --- cache: bucket_key -> q_hat vector ---------------------------------
            bucket_q_cache: dict[str, np.ndarray] = {
                key: np.percentile(np.asarray(vals), percentiles) if vals else np.full(len(self.quantiles), np.nan) for key, vals in item_hist.items()
            }
            dirty: set[str] = set()  # buckets whose history we just ch

            # Decide at which row indices we will actually issue a forecast
            if rolling:
                eval_indices = list(range(0, len(timestamps), window_step))

            else:  # single-shot
                eval_indices = [len(timestamps) - 1]

            forecast_mask = np.zeros(len(timestamps), dtype=bool)
            forecast_mask[eval_indices] = True

            forecasts_per_lt: Dict[int, List[np.ndarray]] = {lt: [] for lt in self.lead_times}

            # 2a. Single pass over the rows: update history, optionally forecast
            for idx, (ts, y) in enumerate(tqdm(zip(timestamps, target_vals), total=len(timestamps), desc=f"RSQP: generate forecasts for item_id {item_id}")):
                # Update history with *current* observation (if not NaN)
                if not np.isnan(y):
                    key_now = self._make_key(ts)
                    item_hist[key_now].append(y)
                    dirty.add(key_now)

                # Skip forecasting if this row is not an evaluation point
                if idx not in eval_indices:
                    continue

                # 3) refresh the cache only for buckets that changed
                for key in dirty:
                    vals = np.asarray(item_hist[key])
                    bucket_q_cache[key] = np.percentile(vals, percentiles) if vals.size else np.full(len(self.quantiles), np.nan)
                dirty.clear()

                # 4) fetch forecasts for all lead-times
                for lt in self.lead_times:
                    future_key = self._make_key(ts + self.offset * lt)
                    q_hat = bucket_q_cache.get(future_key, np.full(len(self.quantiles), np.nan))  # unseen bucket
                    forecasts_per_lt[lt].append(q_hat)

            # 2b. Wrap forecasts for this item in HorizonForecast containers
            horizon_dict: Dict[int, HorizonForecast] = {}
            for lt in self.lead_times:
                preds = np.stack(forecasts_per_lt[lt]) if forecasts_per_lt[lt] else np.empty((0, len(self.quantiles)))  # shape [n_eval, n_q]
                horizon_dict[lt] = HorizonForecast(
                    lead_time=lt,
                    predictions=torch.tensor(preds, dtype=torch.float32),
                )

            ts_forecast[item_id] = TimeSeriesForecast(
                item_id=item_id,
                lead_time_forecasts=horizon_dict,
                data=item_df.copy(),
                freq=freq,
                quantiles=self.quantiles,
                forecast_mask=forecast_mask,
            )

        return ForecastCollection(item_ids=ts_forecast)

    def _fit(
        self,
        data_train: TimeSeriesDataFrame,
        data_val: Optional[TimeSeriesDataFrame] = None,
    ) -> None:
        """
        No fitting required. This predictor uses only historical patterns at predict time.
        """
        logging.info("RollingSeasonalQuantilePredictor: No fit step; provide data in predict() function.")


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
    last_n_samples : int, optional
        Number of most recent samples to use for quantile estimation.
    output_dir : Optional[Union[str, Path]], optional
        Directory to store model outputs or logs.
    """

    def __init__(
        self,
        quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        lead_times: List[int] = Field(default_factory=lambda: [1, 2, 3]),
        last_n_samples: Optional[int] = 100,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(lead_times=lead_times, output_dir=output_dir)
        self.quantiles = quantiles
        self.last_n_samples = last_n_samples

    def _initialize_history(self, item_ids: List[Any]) -> Dict[int, Deque[float]]:
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
        return {item_id: deque(maxlen=self.last_n_samples) for item_id in item_ids}

    def _build_history_from_context(self, context_data: TimeSeriesDataFrame) -> Dict[int, Deque[float]]:
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
            clean_vals = vals.dropna().values[-self.last_n_samples :] if self.last_n_samples else vals.dropna().values
            history[item_id].extend(clean_vals.tolist())  # extend with list of floats

        return history

    def _fit(
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
        rolling: bool = False,
        window_step: int = 1,
    ) -> ForecastCollection:
        """
        Generate forecasts using rolling quantiles over past target values.

        Forecast quantiles for each lead time using rolling history of target values.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            Time series data used to update history and for which forecasts are required.
        previous_context_data : Optional[TimeSeriesDataFrame], optional
            Contextual data used to pre-fill history before forecasting.
        rolling : bool, optional
            Whether to perform rolling forecasts at each window_step.
        window_step : int, optional
            The number of time steps to move the sliding (rolling) prediction window forward between each prediction.
            This controls how densely forecasts are generated across time. A smaller value creates more overlapping
            forecasts, while a larger value skips more observations between windows.
            The rolling procedure is applied independently to each time series in the dataset.

        Returns
        -------
        ForecastCollection
            Forecasted quantiles for each item and lead time.
        """
        if window_step < 1:
            raise ValueError("window_step must be a positive integer (≥1)")

        if previous_context_data is not None:
            logging.info("Building history from provided context_data.")
            history = self._build_history_from_context(previous_context_data)
        else:
            logging.info("Initializing empty history.")
            history = self._initialize_history(data.item_ids)

        ts_forecast: Dict[int, TimeSeriesForecast] = {}
        percentiles = (np.array(self.quantiles) * 100).astype(int)

        freq = pd.tseries.frequencies.to_offset(data.freq)

        for item_id in tqdm(data.item_ids, desc="RollingQuantilePredictor"):
            data_sub = data.loc[[item_id]]
            item_history = history[item_id]
            timestamps = data_sub.index.get_level_values("timestamp")
            target_vals = data_sub["target"].values

            # Decide which rows we forecast at
            if rolling:
                eval_indices = list(range(0, len(timestamps), window_step))
            else:
                eval_indices = [len(timestamps) - 1]

            forecast_mask = np.zeros(len(timestamps), dtype=bool)
            forecast_mask[eval_indices] = True

            forecasts_per_lt: Dict[int, List[np.ndarray]] = {lt: [] for lt in self.lead_times}

            # Use a cache to avoid recomputing percentiles unnecessarily
            cached_q_hat = np.full(len(self.quantiles), np.nan)
            dirty = False

            for idx, (_, y) in enumerate(tqdm(zip(timestamps, target_vals), total=len(timestamps), desc=f"RQP: generate forecasts for item_id {item_id}")):
                # Update history
                if not np.isnan(y):
                    item_history.append(y)
                    dirty = True

                # If not forecasting at this row, continue
                if idx not in eval_indices:
                    continue

                # Refresh cache if history was updated
                if dirty:
                    arr = np.asarray(item_history)
                    if arr.size == 0:
                        cached_q_hat = np.full(len(self.quantiles), np.nan)
                    else:
                        cached_q_hat = np.percentile(arr, percentiles)
                    dirty = False

                # Append the same forecast for all lead times (trivial persistence)
                for lt in self.lead_times:
                    forecasts_per_lt[lt].append(cached_q_hat)

            # Wrap forecasts per lead time
            horizon_dict = {}
            for lt in self.lead_times:
                preds = np.stack(forecasts_per_lt[lt]) if forecasts_per_lt[lt] else np.empty((0, len(self.quantiles)))
                horizon_dict[lt] = HorizonForecast(
                    lead_time=lt,
                    predictions=torch.tensor(preds, dtype=torch.float32),
                )

            ts_forecast[item_id] = TimeSeriesForecast(
                item_id=item_id,
                lead_time_forecasts=horizon_dict,
                data=data_sub.copy(),
                freq=freq,
                quantiles=self.quantiles,
                forecast_mask=forecast_mask,
            )

        return ForecastCollection(item_ids=ts_forecast)


class RandomWalkBenchmark(AbstractPredictor):
    """
    A simple benchmark model based on a random walk with Gaussian innovations in log space.

    This model estimates the standard deviation of log returns for each time series from
    the training data, and generates future quantile forecasts by simulating a random walk
    in log space, scaled by the square root of the lead time.

    Parameters
    ----------
    quantiles : List[float], optional
        List of quantiles to predict (e.g., [0.1, 0.5, 0.9]).
    lead_times : List[int], optional
        List of lead times (in time steps) for which forecasts should be produced.
    output_dir : Optional[Union[str, Path]], optional
        Directory to store model outputs or logs.
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        lead_times: List[int] = [1, 2, 3],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(lead_times=lead_times, output_dir=output_dir)
        self.quantiles = quantiles
        self.sd_yd = {}  # standard deviation for each item id

    def _fit(
        self,
        data_train: TimeSeriesDataFrame,
        data_val: Optional[TimeSeriesDataFrame] = None,
    ) -> None:
        """
        Estimate the standard deviation of the log-differenced target series for each item.

        Parameters
        ----------
        data_train : TimeSeriesDataFrame
            Training data containing time series with 'target' values.
        data_val : Optional[TimeSeriesDataFrame], optional
            Validation data (not used in this implementation).
        """

        for id in data_train.item_ids:
            data_sub = data_train.loc[[id]][TARGET].values

            if any(data_sub <= 0):
                raise ValueError("This model can only be used with strictly positive time series.")

            y = np.log(data_sub)
            y_diff = np.diff(y)
            y_diff = y_diff[~np.isnan(y_diff)]
            self.sd_yd[id] = np.std(y_diff)

        logging.info("RandomWalkBenchmark estimated standard deviation for each time series from training data.")

    def predict(
        self,
        data: TimeSeriesDataFrame,
        previous_context_data: Optional[TimeSeriesDataFrame] = None,
        rolling: bool = False,
        window_step: int = 1,
    ) -> ForecastCollection:
        """
        Generate quantile forecasts using a Gaussian random walk in log space.

        For each time series, forecasts are produced by simulating a driftless
        random walk with standard deviation estimated during training. Forecasts
        are returned in the original (exponentiated) scale.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            Time series data used to update history and for which forecasts are required.
        previous_context_data : Optional[TimeSeriesDataFrame], optional
            Contextual data used to pre-fill history before forecasting. Not used in this implementation.
        rolling : bool, optional
            Whether to forecast repeatedly in a rolling fashion.
        window_step : int, optional
            The number of time steps to move the sliding (rolling) prediction window forward between each prediction.
            This controls how densely forecasts are generated across time. A smaller value creates more overlapping
            forecasts, while a larger value skips more observations between windows.
            The rolling procedure is applied independently to each time series in the dataset.

        Returns
        -------
        ForecastCollection
            Forecasted quantiles for each item and lead time.
        """

        freq = pd.tseries.frequencies.to_offset(data.freq)

        ts_forecast: Dict[int, TimeSeriesForecast] = {}

        h_steps = np.array(self.lead_times).reshape(-1, 1)
        z = stats.norm.ppf(np.array(self.quantiles)).reshape(1, -1)

        for item_id in tqdm(data.item_ids, desc="Predicting using Random Walk Benchmark"):
            data_sub = data.loc[[item_id]]

            timestamps = data_sub.index.get_level_values("timestamp")
            log_targets = np.log(data_sub["target"]).values

            q_fc_matrix = np.sqrt(h_steps) @ z * self.sd_yd[item_id]

            # Decide at which rows to forecast
            if rolling:
                eval_indices = list(range(0, len(timestamps), window_step))
            else:
                eval_indices = [len(timestamps) - 1]

            forecast_mask = np.zeros(len(timestamps), dtype=bool)
            forecast_mask[eval_indices] = True

            q_fc_y = []

            for idx, (timestamp, log_y) in enumerate(zip(timestamps, log_targets)):

                if idx not in eval_indices:
                    continue

                if np.isnan(log_y):
                    q_fc_y.append(np.full_like(q_fc_matrix, np.nan))
                else:
                    q_fc_y.append(q_fc_matrix + log_y)

            q_fc_y = np.exp(np.stack(q_fc_y, axis=0))

            lt_forcast: Dict[int, HorizonForecast] = {}

            for i, lead_time in enumerate(self.lead_times):
                lt_forcast[lead_time] = HorizonForecast(
                    lead_time=lead_time,
                    predictions=torch.tensor(q_fc_y[:, i, :]),
                )

            ts_forecast[item_id] = TimeSeriesForecast(
                item_id=item_id,
                lead_time_forecasts=lt_forcast,
                data=data_sub.copy(),
                freq=freq,
                quantiles=self.quantiles,
                forecast_mask=forecast_mask,
            )

        return ForecastCollection(item_ids=ts_forecast)
