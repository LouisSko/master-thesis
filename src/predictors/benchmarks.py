from autogluon.timeseries import TimeSeriesDataFrame
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Deque, Iterable
import torch
from tqdm.auto import tqdm
from src.core.base import AbstractPredictor
from src.core.utils import set_global_seed
from src.core.timeseries_evaluation import ForecastCollection, TimeSeriesForecast, HorizonForecast, TARGET
import logging
from pydantic import Field
from pathlib import Path
from pandas.tseries.frequencies import to_offset
from collections import deque
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")
set_global_seed()


class SeasonalNaive(AbstractPredictor):
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
    lead_times : Optional[Iterable[int]], default=None
        An iterable of integers specifying the forecast lead times.
        If None, defaults to [1, 2, 3].
    freq : Union[str, pd.DateOffset], Optional
        Frequency of the time series data; can be a pandas-parsable string
        (e.g., "1h", "1D"), or a DateOffset.
    last_n_samples : int, optional
        Number of most recent samples per bucket to use for quantile estimation.
    output_dir : Optional[Union[str, Path]], optional
        Directory to store model outputs or logs.
    name : str, optional
        Name of the model, defaults to the class name
    """

    def __init__(
        self,
        quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        lead_times: Optional[Iterable[int]] = None,
        freq: Union[str, pd.DateOffset] = "1h",
        last_n_samples: Optional[int] = 10,
        output_dir: Optional[Union[str, Path]] = None,
        name: Optional[str] = None,
    ) -> None:
        # Normalize freq into a pandas DateOffset
        self.offset = to_offset(freq)
        super().__init__(lead_times=lead_times, name=name, output_dir=output_dir)
        self.quantiles = quantiles
        self.last_n_samples = last_n_samples

        # Prepare bucket keys and key‐making function based on freq
        self._setup_buckets()

    def _make_day_key(self, ts: pd.Timestamp) -> str:
        return str(ts.weekday())

    def _make_hour_key(self, ts: pd.Timestamp) -> str:
        return f"{ts.weekday()}_{ts.hour}"

    def _make_minute_key(self, ts: pd.Timestamp) -> str:
        slot = ts.minute // self._minute_interval
        return f"{ts.weekday()}_{ts.hour}_{slot}"

    def _setup_buckets(self) -> None:
        """
        Set up bucket keys and the function to map timestamps to bucket keys.

        The bucketing scheme depends on the frequency of the time series.
        """
        fstr = self.offset.rule_code.upper()

        if fstr in ("D", "B"):
            self._make_key = self._make_day_key
            self.bucket_keys = [str(d) for d in range(7)]

        elif fstr == "H":
            self._make_key = self._make_hour_key
            self.bucket_keys = [f"{d}_{h}" for d in range(7) for h in range(24)]

        elif fstr in ("T", "MIN"):
            self._minute_interval = self.offset.n
            self._make_key = self._make_minute_key
            slots_per_hour = 60 // self._minute_interval
            self.bucket_keys = [f"{d}_{h}_{slot}" for d in range(7) for h in range(24) for slot in range(slots_per_hour)]

        else:
            raise ValueError(f"Unsupported frequency '{fstr}' for SeasonalNaive")

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
        **kwargs,
    ) -> None:
        """
        No fitting required. This predictor uses only historical patterns at predict time.
        """
        logging.info("SeasonalNaive: No fit step; predict() will build or update history.")

    def _predict(
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

            # --- Data sufficiency check for historical quantile estimation ---
            num_buckets = len(item_hist)
            required_samples = num_buckets * self.last_n_samples
            available_samples = len(item_df)
            context_samples = len(previous_context_data.loc[[item_id]]) if previous_context_data is not None else 0

            logging.info(
                "[%s] Required samples: %d buckets × %d last_n_samples = %d",
                item_id,
                num_buckets,
                self.last_n_samples,
                required_samples,
            )

            if rolling:
                if previous_context_data is None:
                    logging.warning(
                        "[%s] Rolling mode is enabled but previous_context_data is missing. "
                        "Forecasts will still be generated, but the first ~%d steps may suffer from poor uncertainty estimation.",
                        item_id,
                        required_samples,
                    )
                elif context_samples < required_samples:
                    logging.warning(
                        "[%s] Only %d context samples available (%.1f%% of required %d). " "Uncertainty estimation may be unreliable, especially for tail quantiles.",
                        item_id,
                        context_samples,
                        100 * context_samples / required_samples,
                        required_samples,
                    )
                else:
                    logging.info(
                        "[%s] Sufficient samples available for historical quantile estimation (%d available).",
                        item_id,
                        context_samples,
                    )
            else:
                if available_samples < required_samples:
                    logging.warning(
                        "[%s] Only %d samples available (%.1f%% of required %d). " "Forecast quality may degrade due to insufficient history.",
                        item_id,
                        available_samples,
                        100 * available_samples / required_samples,
                        required_samples,
                    )
                else:
                    logging.info(
                        "[%s] Sufficient samples available for historical quantile estimation (%d available).",
                        item_id,
                        available_samples,
                    )

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
                    bucket_q_cache[key] = np.percentile(vals, percentiles, method="linear") if vals.size else np.full(len(self.quantiles), np.nan)
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
        **kwargs,
    ) -> None:
        """
        No fitting required. This predictor uses only historical patterns at predict time.
        """
        logging.info("SeasonalNaive: No fit step; provide data in predict() function.")


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
    lead_times : Optional[Iterable[int]], default=None
        An iterable of integers specifying the forecast lead times.
        If None, defaults to [1, 2, 3].
    last_n_samples : int, optional
        Number of most recent samples to use for quantile estimation.
    output_dir : Optional[Union[str, Path]], optional
        Directory to store model outputs or logs.
    name : str, optional
        Name of the model, defaults to the class name
    """

    def __init__(
        self,
        quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        lead_times: Optional[Iterable[int]] = None,
        last_n_samples: Optional[int] = 100,
        output_dir: Optional[Union[str, Path]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(lead_times=lead_times, name=name, output_dir=output_dir)
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
        **kwargs,
    ) -> None:
        """
        No fitting required. This predictor uses only historical patterns at predict time.
        """
        logging.info("RollingQuantilePredictor: no fit step; predict() will build or update history.")

    def _predict(
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


class RandomWalk(AbstractPredictor):
    """
    A simple benchmark model based on a driftless random walk in log space with constant volatility.

    This model first estimates the standard deviation of log returns from the training data for each item.
    It then generates future quantile forecasts by simulating a Gaussian random walk in log space, where
    uncertainty increases with the square root of the lead time. The resulting forecasts are returned in
    the original scale by exponentiating the simulated values.

    This version assumes that volatility is stationary over time and does not change during prediction.

    Parameters
    ----------
    quantiles : List[float], optional
        List of quantiles to predict (e.g., [0.1, 0.5, 0.9]).
    lead_times : Optional[Iterable[int]], default=None
        An iterable of integers specifying the forecast lead times.
        If None, defaults to [1, 2, 3].
    output_dir : Optional[Union[str, Path]], optional
        Directory to store model outputs or logs.
    name : str, optional
        Name of the model, defaults to the class name.
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        lead_times: Optional[Iterable[int]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(lead_times=lead_times, name=name, output_dir=output_dir)
        self.quantiles = quantiles
        self.sd_yd = {}  # standard deviation for each item id

    def _fit(
        self,
        data_train: TimeSeriesDataFrame,
        data_val: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
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

        logging.info("RandomWalk estimated standard deviation for each time series from training data.")

    def _predict(
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
            targets = data_sub["target"]
            if (targets <= 0).any():
                logging.warning(f"Item {item_id} in `data` contains non-positive values; log is undefined.")
            log_targets = np.log(targets).values

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


class OnlineRandomWalk(AbstractPredictor):
    """
    A simple benchmark model based on a driftless random walk in log space with dynamic volatility estimation.

    Unlike the standard RandomWalk model, this version computes the standard deviation of log returns dynamically
    at each prediction time step using a rolling or expanding window. This allows the model to adapt to changes
    in volatility over time. Forecasts are generated in log space and exponentiated to return to the original scale.

    This model does not require a fitting step. Instead, it computes rolling statistics on-the-fly during prediction,
    optionally using a context window (e.g., for rolling forecasting tasks).

    Parameters
    ----------
    quantiles : List[float], optional
        List of quantiles to predict (e.g., [0.1, 0.5, 0.9]).
    lead_times : List[int], optional
        List of lead times (in time steps) for which forecasts should be produced.
    last_n_samples : int, optional
        Number of most recent samples used for computing the standard deviation. If None, all available history
        up to the forecast time is used (expanding window).
    output_dir : Optional[Union[str, Path]], optional
        Directory to store model outputs or logs.
    name : str, optional
        Name of the model, defaults to the class name.
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        lead_times: List[int] = [1, 2, 3],
        last_n_samples: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(lead_times=lead_times, name=name, output_dir=output_dir)
        self.quantiles = quantiles
        self.last_n_samples = last_n_samples
        self.sd_yd = {}  # standard deviation for each item id

    def _fit(
        self,
        data_train: TimeSeriesDataFrame,
        data_val: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
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

        logging.info("No fitting needed. Standard deviation will be estimated based on the data in the predict() function.")

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        previous_context_data: Optional[TimeSeriesDataFrame] = None,
        rolling: bool = False,
        window_step: int = 1,
    ) -> ForecastCollection:

        freq = pd.tseries.frequencies.to_offset(data.freq)
        ts_forecast: Dict[int, TimeSeriesForecast] = {}

        h_steps = np.array(self.lead_times).reshape(-1, 1)
        z = stats.norm.ppf(np.array(self.quantiles)).reshape(1, -1)

        for item_id in tqdm(data.item_ids, desc="Predicting using Random Walk without drift"):
            data_sub = data.loc[[item_id]].copy()
            timestamps = data_sub.index.get_level_values("timestamp")

            # Compute rolling std of log returns and utilize the context data
            if previous_context_data is not None:
                context_data_sub = previous_context_data.loc[[item_id]].copy()
                data_merged, skip_first = self._merge_data(data_sub, context_data_sub, len(context_data_sub))
                skip_first = skip_first[item_id]
            else:
                data_merged = data_sub
                skip_first = 0
            targets_merged = data_merged["target"]
            min_target = targets_merged.min()
            if min_target <= 0:
                epsilon = -min_target + 1e-8
                logging.info("Item %s contains non-positive values; log is undefined. Adding an epsilon of %s.", item_id, epsilon)
            else:
                epsilon = 0
            log_targets_merged = np.log(targets_merged + epsilon)
            log_returns = log_targets_merged.diff()
            if self.last_n_samples is None:
                rolling_std = log_returns.expanding(min_periods=2).std()[skip_first:]
            else:
                rolling_std = log_returns.rolling(window=self.last_n_samples, min_periods=2).std()[skip_first:]
            log_targets = log_targets_merged[skip_first:]

            # Decide forecast time steps
            if rolling:
                eval_indices = list(range(0, len(timestamps), window_step))
            else:
                eval_indices = [len(timestamps) - 1]
            forecast_mask = np.zeros(len(timestamps), dtype=bool)
            forecast_mask[eval_indices] = True

            q_fc_y = []

            for idx in eval_indices:
                log_y = log_targets.iloc[idx]
                std_dev = rolling_std.iloc[idx]

                if pd.isna(log_y) or pd.isna(std_dev) or std_dev == 0:
                    q_fc_y.append(np.full((len(self.lead_times), len(self.quantiles)), np.nan))
                else:
                    q_matrix = np.sqrt(h_steps) @ z * std_dev + log_y
                    q_fc_y.append(q_matrix)

            q_fc_y = np.exp(np.stack(q_fc_y, axis=0)) - epsilon

            lt_forecast: Dict[int, HorizonForecast] = {}
            for i, lead_time in enumerate(self.lead_times):
                lt_forecast[lead_time] = HorizonForecast(
                    lead_time=lead_time,
                    predictions=torch.tensor(q_fc_y[:, i, :]),
                )

            ts_forecast[item_id] = TimeSeriesForecast(
                item_id=item_id,
                lead_time_forecasts=lt_forecast,
                data=data_sub.copy(),
                freq=freq,
                quantiles=self.quantiles,
                forecast_mask=forecast_mask,
            )

        return ForecastCollection(item_ids=ts_forecast)
