from autogluon.timeseries import TimeSeriesDataFrame
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Deque
import torch
from tqdm import tqdm
from src.core.base import AbstractPredictor
from src.core.timeseries_evaluation import ForecastCollection, TimeSeriesForecast, HorizonForecast
import logging
from pydantic import Field
from pathlib import Path
from pandas.tseries.frequencies import to_offset
from collections import deque
from scipy import stats

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
        predict_only_last_timestep: bool = False,
    ) -> ForecastCollection:
        """
        Generate forecasts using rolling quantiles over past target values.

        Forecast quantiles for each lead time using seasonal rolling history of target values.

        Update historical buckets with new target values and compute forecasts.

        For each (item_id, timestamp), the method:
        - Updates the appropriate bucket with the latest observation.
        - Computes quantile forecasts for each lead time based on the future timestamp's bucket.

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
        ForecastCollection
            Forecasted quantiles for each item and lead time.
        """
        if previous_context_data is not None:
            logging.info("Building history from provided context_data.")
            history = self._build_history_from_context(previous_context_data)
        else:
            logging.info("Initializing empty history.")
            history = self._initialize_history(data.item_ids)

        ts_forecast: Dict[int, TimeSeriesForecast] = {}
        percentiles = (np.array(self.quantiles) * 100).astype(int)

        for item_id in tqdm(data.item_ids, desc="Predicting using Rolling Window Benchmark"):
            data_sub = data.loc[[item_id]]
            item_history = history[item_id]
            forecasts = {lt: [] for lt in self.lead_times}

            timestamps = data_sub.index.get_level_values("timestamp")
            target_vals = data_sub["target"].values

            for timestamp, target_val in zip(timestamps, target_vals):
                if not np.isnan(target_val):
                    key_now = self._make_key(timestamp)
                    item_history[key_now].append(target_val)

                for lead_time in self.lead_times:
                    pred_timestamp = timestamp + self.offset * lead_time
                    key_pred = self._make_key(pred_timestamp)
                    arr = np.array(item_history.get(key_pred, []))
                    if arr.size == 0:
                        forecasts[lead_time].append(np.full(len(self.quantiles), np.nan))
                    else:
                        forecasts[lead_time].append(np.percentile(arr, percentiles))

            lt_forcast: Dict[int, HorizonForecast] = {}
            for lead_time in self.lead_times:
                lt_forcast[lead_time] = HorizonForecast(lead_time=lead_time, predictions=torch.tensor(np.stack(forecasts[lead_time])))

            ts_forecast[item_id] = TimeSeriesForecast(item_id=item_id, lead_time_forecasts=lt_forcast, data=data_sub.copy(), freq=self.freq, quantiles=self.quantiles)

        return ForecastCollection(item_ids=ts_forecast)


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
        predict_only_last_timestep: bool = False,
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
        predict_only_last_timestep : bool, optional
            Not used in this implementation.

        Returns
        -------
        ForecastCollection
            Forecasted quantiles for each item and lead time.
        """
        if previous_context_data is not None:
            logging.info("Building history from provided context_data.")
            history = self._build_history_from_context(previous_context_data)
        else:
            logging.info("Initializing empty history.")
            history = self._initialize_history(data.item_ids)

        ts_forecast: Dict[int, TimeSeriesForecast] = {}
        percentiles = (np.array(self.quantiles) * 100).astype(int)

        for item_id in tqdm(data.item_ids, desc="Predicting using Rolling Window Benchmark"):
            data_sub = data.loc[[item_id]]
            forecasts = []
            item_history = history[item_id]

            timestamps = data_sub.index.get_level_values("timestamp")
            target_vals = data_sub["target"].values

            for timestamp, target_val in zip(timestamps, target_vals):
                if not np.isnan(target_val):
                    item_history.append(target_val)

                arr = np.array(history.get(item_id, []))
                if arr.size == 0:
                    forecasts.append(np.full(len(self.quantiles), np.nan))
                else:
                    forecasts.append(np.percentile(arr, percentiles))

            forecasts = np.stack(forecasts)
            lt_forcast: Dict[int, HorizonForecast] = {}

            for lead_time in self.lead_times:
                lt_forcast[lead_time] = HorizonForecast(
                    lead_time=lead_time,
                    predictions=torch.tensor(forecasts),  # same trivial forecast for each lead time
                )

            ts_forecast[item_id] = TimeSeriesForecast(item_id=item_id, lead_time_forecasts=lt_forcast, data=data_sub.copy(), freq=self.freq, quantiles=self.quantiles)

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
    freq : Union[str, pd.Timedelta, pd.DateOffset], optional
        Frequency of the time series data; can be a pandas-parsable string
        (e.g., "1h", "1D"), a Timedelta, or a DateOffset.
    output_dir : Optional[Union[str, Path]], optional
        Directory to store model outputs or logs.
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        lead_times: List[int] = [1, 2, 3],
        freq: Union[pd.Timedelta, pd.DateOffset] = pd.Timedelta("1h"),
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        # Normalize freq into a pandas DateOffset
        self.offset = to_offset(freq)
        super().__init__(lead_times=lead_times, freq=freq, output_dir=output_dir)
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
            data_sub = data_train.loc[[id]]["target"].values

            y = np.log(data_sub)
            y_diff = np.diff(y)
            y_diff = y_diff[~np.isnan(y_diff)]
            self.sd_yd[id] = np.std(y_diff)

        logging.info("RandomWalkBenchmark estimated standard deviation for each time series from training data.")

    def predict(
        self,
        data: TimeSeriesDataFrame,
        previous_context_data: Optional[TimeSeriesDataFrame] = None,
        predict_only_last_timestep: bool = False,
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
        predict_only_last_timestep : bool, optional
            Whether to forecast only the final time step. Not used in this implementation.

        Returns
        -------
        ForecastCollection
            Forecasted quantiles for each item and lead time.
        """

        ts_forecast: Dict[int, TimeSeriesForecast] = {}

        h_steps = np.array(self.lead_times).reshape(-1, 1)
        z = stats.norm.ppf(np.array(self.quantiles)).reshape(1, -1)

        for item_id in tqdm(data.item_ids, desc="Predicting using Rolling Window Benchmark"):
            data_sub = data.loc[[item_id]]

            timestamps = data_sub.index.get_level_values("timestamp")
            log_targets = np.log(data_sub["target"]).values

            q_fc_matrix = np.sqrt(h_steps) @ z * self.sd_yd[item_id]

            q_fc_y = []

            for timestamp, log_y in zip(timestamps, log_targets):
                q_fc_y.append(q_fc_matrix + log_y)

            q_fc_y = np.exp(np.stack(q_fc_y, axis=0))

            lt_forcast: Dict[int, HorizonForecast] = {}

            for i, lead_time in enumerate(self.lead_times):
                lt_forcast[lead_time] = HorizonForecast(lead_time=lead_time, predictions=torch.tensor(q_fc_y[:, i, :]))

            ts_forecast[item_id] = TimeSeriesForecast(item_id=item_id, lead_time_forecasts=lt_forcast, data=data_sub.copy(), freq=self.freq, quantiles=self.quantiles)

        return ForecastCollection(item_ids=ts_forecast)
