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
from pandas.tseries.offsets import DateOffset
from pandas.tseries.frequencies import to_offset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")


class NNPredictor(AbstractPredictor):
    """
    Nearest Neighbour Predictor for quantile forecasting based on historical values.

    Parameters
    ----------
    quantiles : List[float], optional
        List of quantiles to predict.
    lead_times : List[int], optional
        List of lead times (in steps) for forecasting.
    freq : Union[str, pd.Timedelta, DateOffset], optional
        Frequency of the time series data; can be a pandas‐parsable string
        (e.g. "1h","1D","1B","15T"), a Timedelta, or a DateOffset.
    last_n_samples : int, optional
        Number of past samples to consider for prediction.
    output_dir : Optional[Union[str, Path]], optional
        Directory to store outputs.
    """

    def __init__(
        self,
        quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        lead_times: List[int] = Field(default_factory=lambda: [1, 2, 3]),
        freq: Union[str, pd.Timedelta, DateOffset] = "1h",
        last_n_samples: int = 10,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        # Normalize freq into a pandas DateOffset
        self.offset = to_offset(freq)
        super().__init__(lead_times=lead_times, freq=self.offset, output_dir=output_dir)
        self.quantiles = quantiles
        self.last_n_samples = last_n_samples

        # Prepare bucket keys and key‐making function based on freq
        self._setup_buckets()

    def _setup_buckets(self) -> None:
        """
        Build the list of all possible "bucket" keys and a function to
        map any timestamp into its bucket, depending on self.offset.
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
        Initialize an empty history dict for each item_id and bucket key.
        """
        return {item_id: {key: np.array([], dtype=float) for key in self.bucket_keys} for item_id in item_ids}

    def _build_history_from_context(self, context_data: TimeSeriesDataFrame) -> Dict[Any, Dict[str, np.ndarray]]:
        """
        Build initial history from provided context_data by grouping past targets
        into buckets defined by self._make_key.
        """
        df = context_data.reset_index()
        df["bucket"] = df["timestamp"].apply(self._make_key)
        grouped = df.groupby(["item_id", "bucket"])["target"]

        history = self._initialize_history(context_data.item_ids)
        for (item_id, bucket), vals in grouped:
            vals = vals.to_numpy()
            history[item_id][bucket] = vals[~np.isnan(vals)]

        return history

    def fit(
        self,
        data_train: TimeSeriesDataFrame,
        data_val: Optional[TimeSeriesDataFrame] = None,
    ) -> None:
        """
        No fitting required—this predictor uses only historical patterns at predict time.
        """
        logging.info("NNPredictor: no fit step; predict() will build or update history.")

    def predict(
        self,
        data: TimeSeriesDataFrame,
        previous_context_data: Optional[TimeSeriesDataFrame] = None,
        predict_only_last_timestep: bool = False,
    ) -> PredictionLeadTimes:
        """
        Generate quantile forecasts for each lead time based on nearest‐neighbour
        bucketed by the normalized frequency offset.
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
        For each (item_id, timestamp), append the observed value to its bucket,
        then predict future buckets at lead times via empirical quantiles.
        """
        forecasts: Dict[int, List[np.ndarray]] = {lt: [] for lt in self.lead_times}
        percentiles = (np.array(self.quantiles) * 100).astype(int)

        for (item_id, ts), row in tqdm(data.iterrows(), total=len(data), desc= "Predicting using Nearest Neighbour"):
            # update history now
            if not np.isnan(row["target"].item()):
                key_now = self._make_key(ts)
                history[item_id][key_now] = np.append(history[item_id][key_now], row["target"].item())

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
