from autogluon.timeseries import TimeSeriesDataFrame
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
import torch
from tqdm import tqdm
from src.core.timeseries_evaluation import PredictionLeadTimes, PredictionLeadTime
from src.core.base import AbstractPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class NNPredictor(AbstractPredictor):
    def __init__(
        self,
        quantiles: np.ndarray,
        lead_times: List[int] = [1, 2, 3],
        freq: pd.Timedelta = pd.Timedelta("1h"),
        last_n_samples: int = 10,
    ) -> None:
        super().__init__(lead_times, freq)
        self.quantiles = quantiles
        self.last_n_samples = last_n_samples

    def fit(self, data_train: TimeSeriesDataFrame, data_val: Optional[TimeSeriesDataFrame] = None) -> None:
        """Fits the model by analyzing historical weekday-hour patterns in the time series data.

        Parameters
        ----------
        data_train : TimeSeriesDataFrame
            time series data, indexed by item_id and timestamp.
        data_val : TimeSeriesDataFrame
            time series data, indexed by item_id and timestamp.
        """

        logging.info("No fit needed. Can call predict directly. Only provide previous_context_data when calling predict().")

    def predict(self, data: TimeSeriesDataFrame, previous_context_data: Optional[TimeSeriesDataFrame] = None, predict_only_last_timestep: bool = False) -> PredictionLeadTimes:
        """Generates predictions for the specified future lead times using historical weekday-hour patterns.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            Time series data for which to generate forecasts.
        previous_context_data : TimeSeriesDataFrame, optional
            Historical context data preceding `data`.
        predict_only_last_timestep : bool, optional
            Whether to predict only for the last timestamp of each series, by default False.

        Returns
        -------
        PredictionLeadTimes
            Object containing predictions for each lead time.
        """

        if previous_context_data is not None:
            logging.info("previous_context_data provided. Use this to build the weekday-hour dictionary.")
            weekday_hour_value_dict = self._build_weekday_hour_dict(previous_context_data)
        else:
            logging.info("No previous_context_data provided. Initialize empty weekday-hour dictionary.")
            weekday_hour_value_dict = self._initialize_weekday_hour_dict(data.item_ids)

        predictions, self.weekday_hour_value_dict = self._forecast_from_weekday_hour_patterns(data, weekday_hour_value_dict)

        return predictions

    def _forecast_from_weekday_hour_patterns(
        self,
        data: TimeSeriesDataFrame,
        weekday_hour_value_dict: Dict[Any, Dict[str, np.ndarray]],
    ) -> Tuple[PredictionLeadTimes, dict]:
        """Forecasts values for each lead time by using past values that occurred in the same weekday-hour bucket.

        It also updates the weekday_hour_value_dict

        Parameters
        ----------
        data : TimeSeriesDataFrame
            The input time series data to update internal state and generate forecasts.
        weekday_hour_value_dict : Dict[Any, Dict[str, np.ndarray]]
            Dictionary mapping each item ID to weekday-hour patterns of observed values.

        Returns
        -------
        Tuple[PredictionLeadTimes, dict]
            A tuple containing:
            - PredictionLeadTimes object with forecasts for each lead time.
            - Updated weekday_hour_value_dict containing the latest observed values.
        """

        forecasts = {lt: [] for lt in self.lead_times}
        results = {lt: {} for lt in self.lead_times}
        percentiles = (self.quantiles * 100).astype(int)

        self.last_train_date = data.index.get_level_values("timestamp").max()

        for (item_id, timestamp), row in tqdm(data.iterrows(), total=len(data)):

            current_timestamp_id = f"{timestamp.weekday()}_{timestamp.hour}"
            weekday_hour_value_dict[item_id][current_timestamp_id] = np.append(weekday_hour_value_dict[item_id][current_timestamp_id], row["target"].item())

            for lt in forecasts.keys():
                prediction_timestamp = timestamp + self.freq * lt
                prediction_timestamp_id = f"{prediction_timestamp.weekday()}_{prediction_timestamp.hour}"

                try:
                    if self.last_n_samples:
                        forecast = np.percentile(
                            weekday_hour_value_dict[item_id][prediction_timestamp_id][-self.last_n_samples :],
                            q=percentiles,
                        )
                    else:
                        forecast = np.percentile(
                            weekday_hour_value_dict[item_id][prediction_timestamp_id],
                            q=percentiles,
                        )
                except:
                    forecast = np.array(len(percentiles) * [np.nan])

                forecasts[lt].append(forecast)

        for lt in forecasts.keys():
            results[lt] = PredictionLeadTime(
                lead_time=lt,
                predictions=torch.tensor(np.vstack(forecasts[lt])),
                freq=self.freq,
                data=data,
            )

        return PredictionLeadTimes(results=results), weekday_hour_value_dict

    def _initialize_weekday_hour_dict(self, item_ids: list) -> Dict[str, Dict[str, np.ndarray]]:
        """Initializes a dictionary with keys for each item and weekday-hour combination
        (e.g., "0_0" to "6_23"), each mapping to an empty numpy array.

        Parameters
        ----------
        item_ids : list
            List of item IDs for which to create the dictionary.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary where each item ID maps to another dictionary of weekday-hour keys and empty arrays.
        """

        hours = 24
        weekdays = 7
        weekday_hour_dict = {item_id: None for item_id in item_ids}
        for item_id in item_ids:
            weekday_hour_dict[item_id] = {f"{weekday}_{hour}": np.array([]) for weekday in range(weekdays) for hour in range(hours)}
        return weekday_hour_dict

    def _build_weekday_hour_dict(self, context_data: TimeSeriesDataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """Builds a nested dictionary of past values grouped by item_id and weekday-hour."""

        # Reset index to access item_id and timestamp easily
        df = context_data.reset_index()
        df["weekday_hour"] = df["timestamp"].dt.weekday.astype(str) + "_" + df["timestamp"].dt.hour.astype(str)

        # Group by item_id and weekday_hour
        grouped = df.groupby(["item_id", "weekday_hour"])["target"]

        # Build the nested dictionary
        result = self._initialize_weekday_hour_dict(context_data.item_ids)
        for (item_id, wh), targets in grouped:
            result[item_id][wh] = targets.to_numpy()

        return result
