from tqdm import tqdm
import torch
from chronos import BaseChronosPipeline
from autogluon.timeseries import TimeSeriesDataFrame
from torch.utils.data import DataLoader
from typing import List, Optional
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from src.core.base import AbstractPredictor
from src.core.timeseries import PredictionLeadTimes, PredictionLeadTime


class ChronosInferenceDataset(Dataset):
    """A dataset for inference with time series data.

    This dataset extracts fixed-length context windows from time series data
    for inference tasks.

    Args:
        target_df (TimeSeriesDataFrame): The time series data containing target values.
        context_length (int): The number of time steps to use as context.
        target_column (str, optional): The column name containing the target values. Defaults to "target".
    """

    def __init__(
        self,
        target_df: TimeSeriesDataFrame,
        context_length: int,
        target_column: str = "target",
    ):
        assert context_length > 0, "context_length must be greater than 0"
        self.context_length = context_length
        self.target_array = target_df[target_column].to_numpy(dtype=np.float32)
        self.freq = target_df.freq

        # Store pointer to start:end of each time series
        cum_sizes = target_df.num_timesteps_per_item().values.cumsum()
        self.indptr = np.append(0, cum_sizes).astype(np.int32)

    def __len__(self):
        """Returns the number of time series in the dataset."""
        return len(self.indptr) - 1

    def _get_context(self, a: np.ndarray, pad_value=np.nan):
        """Extracts the context window, padding with a specified value if needed."""
        a = a[-self.context_length :]
        pad_size = self.context_length - len(a)
        if pad_size > 0:
            pad = np.full(shape=(pad_size,), fill_value=pad_value)
            a = np.concatenate((pad, a))
        return a

    def __getitem__(self, idx) -> np.ndarray:
        """Retrieves the context window for the given index."""
        start_idx = self.indptr[idx]
        end_idx = self.indptr[idx + 1]
        return self._get_context(self.target_array[start_idx:end_idx])


class ChronosBacktestingDataset(Dataset):
    """A dataset for backtesting with time series data.

    This dataset extracts historical context windows for backtesting purposes.

    Args:
        data (TimeSeriesDataFrame): The time series data containing target values.
        context_length (int): The number of time steps to use as context.
        target_column (str, optional): The column name containing the target values. Defaults to "target".
    """

    def __init__(
        self,
        data: TimeSeriesDataFrame,
        context_length: int,
        target_column: str = "target",
    ):
        assert context_length > 0, "context_length must be greater than 0"
        self.context_length = context_length
        self.target_array = data[target_column].to_numpy(dtype=np.float32)
        self.freq = data.freq
        self.item_ids = data.index.get_level_values("item_id").to_numpy()
        cum_sizes = data.num_timesteps_per_item().values.cumsum()
        self.indptr = np.append(0, cum_sizes).astype(np.int32)
        self.item_ids_mask = {item_id: self.item_ids == item_id for item_id in self.item_ids}

    def __len__(self):
        """Returns the total number of time steps in the dataset."""
        return len(self.target_array)

    def _get_context(self, a: np.ndarray, pad_value=np.nan):
        """Extracts the context window, padding with a specified value if needed."""
        a = a[-self.context_length :]
        pad_size = self.context_length - len(a)
        if pad_size > 0:
            pad = np.full(shape=(pad_size,), fill_value=pad_value)
            a = np.concatenate((pad, a))
        return a

    def __getitem__(self, idx) -> np.ndarray:
        """Retrieves the context window for the given index within its corresponding time series."""
        item_id = self.item_ids[idx]
        item_id_start_idx = self.indptr[item_id]
        start_idx = idx - item_id_start_idx
        return self._get_context(self.target_array[self.item_ids_mask[item_id]][: start_idx + 1])


class Chronos(AbstractPredictor):
    """Chronos time series predictor using a pretrained Chronos pipeline (from HuggingFace).
    This class implements prediction logic based on a fixed context window and multiple lead times.

    Parameters:
        model_name (str): Name or path of the pretrained Chronos model.
        device_map (str): Device to run inference on, e.g., "cpu", "cuda", or "mps".
        context_length (int): Number of timesteps used as context for prediction.
        lead_times (List[int]): List of prediction steps ahead (lead times).
        freq (pd.Timedelta): Frequency of the time series data.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-bolt-tiny",
        device_map: str = "mps",
        context_length: int = 2048,
        lead_times: List[int] = [1, 2, 3],
        freq: pd.Timedelta = pd.Timedelta("1h"),
    ) -> None:
        super().__init__(lead_times, freq)

        self.context_length = context_length
        self.prediction_length = max(self.lead_times)

        if self.prediction_length > 64:
            raise ValueError("Maximum supported lead time is 64 currently.")

        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_name,
            device_map=device_map,
        )

    def fit(self, data: TimeSeriesDataFrame) -> None:
        """Placeholder method for training. Not implemented, as Chronos is a pretrained model. Implement Finetuning in the future

        Parameters:
            data (TimeSeriesDataFrame): Training data (not used).

        Raises:
            NotImplementedError: Always raised since training is not supported.
        """

        raise NotImplementedError

    def _merge_data(self, data: TimeSeriesDataFrame, previous_context_data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Merges previous context data with the new prediction data, ensuring time continuity and context length.

        Parameters:
            data (TimeSeriesDataFrame): New data for which predictions will be made.
            previous_context_data (TimeSeriesDataFrame): Historical context data preceding `data`.

        Returns:
            TimeSeriesDataFrame: Combined and sorted dataframe used as input for prediction.

        Raises:
            ValueError: If any time series are not continuous between the two datasets.
        """

        shared_ids = data.item_ids.intersection(previous_context_data.item_ids)

        for item_id in shared_ids:
            prev_series = previous_context_data.loc[item_id]
            curr_series = data.loc[item_id]

            if len(prev_series) == 0 or len(curr_series) == 0:
                continue

            last_prev_time = prev_series.index[-1]
            first_curr_time = curr_series.index[0]

            expected_next_time = last_prev_time + self.freq
            if expected_next_time != first_curr_time:
                raise ValueError(f"Data for item_id '{item_id}' is not consecutive. " f"Expected {expected_next_time}, got {first_curr_time}.")

        previous_context_data = previous_context_data.loc[data.item_ids]
        previous_context_data = previous_context_data.groupby("item_id").tail(self.context_length)
        data_merged = pd.concat([previous_context_data, data]).sort_index()

        return TimeSeriesDataFrame(data_merged)

    def predict(
        self,
        data: TimeSeriesDataFrame,
        predict_only_last_timestep: bool = False,
        previous_context_data: Optional[TimeSeriesDataFrame] = None,
    ) -> PredictionLeadTimes:
        """Predicts future values for the given time series data using a pretrained Chronos model.

        Parameters:
            data (TimeSeriesDataFrame): The target data to forecast.
            predict_only_last_timestep (bool): Whether to forecast only the last timestep of each series.
            previous_context_data (Optional[TimeSeriesDataFrame]): Optional preceding data to provide context.

        Returns:
            PredictionLeadTimes: A dictionary-like object containing lead-time-specific predictions.
        """

        # Add previous context to test data if available
        if previous_context_data is not None:
            data_merged = self._merge_data(data, previous_context_data)
        else:
            data_merged = data

        if predict_only_last_timestep:
            ds = ChronosInferenceDataset(data_merged, self.context_length)
            data = data.slice_by_timestep(start_index=-1)
        else:
            # Make predictions for all timestamps of the test data
            ds = ChronosBacktestingDataset(data_merged, self.context_length)

        dl = DataLoader(ds, batch_size=64)

        results = {ld: None for ld in self.lead_times}
        forecasts = []

        for batch in tqdm(dl, desc="Predicting"):
            forecast = self.pipeline.predict(context=batch, prediction_length=self.prediction_length)
            forecasts.append(forecast)

        forecasts = torch.vstack(forecasts)

        if predict_only_last_timestep is False:
            # Mask to ensure we only return forecasts for the actual prediction set
            mask = data_merged.index.isin(data.index)
            forecasts = forecasts[mask, ...]

        for lt in self.lead_times:
            results[lt] = PredictionLeadTime(lead_time=lt, predictions=forecasts[..., lt - 1], freq=self.freq, data=data)

        return PredictionLeadTimes(results=results)
