from tqdm import tqdm
from timeseries import PredictionLeadTimes, PredictionLeadTime
import torch
from chronos import BaseChronosPipeline
from autogluon.timeseries import TimeSeriesDataFrame
from torch.utils.data import DataLoader
from typing import List, Optional
import pandas as pd
from torch.utils.data import Dataset
import numpy as np


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


def predict_chronos(
    pipeline: BaseChronosPipeline,
    data: TimeSeriesDataFrame,
    previous_context_data: Optional[TimeSeriesDataFrame] = None,
    lead_times: List = [1, 2, 3],
    freq: pd.Timedelta = pd.Timedelta("1h"),
    context_length: int = 2048,
) -> PredictionLeadTimes:
    """Make multi-step forecasts using a Chronos pipeline with optional context extension.

    Parameters:
    ----------
    - pipeline: Trained Chronos forecasting pipeline.
    - data: TimeSeriesDataFrame to predict on (MultiIndex: item_id, timestamp).
    - previous_context_data: Optional TimeSeriesDataFrame for additional past context.
    - lead_times: List of lead times to extract from forecast horizon.
    - freq: Frequency of the time series data.
    - context_length: Number of past timesteps used as input context for prediction.

    Returns:
    --------
    - PredictionLeadTimes containing predictions for each specified lead time.
    """

    prediction_length = max(lead_times)

    if prediction_length > 64:
        raise ValueError("Maximum lead time is 64")

    if previous_context_data is not None:
        # Check for matching item_ids
        shared_ids = data.item_ids.intersection(previous_context_data.item_ids)

        for item_id in shared_ids:
            prev_series = previous_context_data.loc[item_id]
            curr_series = data.loc[item_id]

            if len(prev_series) == 0 or len(curr_series) == 0:
                continue  # Skip empty series

            last_prev_time = prev_series.index[-1]
            first_curr_time = curr_series.index[0]

            expected_next_time = last_prev_time + freq
            if expected_next_time != first_curr_time:
                raise ValueError(f"Data for item_id '{item_id}' is not consecutive. " f"Expected {expected_next_time}, got {first_curr_time}.")

        # Ensure only relevant context is included and item_ids match
        previous_context_data = previous_context_data.loc[data.item_ids]
        previous_context_data = previous_context_data.groupby("item_id").tail(context_length)
        data_merged = pd.concat([previous_context_data, data]).sort_index()
    else:
        data_merged = data

    data_merged = TimeSeriesDataFrame(data_merged)
    ds = ChronosBacktestingDataset(data_merged, context_length)
    dl = DataLoader(ds, batch_size=64)

    results = {ld: None for ld in lead_times}
    forecasts = []

    for batch in tqdm(dl):
        forecast = pipeline.predict(context=batch, prediction_length=prediction_length)
        forecasts.append(forecast)

    forecasts = torch.vstack(forecasts)

    # only get the relevant forecasts for the data, gets rid of the additional data that was added
    mask = data_merged.index.isin(data.index)
    forecasts = forecasts[mask, ...]

    for lt in lead_times:
        results[lt] = PredictionLeadTime(lead_time=lt, predictions=forecasts[..., lt - 1], freq=freq, data=data)

    return PredictionLeadTimes(results=results)
