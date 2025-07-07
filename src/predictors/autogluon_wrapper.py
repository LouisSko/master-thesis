from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from typing import List, Optional, Union, Dict
import pandas as pd
from src.core.base import AbstractPredictor
import logging
from src.core.timeseries_evaluation import ForecastCollection
from pydantic import Field
from pathlib import Path
from gluonts.dataset.common import ListDataset
from torch.utils.data import DataLoader
from src.predictors.chronos import BaseTimeSeriesDataset
from tqdm.auto import tqdm
from src.core.timeseries_evaluation import HorizonForecast, ForecastCollection, TimeSeriesForecast
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")


class GluonTSDataset(BaseTimeSeriesDataset):
    """
    Yields dicts for GluonTS: {"item_id":..., "target":..., "start":...}
    """

    def __getitem__(self, idx) -> np.ndarray:
        """Retrieves the context window for the given index within its corresponding time series."""

        real_idx = self.valid_idx[idx]
        item_id = self.item_ids[real_idx]
        timestamp = self.timestamps[real_idx].timestamp()
        item_start = self.indptr[item_id]
        pos_in_series = real_idx - item_start

        series = self.target_array[self.item_ids_mask[item_id]]
        # get series of corresponding item id
        series = self.target_array[self.item_ids_mask[item_id]]
        context = self._get_context(series[: pos_in_series + 1])

        return {"item_id": item_id, "target": context, "start": timestamp}


class AutogluonPredictor(AbstractPredictor):
    """
    Quantile Regression predictor for time series forecasting.

    This model fits separate quantile regression models for each quantile, lead time, and item ID.

    Parameters
    ----------
    quantiles : List[float], optional
        List of quantiles to predict. Defaults to [0.1, 0.2, ..., 0.9].
    lead_times : List[int], optional
        List of lead times (forecast horizons) to predict. Defaults to [1, 2, 3].
    freq : pd.Timedelta, optional
        Frequency of the time series data. Defaults to 1 hour.
    output_dir : Optional[Path], optional
        Directory to save the fitted model. Defaults to None.
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        lead_times: List[int] = Field(default_factory=lambda: [1, 2, 3]),
        freq: Union[pd.Timedelta, pd.DateOffset] = pd.Timedelta("1h"),
        output_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(lead_times, freq, output_dir)

        self.quantiles = quantiles
        self.predictor: TimeSeriesPredictor = None
        self.context_length = 512

    def _init_model(self):

        return TimeSeriesPredictor(
            prediction_length=self.prediction_length,
            freq=self.freq,
        )

    def _fit(self, data_train: TimeSeriesDataFrame, data_val: Optional[TimeSeriesDataFrame] = None) -> None:
        self.predictor = self._init_model()

        self.predictor.fit(
            train_data=data_train,
            tuning_data=data_val,
            hyperparameters={"PatchTST": {}},
        )  # This means: use the default PatchTST configuration

    def predict(
        self,
        data: TimeSeriesDataFrame,
        previous_context_data: Optional[TimeSeriesDataFrame] = None,
        rolling: bool = False,
        stride: int = 1,
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
        stride : int, default=1
            The stride to advance the sliding window when rolling=True.

        Returns
        -------
        ForecastCollection
            A nested dictionary mapping each item_id to lead time forecasts.
        """
        # Combine context data if given
        if previous_context_data is not None:
            # skip_first: Dict[item_id -> how many prepended rows], used for dataset indexing
            data_merged, skip_first = self._merge_data(data, previous_context_data, self.context_length)
        else:
            data_merged = data
            skip_first = None

        # Choose the appropriate dataset for single-shot or rolling prediction
        if rolling:
            ds = GluonTSDataset(
                data_merged,
                self.context_length,
                stride,
                skip_first,
                rolling=rolling,
            )

            dl = DataLoader(ds, batch_size=128)

            all_forecasts = []
            current_sample = 0

            for batch in tqdm(dl):
                gluonts_batch = ListDataset(
                    [
                        {"item_id": item_id, "target": target, "start": pd.to_datetime(start.item(), unit="s", utc=True), "feat_static_cat": [item_id]}
                        for (item_id, target, start) in zip(batch["item_id"], batch["target"], batch["start"])
                    ],
                    freq=self.freq,
                )

                # item_id refers to sample in this case
                batch_ts = TimeSeriesDataFrame.from_iterable_dataset(gluonts_batch)

                # update idxs
                idx = batch_ts.index
                # get unique old labels
                old_levels = idx.levels[0]
                # increment them
                new_levels = old_levels + current_sample

                # build new index with same codes but shifted labels
                new_index = pd.MultiIndex(levels=[new_levels, idx.levels[1]], codes=idx.codes, names=idx.names)

                batch_ts.index = new_index

                forecasts = self.predictor.predict(batch_ts, model="PatchTST")

                forecasts["item_ids"] = batch["item_id"].repeat_interleave(self.prediction_length)

                current_sample += len(batch["target"])

                all_forecasts.append(forecasts)

            all_forecasts = pd.concat(all_forecasts)

        else:
            all_forecasts = self.predictor.predict(data, model="PatchTST")
            ds = None

        quantile_names = [str(q) for q in self.quantiles]
        all_forecasts = np.stack([group[quantile_names].values.T for sample, group in all_forecasts.groupby(level=0)])

        assert all_forecasts.shape[0] == len(ds), "row count mismatch"

        # If rolling, output data covers all input rows
        if rolling:
            output_data = data
        else:
            # Only the most recent timestep per series
            output_data = data.slice_by_timestep(start_index=-1)

        return ds.to_forecast_collection(predictions=torch.tensor(all_forecasts), lead_times=self.lead_times, output_data=output_data, freq=self.freq)