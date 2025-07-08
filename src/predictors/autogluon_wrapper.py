from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from typing import List, Optional, Union
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
from src.core.timeseries_evaluation import ForecastCollection
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")

# TODO: reload trained models


class GluonTSDataset(BaseTimeSeriesDataset):
    """
    Yields dicts for GluonTS: {"item_id":..., "target":..., "start":...}
    """

    def __getitem__(self, idx) -> np.ndarray:
        """
        Retrieve the context window for the specified index.

        Parameters
        ----------
        idx : int
            Index within the dataset.

        Returns
        -------
        dict
            A dictionary containing:
            - 'item_id' : int
                The ID of the time series this observation belongs to.
            - 'target' : np.ndarray
                The array of target values up to and including the given position.
            - 'start' : float
                The timestamp (POSIX float) of the first observation in the window.
        """
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
    Wrapper around AutoGluon TimeSeriesPredictor.

    Only supports univariate time series.
    Currently does not support:
        - known_covariates
        - past covariates
        - static_features

    Parameters
    ----------
    quantiles : list of float, optional
        List of quantiles to predict. Defaults to [0.1, 0.2, ..., 0.9].
    lead_times : list of int, optional
        List of lead times (forecast horizons) to predict. Defaults to [1, 2, 3].
    freq : str or pd.DateOffset, optional
        Frequency of the time series data. Defaults to '1h'.
    output_dir : Path or None, optional
        Directory to save the fitted model. Defaults to None.
    predictor_kwargs : dict or None, optional
        Additional keyword arguments passed to TimeSeriesPredictor constructor.
    predict_kwargs : dict or None, optional
        Additional keyword arguments passed to TimeSeriesPredictor.predict.
    fit_kwargs : dict or None, optional
        Additional keyword arguments passed to TimeSeriesPredictor.fit.
    context_length : int or None, optional
        Max context length of the time series produced by the dataset/dataloader during predictions.

    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        lead_times: List[int] = Field(default_factory=lambda: [1, 2, 3]),
        freq: Union[str, pd.DateOffset] = "1h",
        output_dir: Optional[Path] = None,
        predictor_kwargs: Optional[dict] = None,
        predict_kwargs: Optional[dict] = None,
        fit_kwargs: Optional[dict] = None,
        context_length: Optional[int] = 512,
    ) -> None:
        super().__init__(lead_times, output_dir)

        self.quantiles = quantiles
        self.predictor: TimeSeriesPredictor = None
        self.predictor_kwargs = predictor_kwargs or {}
        self.fit_kwargs = fit_kwargs or {}
        self.predict_kwargs = predict_kwargs or {}
        self.context_length = context_length
        self.freq = freq

    def _init_model(self):

        return TimeSeriesPredictor(
            prediction_length=self.prediction_length,
            freq=self.freq,
            **self.predictor_kwargs,
        )

    def _fit(self, data_train: TimeSeriesDataFrame, data_val: Optional[TimeSeriesDataFrame] = None) -> None:
        self.predictor = self._init_model()

        self.predictor.fit(
            train_data=data_train,
            tuning_data=data_val,
            verbosity=4,
            **self.fit_kwargs,
        )

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

        ds = GluonTSDataset(
            data_merged,
            self.context_length,
            stride,
            skip_first,
            rolling=rolling,
        )

        dl = DataLoader(ds, batch_size=512)

        all_forecast_chunks = []
        next_item_id = 0

        for batch in tqdm(dl, desc="Predicting rolling windows"):
            # Build ListDataset
            list_ds = ListDataset(
                [
                    {
                        "target": target.numpy(),
                        "start": pd.Timestamp.utcfromtimestamp(float(start)),
                    }
                    for target, start in zip(batch["target"], batch["start"])
                ],
                freq=self.freq,
            )

            ts_data = TimeSeriesDataFrame.from_iterable_dataset(list_ds)

            # Reindex: shift item_ids by next_item_id
            idx = ts_data.index
            # Build new MultiIndex
            new_index = pd.MultiIndex(
                levels=[idx.levels[0] + next_item_id, idx.levels[1]],
                codes=idx.codes,
                names=idx.names,
            )
            ts_data.index = new_index

            forecasts_df = self.predictor.predict(ts_data, **self.predict_kwargs)

            all_forecast_chunks.append(forecasts_df)

            next_item_id += len(ts_data.item_ids)

        all_forecasts_df = pd.concat(all_forecast_chunks)

        quantile_names = [str(q) for q in self.quantiles]
        forecasts_array = np.stack([group[quantile_names].values.T for sample, group in all_forecasts_df.groupby(level=0)])

        assert forecasts_array.shape[0] == len(ds), "row count mismatch"

        # If rolling, output data covers all input rows
        if rolling:
            output_data = data
        else:
            # Only the most recent timestep per series
            output_data = data.slice_by_timestep(start_index=-1)

        return ds.to_forecast_collection(predictions=torch.tensor(forecasts_array), lead_times=self.lead_times, output_data=output_data)


class PatchTST_Ag(AutogluonPredictor):

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        lead_times: List[int] = Field(default_factory=lambda: [1, 2, 3]),
        freq: Union[str, pd.DateOffset] = "1h",
        output_dir: Optional[Path] = None,
    ) -> None:

        predictor_kwargs = {}
        fit_kwargs = {"hyperparameters": {"PatchTST": {}}}
        predict_kwargs = {"model": "PatchTST"}
        super().__init__(quantiles, lead_times, freq, output_dir, predictor_kwargs, predict_kwargs, fit_kwargs, 96)


class TiDE_Ag(AutogluonPredictor):

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        lead_times: List[int] = Field(default_factory=lambda: [1, 2, 3]),
        freq: Union[str, pd.DateOffset] = "1h",
        output_dir: Optional[Path] = None,
    ) -> None:

        predictor_kwargs = {}
        fit_kwargs = {"hyperparameters": {"TiDE": {}}}
        predict_kwargs = {"model": "TiDE"}
        super().__init__(quantiles, lead_times, freq, output_dir, predictor_kwargs, predict_kwargs, fit_kwargs, 512)


class SeasonalNaive_Ag(AutogluonPredictor):

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        lead_times: List[int] = Field(default_factory=lambda: [1, 2, 3]),
        freq: Union[str, pd.DateOffset] = "1h",
        seasonal_period: int = 7,
        output_dir: Optional[Path] = None,
    ) -> None:

        predictor_kwargs = {}
        fit_kwargs = {"hyperparameters": {"SeasonalNaive": {"seasonal_period": seasonal_period}}}
        predict_kwargs = {}
        super().__init__(quantiles, lead_times, freq, output_dir, predictor_kwargs, predict_kwargs, fit_kwargs, 2048)


class Chronos_Ag(AutogluonPredictor):

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        lead_times: List[int] = Field(default_factory=lambda: [1, 2, 3]),
        freq: Union[str, pd.DateOffset] = "1h",
        output_dir: Optional[Path] = None,
    ) -> None:

        predictor_kwargs = {}
        fit_kwargs = {"hyperparameters": {"Chronos": {"model_path": "amazon/chronos-bolt-tiny"}}}
        predict_kwargs = {}
        super().__init__(quantiles, lead_times, freq, output_dir, predictor_kwargs, predict_kwargs, fit_kwargs, 2048)
