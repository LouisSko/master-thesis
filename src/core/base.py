from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
from src.core.timeseries_evaluation import ForecastCollection, TimeSeriesForecast, TabularDataFrame
from autogluon.timeseries import TimeSeriesDataFrame
from typing import Dict, List, Optional, Type, Union, Literal, Iterable, Any
from pathlib import Path
import joblib
import logging
import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
import time
import importlib
from multiprocessing.resource_tracker import ResourceTracker
from pydantic import BaseModel, computed_field
import torch
from typing import Optional
from torch import nn
from transformers.utils import ModelOutput


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")

# Avoid resource_tracker ChildProcessError on shutdown
# Keep a reference to the original _stop method so we can call it later
_orig_stop = ResourceTracker._stop


def _safe_stop(self, *args, **kwargs):
    """
    Replacement for ResourceTracker._stop that suppresses the benign
    ChildProcessError which occurs if the tracker tries to reap a
    child PID that's already gone.
    """
    try:
        _orig_stop(self, *args, **kwargs)
    except ChildProcessError:
        # Ignore the error, since it simply means "no child processes"
        # are left to clean up—and it’s harmless.
        pass


# Overwrite the tracker’s stop method with our safe version
ResourceTracker._stop = _safe_stop


class AbstractPredictor(ABC):
    def __init__(
        self,
        lead_times: Optional[Iterable[int]] = None,
        name: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Parameters
        ----------
        lead_times : Optional[Iterable[int]], default=None
            An iterable of integers specifying the forecast lead times.
            If None, defaults to [1, 2, 3].

        name : Optional[str], default=None
            Optional human-readable name for the predictor.

        output_dir : Optional[Union[str, Path]], default=None
            Directory where model artifacts (e.g., checkpoints, logs) will be stored.
        """
        self.lead_times = self._parse_lead_times(lead_times)

        self.prediction_length = max(lead_times)
        self.output_dir = None
        self.name = name or self.__class__.__name__

        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir = output_dir / self.name
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logging.info("Using the following output directory to store results: %s", self.output_dir)
        else:
            self.output_dir = None
            logging.info("No output directory provided. Models will not be saved or loaded from disk.")
        self.train_time_seconds = None

    @staticmethod
    def _parse_lead_times(lead_times: Optional[Iterable[int]]) -> List[int]:
        # default
        if lead_times is None:
            lead_list = [1, 2, 3]
        else:
            # single integer → list
            if isinstance(lead_times, int):
                lead_list = [lead_times]
            else:
                # objects like numpy.ndarray, torch.Tensor, pandas.Series have .tolist()
                if hasattr(lead_times, "tolist") and not isinstance(lead_times, (str, bytes)):
                    lead_list = lead_times.tolist()
                else:
                    # try to cast any other iterable to list
                    try:
                        lead_list = list(lead_times)
                    except TypeError:
                        raise TypeError(f"lead_times must be an int or an iterable of ints, " f"got {type(lead_times).__name__}")

        # validate each element
        for i, lt in enumerate(lead_list):
            if not isinstance(lt, int):
                raise ValueError(f"lead_times[{i}] is not an integer: {lt!r}")
            if lt < 1:
                raise ValueError(f"lead_times[{i}] must be ≥ 1, got {lt}")

        # dedupe & sort
        return sorted(set(lead_list))

    def fit(
        self,
        data_train: TimeSeriesDataFrame,
        data_val: Optional[TimeSeriesDataFrame] = None,
        train_window_step: int = 1,
        val_window_step: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        data_train : TimeSeriesDataFrame
            Training data used to construct rolling windows for model training.
        data_val : Optional[TimeSeriesDataFrame], default=None
            Optional validation data used for early stopping and evaluation.
        train_window_step : int, default=1
            Number of time steps to shift the rolling window between training samples.
        val_window_step : Optional[int], default=None
            Number of time steps to shift the rolling window between validation samples.

        Note: Parameters are used only by certain predictors.
        """
        start_time = time.time()
        self._fit(
            data_train=data_train,
            data_val=data_val,
            train_window_step=train_window_step,
            val_window_step=val_window_step,
            **kwargs,
        )
        end_time = time.time()

        self.train_time_seconds = np.round(end_time - start_time, 2)

        logging.info("Time to fit %s in seconds: %s", self.__class__.__name__, self.train_time_seconds)

    @abstractmethod
    def _fit(
        self,
        data_train: TimeSeriesDataFrame,
        dat_val: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> None:
        pass

    def predict(
        self,
        data: TimeSeriesDataFrame,
        previous_context_data: Optional[TimeSeriesDataFrame] = None,
        rolling: bool = False,
        window_step: int = 1,
    ) -> ForecastCollection:
        start_time = time.time()
        forecasts = self._predict(data, previous_context_data, rolling, window_step)
        end_time = time.time()

        # add execution time
        forecasts.inference_time_seconds = np.round(end_time - start_time, 2)

        logging.info("%s: Time to generate forecasts in seconds: %s", self.__class__.__name__, forecasts.inference_time_seconds)
        return forecasts

    @abstractmethod
    def _predict(
        self,
        data: TimeSeriesDataFrame,
        previous_context_data: Optional[TimeSeriesDataFrame] = None,
        rolling: bool = False,
        window_step: int = 1,
    ) -> ForecastCollection:
        pass

    @property
    @abstractmethod
    def model_internal_prediction_length(self) -> int:
        """Length of prediction that the model can produce internally."""
        pass
    
    def _merge_data(
        self,
        data: TimeSeriesDataFrame,
        previous_context_data: TimeSeriesDataFrame,
        context_length=int,
    ) -> TimeSeriesDataFrame:
        """
        Merges previous context data with the new prediction data, ensuring time continuity and context length.

        Parameters
            data : TimeSeriesDataFrame
                New data for which predictions will be made.
            previous_context_data : TimeSeriesDataFrame
                Historical context data preceding `data`.
            context_length : int
                Context length to be considered.

        Returns
            TimeSeriesDataFrame: Combined and sorted dataframe used as input for prediction.

        Raises:
            ValueError: If any time series are not continuous between the two datasets.
        """

        # get item ids (unique time series)
        shared_ids = data.item_ids.intersection(previous_context_data.item_ids)

        freq_as_offset = pd.tseries.frequencies.to_offset(data.freq)

        # verify if there are gaps between data and previous_context_data
        for item_id in shared_ids:
            prev_series = previous_context_data.loc[item_id]
            curr_series = data.loc[item_id]

            if len(prev_series) == 0 or len(curr_series) == 0:
                continue

            last_prev_time = prev_series.index[-1]
            first_curr_time = curr_series.index[0]

            expected_next_time = last_prev_time + freq_as_offset
            if expected_next_time != first_curr_time:
                logging.warning(f"Data for item_id '{item_id}' is not consecutive. " f"Expected {expected_next_time}, got {first_curr_time}.")

        # add the context length to data
        previous_context_data = previous_context_data.loc[shared_ids]
        previous_context_data = previous_context_data.groupby("item_id").tail(context_length)
        prepended_timesteps_per_item = previous_context_data.groupby("item_id").size().to_dict()
        data_merged = pd.concat([previous_context_data, data]).sort_index()
        return TimeSeriesDataFrame(data_merged), prepended_timesteps_per_item

    def save(self, file_path: Optional[Path] = None) -> None:

        if file_path is None and self.output_dir is None:
            raise ValueError("No file path provided and no default output_dir set.")

        file_path = file_path or self.output_dir / f"{self.name}.joblib"

        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
            logging.info("Created new directory: %s", file_path.parent)

        joblib.dump(self, file_path)
        logging.info("%s successfully saved to: %s", self.name, file_path)


class AbstractPostprocessor(ABC):
    def __init__(self, output_dir: Optional[Path] = None, name: Optional[str] = None, n_jobs: int = 1) -> None:
        self.ignore_first_n_train_entries = 0
        self.name = name or self.__class__.__name__
        self.params = {}
        self.additional_info = {}
        self.n_jobs = n_jobs

        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir = output_dir / self.name
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logging.info("Using the following output directory to store results: %s", self.output_dir)
        else:
            self.output_dir = None
            logging.info("No output directory provided. Models will not be saved or loaded from disk.")

    def fit(self, data: ForecastCollection) -> None:
        """Fit postprocessor for each time series item and save the model if output_dir is set.

        Fit each series in parallel with joblib.
        """
        start_time = time.time()
        item_ids = data.get_item_ids()

        def _fit_one(item_id: int, forecast: TimeSeriesForecast):
            params = self._fit(forecast)
            self.save_model(params, item_id)
            return item_id, params

        # sequential processing
        if self.n_jobs == 1:
            for item_id in tqdm(data.get_item_ids(), desc=f"Fitting {self.name} for each time series (item)"):
                forecast = data.get_time_series_forecast(item_id)
                _, self.params[item_id] = _fit_one(item_id, forecast)

        # or parallelize
        else:
            # wrap the Parallel call in the tqdm_joblib context manager
            with tqdm_joblib(tqdm(desc=f"Fitting {self.name} for each time series (item)", total=len(item_ids))):
                results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                    delayed(_fit_one)(
                        iid,
                        data.get_time_series_forecast(iid),
                    )
                    for iid in item_ids
                )
            self.params = dict(results)  # collect back into self.params

        end_time = time.time()

        self.train_time_seconds = np.round(end_time - start_time, 2)

        logging.info("Time to fit %s in seconds: %s", self.name, self.train_time_seconds)

    def postprocess(self, data: ForecastCollection) -> ForecastCollection:
        """Apply postprocessor to each item using available or saved models."""
        start_time = time.time()
        results = {}
        # TODO: load all parameters at once.
        # TODO: parallelize postprocessing
        # TODO: If No params available execution takes really a long time
        for item_id in tqdm(data.get_item_ids(), desc=f"Postprocessing with {self.name}"):
            forecast = data.get_time_series_forecast(item_id)
            params = self.get_params(item_id)
            results[item_id] = self._postprocess(forecast, params)
        end_time = time.time()
        inference_time_seconds = np.round(end_time - start_time, 2)
        return ForecastCollection(item_ids=results, inference_time_seconds=inference_time_seconds)

    def get_params(self, item_id: Union[int, str]) -> Any:
        """Returns model parameters from memory or loads them from disk if output_dir is set."""
        if item_id in self.params:
            return self.params[item_id]
        if self.output_dir is not None:
            params = self.load_model(item_id)
            self.params[item_id] = params
            return params
        raise ValueError(f"No parameters available for item_id={item_id}. Either call `fit()` or provide an output_dir with saved models.")

    @abstractmethod
    def _fit(self, data: TimeSeriesForecast) -> Any:
        """Fit method to be implemented. Should return model parameters."""
        pass

    @abstractmethod
    def _postprocess(self, data: TimeSeriesForecast, params: Any) -> TimeSeriesForecast:
        """Apply postprocessing using provided parameters."""
        pass

    def _get_model_path(self, item_id: Union[int, str]) -> Path:
        return self.output_dir / "models" / f"model_item_id_{item_id}.joblib"

    def save_model(self, model: Any, item_id: int) -> None:
        """Save model for a specific item ID to disk using joblib."""
        save_dir = self._get_model_path(item_id)
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_dir)

    def load_model(self, item_id: int) -> Any:
        """Load model for a specific item id"""
        return joblib.load(self._get_model_path(item_id))

    def save(self, file_path: Optional[Path] = None) -> None:
        """Save entire Postprocessor"""
        if file_path is None and self.output_dir is None:
            raise ValueError("No file path provided and no default output_dir set.")

        file_path = file_path or self.output_dir / f"{self.name}.joblib"

        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
            logging.info("Created new directory: %s", file_path.parent)

        joblib.dump(self, file_path)
        logging.info("%s successfully saved to: %s", self.name, file_path)


class AbstractDataTransformer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """Fit the transformer if required (only for PowerTransformer)."""

    @abstractmethod
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Apply the forward transformation, with column-wise support for single-feature fits."""

    @abstractmethod
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Fit and transform in one step."""

    @abstractmethod
    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Invert the transformation."""


class AbstractPipeline(ABC):
    def __init__(
        self,
        model: Type[AbstractPredictor],
        model_kwargs: Dict,
        freq: Union[str, pd.DateOffset],
        postprocessors: Optional[List[Type[AbstractPostprocessor]]] = None,
        postprocessor_kwargs: Optional[List[Dict]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the pipeline with model, data, and optional postprocessor.

        Parameters
        ----------
        model : Type[AbstractPredictor]
            The predictor model class to use
        model_kwargs : Dict
            Keyword arguments to pass to the model constructor
        freq : Union[str, pd.DateOffset]
            Frequency of the data
        postprocessors : Optional[List[Type[AbstractPostprocessor]]], default=None
            Optional list of postprocessors for refining predictions
        output_dir : Optional[Union[str, Path]], default=None
            output directory for the pipeline.
        """

        if output_dir:
            self.output_dir = Path(output_dir)
        self.model = model
        self.model_kwargs = model_kwargs
        self.postprocessors = postprocessors
        self.postprocessor_kwargs = postprocessor_kwargs

        if isinstance(freq, str):
            self.freq = pd.tseries.frequencies.to_offset(freq)
        elif isinstance(freq, pd.DateOffset):
            self.freq = freq
        else:
            raise ValueError("freq needs to be a str or pd.DateOffset")

    @abstractmethod
    def backtest(
        self,
        data: Union[TimeSeriesDataFrame, TabularDataFrame],
        test_start_date: pd.Timestamp,
        test_end_date: Optional[pd.Timestamp] = None,
        rolling_window_eval: bool = False,
        train_window_size: Optional[pd.DateOffset] = None,
        val_window_size: Optional[pd.DateOffset] = None,
        test_window_size: Optional[pd.DateOffset] = None,
        train: bool = False,
        calibration_based_on: Optional[Union[Literal["val", "train", "train_val"], pd.DateOffset]] = None,
        save_results: bool = False,
    ) -> ForecastCollection:
        """
        Run a backtest over the specified time period.

        Parameters
        ----------
        data : Union[TimeSeriesDataFrame, TabularDataFrame]
            The dataset to use for training and prediction
        test_start_date : pd.Timestamp
            Start date of the test set.
        test_end_date : Optional[pd.Timestamp], optional
            End date of the test set. Defaults to last available timestamp.
        rolling_window_eval : bool, optional
            Whether to perform rolling window evaluation. Defaults to False.
        train_window_size : Optional[pd.DateOffset], optional
            Size of the training window. Defaults to None.
        val_window_size : Optional[pd.DateOffset], optional
            Size of the validation window. Defaults to None.
        test_window_size : Optional[pd.DateOffset], optional
            Size of the test window. Defaults to None.
        train : bool, optional
            Whether to train the model during backtesting. Defaults to False.
        calibration_based_on : Optional[Union[Literal["val", "train", "train_val"], pd.DateOffset]], optional
            Strategy for calibrating postprocessors. Defaults to None.
        save_results : bool
            Whether to save backtest prediction results. Defaults to False.

        Returns
        -------
        ForecastCollection
            The predictions over the backtest period.
        """

    @abstractmethod
    def train_predictor_model(
        self,
        data_train: Union[TimeSeriesDataFrame, TabularDataFrame],
        data_val: Optional[Union[TimeSeriesDataFrame, TabularDataFrame]] = None,
    ) -> None:
        """
        Train the predictor.

        Parameters
        ----------
        data_train : Union[TimeSeriesDataFrame, TabularDataFrame]
            The training data.
        data_val : Optional[Union[TimeSeriesDataFrame, TabularDataFrame]], optional
            Optional validation data. Defaults to None.

        Returns
        -------
        None
        """

    @abstractmethod
    def generate_forecasts(
        self,
        data_test: Union[TimeSeriesDataFrame, TabularDataFrame],
        data_previous_context: Optional[Union[TimeSeriesDataFrame, TabularDataFrame]] = None,
        rolling: bool = False,
        window_step: int = 1,
    ) -> Dict[str, ForecastCollection]:
        """
        Generates forecasts for each time series using the predictor.

        This method can perform either:
        - *single-shot prediction* (predicting from the most recent context window), or
        - *rolling backtesting* (sliding a window across the time series to predict at each time point).

        Parameters
        ----------
        data_test : Union[TimeSeriesDataFrame, TabularDataFrame]
            The test dataset.
        data_previous_context : Union[TimeSeriesDataFrame, TabularDataFrame]
            The previous context data. This is used by some predictors.
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
        Dict[str, ForecastCollection]
            Dictionary with predictions stored in a ForecastCollection object.
        """

    @abstractmethod
    def train_postprocessors(self, calibration_data: Union[TimeSeriesDataFrame, TabularDataFrame]) -> None:
        """
        Fit the postprocessors based on calibration data.

        Parameters
        ----------
        calibration_data : Union[TimeSeriesDataFrame, TabularDataFrame]]
            The calibration data. Used to fit the postprocessor.
        """

    @abstractmethod
    def apply_postprocessors_to_forecasts(self, predictions: Dict[str, ForecastCollection]) -> Dict[str, ForecastCollection]:
        """
        Apply postprocessing to forecasts.

        Parameters
        ----------
        predictions : Dict[str, ForecastCollection]
            The predictions. Keys correspond to the utilized model, e.g. `Chronos` or `QuantileRegression`

        Returns
        -------
        Dict[str, ForecastCollection]
            Dictionary with processed predictions.
        """

    def get_init_params(self) -> Dict:
        """Get the init params information"""
        return {
            "model": get_class_path(self.model),
            "model_kwargs": self.model_kwargs,
            "postprocessors": [get_class_path(postprocessor) for postprocessor in (self.postprocessors or [])],
            "postprocessors_kwargs": self.postprocessor_kwargs,
            "freq": self.freq,
        }


def get_class_path(cls: Type) -> str:
    """
    Get the full import path of a class.
    Example: mypackage.module.MyClass

    Parameters:
    -----------
    cls : Type
        The class to get the path from.

    Returns:
    --------
    str
        Full import path of the class.
    """
    return cls.__module__ + "." + cls.__name__


def load_class_from_path(class_path: str) -> Type:
    """
    Dynamically load a class from a full import path.

    Parameters:
    -----------
    class_path : str
        Full import path (e.g., 'mypackage.module.MyClass').

    Returns:
    --------
    Type
        The loaded class.
    """
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    logging.info("Load %s from %s", class_name, module)
    return getattr(module, class_name)


class ExecutionTimePredictor(BaseModel):
    predictor_name: str
    predictor_train_time: Optional[float]
    predictor_inference_time: float

    def get_total_train_time(self):
        return self.predictor_train_time

    def get_total_inference_time(self):
        return self.predictor_inference_time

    @computed_field
    def total_train_time(self) -> float:
        return self.get_total_train_time()

    @computed_field
    def total_inference_time(self) -> float:
        return self.get_total_inference_time()


class ExecutionTimePostprocessor(BaseModel):
    postprocessor_name: str
    execution_time_predictor: ExecutionTimePredictor
    calibration_inference_time: float
    postprocessor_train_time: float
    postprocessor_inference_time: float

    @property
    def predictor_name(self):
        return self.execution_time_predictor.predictor_name

    @property
    def predictor_train_time(self):
        return self.execution_time_predictor.predictor_train_time

    @property
    def predictor_inference_time(self):
        return self.execution_time_predictor.predictor_inference_time

    def get_total_train_time(self):
        predictor_train_time = self.predictor_train_time or 0
        t = predictor_train_time + self.calibration_inference_time + self.postprocessor_train_time
        return round(t, 2)

    def get_total_inference_time(self):
        t = self.predictor_inference_time + self.postprocessor_inference_time
        return round(t, 2)

    @computed_field
    def total_train_time(self) -> float:
        return self.get_total_train_time()

    @computed_field
    def total_inference_time(self) -> float:
        return self.get_total_inference_time()


def aggregate_execution_time_objects(
    objs: Iterable[Union["ExecutionTimePredictor", "ExecutionTimePostprocessor"]],
    *,
    skipna: bool = True,
) -> Union["ExecutionTimePredictor", "ExecutionTimePostprocessor"]:
    """
    Aggregate ExecutionTimePredictor or ExecutionTimePostprocessor objects by summing their times.

    Parameters
    ----------
    objs : iterable
        List of either ExecutionTimePredictor or ExecutionTimePostprocessor.
    skipna : bool
        If True, ignore None values when summing train times.

    Returns
    -------
    Aggregated ExecutionTimePredictor or ExecutionTimePostprocessor.
    """
    objs = list(objs)
    if not objs:
        raise ValueError("No execution time objects provided.")

    first = objs[0]
    is_post = isinstance(first, ExecutionTimePostprocessor)

    if not all(isinstance(o, type(first)) for o in objs):
        raise TypeError("All objects must be the same type (all predictors or all postprocessors).")

    if is_post:
        name = first.postprocessor_name
        preds = [o.execution_time_predictor for o in objs]

        # Aggregate postprocessor fields
        calibration_sum = round(sum(o.calibration_inference_time for o in objs), 2)
        post_train_sum = round(sum(o.postprocessor_train_time for o in objs), 2)
        post_inf_sum = round(sum(o.postprocessor_inference_time for o in objs), 2)

        # Reuse this function to aggregate predictors
        agg_pred = aggregate_execution_time_objects(preds, skipna=skipna)

        return ExecutionTimePostprocessor(
            postprocessor_name=name,
            execution_time_predictor=agg_pred,
            calibration_inference_time=calibration_sum,
            postprocessor_train_time=post_train_sum,
            postprocessor_inference_time=post_inf_sum,
        )

    else:
        # Predictor case
        names = {p.predictor_name for p in objs if p.predictor_name is not None}
        name = next(iter(names)) if len(names) == 1 else "+".join(sorted(names))

        train_times = [p.predictor_train_time for p in objs]
        if skipna:
            train_times = [t for t in train_times if t is not None]
        train_total = round(sum(train_times), 2) if train_times else None

        inf_total = round(sum(p.predictor_inference_time for p in objs), 2)

        return ExecutionTimePredictor(
            predictor_name=name,
            predictor_train_time=train_total,
            predictor_inference_time=inf_total,
        )


@dataclass
class ModelOutput(ModelOutput):
    quantile_preds: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None


class AbstractPytorchCalibrator(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        quantiles: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> "ModelOutput":
        raise NotImplementedError


class AbstractPytorchCalibrator(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        quantiles: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> "ModelOutput":
        raise NotImplementedError

    def fit(
        self,
        x: torch.Tensor,  # (T, H, Q)
        y_true: torch.Tensor,  # (T, H)
        quantiles: torch.Tensor,  # (Q,)
        num_steps: int = 1000,
        lr: float = 0.05,
        weight_decay: float = 0.0,
        lambda_noncross: float = 0.0,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ) -> torch.Tensor:  # returns invalid_h
        """
        Trains the calibrator in-place. Returns invalid_horizons_mask (H,)
        """
        assert x.dim() == 3 and y_true.dim() == 2
        T, H, Q = x.shape
        assert y_true.shape == (T, H)
        assert quantiles.shape == (Q,)

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        self.to(device)
        x = x.to(device)
        y_true = y_true.to(device, dtype=torch.float32)
        quantiles = quantiles.to(device, dtype=torch.float32)
        mask = ~torch.isnan(y_true)
        y_true = torch.where(mask, y_true, torch.zeros_like(y_true))

        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_loss = float("inf")
        patience, bad = 10, 0
        best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}

        for step in range(num_steps):
            opt.zero_grad()
            out = self(x, quantiles, target=y_true, mask=mask)
            loss = out.loss

            # Optional non-crossing penalty for quantile models
            if lambda_noncross > 0 and hasattr(out, "quantile_preds"):
                diff = out.quantile_preds[..., 1:] - out.quantile_preds[..., :-1]
                viol = torch.relu(-diff)
                loss = loss + lambda_noncross * viol.pow(2).mean()

            loss.backward()
            opt.step()

            if verbose and (step % 50 == 0 or step == num_steps - 1):
                print(f"step {step:4d}  loss {loss.item():.6f}")

            if loss.item() < best_loss - 1e-3:
                best_loss = loss.item()
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        self.load_state_dict(best_state)
        invalid_h = _horizons_with_no_data(mask)
        return invalid_h


@torch.no_grad()
def _horizons_with_no_data(mask: torch.Tensor) -> torch.Tensor:
    """Return boolean mask (H,) True where a horizon has no valid targets."""
    # mask: (T, H)
    return mask.sum(dim=0) == 0
