from abc import ABC, abstractmethod
from typing import List, Optional, Any
import pandas as pd
from src.core.timeseries_evaluation import ForecastCollection, TimeSeriesForecast, TabularDataFrame
from autogluon.timeseries import TimeSeriesDataFrame
from typing import Dict, List, Optional, Type, Union, Literal
from pathlib import Path
import joblib
import logging
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed, cpu_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")

class AbstractPredictor(ABC):
    def __init__(
        self,
        lead_times: List[int] = [1, 2, 3],
        freq: Union[pd.Timedelta, pd.DateOffset] = pd.Timedelta("1h"),
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:

        self.lead_times = lead_times
        self.freq = freq
        self.output_dir = None
        if output_dir:
            self.output_dir = Path(output_dir)

    @abstractmethod
    def fit(self, data_train: TimeSeriesDataFrame, dat_val: Optional[TimeSeriesDataFrame] = None) -> None:
        pass

    @abstractmethod
    def predict(self, data: TimeSeriesDataFrame, previous_context_data: Optional[TimeSeriesDataFrame] = None, predict_only_last_timestep: bool = False) -> ForecastCollection:
        pass

    def _merge_data(self, data: TimeSeriesDataFrame, previous_context_data: TimeSeriesDataFrame, context_length=int) -> TimeSeriesDataFrame:
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

        # verify if there are gaps between data and previous_context_data
        for item_id in shared_ids:
            prev_series = previous_context_data.loc[item_id]
            curr_series = data.loc[item_id]

            if len(prev_series) == 0 or len(curr_series) == 0:
                continue

            last_prev_time = prev_series.index[-1]
            first_curr_time = curr_series.index[0]

            expected_next_time = last_prev_time + self.freq
            if expected_next_time != first_curr_time:
                logging.warning(f"Data for item_id '{item_id}' is not consecutive. " f"Expected {expected_next_time}, got {first_curr_time}.")

        # add the context length to data
        previous_context_data = previous_context_data.loc[data.item_ids]
        previous_context_data = previous_context_data.groupby("item_id").tail(context_length)
        data_merged = pd.concat([previous_context_data, data]).sort_index()

        return TimeSeriesDataFrame(data_merged)

    def save(self, file_path: Optional[Path] = None) -> None:

        if file_path is None and self.output_dir is None:
            raise ValueError("No file path provided and no default output_dir set.")

        file_path = file_path or self.output_dir / f"{self.__class__.__name__}.joblib"

        joblib.dump(self, file_path)
        logging.info("%s successfully saved to: %s", self.__class__.__name__, file_path)


class AbstractPostprocessor(ABC):
    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.ignore_first_n_train_entries = 200
        self.class_name = self.__class__.__name__
        self.params = {}

        if output_dir is not None:
            self.output_dir = output_dir / self.class_name / "models"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logging.info("Using the following output directory to store results: %s", self.output_dir)
        else:
            self.output_dir = None
            logging.info("No output directory provided. Models will not be saved or loaded from disk.")

    def fit(self, data: ForecastCollection, n_jobs: Optional[int] = None) -> None:
        """Fit postprocessor for each time series item and save the model if output_dir is set.

        Fit each series in parallel with joblib.
        """

        n_jobs = n_jobs or min(4, cpu_count())

        item_ids = data.get_item_ids()

        # ensure output directory exists before forking
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        def _fit_one(item_id):
            # each process/worker will run this
            forecast = data.get_time_series_forecast(item_id)
            params = self._fit(forecast)
            if self.output_dir is not None:
                self.save_model(params, item_id)
            return item_id, params

        # wrap the Parallel call in the tqdm_joblib context manager
        with tqdm_joblib(tqdm(desc=f"Fitting {self.class_name}", total=len(item_ids))):
            results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_fit_one)(iid) for iid in item_ids)

        # collect back into self.params
        self.params = dict(results)

    def fit_single_thread(self, data: ForecastCollection) -> None:
        """Fit postprocessor for each time series item and save the model if output_dir is set."""
        for item_id in tqdm(data.get_item_ids(), desc=f"Fitting {self.class_name} for each time series (item)"):
            forecast = data.get_time_series_forecast(item_id)
            params = self._fit(forecast)
            self.params[item_id] = params

            if self.output_dir is not None:
                self.save_model(params, item_id)

    def postprocess(self, data: ForecastCollection) -> ForecastCollection:
        """Apply postprocessor to each item using available or saved models."""
        results = {}
        for item_id in tqdm(data.get_item_ids(), desc=f"Postprocessing with {self.class_name}"):
            forecast = data.get_time_series_forecast(item_id)
            params = self.get_params(item_id)
            results[item_id] = self._postprocess(forecast, params)
        return ForecastCollection(item_ids=results)

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
        return self.output_dir / f"models_item_id_{item_id}.joblib"

    def save_model(self, model: Any, item_id: int) -> None:
        """Save model for a specific item ID to disk using joblib."""
        joblib.dump(model, self._get_model_path(item_id))

    def load_model(self, item_id: int) -> Any:
        """Load model for a specific item id"""
        return joblib.load(self._get_model_path(item_id))

    def save(self, file_path: Path) -> None:
        joblib.dump(self, file_path)
        logging.info("%s successfully saved to: %s", self.__class__.__name__, file_path)


class AbstractPipeline(ABC):
    def __init__(
        self,
        model: Type[AbstractPredictor],
        model_kwargs: Dict,
        postprocessors: Optional[List[Type[AbstractPostprocessor]]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the pipeline with model, data, and optional postprocessor.

        Parameters
        ----------
        model : Type[AbstractPredictor]
            The predictor model class to use
        model_kwargs : Dict
            Keyword arguments to pass to the model constructor
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
    def train(
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
    def predict(
        self,
        data_test: Union[TimeSeriesDataFrame, TabularDataFrame],
        data_previous_context: Optional[Union[TimeSeriesDataFrame, TabularDataFrame]] = None,
    ) -> Dict[str, ForecastCollection]:
        """
        predict on the test data.

        Parameters
        ----------
        data_test : Union[TimeSeriesDataFrame, TabularDataFrame]
            The test dataset.
        data_previous_context : Union[TimeSeriesDataFrame, TabularDataFrame]
            The previous context data. This is used by some predictors.
        Returns
        -------
        Dict[str, ForecastCollection]
            Dictionary with raw predictions.
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
    def apply_postprocessing(self, predictions: Dict[str, ForecastCollection]) -> Dict[str, ForecastCollection]:
        """
        Apply postprocessing to predictions.

        Parameters
        ----------
        predictions : Dict[str, ForecastCollection]
            The predictions. Keys correspond to the utilized model, e.g. `Chronos` or `QuantileRegression`

        Returns
        -------
        Dict[str, ForecastCollection]
            Dictionary with processed predictions.
        """

    def get_config(
        self,
        backtest_params: Dict,
        additional_config_info: Optional[Dict] = None,
    ) -> Dict:
        """Save configuration information to a JSON file.

        Parameters
        ----------
        backtest_params : Dict
            Dictionary containing the backtest parameters
        additional_config_info : Optional[Dict], default=None
            Optional additional information to include in the config file

        Returns
        -------

        Dict
            A dictionary containing the configuration
        """

        # Create base config with model information
        config = {
            "init_params": {
                "model": self.model.__name__,
                "model_kwargs": self.model_kwargs,
                "postprocessors": [p.__name__ for p in self.postprocessors] if self.postprocessors is not None else None,
            }
        }

        # Add backtest parameters
        config.update({"backtest_params": backtest_params})

        # Add any additional information
        if additional_config_info:
            config.update({"additional_info": additional_config_info})

        return config
