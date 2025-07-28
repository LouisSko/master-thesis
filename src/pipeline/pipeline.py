import torch
from typing import Dict, List, Optional, Type, Union, Literal, Tuple
from src.core.timeseries_evaluation import (
    ForecastCollection,
    TimeSeriesForecast,
    HorizonForecast,
    TabularDataFrame,
    DIR_BACKTESTS,
    DIR_MODELS,
    DIR_POSTPROCESSORS,
    ITEMID,
    TARGET,
    PIPELINE_CONFIG_FILE_NAME,
    BACKTEST_CONFIG_FILENAME,
    PREDICTIONS_FILENAME,
    EVAL_CONFIG_FILENAME,
)
from src.core.base import AbstractPostprocessor, AbstractPredictor, ExecutionTimePostprocessor, ExecutionTimePredictor, load_class_from_path, aggregate_execution_time_objects
from src.core.utils import CustomJSONEncoder, set_global_seed
from autogluon.timeseries import TimeSeriesDataFrame
import pandas as pd
from src.core.base import AbstractPipeline
from pathlib import Path
import json
from typing import Type
import joblib
import logging
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")
set_global_seed()


class ForecastingPipeline(AbstractPipeline):
    def __init__(
        self,
        model: Type[AbstractPredictor],
        model_kwargs: Dict,
        freq: Union[str, pd.DateOffset],
        postprocessors: Optional[List[Type[AbstractPostprocessor]]] = None,
        postprocessor_kwargs: Optional[List[Dict]] = None,
        output_dir: Optional[Union[Path, str]] = None,
    ):
        super().__init__(model, model_kwargs, freq, postprocessors, postprocessor_kwargs, output_dir)

        # define storage directory
        self.pipeline_dir_models = self.output_dir / DIR_MODELS
        self.pipeline_dir_postprocessors = self.output_dir / DIR_POSTPROCESSORS
        self.pipeline_dir_backtests = self.output_dir / DIR_BACKTESTS

        # add output directory
        self.postprocessors = postprocessors
        self.model = model
        self.model_kwargs = model_kwargs

        # Need to be fitted
        self.predictor = None
        self.postprocessor_dict: Dict[str, AbstractPostprocessor] = {}

        # create predictor and postprocessor
        self._initialize_predictor()
        if self.postprocessors is not None:
            self._initialize_postprocessors()

    def _initialize_predictor(self):
        self.model_kwargs.update({"output_dir": self.pipeline_dir_models})
        self.predictor = self.model(**self.model_kwargs)

    def _initialize_postprocessors(self):
        if self.postprocessors is None:
            raise ValueError("No postprocessors specified.")

        if self.postprocessor_kwargs is None:
            self.postprocessor_kwargs = [{} for p in self.postprocessors]

        assert len(self.postprocessor_kwargs) == len(self.postprocessors)

        for p_kwarg in self.postprocessor_kwargs:
            p_kwarg.update({"output_dir": self.pipeline_dir_postprocessors})

        self.postprocessor_dict: Dict[str, AbstractPostprocessor] = {}
        for pp, kwargs in zip(self.postprocessors, self.postprocessor_kwargs):
            pp_instance = pp(**kwargs)
            self.postprocessor_dict.update({pp_instance.name: pp_instance})

    def save_pipeline(self) -> None:
        """Save the pipeline configuration to a JSON file and store predictors and postprocessors as joblib."""

        logging.info("Saving Pipeline to specified output directory: %s", self.output_dir)

        config = self.get_init_params()

        # save pipeline configuration
        config_file_path = self.output_dir / PIPELINE_CONFIG_FILE_NAME
        with open(config_file_path, "w") as f:
            json.dump(config, f, indent=4, cls=CustomJSONEncoder)
        logging.info("Pipeline configuration saved to: %s", config_file_path)

        self.predictor.save()

        if self.postprocessor_dict is not None:
            for postprocessor in self.postprocessor_dict.values():
                postprocessor.save()

        logging.info('Pipeline saved successfully. Reload Pipeline using: ForecastingPipeline.from_pretrained("%s")', self.output_dir)

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> "ForecastingPipeline":
        """
        Load a ForecastingPipeline from a saved directory.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the directory containing the saved pipeline configuration, model, and postprocessors.

        Returns
        -------
        ForecastingPipeline
            The loaded ForecastingPipeline instance.
        """
        pipeline_dir = Path(path)

        with open(pipeline_dir / PIPELINE_CONFIG_FILE_NAME, "r") as f:
            config: dict = json.load(f)

        # Load model class
        config["model"] = load_class_from_path(config["model"])

        # Handle special fields in model_kwargs
        model_kwargs = config.get("model_kwargs", {})
        if "freq" in model_kwargs:  # TODO: also support DateOffset
            model_kwargs["freq"] = pd.Timedelta(model_kwargs["freq"])
        config["model_kwargs"] = model_kwargs

        # Load postprocessor classes
        config["postprocessors"] = [load_class_from_path(pp) for pp in config.get("postprocessors", [])]

        # Recreate the pipeline
        pipeline = ForecastingPipeline(
            model=config["model"],
            model_kwargs=config["model_kwargs"],
            postprocessors=config["postprocessors"],
            postprocessor_kwargs=config["postprocessors_kwargs"],
            output_dir=pipeline_dir,
            freq=config["freq"]["freqstr"],
        )

        # Load predictor
        models = list((pipeline_dir / DIR_MODELS).rglob("*.joblib"))
        if not models:
            logging.info("No models found in %s. Skipping model loading.", pipeline_dir / DIR_MODELS)
        else:
            pipeline.predictor = joblib.load(models[0])

        # Load postprocessors TODO: this might be not correct, since we save individual models as joblib as well
        postprocessor_files = list((pipeline_dir / DIR_POSTPROCESSORS).rglob("*.joblib"))
        if not postprocessor_files:
            logging.info("No postprocessors found in %s. Skipping postprocessor loading.", pipeline_dir / DIR_POSTPROCESSORS)
        else:
            for pp_file in postprocessor_files:
                name = pp_file.stem
                pipeline.postprocessor_dict[name] = joblib.load(pp_file)

        return pipeline

    def backtest(
        self,
        data: Union[TimeSeriesDataFrame, TabularDataFrame],
        test_start_date: pd.Timestamp,
        test_end_date: Optional[pd.Timestamp] = None,
        rolling_window_eval: bool = False,
        train_window_size: Optional[pd.DateOffset] = None,
        val_window_size: Optional[pd.DateOffset] = None,
        test_window_size: Optional[pd.DateOffset] = None,
        test_window_step: int = 1,
        calibration_window_step: int = 1,
        train: bool = False,
        calibration_based_on: Optional[Union[Literal["val", "train", "train_val"], pd.DateOffset]] = None,
        save_results: bool = False,
    ) -> Tuple[Dict[str, ForecastCollection], Dict]:
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
            Only the latest model/postprocessor get saved.
        train_window_size : Optional[pd.DateOffset], optional
            Size of the training window. Defaults to None.
        val_window_size : Optional[pd.DateOffset], optional
            Size of the validation window. Defaults to None.
        test_window_size : Optional[pd.DateOffset], optional
            Size of the test window. Defaults to None.
        test_window_step : int, Defaults to 1.
            The number of time steps to move the sliding prediction window forward between each prediction.
            This controls how densely forecasts are generated across time. A smaller value creates more overlapping
            forecasts, while a larger value skips more observations between windows.
            The rolling procedure is applied independently to each time series in the dataset.
        calibration_window_step : int, Defaults to 1.
            The number of time steps to move the sliding prediction window forward between each prediction on the calibration dataset.
            Smaller values creates more calibration datapoints, while a larger value speeds up calibration process.
        train : bool, optional
            Whether to train the model during backtesting. Defaults to False.
        calibration_based_on : Optional[Union[Literal["val", "train", "train_val"], pd.DateOffset]], optional
            Strategy for calibrating postprocessors. Defaults to None.
        save_results : bool
            Whether to save backtest results. Defaults to False.

        Returns
        -------
        Tuple[Dict[str, ForecastCollection], Dict]
                    A tuple containing:
                    - A dictionary containing the forecasts for predictors/postprocessors inside a dict.
                    - An info dictionary containing metadata and details about the backtest.
        """

        logging.info("Start E2E backtesting...")

        results = {}
        execution_times = {}

        if test_end_date is None:
            test_end_date = data.index.get_level_values("timestamp").max()

        if rolling_window_eval:
            if not test_window_size:
                logging.info("test_window_size not specified. Set it to 1 year as default")
                test_window_size = pd.DateOffset(years=1)

            i = 0
            while test_start_date < test_end_date:
                results[test_start_date], execution_times[f"backtest_{i}"] = self._run_backtest_iteration(
                    data,
                    test_start_date,
                    train,
                    train_window_size,
                    val_window_size,
                    test_window_size,
                    test_window_step,
                    calibration_window_step,
                    calibration_based_on,
                )
                execution_times[f"backtest_{i}"]["start"] = test_start_date
                test_start_date += test_window_size
                execution_times[f"backtest_{i}"]["end"] = test_start_date
                i += 1

            results = self._combine_backtest_results(results)
            execution_times = self._combine_execution_time(execution_times)

        else:
            results, execution_times = self._run_backtest_iteration(
                data,
                test_start_date,
                train,
                train_window_size,
                val_window_size,
                test_window_size,
                test_window_step,
                calibration_window_step,
                calibration_based_on,
            )

        if save_results:
            backtest_params = {
                "test_start_date": test_start_date,
                "test_end_date": test_end_date,
                "rolling_window_eval": rolling_window_eval,
                "train_window_size": train_window_size,
                "val_window_size": val_window_size,
                "test_window_size": test_window_size,
                "test_window_step": test_window_step,
                "calibration_window_step": calibration_window_step,
                "train": train,
                "calibration_based_on": calibration_based_on,
            }
            self._store_backtest_outputs(results, backtest_params, execution_times)

        return results, execution_times

    def _store_backtest_outputs(self, results: Dict[str, ForecastCollection], backtest_params: Dict, execution_times: Dict) -> None:
        """Save backtest results and config."""

        logging.info("Storing backtest results...")

        config = {
            "pipeline_config": self.get_init_params(),
            "backtest_params": backtest_params,
        }

        # store general results
        save_path = self.pipeline_dir_backtests
        create_dir(save_path)
        config_path = save_path / BACKTEST_CONFIG_FILENAME
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4, cls=CustomJSONEncoder)
        logging.info("Saved pipeline and backtest configuration to: %s.", config_path)

        for method, result in results.items():
            save_path = self.pipeline_dir_backtests / method
            create_dir(save_path)

            # Add basic information
            eval_config = {}
            eval_config = {"method": method}
            eval_config.update(result.get_crps(mean_time=True, mean_lead_times=True, mean_item_ids=True).to_dict())
            eval_config.update(result.get_empirical_coverage_rates(mean_lead_times=True).to_dict())
            eval_config.update(result.get_quantile_scores(mean_lead_times=True).to_dict())
            # add execution time
            eval_config["execution_time"] = execution_times[method]

            # Save config
            config_path = save_path / EVAL_CONFIG_FILENAME
            with open(config_path, "w") as f:
                json.dump(eval_config, f, indent=4, cls=CustomJSONEncoder)
            logging.info("Saved backtest evaluation results for `%s` to: %s.", method, config_path)
            # Save predictions
            result.save(save_path / PREDICTIONS_FILENAME)

    def split_time_series_data(
        self,
        data: Union[TimeSeriesDataFrame, TabularDataFrame],
        test_start_date: pd.Timestamp,
        train_window_size: Optional[pd.DateOffset],
        val_window_size: Optional[pd.DateOffset],
        test_window_size: Optional[pd.DateOffset],
    ) -> Tuple[Union[TimeSeriesDataFrame, TabularDataFrame], Union[TimeSeriesDataFrame, TabularDataFrame, None], Union[TimeSeriesDataFrame, TabularDataFrame]]:
        """
        Split data into training, validation, and testing sets.

        Parameters
        ----------
        data : Union[TimeSeriesDataFrame, TabularDataFrame]
            The dataset to use for training and prediction
        test_start_date : pd.Timestamp
            Start date of the test set.
        train_window_size : Optional[pd.DateOffset]
            Size of the training window.
        val_window_size : Optional[pd.DateOffset]
            Size of the validation window.
        test_window_size : Optional[pd.DateOffset]
            Size of the test window.

        Returns
        -------
        Tuple[data_train, data_val, data_test]
            Split datasets for training, validation (optional), and testing.
        """
        logging.info("Starting data split operation.")

        data_val = None
        logging.debug("Splitting data based on the test start date: %s", test_start_date)
        data_train, data_test = data.split_by_time(test_start_date)
        logging.debug("Training set and testing set created.")

        if val_window_size:
            logging.debug("Splitting training data for validation with window size: %s", val_window_size)
            data_train, data_val = data_train.split_by_time(test_start_date - val_window_size)
            logging.debug("Validation set created.")

        if train_window_size:
            logging.debug("Adjusting training data based on window size: %s", train_window_size)
            _, data_train = data_train.split_by_time(test_start_date - train_window_size)
            logging.debug("Training set adjusted.")

        if test_window_size:
            logging.debug("Adjusting test set based on window size: %s", test_window_size)
            data_test, _ = data_test.split_by_time(test_start_date + test_window_size)
            logging.debug("Test set adjusted.")

        logging.info("Data split operation completed successfully.")

        return data_train, data_val, data_test

    def validate_data(self, data: TimeSeriesDataFrame):

        if not data.index.is_monotonic_increasing:
            logging.info("Data index is not monotonic increasing. Reorder dataframe")
            data = data.sort_index()
            data.freq = None

        # check frequency of the df
        if data.freq != self.freq.freqstr:
            logging.info("CAUTION: Frequency of data '%s' does not match frequency defined in the pipeline '%s'.", data.freq, self.freq.freqstr)
            data = data.convert_frequency(self.freq)
            logging.info("Data resampled to %s", data.freq)
        return data

    def auto_split_train_val(
        self,
        data: TimeSeriesDataFrame,
        prediction_length: int,
        min_val_windows: int = 1,
        max_val_windows: int = 10,
        min_train_windows: int = 1,
        min_train_fraction: float = 0.7,
    ) -> Tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame], int]:
        """
        Split time series data into training and validation sets using a sliding window approach.

        Enforces at least `min_val_windows` in the validation set. Attempts to increase the number
        of validation windows (up to `max_val_windows`) as long as the training set satisfies:
        - At least `min_train_windows`
        - Preferably also `min_train_fraction` of the total series length

        The `min_train_fraction` constraint is relaxed if necessary to allow `min_val_windows`.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            Full dataset to be split.
        prediction_length : int
            Size of each prediction window.
        min_val_windows : int
            Minimum number of validation windows (mandatory).
        max_val_windows : int
            Maximum number of validation windows to consider.
        min_train_windows : int
            Minimum number of prediction windows in training set.
        min_train_fraction : float
            Preferred minimum fraction of series reserved for training.

        Returns
        -------
        data_train : TimeSeriesDataFrame
            Training portion of the data.
        data_val : Optional[TimeSeriesDataFrame]
            Validation portion of the data, or None if no valid split is possible.
        val_windows : int
            Number of validation windows
        """

        logging.info("Automatically splitting data into training and validation data...")
        timesteps_per_item = data.num_timesteps_per_item()
        min_timesteps = timesteps_per_item.min()
        max_timesteps = timesteps_per_item.max()
        if min_timesteps != max_timesteps:
            logging.warning(
                "Time series have varying lengths: min=%d, max=%d. " "Splitting is based on the longest series only.",
                min_timesteps,
                max_timesteps,
            )

        min_train_len = prediction_length * min_train_windows
        preferred_train_len = max(min_train_len, int(max_timesteps * min_train_fraction))
        max_possible_val_windows = (max_timesteps - min_train_len) // prediction_length
        effective_max_val_windows = min(max_val_windows, max_possible_val_windows)

        if max_timesteps < prediction_length * (min_val_windows + min_train_windows):
            logging.warning(
                "Not enough timesteps to create train/val split. " "Required at least %d, but got %d.",
                prediction_length * (min_val_windows + min_train_windows),
                max_timesteps,
            )
            return data, None, 1

        best_w = None
        for w in reversed(range(min_val_windows, effective_max_val_windows + 1)):
            val_len = prediction_length * w
            train_len = max_timesteps - val_len

            if train_len >= min_train_len:
                # Acceptable: we always meet min_train_windows
                # Prefer: also meets min_train_fraction
                if train_len >= preferred_train_len or w == min_val_windows:
                    best_w = w
                    break

        if best_w is None:
            logging.warning("Could not find a suitable validation split. Keeping all data for training.")
            return data, None, 1

        split_idx = int(prediction_length * best_w)
        data_val = data.slice_by_timestep(start_index=-split_idx)
        data_train = data.slice_by_timestep(end_index=-split_idx)

        val_pct = len(data_val) / (len(data_train) + len(data_val)) * 100
        train_pct = 100 - val_pct
        num_val_windows = split_idx // prediction_length
        num_train_windows = (max_timesteps - split_idx) // prediction_length
        logging.info(
            "Split result: %d timesteps for training (%.1f%%), %d timesteps for validation (%.1f%%)",
            len(data_train),
            train_pct,
            len(data_val),
            val_pct,
        )
        logging.info(
            "Split sizes per series: %d raw values for training, %d for validation",
            max_timesteps - split_idx,
            split_idx,
        )
        logging.info(
            "Sliding windows: %d training windows, %d validation windows (window size = %d)",
            num_train_windows,
            num_val_windows,
            prediction_length,
        )
        return data_train, data_val, best_w

    def train_predictor_model(
        self,
        data_train: Union[TimeSeriesDataFrame, TabularDataFrame],
        data_val: Optional[Union[TimeSeriesDataFrame, TabularDataFrame]] = None,
        train_window_step: int = 1,
        val_window_step: Optional[int] = None,
        auto_determine_val_set: bool = True,
        max_val_windows: int = 1,
    ) -> None:
        """
        Trains the predictor using the provided training data, optionally with validation data.

        Supports both manual and automatic validation data generation. If no validation data is
        provided and `auto_determine_val_set=True`, a validation set will be split off from the
        tail end of the training data. Sliding window sampling is used for both training and
        validation sets, with configurable step sizes.

        Parameters
        ----------
        data_train : Union[TimeSeriesDataFrame, TabularDataFrame]
            The dataset used for training the predictor. Must include the target variable(s).
        data_val : Optional[Union[TimeSeriesDataFrame, TabularDataFrame]], default=None
            Optional dataset used for validation during training. If None and
            `auto_determine_val_set=True`, a validation set will be split automatically from `data_train`.
        train_window_step : int, default=1
            The number of time steps to move the sliding window forward between each training sample.
            Smaller values create more overlapping windows (more training samples); larger values
            create sparser training data.
        val_window_step : Optional[int], default=None
            The step size for generating rolling windows on the validation set. If None, defaults to
            the model's `prediction_length`. Ignored if `auto_determine_val_set=True`.
        auto_determine_val_set : bool, default=True
            Whether to automatically create a validation set from the training data if `data_val`
            is not provided based on `max_val_windows`.
        max_val_windows : int, default=1
            The maximum number of forecast windows to include in the automatically generated validation set.
            Only used when `auto_determine_val_set=True`.
            A smaller number might be chosen, if not enough data is available.

        Returns
        -------
        None
        """
        logging.info("Initializing predictor with model: %s", self.model.__name__)
        self._initialize_predictor()

        data_train = self.validate_data(data_train)

        if data_val is not None:
            logging.info("Validation data is provided.")
            data_val = self.validate_data(data_val)

        elif auto_determine_val_set:
            logging.info("Inferring validation set from training data...")
            data_train, data_val, val_windows = self.auto_split_train_val(data_train, self.predictor.prediction_length, max_val_windows=max_val_windows)
            val_window_step = self.predictor.prediction_length  # TODO: potentially not hardcode this but change it based on number of validation windows
        logging.info("Training data from %s to %s", data_train.index.get_level_values("timestamp").min(), data_train.index.get_level_values("timestamp").max())

        if data_val is not None:
            logging.info("Validation data from %s to %s", data_val.index.get_level_values("timestamp").min(), data_val.index.get_level_values("timestamp").max())
        else:
            logging.info("No validation data will be used.")

        logging.info("Fitting predictor to the training data...")
        self.predictor.fit(data_train, data_val, train_window_step, val_window_step)

    def generate_forecasts(
        self,
        data_test: Union[TimeSeriesDataFrame, TabularDataFrame],
        data_previous_context: Optional[Union[TimeSeriesDataFrame, TabularDataFrame]] = None,
        rolling: bool = False,
        window_step: int = 1,
        max_calibration_samples: Optional[int] = None,
    ) -> Dict[str, ForecastCollection]:
        """
        Generates forecasts using the predictor, supporting both single-shot and rolling modes.

        Parameters
        ----------
        data_test : Union[TimeSeriesDataFrame, TabularDataFrame]
            The dataset to generate forecasts on. Must include the target values.
        data_previous_context : Optional[Union[TimeSeriesDataFrame, TabularDataFrame]], default=None
            Optional historical context preceding `data_test`. Required by some models.
        rolling : bool, default=False
            Whether to perform rolling forecasts across all time steps, or just a single-shot forecast
            using the latest available context window.
        window_step : int, default=1
            Step size for rolling forecast windows. Smaller values create denser forecasts.
            **Note**: This parameter is ignored if `max_calibration_samples` is set.
        max_calibration_samples : Optional[int], default=None
            If provided, automatically creates a adjusts `data_test` and `window_step` so that at most this many
            rolling window predictions are generated. This is typically used when forecasts are needed for
            downstream calibration postprocessors. In this case, the `window_step` argument is ignored. If not set, the full `data_test` is used
            as-is with the specified `window_step`.

        Returns
        -------
        Dict[str, ForecastCollection]
            Dictionary mapping the predictor name to its generated ForecastCollection.
        """

        def _get_divisors(n: int) -> List[int]:
            """Returns all positive divisors of `n`, sorted ascending."""
            return [i for i in range(1, n + 1) if n % i == 0]

        def _compute_window_step(samples: int, prediction_length: int, max_samples: int) -> int:
            """
            Computes the largest possible window step size such that the number of rolling
            forecast windows does not exceed a specified maximum.

            The function tries all divisors of `prediction_length` (to ensure alignment of windows)
            and selects the smallest one that results in at most `max_samples` windows.

            Parameters
            ----------
            samples : int
                Total number of timesteps available in the time series.
            prediction_length : int
                The forecast horizon of the model.
            max_samples : int
                The maximum number of calibration forecast windows allowed.

            Returns
            -------
            int
                The step size for rolling forecasting that respects the calibration constraint.
            """
            for step in _get_divisors(prediction_length):
                if samples // step <= max_samples:
                    return step
            return prediction_length

        if max_calibration_samples is not None:
            logging.info("`max_calibration_samples` is set to true, ignoring `window_step` and setting `rolling`=True")
            rolling = True
            samples = data_test.num_timesteps_per_item().max()
            window_step = _compute_window_step(
                samples,
                self.predictor.prediction_length,
                max_calibration_samples,
            )
            idx_split = max_calibration_samples * window_step
            logging.info("Automatically determined window_step: %s", window_step)

            # Prepare truncated calibration set from the tail of data_test
            other_data = data_test.slice_by_timestep(end_index=-idx_split)
            data_test = data_test.slice_by_timestep(start_index=-idx_split)

            if data_previous_context is not None:
                data_previous_context = pd.concat([data_previous_context, other_data]).sort_index()
            else:
                data_previous_context = other_data

        data_test = self.validate_data(data_test)
        if data_previous_context is not None:
            data_previous_context = self.validate_data(data_previous_context)

        logging.info(
            "Starting forecast generation of model %s for data_test from %s to %s",
            self.predictor.name,
            data_test.index.get_level_values("timestamp").min(),
            data_test.index.get_level_values("timestamp").max(),
        )
        predictions = self.predictor.predict(data_test, data_previous_context, rolling, window_step)

        return {self.predictor.name: predictions}

    def train_postprocessors(self, calibration_predictions: ForecastCollection) -> None:
        """
        Fit the postprocessors based on calibration data.

        Parameters
        ----------
        calibration_predictions : ForecastCollection
            The generated forecasts of the predictor on the calibration data.
        """
        logging.info("Start training postprocessors...")
        start_time = pd.Timestamp.now()
        self._initialize_postprocessors()
        for name, postprocessor in self.postprocessor_dict.items():
            postprocessor.fit(data=calibration_predictions)
        end_time = pd.Timestamp.now()
        logging.info("Postprocessors training completed in %s seconds.", (end_time - start_time).total_seconds())

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
        if not self.postprocessor_dict:
            logging.info("No postprocessors configured. Returning raw predictions.")
            return predictions

        logging.info("Applying postprocessing to predictions...")
        for name, postprocessor in self.postprocessor_dict.items():
            logging.info("Postprocessing predictions using postprocessor: %s", name)

            # Apply the postprocessing and store predictions as additional key value pair
            predictions[name] = postprocessor.postprocess(data=predictions[self.predictor.name])
            logging.info("Postprocessing complete for %s", name)

        logging.info("Postprocessing completed for all models.")
        return predictions

    def _run_backtest_iteration(
        self,
        data: Union[TimeSeriesDataFrame, TabularDataFrame],
        test_start_date: pd.Timestamp,
        train: bool,
        train_window_size: Optional[pd.DateOffset],
        val_window_size: Optional[pd.DateOffset],
        test_window_size: Optional[pd.DateOffset],
        test_window_step: int,
        calibration_window_step: int,
        calibration_based_on: Optional[Union[Literal["val", "train", "train_val"], pd.DateOffset]],
    ) -> Tuple[Dict[str, ForecastCollection], Dict]:
        """
        Train, predict, and postprocess wrapper for internal backtesting.

        - train window step is set to 1
        - validation window step is hardcoded to be aligned with test_window_step
        """

        execution_times = {}

        data_train, data_val, data_test = self.split_time_series_data(data, test_start_date, train_window_size, val_window_size, test_window_size)

        logging.info("Removing all item_ids which contain only nans in target column.")
        data_test = data_test[data_test.groupby(level=ITEMID)[TARGET].transform(lambda x: not x.isna().all())].copy()
        if len(data_test) == 0:
            raise ValueError("Test data is empty after dropping all rows with target=nan. Check data.")

        # ---------- train the predictor ----------
        if train:
            self.train_predictor_model(data_train, data_val, 1, test_window_step)
            # TODO: save model directly
        else:
            logging.info("Skipping model training because `train=False`.")

        # ---------- predictions generated by predictor ----------
        predictions_test_data = self.generate_forecasts(
            data_test=data_test,
            data_previous_context=data.split_by_time(data_test.index.get_level_values("timestamp").min())[0],
            rolling=True,
            window_step=test_window_step,
        )  # TODO: save predictions directly

        # store information on training and inference time
        execution_times[self.predictor.name] = ExecutionTimePredictor(
            predictor_name=self.predictor.name,
            predictor_train_time=self.predictor.train_time_seconds,
            predictor_inference_time=predictions_test_data[self.predictor.name].inference_time_seconds,
        )

        # ---------- define calibration dataset ----------
        if self.postprocessors is not None:
            if calibration_based_on == "val":
                calibration_data = data_val
                context_data = data_train
            elif calibration_based_on == "train":
                calibration_data = data_train
                context_data = None
            elif calibration_based_on == "train_val":
                calibration_data = pd.concat([data_train, data_val]).sort_index()
                context_data = None
            else:
                raise ValueError(f"Invalid calibration_based_on: {calibration_based_on}")

            logging.info(
                "Use calibration data from %s to %s for fitting postprocessors",
                calibration_data.index.get_level_values("timestamp").min(),
                calibration_data.index.get_level_values("timestamp").max(),
            )

            # ---------- generate forecasts on calibration dataset using the predictor ----------
            logging.info("Generating forecasts on calibration data...")
            predictions_calibration_data = self.generate_forecasts(
                data_test=calibration_data,
                data_previous_context=context_data,
                rolling=True,
                window_step=calibration_window_step,
            )

            # ---------- train postprocessors ----------
            self.train_postprocessors(predictions_calibration_data[self.predictor.name])

            # ---------- create postprocessed forecasts ----------
            predictions_test_data = self.apply_postprocessors_to_forecasts(predictions_test_data)

            # store information on training and inference time
            for name, postprocessor in self.postprocessor_dict.items():
                execution_times[name] = ExecutionTimePostprocessor(
                    postprocessor_name=name,
                    execution_time_predictor=execution_times[self.predictor.name],
                    calibration_inference_time=predictions_calibration_data[self.predictor.name].inference_time_seconds,
                    postprocessor_train_time=postprocessor.train_time_seconds,
                    postprocessor_inference_time=predictions_test_data[name].inference_time_seconds,
                )

        return predictions_test_data, execution_times

    def _combine_execution_time(self, execution_times: dict) -> dict:
        """
        Aggregate a dict of execution_time objects (from multiple backtests) into one.
        """
        all_backtest_keys = list(execution_times.keys())
        all_methods = set()
        # get all predictor and postprocessors
        for bt in all_backtest_keys:
            all_methods.update(execution_times[bt].keys())

        merged = {}
        # merge values
        for name in all_methods:
            all_obj = []
            for bt in all_backtest_keys:
                all_obj.append(execution_times[bt][name])
            merged[name] = aggregate_execution_time_objects(all_obj)

        return merged

    def _combine_backtest_results(self, backtest_results: Dict[pd.Timestamp, Dict[str, ForecastCollection]]) -> Dict[str, ForecastCollection]:

        all_predictors = {key for result in backtest_results.values() for key in result}

        logging.info("Merge results for each of the following predictors/postprocessors %s", all_predictors)

        merged_results = {}
        for predictor_name in all_predictors:
            merged_results[predictor_name] = self.combine_forecast_windows(backtest_results, predictor_name)

        logging.info("Merge completed.")

        return merged_results

    def combine_forecast_windows(
        self,
        backtest_results: Dict[pd.Timestamp, Dict[str, ForecastCollection]],
        predictor_name: str,
    ) -> ForecastCollection:
        """
        Merge a single predictor's backtest windows into one ForecastCollection.

        Parameters
        ----------
        backtest_results : Dict[pd.Timestamp, Dict[str, ForecastCollection]]
            mapping from window-start date to all predictors' ForecastCollection.
        predictor_name : str
            which predictor to merge.

        Returns
        -------
            A ForecastCollection whose TimeSeriesForecasts have all windows stitched together.
        """
        # 1) sort by window-start date
        dates = sorted(backtest_results.keys())

        # 2) grab each window's ForecastCollection for this predictor
        per_window_fc = [backtest_results[d][predictor_name] for d in dates]

        # 3) identify all item_ids
        item_ids = per_window_fc[0].get_item_ids()

        merged_ts: Dict[int, TimeSeriesForecast] = {}

        for item_id in item_ids:
            # 4) collect this item’s TimeSeriesForecast from each window
            ts_list = [fc.get_time_series_forecast(item_id) for fc in per_window_fc]

            lead_times = ts_list[0].get_lead_times()
            quantiles = ts_list[0].quantiles
            freq = ts_list[0].freq

            # 5) concatenate underlying dataframes
            data_concat = pd.concat([ts.data for ts in ts_list])
            forecast_mask = np.concatenate([ts.forecast_mask for ts in ts_list])
            # 6) for each lead time, stack predictions
            lead_time_forecasts: Dict[int, HorizonForecast] = {}
            for lt in lead_times:
                # gather the raw prediction tensors in window order
                preds = torch.vstack([ts.get_lead_time_forecast(lt).predictions for ts in ts_list])
                # build a new HorizonForecast
                lead_time_forecasts[lt] = HorizonForecast(lead_time=lt, predictions=preds)

            # 7) re-create the merged TimeSeriesForecast
            merged_ts[item_id] = TimeSeriesForecast(
                item_id=item_id,
                lead_time_forecasts=lead_time_forecasts,
                data=data_concat,
                quantiles=quantiles,
                freq=freq,
                forecast_mask=forecast_mask,
            )

        return ForecastCollection(item_ids=merged_ts)


def create_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        logging.info("Created new directory: %s", path)
