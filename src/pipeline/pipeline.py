import torch
from typing import Dict, Any, List, Optional, Type, Union, Literal, Tuple
from src.core.timeseries_evaluation import PredictionLeadTime, PredictionLeadTimes, TabularDataFrame
from src.core.base import AbstractPostprocessor, AbstractPredictor
from src.core.utils import CustomJSONEncoder
from autogluon.timeseries import TimeSeriesDataFrame
import pandas as pd
from src.core.base import AbstractPipeline
from pathlib import Path
import json
import importlib
from typing import Type
import joblib

DIR_BACKTESTS = "backtest"
DIR_MODELS = "models"
DIR_POSTPROCESSORS = "postprocessors"
PIPELINE_CONFIG_FILE_NAME = "pipeline_config.json"


class ForecastingPipeline(AbstractPipeline):
    def __init__(
        self,
        model: Type[AbstractPredictor],
        model_kwargs: Dict,
        postprocessors: Optional[List[Type[AbstractPostprocessor]]] = None,
        output_dir: Optional[Union[Path, str]] = None,
    ):
        super().__init__(model, model_kwargs, postprocessors, output_dir)

        # initialize None predictor
        self.predictor = None
        self.postprocessor_dict: Dict[str, AbstractPostprocessor] = {}

        # define storage directory
        self.pipeline_dir_models = self.output_dir / DIR_MODELS
        self.pipeline_dir_postprocessors = self.output_dir / DIR_POSTPROCESSORS
        self.pipeline_dir_backtests = self.output_dir / DIR_BACKTESTS
        self.model_kwargs.update({"output_dir": self.pipeline_dir_models})

    def save(self) -> None:
        """Save the pipeline configuration to a JSON file and store predictors and postprocessors as joblib."""

        create_dir(self.pipeline_dir_models)
        create_dir(self.pipeline_dir_postprocessors)

        config = {
            "model": get_class_path(self.model),
            "model_kwargs": self.model_kwargs,
            "postprocessors": [get_class_path(postprocessor) for postprocessor in (self.postprocessors or [])],
        }
        with open(self.output_dir / "pipeline_config.json", "w") as f:
            json.dump(config, f, indent=4, cls=CustomJSONEncoder)

        if self.predictor is not None:
            self.predictor.save()

        if self.postprocessor_dict is not None:
            for name, postprocessor in self.postprocessor_dict.items():
                # TODO: storing should be done in the same way as for predictor
                postprocessor.save(self.pipeline_dir_postprocessors / f"{name}.joblib")

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
        if "freq" in model_kwargs:
            model_kwargs["freq"] = pd.Timedelta(model_kwargs["freq"])
        config["model_kwargs"] = model_kwargs

        # Load postprocessor classes
        config["postprocessors"] = [load_class_from_path(pp) for pp in config.get("postprocessors", [])]

        # Recreate the pipeline
        pipeline = ForecastingPipeline(
            model=config["model"],
            model_kwargs=config["model_kwargs"],
            postprocessors=config["postprocessors"],
            output_dir=pipeline_dir,
        )

        # Load predictor
        models = list((pipeline_dir / DIR_MODELS).glob("*.joblib"))
        if not models:
            print(f"No models found in {pipeline_dir / DIR_MODELS}. Skipping model loading.")
        else:
            pipeline.predictor = joblib.load(models[0])

        # Load postprocessors
        postprocessor_files = list((pipeline_dir / DIR_POSTPROCESSORS).glob("*.joblib"))
        if not postprocessor_files:
            print(f"No postprocessors found in {pipeline_dir / DIR_POSTPROCESSORS}. Skipping postprocessor loading.")
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
        train: bool = False,
        calibration_based_on: Optional[Union[Literal["val", "train", "train_val"], pd.DateOffset]] = None,
        save_results: bool = False,
    ) -> PredictionLeadTimes:
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
            Whether to save backtest results. Defaults to False.

        Returns
        -------
        PredictionLeadTimes
            The predictions over the backtest period.
        """
        results = {}

        if test_end_date is None:
            test_end_date = data.index.get_level_values("timestamp").max()

        if rolling_window_eval:
            if not test_window_size:
                print("test_window_size not specified. Set it to 1 year as default")
                test_window_size = pd.DateOffset(years=1)

            while test_start_date < test_end_date:
                results[test_start_date] = self._train_predict_postprocess(data, test_start_date, train, train_window_size, val_window_size, test_window_size, calibration_based_on)
                test_start_date += test_window_size

            results = self._merge_results(results)
        else:
            results = self._train_predict_postprocess(data, test_start_date, train, train_window_size, val_window_size, test_window_size, calibration_based_on)

        if save_results:
            backtest_params = {
                "test_start_date": test_start_date,
                "test_end_date": test_end_date,
                "rolling_window_eval": rolling_window_eval,
                "train_window_size": train_window_size,
                "val_window_size": val_window_size,
                "test_window_size": test_window_size,
                "train": train,
                "calibration_based_on": calibration_based_on,
            }
            self._save_backtest_results(results, backtest_params)

        return results

    def split_data(
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
        data_val = None
        data_train, data_test = data.split_by_time(test_start_date)

        if val_window_size:
            data_train, data_val = data_train.split_by_time(test_start_date - val_window_size)
        if test_window_size:
            data_test, _ = data_test.split_by_time(test_start_date + test_window_size)
        if train_window_size:
            _, data_train = data_train.split_by_time(test_start_date - train_window_size)

        return data_train, data_val, data_test

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
        self.predictor = self.model(**self.model_kwargs)
        print(f"Training from {data_train.index.get_level_values('timestamp').min()} to {data_train.index.get_level_values('timestamp').max()}")
        if data_val is not None:
            print(f"Validation from {data_val.index.get_level_values('timestamp').min()} to {data_val.index.get_level_values('timestamp').max()}")
        self.predictor.fit(data_train, data_val)

    def predict(
        self,
        data_test: Union[TimeSeriesDataFrame, TabularDataFrame],
        data_previous_context: Optional[Union[TimeSeriesDataFrame, TabularDataFrame]] = None,
    ) -> Dict[str, PredictionLeadTimes]:
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
        Dict[str, PredictionLeadTimes]
            Dictionary with raw predictions.
        """
        print(f"Prediction from {data_test.index.get_level_values('timestamp').min()} to {data_test.index.get_level_values('timestamp').max()}")

        if self.predictor is None:
            print("Predictor has not been fitted yet. Trying to predict anyways...")
            self.predictor = self.model(**self.model_kwargs)

        return {self.predictor.__class__.__name__: self.predictor.predict(data=data_test, previous_context_data=data_previous_context)}

    def train_postprocessors(self, calibration_data: Union[TimeSeriesDataFrame, TabularDataFrame]) -> None:
        """
        Fit the postprocessors based on calibration data.

        Parameters
        ----------
        calibration_data : Union[TimeSeriesDataFrame, TabularDataFrame]]
            The calibration data. Used to fit the postprocessor.
        """
        if not self.postprocessors:
            raise ValueError("No postprocessors specified.")

        calibration_predictions = self.predictor.predict(calibration_data)

        for postprocessor in self.postprocessors:
            self.postprocessor_dict[postprocessor.__name__] = postprocessor()
            print(f"Fit postprocessor: {postprocessor.__name__}")
            self.postprocessor_dict[postprocessor.__name__].fit(data=calibration_predictions)

    def apply_postprocessing(self, predictions: Dict[str, PredictionLeadTimes]) -> Dict[str, PredictionLeadTimes]:
        """
        Apply postprocessing to predictions.

        Parameters
        ----------
        predictions : Dict[str, PredictionLeadTimes]
            The predictions. Keys correspond to the utilized model, e.g. `Chronos` or `QuantileRegression`

        Returns
        -------
        Dict[str, PredictionLeadTimes]
            Dictionary with processed predictions.
        """
        if not self.postprocessors:
            return predictions

        if self.postprocessor_dict is None:
            raise ValueError("Postprocessora need to be fitted via `.train_postprocessors()` first.")

        for postprocessor in self.postprocessors:
            print(f"Postprocessing predictions using {postprocessor.__name__}")
            predictions[postprocessor.__name__] = self.postprocessor_dict[postprocessor.__name__].postprocess(data=predictions[self.predictor.__class__.__name__])

        return predictions

    def _train_predict_postprocess(
        self,
        data: Union[TimeSeriesDataFrame, TabularDataFrame],
        test_start_date: pd.Timestamp,
        train: bool,
        train_window_size: Optional[pd.DateOffset],
        val_window_size: Optional[pd.DateOffset],
        test_window_size: Optional[pd.DateOffset],
        calibration_based_on: Optional[Union[Literal["val", "train", "train_val"], pd.DateOffset]],
    ) -> Dict[str, PredictionLeadTimes]:
        """Train, predict, and postprocess wrapper for internal backtesting."""
        data_train, data_val, data_test = self.split_data(data, test_start_date, train_window_size, val_window_size, test_window_size)

        if train:
            self.train(data_train, data_val)

        if self.postprocessors is not None:
            if calibration_based_on == "val":
                calibration_data = data_val
            elif calibration_based_on == "train":
                calibration_data = data_train
            elif calibration_based_on == "train_val":
                calibration_data = pd.concat([data_train, data_val]).sort_index()
            else:
                raise ValueError(f"Invalid calibration_based_on: {calibration_based_on}")

            self.train_postprocessors(calibration_data)
            predictions = self.predict(data_test, data.split_by_time(data_test.index.get_level_values("timestamp").min())[0])
            predictions = self.apply_postprocessing(predictions)

        return predictions

    def _merge_results(self, results: Dict[pd.Timestamp, Dict[str, PredictionLeadTimes]]) -> Dict[str, PredictionLeadTimes]:

        print(f"Merging test results from the {len(results.keys())} different runs: {list(results.keys())}")

        results_all = {}

        # Collect all unique postprocessor names
        all_postprocessors = set()
        for result in results.values():
            all_postprocessors.update(result.keys())

        # extract results for each postprocessor and merge those
        for postprocessor_name in all_postprocessors:

            pp_results = {date: result[postprocessor_name] for date, result in results.items() if postprocessor_name in result}
            # extract lead times
            lead_times: List[int] = sorted({lt for prediction_lead_times in pp_results.values() for lt in prediction_lead_times.results.keys()})

            results_merged = {}

            # create merged PredictionLeadTimes object by merging individual PredictionLeadTime objects
            for lt in lead_times:
                results_lead_time = [result.results[lt] for result in pp_results.values()]

                q = _get_quantiles(results_lead_time)
                f = _get_freq(results_lead_time)
                l = _get_lead_time(results_lead_time)
                p = torch.tensor(pd.concat([result.to_dataframe() for result in results_lead_time]).sort_index()[q].values)
                d = pd.concat([result.data for result in results_lead_time]).sort_index()

                if all(isinstance(result.data, TabularDataFrame) for result in results_lead_time):
                    d = TabularDataFrame(d)
                elif all(isinstance(result.data, TimeSeriesDataFrame) for result in results_lead_time):
                    d = TimeSeriesDataFrame(d)

                results_merged[lt] = PredictionLeadTime(lead_time=l, freq=f, quantiles=q, predictions=p, data=d)

                results_all[postprocessor_name] = PredictionLeadTimes(results=results_merged)

        return results_all

    def _save_backtest_results(self, results: Dict[str, PredictionLeadTimes], backtest_params: Dict) -> None:
        """Save backtest results and config."""

        for method, result in results.items():
            save_path = self.pipeline_dir_backtests / method

            create_dir(save_path)

            # Add information
            eval_config_info = {}
            eval_config_info = {"applied_postprocessor": None if method == "raw" else method}
            eval_config_info.update(result.get_crps(mean_lead_times=True, mean_time=True).to_dict())
            eval_config_info.update(result.get_empirical_coverage_rates(mean_lead_times=True).to_dict())
            eval_config_info.update(result.get_quantile_scores(mean_lead_times=True).to_dict())
            config = self.get_config(backtest_params, eval_config_info)

            # Save config
            with open(save_path / "config.json", "w") as f:
                json.dump(config, f, indent=4, cls=CustomJSONEncoder)

            # Save predictions
            result.save(save_path / "predictions.joblib")


# check if all lead times, quantiles and freq are the same
def _get_common_attr(results: List[PredictionLeadTime], attr: str) -> Any:
    values = [getattr(result, attr) for result in results]
    if all(val == values[0] for val in values):
        return values[0]
    else:
        raise ValueError(f"Not all {attr}s match.")


def _get_lead_time(results: List[PredictionLeadTime]) -> int:
    return _get_common_attr(results, "lead_time")


def _get_freq(results: List[PredictionLeadTime]) -> pd.Timedelta:
    return _get_common_attr(results, "freq")


def _get_quantiles(results: List[PredictionLeadTime]) -> List[float]:
    return _get_common_attr(results, "quantiles")


def create_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        print(f"Created new directory: {path}")


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
    return getattr(module, class_name)
