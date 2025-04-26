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

PATH_PREDICTIONS = Path("./data/predictions/")


class ForecastingPipeline(AbstractPipeline):
    def __init__(
        self,
        model: Type[AbstractPredictor],
        model_kwargs: Dict,
        data: Union[TimeSeriesDataFrame, TabularDataFrame],
        postprocessors: Optional[List[Type[AbstractPostprocessor]]] = None,
    ):
        super().__init__(model, model_kwargs, data, postprocessors)

        # initialize None predictor
        self.predictor = None

    def backtest(
        self,
        test_start_date: pd.Timestamp,
        test_end_date: Optional[pd.Timestamp] = None,
        rolling_window_eval: bool = False,
        train_window_size: Optional[pd.DateOffset] = None,
        val_window_size: Optional[pd.DateOffset] = None,
        test_window_size: Optional[pd.DateOffset] = None,
        train: bool = False,
        calibration_based_on: Optional[Union[Literal["val", "train", "train_val"], pd.DateOffset]] = None,
        output_dir: Optional[Path] = None,
    ) -> PredictionLeadTimes:
        """
        Run a backtest over the specified time period.

        Parameters
        ----------
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
        output_dir : Optional[Path], optional
            Directory to save backtest results. Defaults to None.

        Returns
        -------
        PredictionLeadTimes
            The predictions over the backtest period.
        """
        results = {}

        if test_end_date is None:
            test_end_date = self.data.index.get_level_values("timestamp").max()

        if rolling_window_eval:
            if not test_window_size:
                print("test_window_size not specified. Set it to 1 year as default")
                test_window_size = pd.DateOffset(years=1)

            while test_start_date < test_end_date:
                results[test_start_date] = self._train_predict_postprocess(test_start_date, train, train_window_size, val_window_size, test_window_size, calibration_based_on)
                test_start_date += test_window_size

            results = self._merge_results(results)
        else:
            results = self._train_predict_postprocess(test_start_date, train, train_window_size, val_window_size, test_window_size, calibration_based_on)

        if output_dir:
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
            self._save_backtest_results(results, output_dir, backtest_params)

        return results

    def split_data(
        self,
        test_start_date: pd.Timestamp,
        train_window_size: Optional[pd.DateOffset],
        val_window_size: Optional[pd.DateOffset],
        test_window_size: Optional[pd.DateOffset],
    ) -> Tuple[Union[TimeSeriesDataFrame, TabularDataFrame], Union[TimeSeriesDataFrame, TabularDataFrame, None], Union[TimeSeriesDataFrame, TabularDataFrame]]:
        """
        Split data into training, validation, and testing sets.

        Parameters
        ----------
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
        data_train, data_test = self.data.split_by_time(test_start_date)

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
    ) -> Dict[str, PredictionLeadTimes]:
        """
        predict on the test data.

        Parameters
        ----------
        data_test : Union[TimeSeriesDataFrame, TabularDataFrame]
            The test dataset.

        Returns
        -------
        Dict[str, PredictionLeadTimes]
            Dictionary with raw predictions.
        """

        print(f"Prediction from {data_test.index.get_level_values('timestamp').min()} to {data_test.index.get_level_values('timestamp').max()}")

        if self.predictor is None:
            print("Predictor has not been fitted yet. Trying to predict anyways...")
            self.predictor = self.model(**self.model_kwargs)

        return {
            self.predictor.__class__.__name__: self.predictor.predict(
                data=data_test,
                previous_context_data=self.data.split_by_time(data_test.index.get_level_values("timestamp").min())[0],
            )
        }

    def postprocess(
        self,
        predictions: Dict[str, PredictionLeadTimes],
        data_train: Optional[Union[TimeSeriesDataFrame, TabularDataFrame]] = None,
        data_val: Optional[Union[TimeSeriesDataFrame, TabularDataFrame]] = None,
        calibration_based_on: Optional[Literal["val", "train", "train_val"]] = None,
    ) -> Dict[str, PredictionLeadTimes]:
        """
        Apply postprocessing to predictions.

        Parameters
        ----------
        predictions : Dict[str, PredictionLeadTimes]
            The predictions. Keys correspond to the utilized model, e.g. `Chronos` or `QuantileRegression`
        data_train : Optional[Union[TimeSeriesDataFrame, TabularDataFrame]], optional
            The training data. Defaults to None.
        data_val : Optional[Union[TimeSeriesDataFrame, TabularDataFrame]], optional
            The validation data, if available. Defaults to None.
        calibration_based_on : Optional[Literal["val", "train", "train_val"]], optional
            Calibration strategy. Defaults to None.

        Returns
        -------
        Dict[str, PredictionLeadTimes]
            Dictionary with processed predictions.
        """
        if not self.postprocessors:
            return predictions

        for postprocessor in self.postprocessors:
            if calibration_based_on == "val":
                calibration_predictions = self.predictor.predict(data_val, data_train)
            elif calibration_based_on == "train":
                calibration_predictions = self.predictor.predict(data_train)
            elif calibration_based_on == "train_val":
                calibration_predictions = self.predictor.predict(pd.concat([data_train, data_val]).sort_index())
            else:
                raise ValueError(f"Invalid calibration_based_on: {calibration_based_on}")

            pp = postprocessor()
            print(f"Fit postprocessor: {postprocessor.__name__}")
            pp.fit(data=calibration_predictions)
            print(f"Postprocessing predictions using {postprocessor.__name__}")
            predictions[postprocessor.__name__] = pp.postprocess(data=predictions[self.predictor.__class__.__name__])

        return predictions

    def _train_predict_postprocess(
        self,
        test_start_date: pd.Timestamp,
        train: bool,
        train_window_size: Optional[pd.DateOffset],
        val_window_size: Optional[pd.DateOffset],
        test_window_size: Optional[pd.DateOffset],
        calibration_based_on: Optional[Union[Literal["val", "train", "train_val"], pd.DateOffset]],
    ) -> Dict[str, PredictionLeadTimes]:
        """Train, predict, and postprocess wrapper for internal backtesting."""
        data_train, data_val, data_test = self.split_data(test_start_date, train_window_size, val_window_size, test_window_size)

        if train:
            self.train(data_train, data_val)

        predictions = self.predict(data_test)
        predictions = self.postprocess(predictions, data_train, data_val, calibration_based_on)

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

    def _save_backtest_results(self, results: Dict[str, PredictionLeadTimes], output_dir: Path, backtest_params: Dict) -> None:
        """Save backtest results and config."""

        for method, result in results.items():
            save_path = PATH_PREDICTIONS / output_dir / method

            if not save_path.exists():
                save_path.mkdir(parents=True)
                print(f"Created new directory: {save_path}")

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
