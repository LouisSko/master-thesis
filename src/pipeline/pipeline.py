import torch
from typing import Dict, Any, List, Optional, Type, Union, Literal
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

    def _train_predict_postprocess(
        self,
        test_start_date: pd.Timestamp,
        train: bool,
        train_window_size: Optional[pd.DateOffset],
        val_window_size: Optional[pd.DateOffset],
        test_window_size: Optional[pd.DateOffset],
        calibration_based_on: Optional[Union[Literal["val", "train", "train_val"], pd.DateOffset]],
    ) -> Dict[str, PredictionLeadTimes]:
        """Train, predict and postprocess."""
        # get datasets
        data_val = None
        data_train, data_test = self.data.split_by_time(test_start_date)
        if val_window_size:
            data_train, data_val = data_train.split_by_time(test_start_date - val_window_size)
        if test_window_size:
            data_test, _ = data_test.split_by_time(test_start_date + test_window_size)
        if train_window_size:
            _, data_train = data_train.split_by_time(test_start_date - train_window_size)

        # load predictor
        predictor = self.model(**self.model_kwargs)

        # optionally train model
        if train:
            print(f"training from {data_train.index.get_level_values('timestamp').min()} to {data_train.index.get_level_values('timestamp').max()}")
            if data_val is not None:
                print(f"validation from {data_val.index.get_level_values('timestamp').min()} to {data_val.index.get_level_values('timestamp').max()}")
            predictor.fit(data_train, data_val)

        # make predictions
        predictions_test = {}

        print(f"prediction from {data_test.index.get_level_values('timestamp').min()} to {data_test.index.get_level_values('timestamp').max()}")
        predictions_test["raw"] = predictor.predict(
            data=data_test,
            previous_context_data=self.data.split_by_time(data_test.index.get_level_values("timestamp").min())[0],
        )

        # optionally postprocess
        if self.postprocessors:
            for postprocessor in self.postprocessors:
                if calibration_based_on == "val":
                    calibration_predictions = predictor.predict(data_val, data_train)
                elif calibration_based_on == "train":
                    calibration_predictions = predictor.predict(data_train)
                elif calibration_based_on == "train_val":
                    calibration_predictions = predictor.predict(pd.concat([data_train, data_val]).sort_index())
                else:
                    raise ValueError(f"Invalid calibration_based_on: {calibration_based_on}")

                pp = postprocessor()

                print(f"Fit postprocessor: {postprocessor.__name__}")
                pp.fit(data=calibration_predictions)
                print(f"Postprocessing predictions using {postprocessor.__name__}")
                predictions_test[postprocessor.__name__] = pp.postprocess(data=predictions_test["raw"])

        return predictions_test

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

    def inference(self) -> PredictionLeadTimes:
        raise NotImplementedError

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
