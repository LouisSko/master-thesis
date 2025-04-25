import torch
from typing import Dict, Any, List, Optional, Type, Union, Literal
from src.core.timeseries_evaluation import PredictionLeadTime, PredictionLeadTimes, TabularDataFrame
from src.core.base import AbstractPostprocessor, AbstractPredictor
from autogluon.timeseries import TimeSeriesDataFrame
import pandas as pd


# check if all lead times, quantiles and freq are the same
def get_common_attr(results: Dict[str, PredictionLeadTime], attr: str) -> Any:
    values = [getattr(result, attr) for result in results]
    if all(val == values[0] for val in values):
        return values[0]
    else:
        raise ValueError(f"Not all {attr}s match.")


def get_lead_time(results: Dict[str, PredictionLeadTime]) -> int:
    return get_common_attr(results, "lead_time")


def get_freq(results: Dict[str, PredictionLeadTime]) -> pd.Timedelta:
    return get_common_attr(results, "freq")


def get_quantiles(results: Dict[str, PredictionLeadTime]) -> List[float]:
    return get_common_attr(results, "quantiles")


def merge_results(results: Dict[str, PredictionLeadTimes]):
    results_merged = {}

    # extract lead times
    lead_times: List[int] = sorted({lt for prediction_lead_times in results.values() for lt in prediction_lead_times.keys()})

    for lt in lead_times:
        results_lead_time = [result.results[lt] for result in results.values()]

        q = get_quantiles(results_lead_time)
        f = get_freq(results_lead_time)
        l = get_lead_time(results_lead_time)
        p = torch.tensor(pd.concat([result.to_dataframe() for result in results_lead_time]).sort_index()[q].values)
        d = pd.concat([result.data for result in results_lead_time]).sort_index()

        if all(isinstance(result.data, TabularDataFrame) for result in results_lead_time):
            d = TabularDataFrame(d)
        elif all(isinstance(result.data, TimeSeriesDataFrame) for result in results_lead_time):
            d = TimeSeriesDataFrame(d)

        results_merged[lt] = PredictionLeadTime(lead_time=l, freq=f, quantiles=q, predictions=p, data=d)

    return PredictionLeadTimes(results=results_merged)


def backtest(
    model: Type[AbstractPredictor],
    model_kwargs: Dict,
    data: Union[TimeSeriesDataFrame, TabularDataFrame],
    test_start_date: pd.Timestamp,
    end_date: Optional[pd.Timestamp] = None,
    rolling_window: bool = False,
    train_window_size: Optional[pd.DateOffset] = None,
    val_window_size: Optional[pd.DateOffset] = None,
    test_window_size: Optional[pd.DateOffset] = None,
    train: bool = False,
    postprocessor: Optional[Type[AbstractPostprocessor]] = None,
    calibration_based_on: Optional[Union[Literal["val", "train", "val_train"], pd.DateOffset]] = None,
) -> PredictionLeadTimes:

    def train_predict_postprocess():
        """Helper function for training, predicting and postprocessing."""

        # get datasets
        data_val = None
        data_train, data_test = data.split_by_time(test_start_date)
        if val_window_size:
            data_train, data_val = data_train.split_by_time(test_start_date - val_window_size)
        if test_window_size:
            data_test, _ = data_test.split_by_time(test_start_date + test_window_size)
        if train_window_size:
            _, data_train = data_train.split_by_time(test_start_date - train_window_size)

        # load predictor
        predictor = model(**model_kwargs)

        # optionally train model
        if train:
            print(f"training from {data_train.index.get_level_values('timestamp').min()} to {data_train.index.get_level_values('timestamp').max()}")
            if data_val:
                print(f"validation from {data_val.index.get_level_values('timestamp').min()} to {data_val.index.get_level_values('timestamp').max()}")
            predictor.fit(data_train, data_val)

        # make predictions
        print(f"prediction from {data_test.index.get_level_values('timestamp').min()} to {data_test.index.get_level_values('timestamp').max()}")
        predictions_test = predictor.predict(data=data_test, previous_context_data=data.split_by_time(data_test.index.get_level_values("timestamp").min())[0])

        # optionally post process predictions. For that a calibration dataset is necessary. Use either train, val or both
        if postprocessor:

            if calibration_based_on == "val":
                calibration_predictions = predictor.predict(data_val, data_train)
            elif calibration_based_on == "train":
                calibration_predictions = predictor.predict(data_train)
            elif calibration_based_on == "train_val":
                calibration_predictions = predictor.predict(pd.concat([data_train, data_val]).sort_index())

            # fit postprocessor and postprocess predictions
            pp = postprocessor()
            print("Fit postprocessor")
            pp.fit(data=calibration_predictions)
            print("Postprocessing predictions")
            predictions_test_post = pp.postprocess(data=predictions_test)

            return predictions_test_post

        return predictions_test

    results = {}

    if end_date is None:
        end_date = data.index.get_level_values("timestamp").max()

    if rolling_window:

        if not test_window_size:
            print("test_window_size not specified. Set it to 1 year as default")
            test_window_size = pd.DateOffset(years=1)

        while test_start_date < end_date:
            results[test_start_date] = train_predict_postprocess()
            test_start_date += test_window_size
        results = merge_results(results)
    else:
        results = train_predict_postprocess()

    return results
