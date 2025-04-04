from timeseries import PredictionLeadTimes, PredictionLeadTime, TabularDataFrame
from autogluon.timeseries import TimeSeriesDataFrame
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import torch
from fastai.tabular.core import add_datepart
from tqdm import tqdm
import statsmodels.api as sm

############# Nearest Neighbour #############


def forecast_from_weekday_hour_patterns(
    data: TimeSeriesDataFrame, weekday_hour_value_dict: Dict[str, np.ndarray], lead_times: np.ndarray, percentiles: np.ndarray, freq: pd.Timedelta = pd.Timedelta("1h")
) -> Tuple[PredictionLeadTimes, dict]:
    """Generates percentile-based forecasts for multiple lead times based on historical patterns
    observed at the same weekday and hour.

    This function uses a weekday-hour-based pattern dictionary to compute forecasts for each
    lead time into the future. For each timestamp in the data, it appends the observed value
    to the corresponding weekday-hour slot and computes future forecasts using the updated
    dictionary.

    Parameters:
    ----------
    data : TimeSeriesDataFrame
        A time-indexed DataFrame containing the target variable to forecast.
    weekday_hour_value_dict : dict
        A dictionary mapping weekday-hour combinations (e.g., "0_13" for Monday 1PM)
        to arrays of past values seen at those times.
    lead_times : np.ndarray
        Array of integers specifying the lead times (in `freq` units) to forecast.
    percentiles : np.ndarray
        Percentile values (between 0 and 100) to compute for the forecast at each lead time.
    freq : pd.Timedelta, optional
        Frequency of the time series data (default is 1 hour).

    Returns:
    -------
    Tuple[PredictionLeadTimes, dict]
        A tuple containing:
        - PredictionLeadTimes: Object containing forecast results for each lead time.
        - dict: Updated `weekday_hour_value_dict` with appended observations.
    """

    forecasts = {lt: [] for lt in lead_times}
    results = {lt: {} for lt in lead_times}

    for (_, timestamp), target in data.iterrows():
        # Add the current observation to the weekday-hour bucket
        current_timestamp_id = f"{timestamp.weekday()}_{timestamp.hour}"
        weekday_hour_value_dict[current_timestamp_id] = np.append(weekday_hour_value_dict[current_timestamp_id], target.item())

        # Forecast for each lead time using the appropriate future weekday-hour bucket
        for lt in forecasts.keys():
            prediction_timestamp = timestamp + freq * lt
            prediction_timestamp_id = f"{prediction_timestamp.weekday()}_{prediction_timestamp.hour}"

            try:
                forecast = np.percentile(weekday_hour_value_dict[prediction_timestamp_id], q=percentiles)
            except:
                # If no data is available for the future time slot, use zeros
                forecast = np.zeros(len(percentiles))

            forecasts[lt].append(forecast)

    for lt in forecasts.keys():
        results[lt] = PredictionLeadTime(lead_time=lt, predictions=torch.tensor(np.vstack(forecasts[lt])), freq=freq, data=data)

    return PredictionLeadTimes(results=results), weekday_hour_value_dict


def initialize_weekday_hour_dict() -> Dict[str, np.ndarray]:
    """Initializes a dictionary with keys for each weekday-hour combination (e.g., "0_0" to "6_23"),
    mapping to empty numpy arrays for accumulating historical values.

    Returns:
    -------
    dict
        Dictionary with keys formatted as "<weekday>_<hour>", each mapped to an empty array.
    """

    hours = 24
    weekdays = 7
    weekday_hour_dict = {f"{weekday}_{hour}": np.array([]) for weekday in range(weekdays) for hour in range(hours)}
    return weekday_hour_dict


############# Quantile Regression #############


def add_cyclic_encoding(data: pd.DataFrame, colname: str, period: int, drop: bool = False):
    """Adds sine and cosine encoding for a cyclical feature."""
    data[f"sin_{colname}"] = np.sin(2 * np.pi * data[colname] / period)
    data[f"cos_{colname}"] = np.cos(2 * np.pi * data[colname] / period)

    if drop:
        data = data.drop(columns=[colname])


def create_cyclic_features(data: TimeSeriesDataFrame, lead_time: int = 1) -> TabularDataFrame:
    """Creates multiple time series related features, including relative time deltas for linear models."""

    freq = pd.Timedelta("1h")

    # Shift target to reflect prediction lead time
    data = data.reset_index(level=0)
    data["target"] = data["target"].shift(periods=-lead_time, freq=freq)
    data = data.reset_index()

    # === Features from CURRENT timestamp ===
    add_datepart(data, "timestamp", prefix="timestamp_", drop=False)
    data["timestamp_Hour"] = data["timestamp"].dt.hour
    add_cyclic_encoding(data, "timestamp_Dayofweek", 7)
    add_cyclic_encoding(data, "timestamp_Week", 52)
    add_cyclic_encoding(data, "timestamp_Month", 12)
    add_cyclic_encoding(data, "timestamp_Hour", 24)

    # === Features from PREDICTION date ===
    data["prediction_date"] = data["timestamp"] + lead_time * freq
    add_datepart(data, "prediction_date", prefix="prediction_date_", drop=False)
    data["prediction_date_Hour"] = data["prediction_date"].dt.hour
    add_cyclic_encoding(data, "prediction_date_Dayofweek", 7)
    add_cyclic_encoding(data, "prediction_date_Week", 52)
    add_cyclic_encoding(data, "prediction_date_Month", 12)
    add_cyclic_encoding(data, "prediction_date_Hour", 24)

    # === Relative delta features (trig deltas and their encodings) ===
    data["hours_ahead"] = lead_time
    data["delta_Hour"] = data["prediction_date_Hour"] - data["timestamp_Hour"]
    data["delta_Dayofweek"] = data["prediction_date_Dayofweek"] - data["timestamp_Dayofweek"]
    data["delta_Week"] = data["prediction_date_Week"] - data["timestamp_Week"]
    data["delta_Month"] = data["prediction_date_Month"] - data["timestamp_Month"]

    add_cyclic_encoding(data, "delta_Hour", 24, True)
    add_cyclic_encoding(data, "delta_Dayofweek", 7, True)
    add_cyclic_encoding(data, "delta_Week", 52, True)
    add_cyclic_encoding(data, "delta_Month", 12, True)

    # Drop unused prediction_date fields
    data = data.drop(
        columns=[
            "prediction_date_Elapsed",
            "prediction_date_Year",
            "prediction_date_Month",
            "prediction_date_Week",
            "prediction_date_Day",
            "prediction_date_Hour",
            "prediction_date_Dayofyear",
            "prediction_date_Dayofweek",
            "prediction_date",
            "prediction_date_Is_month_end",
            "prediction_date_Is_month_start",
            "prediction_date_Is_quarter_end",
            "prediction_date_Is_quarter_start",
            "prediction_date_Is_year_end",
            "prediction_date_Is_year_start",
        ]
    )

    # Drop unused timestamp fields
    data = data.drop(
        columns=[
            "timestamp_Elapsed",
            "timestamp_Year",
            "timestamp_Month",
            "timestamp_Week",
            "timestamp_Day",
            "timestamp_Hour",
            "timestamp_Dayofyear",
            "timestamp_Dayofweek",
            "timestamp_Is_month_end",
            "timestamp_Is_month_start",
            "timestamp_Is_quarter_end",
            "timestamp_Is_quarter_start",
            "timestamp_Is_year_end",
            "timestamp_Is_year_start",
        ]
    )

    # Final prep
    data = data.set_index(["item_id", "timestamp"])
    data = data.dropna()

    return TabularDataFrame(data)


def quantile_regression(data_train: TimeSeriesDataFrame, data_test: TimeSeriesDataFrame, quantiles: np.ndarray, lead_times: np.ndarray, freq: pd.Timedelta) -> PredictionLeadTimes:

    results_benchmark = {}

    for lt in tqdm(lead_times):

        data_train_lt = create_cyclic_features(data_train, lt)
        data_test_lt = create_cyclic_features(data_test, lt)

        # store results
        qr_models = {}
        test_results = {}

        # fit a quantile regression for each quantile and make predictions on test dataset
        for q in quantiles:

            x_train = data_train_lt.drop(columns="target").astype(float).to_numpy()
            y_train = np.log(data_train_lt["target"].values)
            x_test = data_test_lt.drop(columns="target").astype(float).to_numpy()
            y_test = np.log(data_test_lt["target"].values)

            # Add constant for intercept
            x_train = sm.add_constant(x_train)
            x_test = sm.add_constant(x_test)

            # Fit quantile regression model
            model = sm.QuantReg(y_train, x_train)
            qr_models[q] = model.fit(q=q, max_iter=2000)

            predictions = qr_models[q].predict(x_test)
            test_results[q] = np.exp(predictions)

        results_benchmark[lt] = PredictionLeadTime(lead_time=lt, predictions=torch.tensor(np.array(list(test_results.values())).T), freq=freq, data=data_test_lt)

    return PredictionLeadTimes(results=results_benchmark)
