from timeseries import PredictionLeadTimes, PredictionLeadTime, TabularDataFrame
from autogluon.timeseries import TimeSeriesDataFrame
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Any
import torch
from fastai.tabular.core import add_datepart
from tqdm import tqdm
import statsmodels.api as sm

############# Nearest Neighbour #############


def forecast_from_weekday_hour_patterns(
    data: TimeSeriesDataFrame,
    weekday_hour_value_dict: Dict[Any, Dict[str, np.ndarray]],
    lead_times: np.ndarray,
    quantiles: np.ndarray,
    freq: pd.Timedelta = pd.Timedelta("1h"),
    last_n_samples: Optional[int] = None,
) -> Tuple[PredictionLeadTimes, dict]:
    """Generates quantile-based forecasts for multiple lead times based on historical patterns
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
    quantiles : np.ndarray
        quantiles values to compute for the forecast at each lead time.
    freq : pd.Timedelta, optional
        Frequency of the time series data (default is 1 hour).
    last_n_samples: int, optional
        Number of most recent similar samples to consider when calculating empirical quantiles

    Returns:
    -------
    Tuple[PredictionLeadTimes, dict]
        A tuple containing:
        - PredictionLeadTimes: Object containing forecast results for each lead time.
        - dict: Updated `weekday_hour_value_dict` with appended observations.
    """


    forecasts = {lt: [] for lt in lead_times}
    results = {lt: {} for lt in lead_times}

    percentiles = (quantiles * 100).astype(int)  # convert to percentiles

    for (item_id, timestamp), target in tqdm(data.iterrows(), total=len(data)):
        # Add the current observation to the weekday-hour bucket
        current_timestamp_id = f"{timestamp.weekday()}_{timestamp.hour}"
        weekday_hour_value_dict[item_id][current_timestamp_id] = np.append(weekday_hour_value_dict[item_id][current_timestamp_id], target.item())

        # Forecast for each lead time using the appropriate future weekday-hour bucket
        for lt in forecasts.keys():
            prediction_timestamp = timestamp + freq * lt
            prediction_timestamp_id = f"{prediction_timestamp.weekday()}_{prediction_timestamp.hour}"

            try:
                if last_n_samples:
                    forecast = np.percentile(weekday_hour_value_dict[item_id][prediction_timestamp_id][-last_n_samples:], q=percentiles)
                else:
                    forecast = np.percentile(weekday_hour_value_dict[item_id][prediction_timestamp_id], q=percentiles)
            except:
               # If no data is available for the future time slot, use zeros
               forecast = np.zeros(len(percentiles))

            forecasts[lt].append(forecast)

    for lt in forecasts.keys():
        results[lt] = PredictionLeadTime(lead_time=lt, predictions=torch.tensor(np.vstack(forecasts[lt])), freq=freq, data=data)

    return PredictionLeadTimes(results=results), weekday_hour_value_dict


def initialize_weekday_hour_dict(item_ids: list) -> Dict[str, np.ndarray]:
    """Initializes a dictionary with keys for each weekday-hour combination (e.g., "0_0" to "6_23"),
    mapping to empty numpy arrays for accumulating historical values.

    Returns:
    -------
    dict
        Dictionary with keys formatted as "<weekday>_<hour>", each mapped to an empty array.
    """

    hours = 24
    weekdays = 7
    weekday_hour_dict = {item_id: None for item_id in item_ids}
    for item_id in item_ids:
        weekday_hour_dict[item_id] = {f"{weekday}_{hour}": np.array([]) for weekday in range(weekdays) for hour in range(hours)}
    return weekday_hour_dict


############# Quantile Regression #############


def add_cyclic_encoding(data: pd.DataFrame, colname: str, period: int, drop: bool = False):
    """Adds sine and cosine encoding for a cyclical feature to a DataFrame.

    Parameters:
    ----------
    data : pd.DataFrame
        The DataFrame containing the cyclical column to be encoded.
    colname : str
        Name of the column to encode.
    period : int
        The period of the cycle (e.g., 24 for hours in a day).
    drop : bool, optional
        Whether to drop the original column after encoding. Default is False.

    Returns:
    -------
    None
        Modifies the DataFrame in-place.
    """

    data[f"sin_{colname}"] = np.sin(2 * np.pi * data[colname] / period)
    data[f"cos_{colname}"] = np.cos(2 * np.pi * data[colname] / period)

    if drop:
        data = data.drop(columns=[colname])


def create_cyclic_features(data: TimeSeriesDataFrame, lead_time: int = 1, freq: pd.Timedelta = pd.Timedelta("1h")) -> TabularDataFrame:
    """Generates timestamp-based and relative time delta features for prediction tasks.

    Parameters:
    ----------
    data : TimeSeriesDataFrame
        Input time series data indexed by item_id and timestamp.
    lead_time : int, optional
        Forecasting lead time
    freq : pd.Timedelta, optional
        Frequency of the time series

    Returns:
    -------
    TabularDataFrame
        Tabular format of the input time series with created features.
    """

    data_w_features = []

    for i, data_subset in data.groupby("item_id"):
        # Shift target to reflect prediction lead time
        data_subset = data_subset.reset_index(level=0)
        data_subset["target"] = data_subset["target"].shift(periods=-lead_time, freq=freq)
        data_subset = data_subset.reset_index()

        # === Features from CURRENT timestamp ===
        add_datepart(data_subset, "timestamp", prefix="timestamp_", drop=False)
        data_subset["timestamp_Hour"] = data_subset["timestamp"].dt.hour
        add_cyclic_encoding(data_subset, "timestamp_Dayofweek", 7)
        add_cyclic_encoding(data_subset, "timestamp_Week", 52)
        add_cyclic_encoding(data_subset, "timestamp_Month", 12)
        add_cyclic_encoding(data_subset, "timestamp_Hour", 24)

        # === Features from PREDICTION date ===
        data_subset["prediction_date"] = data_subset["timestamp"] + lead_time * freq
        add_datepart(data_subset, "prediction_date", prefix="prediction_date_", drop=False)
        data_subset["prediction_date_Hour"] = data_subset["prediction_date"].dt.hour
        add_cyclic_encoding(data_subset, "prediction_date_Dayofweek", 7)
        add_cyclic_encoding(data_subset, "prediction_date_Week", 52)
        add_cyclic_encoding(data_subset, "prediction_date_Month", 12)
        add_cyclic_encoding(data_subset, "prediction_date_Hour", 24)

        # === Relative delta features (trig deltas and their encodings) ===
        data_subset["hours_ahead"] = lead_time
        data_subset["delta_Hour"] = data_subset["prediction_date_Hour"] - data_subset["timestamp_Hour"]
        data_subset["delta_Dayofweek"] = data_subset["prediction_date_Dayofweek"] - data_subset["timestamp_Dayofweek"]
        data_subset["delta_Week"] = data_subset["prediction_date_Week"] - data_subset["timestamp_Week"]
        data_subset["delta_Month"] = data_subset["prediction_date_Month"] - data_subset["timestamp_Month"]

        add_cyclic_encoding(data_subset, "delta_Hour", 24, True)
        add_cyclic_encoding(data_subset, "delta_Dayofweek", 7, True)
        add_cyclic_encoding(data_subset, "delta_Week", 52, True)
        add_cyclic_encoding(data_subset, "delta_Month", 12, True)

        # Drop unused prediction_date fields
        data_subset = data_subset.drop(
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
        data_subset = data_subset.drop(
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
        data_subset = data_subset.set_index(["item_id", "timestamp"])
        data_subset = data_subset.dropna()

        data_w_features.append(data_subset)

    return TabularDataFrame(pd.concat(data_w_features))


def quantile_regression(data_train: TimeSeriesDataFrame, data_test: TimeSeriesDataFrame, quantiles: np.ndarray, lead_times: np.ndarray, freq: pd.Timedelta) -> PredictionLeadTimes:
    """Performs quantile regression on time series data for each item and lead time using with cyclic and timestamp-based features.

    Parameters:
    ----------
    data_train : TimeSeriesDataFrame
        Training dataset for fitting the quantile regression models.
    data_test : TimeSeriesDataFrame
        Test dataset to evaluate model performance.
    quantiles : np.ndarray
        Array of quantiles to predict (e.g., [0.1, 0.5, 0.9]).
    lead_times : np.ndarray
        Array of lead times (in hours) for which forecasts should be made.
    freq : pd.Timedelta
        Frequency of the time series data (e.g., 1 hour).

    Returns:
    -------
    PredictionLeadTimes
        Predictions for each lead time and item_id in test set, with quantile estimates.
    """

    results_benchmark = {}

    # iterate over all lead times
    for lt in tqdm(lead_times):

        data_train_lt = create_cyclic_features(data_train, lt)
        data_test_lt = create_cyclic_features(data_test, lt)
        epsilon = 100_000

        # store results
        qr_models = {}
        test_results = {}

        predictions_item_ids = []

        # Postprocess each time series separately
        for item_id in data_test_lt.item_ids:

            # get data for the specific time series
            data_train_lt_item = data_train_lt.loc[[item_id]]
            data_test_lt_item = data_test_lt.loc[[item_id]]

            # retrieve constant to add
            # epsilon = -data_train_lt_item["target"].min() + 1
            epsilon = data_train_lt_item["target"].max() * 5
            epsilon = 100_000
            # fit a quantile regression for each quantile and make predictions on test dataset
            for q in quantiles:

                x_train = data_train_lt_item.drop(columns="target").astype(float).to_numpy()
                y_train = np.log(data_train_lt_item["target"].values + epsilon)
                x_test = data_test_lt_item.drop(columns="target").astype(float).to_numpy()

                # Add constant for intercept
                x_train = sm.add_constant(x_train)
                x_test = sm.add_constant(x_test)

                # Fit quantile regression model
                model = sm.QuantReg(y_train, x_train)
                qr_models[q] = model.fit(q=q)

                # make predictions
                predictions = qr_models[q].predict(x_test)
                test_results[q] = np.exp(predictions) - epsilon

            predictions_complete = np.array(list(test_results.values())).T
            predictions_item_ids.append(predictions_complete)

        predictions_item_ids = np.vstack(predictions_item_ids)
        results_benchmark[lt] = PredictionLeadTime(lead_time=lt, predictions=torch.tensor(predictions_item_ids), freq=freq, data=data_test_lt)

    return PredictionLeadTimes(results=results_benchmark)
