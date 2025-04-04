from timeseries import PredictionLeadTimes, PredictionLeadTime
from autogluon.timeseries import TimeSeriesDataFrame
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import torch


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
