from tqdm import tqdm
import torch
from autogluon.timeseries import TimeSeriesDataFrame
from typing import List, Optional
import pandas as pd
import numpy as np
from src.core.base import AbstractPredictor
from src.core.timeseries_evaluation import PredictionLeadTimes, PredictionLeadTime, TabularDataFrame
from fastai.tabular.core import add_datepart
import statsmodels.api as sm
from pydantic import Field


class QuantileRegression(AbstractPredictor):
    def __init__(
        self,
        quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        lead_times: List[int] = Field(default_factory=lambda: [1, 2, 3]),
        freq: pd.Timedelta = pd.Timedelta("1h"),
    ) -> None:
        super().__init__(lead_times, freq)

        self.quantiles = quantiles
        self.models_qr = {}
        self.epsilon = 100_000

    def fit(self, data_train: PredictionLeadTimes, data_val: Optional[TimeSeriesDataFrame] = None) -> None:

        if data_val is not None:
            print("data_val is not used.")

        for lt in tqdm(self.lead_times):

            data_lt = self._create_cyclic_features(data_train, lt, dropna=True)

            # Postprocess each time series separately
            for item_id in data_lt.item_ids:
                # get data for the specific time series
                data_lt_item = data_lt.loc[[item_id]]
                # fit a quantile regression for each quantile and make predictions on test dataset
                for q in self.quantiles:

                    x_train = data_lt_item.drop(columns="target").astype(float).to_numpy()
                    y_train = np.log(data_lt_item["target"].values + self.epsilon)
                    # Add constant for intercept
                    x_train = sm.add_constant(x_train)
                    # fit model
                    model = sm.QuantReg(y_train, x_train)
                    self.models_qr[(lt, item_id, q)] = model.fit(q=q)

    def predict(
        self, data_train: PredictionLeadTimes, previous_context_data: Optional[TimeSeriesDataFrame] = None, predict_only_last_timestep: bool = False
    ) -> PredictionLeadTimes:

        if self.models_qr is None:
            raise ValueError("Need to fit models first.")

        results = {ld: None for ld in self.lead_times}

        for lt in tqdm(self.lead_times):

            data_lt = self._create_cyclic_features(data_train, lt, dropna=False)

            # store results
            test_results = {}
            predictions_item_ids = []

            # Postprocess each time series separately
            for item_id in data_lt.item_ids:
                # get data for the specific time series
                data_lt_item = data_lt.loc[[item_id]]
                # fit a quantile regression for each quantile and make predictions on test dataset
                for q in self.quantiles:
                    x_test = data_lt_item.drop(columns="target").astype(float).to_numpy()
                    x_test = sm.add_constant(x_test)
                    predictions = self.models_qr[(lt, item_id, q)].predict(x_test)
                    test_results[q] = np.exp(predictions) - self.epsilon

                predictions_complete = np.array(list(test_results.values())).T
                predictions_item_ids.append(predictions_complete)

            predictions_item_ids = np.vstack(predictions_item_ids)
            results[lt] = PredictionLeadTime(lead_time=lt, predictions=torch.tensor(predictions_item_ids), freq=self.freq, data=data_lt)

        return PredictionLeadTimes(results=results)

    def _add_cyclic_encoding(self, data: pd.DataFrame, colname: str, period: int, drop: bool = False):
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

    def _create_cyclic_features(self, data: TimeSeriesDataFrame, lead_time: int = 1, freq: pd.Timedelta = pd.Timedelta("1h"), dropna: bool = True) -> TabularDataFrame:
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
            self._add_cyclic_encoding(data_subset, "timestamp_Dayofweek", 7)
            self._add_cyclic_encoding(data_subset, "timestamp_Week", 52)
            self._add_cyclic_encoding(data_subset, "timestamp_Month", 12)
            self._add_cyclic_encoding(data_subset, "timestamp_Hour", 24)

            # === Features from PREDICTION date ===
            data_subset["prediction_date"] = data_subset["timestamp"] + lead_time * freq
            add_datepart(data_subset, "prediction_date", prefix="prediction_date_", drop=False)
            data_subset["prediction_date_Hour"] = data_subset["prediction_date"].dt.hour
            self._add_cyclic_encoding(data_subset, "prediction_date_Hour", 24)
            self._add_cyclic_encoding(data_subset, "prediction_date_Dayofweek", 7)
            self._add_cyclic_encoding(data_subset, "prediction_date_Week", 52)
            self._add_cyclic_encoding(data_subset, "prediction_date_Month", 12)

            # === Relative delta features (trig deltas and their encodings) ===
            data_subset["hours_ahead"] = lead_time
            data_subset["delta_Hour"] = data_subset["prediction_date_Hour"] - data_subset["timestamp_Hour"]
            data_subset["delta_Dayofweek"] = data_subset["prediction_date_Dayofweek"] - data_subset["timestamp_Dayofweek"]

            self._add_cyclic_encoding(data_subset, "delta_Hour", 24, True)
            self._add_cyclic_encoding(data_subset, "delta_Dayofweek", 7, True)

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

            data_subset = data_subset.set_index(["item_id", "timestamp"])

            if dropna:
                data_subset = data_subset.dropna()

            data_w_features.append(data_subset)

        return TabularDataFrame(pd.concat(data_w_features))
