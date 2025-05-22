import numpy as np
import statsmodels.api as sm
import torch
from src.core.base import AbstractPostprocessor
from src.core.timeseries_evaluation import TimeSeriesForecast, HorizonForecast, TabularDataFrame
from pathlib import Path
import logging
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")


class PostprocessorQR(AbstractPostprocessor):
    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir)
        self.epsilon = 100_000

    def _transform(self, x: np.ndarray) -> np.ndarray:
        """Apply arcsinh transform to any real-valued input."""
        # return np.arcsinh(x)
        return np.log(x + self.epsilon)

    def _inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Apply arcsinh transform to any real-valued input."""
        # return np.sinh(x)
        return np.exp(x) - self.epsilon

    def _create_features(self, data: TabularDataFrame, q: float) -> np.ndarray:
        """Creates Features for Quantile Regression"""
        x_raw = data[f"feature_{q}"].values
        x_transformed = self._transform(x_raw)
        x_train = x_transformed.reshape(-1, 1)
        x_train = sm.add_constant(x_train, has_constant="add")
        return x_train

    def _create_features2(self, data: TabularDataFrame, q: float) -> np.ndarray:
        """Alternative feature construction with multiple inputs."""
        trans_q = self._transform(data[f"feature_{q}"].values)
        trans_iqr = self._transform(data["feature_0.8"].values - data["feature_0.2"].values)
        x_train = np.column_stack((trans_q, trans_iqr))
        x_train = sm.add_constant(x_train, has_constant="add")
        return x_train

    def _prepare_dataframe(self, ts_forecast: TimeSeriesForecast, lead_time: int, train: bool = False) -> TabularDataFrame:
        """Creates Dataframe from Predictions"""
        if train:
            data = ts_forecast.to_dataframe(lead_time).iloc[self.ignore_first_n_train_entries :].dropna()
        else:
            data = ts_forecast.to_dataframe(lead_time)

        cols_rename = {q: f"feature_{q}" for q in ts_forecast.quantiles}
        data = data.rename(columns=cols_rename)
        cols_to_keep = list(cols_rename.values()) + ["target"]
        data = data[cols_to_keep]

        return TabularDataFrame(data)

    def _fit(self, data: TimeSeriesForecast) -> Any:
        """Fit the quantile regression models for each lead time and quantile."""

        qr_params = {}
        for lead_time in data.get_lead_times():
            qr_params[lead_time] = {}

            for q in data.quantiles:
                # Prepare training data
                train_data = self._prepare_dataframe(data, lead_time, train=True)
                y_train_raw = train_data["target"].values
                y_train = self._transform(y_train_raw)
                x_train = self._create_features(train_data, q)

                if len(x_train) == 0:
                    logging.info("No calibration data available for item_id: %s, lead time: %s.", data.item_id, lead_time)
                    qr_params[lead_time][q] = None
                    continue

                # Fit model
                model = sm.QuantReg(y_train, x_train)
                fit_result = model.fit(q=q)
                qr_params[lead_time][q] = fit_result.params  # only save the coefficients

        return qr_params

    def _postprocess(self, data: TimeSeriesForecast, params: Any) -> TimeSeriesForecast:
        """Apply the fitted quantile regression models to generate predictions on the test data."""

        results_lt = {}

        for lead_time in data.get_lead_times():
            df = self._prepare_dataframe(data, lead_time, train=False)
            adjusted_predictions = []

            for quantile in data.quantiles:
                x_test = self._create_features(df, quantile)

                lt_params = params[lead_time][quantile]

                if lt_params is None:
                    logging.info("No params available for item: %s, lead time: %s, quantile: %s. Keeping original predictions.", data.item_id, lead_time, quantile)
                    predictions = df[quantile].values
                else:
                    predictions = x_test @ lt_params  # only use the stored params
                    predictions = self._inverse_transform(predictions)  # inverse of arcsinh
                adjusted_predictions.append(predictions)

            adjusted_predictions = np.column_stack(adjusted_predictions)
            results_lt[lead_time] = HorizonForecast(lead_time=lead_time, predictions=torch.tensor(adjusted_predictions))

        return TimeSeriesForecast(item_id=data.item_id, lead_time_forecasts=results_lt, data=data.data, freq=data.freq, quantiles=data.quantiles)
