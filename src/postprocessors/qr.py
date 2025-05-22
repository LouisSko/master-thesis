import numpy as np
import statsmodels.api as sm
import torch
from tqdm import tqdm
from src.core.base import AbstractPostprocessor
from src.core.timeseries_evaluation import ForecastCollection, TimeSeriesForecast, HorizonForecast, TabularDataFrame
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")


class PostprocessorQR(AbstractPostprocessor):
    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir)
        self.qr_models = {}
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

    def fit(self, data: ForecastCollection) -> None:
        """Fit the quantile regression models for each lead time and quantile."""

        for item_id in tqdm(data.get_item_ids(), desc="Fitting QR Postprocessor for each time series (item)"):
            self.qr_models = {}
            item = data.get_time_series_forecast(item_id)

            for lead_time in item.get_lead_times():
                self.qr_models[lead_time] = {}

                for q in item.quantiles:
                    # Prepare training data
                    train_data = self._prepare_dataframe(item, lead_time, train=True)
                    y_train_raw = train_data["target"].values
                    y_train = self._transform(y_train_raw)
                    x_train = self._create_features(train_data, q)

                    if len(x_train) == 0:
                        logging.info("No calibration data available for item_id: %s, lead time: %s.", item_id, lead_time)
                        self.qr_models[lead_time][q] = None
                        continue

                    # Fit model
                    model = sm.QuantReg(y_train, x_train)
                    fit_result = model.fit(q=q)
                    self.qr_models[lead_time][q] = fit_result.params
            self.save_model(model=self.qr_models, item_id=item_id)

    def postprocess(self, data: ForecastCollection) -> ForecastCollection:
        """Apply the fitted quantile regression models to generate predictions on the test data."""
        results_item_ids = {}

        for item_id in tqdm(data.get_item_ids(), desc="Updating Forecasts using QR Postprocessor."):
            self.qr_models = self.load_model(item_id=item_id)
            results_lt = {}
            item = data.get_time_series_forecast(item_id)

            for lead_time in item.get_lead_times():
                df = self._prepare_dataframe(item, lead_time, train=False)
                adjusted_predictions = []

                for quantile in item.quantiles:
                    x_test = self._create_features(df, quantile)

                    params = self.qr_models[lead_time][quantile]

                    if params is None:
                        logging.info("No params available for item: %s, lead time: %s, quantile: %s. Keeping original predictions.", item_id, lead_time, quantile)
                        predictions = df[quantile].values
                    else:
                        predictions = x_test @ params
                        predictions = self._inverse_transform(predictions)  # inverse of arcsinh
                    adjusted_predictions.append(predictions)

                adjusted_predictions = np.column_stack(adjusted_predictions)
                results_lt[lead_time] = HorizonForecast(lead_time=lead_time, predictions=torch.tensor(adjusted_predictions))

            results_item_ids[item_id] = TimeSeriesForecast(item_id=item_id, lead_time_forecasts=results_lt, data=item.data, freq=item.freq, quantiles=item.quantiles)

        return ForecastCollection(item_ids=results_item_ids)
