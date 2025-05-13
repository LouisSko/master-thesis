import numpy as np
import statsmodels.api as sm
import torch
from tqdm import tqdm
from src.core.base import AbstractPostprocessor
from src.core.timeseries_evaluation import ForecastCollection, TimeSeriesForecast, HorizonForecast, TabularDataFrame
from typing import List


class PostprocessorQR(AbstractPostprocessor):
    def __init__(self) -> None:
        super().__init__()
        self.epsilon = 100_000
        self.qr_models = {}

    def _create_features(self, data: TabularDataFrame, q: float) -> np.ndarray:
        """Creates Features for Quantile Regression"""
        log_q = np.log(data[f"feature_{q}"] + self.epsilon)
        x_train = np.array(log_q).reshape(-1, 1)
        x_train = sm.add_constant(x_train, has_constant="add")
        return x_train

    def _create_features2(self, data: TabularDataFrame, q: float) -> np.ndarray:
        """Creates Features for Quantile Regression"""
        log_q = np.log(data[f"feature_{q}"] + self.epsilon)
        log_iqr = np.log(data["feature_0.8"] - data["feature_0.2"])
        x_train = np.column_stack((log_q, log_iqr))

        # x_train = np.log(data[f"feature_{q}"].values.reshape(-1, 1) + self.epsilon)
        x_train = sm.add_constant(x_train, has_constant="add")
        return x_train

    def _prepare_dataframe(self, result_lead_time: HorizonForecast, quantiles: List[float], train: bool = False) -> TabularDataFrame:
        """Creates Dataframe from Predictions"""
        if train:
            data = result_lead_time.to_dataframe().iloc[self.ignore_first_n_train_entries :].dropna()
        else:
            data = result_lead_time.to_dataframe()

        cols_rename = {q: f"feature_{q}" for q in quantiles}
        data = data.rename(columns=cols_rename)
        cols_to_keep = list(cols_rename.values()) + ["target"]
        data = data[cols_to_keep]

        return TabularDataFrame(data)

    def fit(self, data: ForecastCollection) -> None:
        """Fit the quantile regression models for each lead time and quantile."""

        # Fit model for each lead time, timeseries (item_id) and quantile
        self.qr_models = {}
        for item_id in tqdm(data.get_item_ids(), desc="Fitting QR Postprocessor for each time series (item)"):
            self.qr_models[item_id] = {}
            item = data.get_time_series_forecast(item_id)
            for lead_time in item.get_lead_times():
                self.qr_models[item_id][lead_time] = {}
                lt_item = item.get_lead_time_forecast(lead_time)

                #### specific code #####
                for q in lt_item.quantiles:
                    # Prepare the training data
                    train_data = self._prepare_dataframe(lt_item, lt_item.quantiles, True)
                    y_train = np.log(train_data["target"].values + self.epsilon)
                    x_train = self._create_features(train_data, q)
                    # Fit quantile regression model for each quantile
                    model = sm.QuantReg(y_train, x_train)
                    self.qr_models[item_id][lead_time][q] = model.fit(q=q, max_iter=2000)
                #### specific code #####

    def postprocess(self, data: ForecastCollection) -> ForecastCollection:
        """Apply the fitted quantile regression models to generate predictions on the test data."""

        results_item_ids = {}

        for item_id in tqdm(data.get_item_ids(), desc="Updating Forecasts using QR Postprocessor."):
            results_lt = {}
            item = data.get_time_series_forecast(item_id)
            for lead_time in item.get_lead_times():
                lt_item = item.get_lead_time_forecast(lead_time)

                #### specific code #####
                df = self._prepare_dataframe(lt_item, lt_item.quantiles)
                adjusted_predictions = []
                for quantile in lt_item.quantiles:
                    x_test = self._create_features(df, quantile)
                    # Retrieve the fitted model for the quantile and lead time
                    try:
                        model: sm.QuantReg = self.qr_models[item_id][lead_time][quantile]
                    except:
                        raise ValueError(f"No model for item_id: {item_id}, lead time:{lead_time}, quantile:{q}.")
                    else:
                        predictions = model.predict(x_test)
                        adjusted_predictions.append(np.exp(predictions) - self.epsilon)
                adjusted_predictions = np.column_stack(adjusted_predictions)
                #### specific code #####

                results_lt[lead_time] = HorizonForecast(
                    lead_time=lt_item.lead_time,
                    predictions=torch.tensor(adjusted_predictions),
                    quantiles=lt_item.quantiles,
                    freq=lt_item.freq,
                )

            results_item_ids[item_id] = TimeSeriesForecast(item_id=item_id, lead_time_forecasts=results_lt, data=item.data)

        return ForecastCollection(items=results_item_ids)
