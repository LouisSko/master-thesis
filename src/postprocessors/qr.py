import numpy as np
import statsmodels.api as sm
from src.core.timeseries_evaluation import PredictionLeadTimes, PredictionLeadTime
import torch
from tqdm import tqdm
from src.core.base import AbstractPostprocessor
from src.core.timeseries_evaluation import PredictionLeadTimes, PredictionLeadTime, TabularDataFrame
import pandas as pd
from typing import List


class PostprocessorQR(AbstractPostprocessor):
    def __init__(self) -> None:
        super().__init__()
        self.ignore_first_n_train_entries = 500
        self.epsilon = 100_000
        self.qr_models = {}

    def _create_features(self, data: TabularDataFrame, q: float) -> np.ndarray:

        log_q = np.log(data[f"feature_{q}"] + self.epsilon)
        log_iqr = np.log(data["feature_0.8"] - data["feature_0.2"])
        x_train = np.column_stack((log_q, log_iqr))

        # x_train = np.log(data[f"feature_{q}"].values.reshape(-1, 1) + self.epsilon)
        x_train = sm.add_constant(x_train)
        return x_train

    def _prepare_dataframe(self, result_lead_time: PredictionLeadTime, item_id: int, quantiles: List[float]) -> TabularDataFrame:

        data = result_lead_time.to_dataframe(item_ids=[item_id]).iloc[self.ignore_first_n_train_entries :].dropna()
        cols_rename = {q: f"feature_{q}" for q in quantiles}
        data = data.rename(columns=cols_rename)
        cols_to_keep = list(cols_rename.values()) + ["target"]
        data = data[cols_to_keep]

        return TabularDataFrame(data)

    def fit(self, data: PredictionLeadTimes) -> None:
        """Fit the quantile regression models for each lead time and quantile."""

        # Fit model for each lead time, timeseries (item_id) and quantile
        self.qr_models = {}  # Reset the models
        for lt, results in tqdm(data.results.items(), desc="Fitting QR Postprocessor"):

            # extract quantiles and item_ids
            quantiles = results.quantiles
            item_ids = results.data.item_ids

            for item_id in item_ids:
                for q in quantiles:
                    # Prepare the training data
                    train_data = self._prepare_dataframe(results, item_id, quantiles)

                    y_train = np.log(train_data["target"].values + self.epsilon)
                    x_train = self._create_features(train_data, q)

                    # Fit quantile regression model for each quantile
                    model = sm.QuantReg(y_train, x_train)
                    self.qr_models[(lt, item_id, q)] = model.fit(q=q, max_iter=2000)

    def postprocess(self, data: PredictionLeadTimes) -> PredictionLeadTimes:
        """Apply the fitted quantile regression models to generate predictions on the test data."""

        postprocessing_results = {}

        for lt, results in tqdm(data.results.items()):

            # extract quantiles and item_ids
            quantiles = results.quantiles
            item_ids = results.data.item_ids
            freq = results.freq

            # use lists to store results
            predictions_item_ids = []
            test_data_item_ids = []

            for item_id in item_ids:

                test_data = self._prepare_dataframe(results, item_id, quantiles)

                test_results = {}

                for q in quantiles:
                    x_test = self._create_features(test_data, q)

                    # Retrieve the fitted model for the quantile and lead time
                    model: sm.QuantReg = self.qr_models.get((lt, item_id, q))
                    if model is None:
                        raise ValueError(f"No model for lead time:{lt}, item_id: {item_id}, quantile:{q}.")
                    else:
                        predictions = model.predict(x_test)
                        test_results[q] = np.exp(predictions) - self.epsilon

                predictions = np.array(list(test_results.values())).T
                predictions_item_ids.append(predictions)
                test_data_item_ids.append(test_data)

            combined_prediction_data = np.vstack(predictions_item_ids)
            combined_test_data = TabularDataFrame(pd.concat(test_data_item_ids))

            postprocessing_results[lt] = PredictionLeadTime(
                lead_time=lt, predictions=torch.tensor(combined_prediction_data), quantiles=quantiles, freq=freq, data=combined_test_data
            )

        return PredictionLeadTimes(results=postprocessing_results)
