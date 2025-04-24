import numpy as np
import statsmodels.api as sm
from src.core.timeseries_evaluation import PredictionLeadTimes, PredictionLeadTime
import torch
from tqdm import tqdm
from src.core.base import AbstractPostprocessor
from src.core.timeseries_evaluation import PredictionLeadTimes, PredictionLeadTime, TabularDataFrame
import pandas as pd


class PostprocessorQR(AbstractPostprocessor):
    def __init__(self) -> None:
        super().__init__()
        self.ignore_first_n_train_entries = 500
        self.epsilon = 100_000
        self.qr_models = {}

    def fit(self, data: PredictionLeadTimes) -> None:
        """Fit the quantile regression models for each lead time and quantile."""

        # Fit model for each lead time, timeseries (item_id) and quantile
        self.qr_models = {}  # Reset the models
        lead_times = data.results.keys()
        for lt in tqdm(lead_times):

            # extract quantiles and item_ids
            quantiles = data.results[lt].quantiles
            item_ids = data.results[lt].data.item_ids

            for item_id in item_ids:
                for q in quantiles:
                    # Prepare the training data
                    df_train = data.results[lt].to_dataframe(item_ids=[item_id]).iloc[self.ignore_first_n_train_entries :].dropna()
                    df_train = df_train.dropna()
                    cols_rename = {q: f"feature_{q}" for q in quantiles}
                    df_train = df_train.rename(columns=cols_rename)
                    cols_to_keep = list(cols_rename.values()) + ["target"]
                    df_train = df_train[cols_to_keep]
                    # df_train.index = pd.MultiIndex.from_arrays([[item_id] * len(df_train), df_train.index], names=["item_id", df_train.index.name])
                    train_data = TabularDataFrame(df_train)

                    x_train = np.log(train_data[f"feature_{q}"].values.reshape(-1, 1) + self.epsilon)
                    y_train = np.log(train_data["target"].values + self.epsilon)
                    x_train = sm.add_constant(x_train)

                    # Fit quantile regression model for each quantile
                    model = sm.QuantReg(y_train, x_train)
                    self.qr_models[(lt, item_id, q)] = model.fit(q=q, max_iter=2000)

    def postprocess(self, data: PredictionLeadTimes) -> PredictionLeadTimes:
        """Apply the fitted quantile regression models to generate predictions on the test data."""

        postprocessing_results = {}
        lead_times = list(data.results.keys())

        for lt in tqdm(lead_times):

            # extract quantiles and item_ids
            quantiles = data.results[lt].quantiles
            item_ids = data.results[lt].data.item_ids
            freq = data.results[lt].freq

            # use lists to store results
            predictions_item_ids = []
            test_data_item_ids = []

            for item_id in item_ids:
                # prepare the test data
                df_test = data.results[lt].to_dataframe(item_ids=[item_id])
                cols_rename = {q: f"feature_{q}" for q in quantiles}
                df_test = df_test.rename(columns=cols_rename)
                cols_to_keep = list(cols_rename.values()) + ["target"]
                # df_test.index = pd.MultiIndex.from_arrays([[item_id] * len(df_test), df_test.index], names=["item_id", df_test.index.name])
                test_data = TabularDataFrame(df_test[cols_to_keep])

                test_results = {}

                for q in quantiles:
                    x_test = np.log(test_data[f"feature_{q}"].values.reshape(-1, 1) + self.epsilon)
                    x_test = sm.add_constant(x_test)

                    # Retrieve the fitted model for the quantile and lead time
                    model = self.qr_models.get((lt, item_id, q))
                    if model is not None:
                        predictions = model.predict(x_test)
                        test_results[q] = np.exp(predictions) - self.epsilon
                    if model is None:
                        raise ValueError(f"No model for lead time:{lt}, item_id: {item_id}, quantile:{q}.")

                predictions = np.array(list(test_results.values())).T
                predictions_item_ids.append(predictions)
                test_data_item_ids.append(test_data)

            combined_prediction_data = np.vstack(predictions_item_ids)
            combined_test_data = TabularDataFrame(pd.concat(test_data_item_ids))

            postprocessing_results[lt] = PredictionLeadTime(
                lead_time=lt, predictions=torch.tensor(combined_prediction_data), quantiles=quantiles, freq=freq, data=combined_test_data
            )

        return PredictionLeadTimes(results=postprocessing_results)
