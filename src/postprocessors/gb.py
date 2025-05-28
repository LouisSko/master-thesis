import numpy as np
import torch
from src.core.base import AbstractPostprocessor
from src.core.timeseries_evaluation import TimeSeriesForecast, HorizonForecast
from pathlib import Path
import logging
from typing import Any, Optional
from lightgbm import LGBMRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")


class PostprocessorGB(AbstractPostprocessor):
    """Gradient boosting postprocessor"""

    def __init__(self, output_dir: Optional[Path] = None, name: Optional[str] = None, n_jobs: int = 1) -> None:
        super().__init__(output_dir, name, n_jobs)
        self.n_jobs_lgbm = -1

    def _fit(self, data: TimeSeriesForecast) -> Any:
        """Fit the quantile regression models for each lead time and quantile."""

        gb_params = {}

        for lead_time in data.get_lead_times():
            gb_params[lead_time] = {}
            df = data.to_dataframe(lead_time)
            x_train = df[data.quantiles]
            y_train = df["target"]

            for quantile in data.quantiles:
                model = LGBMRegressor(objective="quantile", alpha=quantile, n_jobs=self.n_jobs_lgbm, verbose=-1)

                if len(x_train) == 0:
                    logging.info("No calibration data available for item_id: %s, lead time: %s.", data.item_id, lead_time)
                    gb_params[lead_time][quantile] = None
                    continue

                model.fit(x_train, y_train)
                gb_params[lead_time][quantile] = model

        return gb_params

    def _postprocess(self, data: TimeSeriesForecast, params: Any) -> TimeSeriesForecast:
        """Apply the fitted quantile regression models to generate predictions on the test data."""

        results_lt = {}

        for lead_time in data.get_lead_times():
            df = data.to_dataframe(lead_time)
            x_train = df[data.quantiles].to_numpy()
            adjusted_predictions = []

            for quantile in data.quantiles:

                model: LGBMRegressor = params[lead_time][quantile]

                if model is None:
                    logging.info("No model available for item: %s, lead time: %s, quantile: %s. Keeping original predictions.", data.item_id, lead_time, quantile)
                    predictions = df[quantile].values
                else:
                    predictions = model.predict(x_train)

                adjusted_predictions.append(predictions)

            adjusted_predictions = np.column_stack(adjusted_predictions)
            results_lt[lead_time] = HorizonForecast(lead_time=lead_time, predictions=torch.tensor(adjusted_predictions))

        return TimeSeriesForecast(item_id=data.item_id, lead_time_forecasts=results_lt, data=data.data, freq=data.freq, quantiles=data.quantiles)
