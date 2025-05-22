import numpy as np
from src.core.base import AbstractPostprocessor
from src.core.timeseries_evaluation import TimeSeriesForecast, HorizonForecast, TARGET
import torch
from pathlib import Path
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")


class PostprocessorEQC(AbstractPostprocessor):
    """
    EmpiricalQuantileCalibrator.

    This postprocessor adjusts quantile regression outputs by computing empirical offsets to improve quantile coverage.
    """
    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir)

    def _fit(self, data: TimeSeriesForecast) -> Dict[int, Dict[float, float]]:
        """
        Calibrates predicted quantiles by computing empirical offsets for each lead time.

        This method estimates how much each predicted quantile should be shifted so that
        the resulting quantile forecasts achieve the correct empirical coverage on the
        calibration set. For each lead time and quantile, it calculates the empirical
        error between the predicted quantile and the true target and stores the
        corresponding offset.

        Parameters
        ----------
        data : TimeSeriesForecast
            Forecast data for a single item, including predicted quantiles and targets,
            used for calibration.

        Returns
        -------
        Dict[int, Dict[float, float]]
            A nested dictionary of empirical quantile offsets structured as:
            {
                lead_time_1: {
                    quantile_1: offset,
                    quantile_2: offset,
                    ...
                },
                ...
            }
            where each offset can be used to shift the corresponding quantile prediction.
        """
        conf_thresholds = {}
        for lead_time in data.get_lead_times():
            conf_thresholds[lead_time] = {}
            # TODO: could be made more efficient by accessing the predictions directly
            df = data.to_dataframe(lead_time).iloc[self.ignore_first_n_train_entries :].dropna().copy()

            if len(df) == 0:
                logging.info("No calibration data available for item_id: %s, lead time: %s.", data.item_id, lead_time)
                for q in data.quantiles:
                    conf_thresholds[lead_time][q] = None
                continue

            for q in data.quantiles:
                scores = df[TARGET] - df[q]
                conf_thresholds[lead_time][q] = np.quantile(scores, q=q)

        return conf_thresholds

    def _postprocess(self, data: TimeSeriesForecast, params: Dict[int, Dict[float, float]]) -> TimeSeriesForecast:
        """
        Applies the empirical quantile offsets to adjust predictions.

        Parameters
        -----------
        data : TimeSeriesForecast
            Prediction data to be postprocessed, containing raw quantile predictions.

        params : Dict[int, Dict[float, float]]
            A nested dictionary of empirical quantile offsets

        Returns
        --------
        TimeSeriesForecast
            A new `TimeSeriesForecast` object with calibrated quantile predictions.
        """
        results_lt = {}
        for lead_time in data.get_lead_times():
            df = data.to_dataframe(lead_time)  # TODO: could be made more efficient by accessing the predictions directly
            adjusted_predictions = []
            for quantile in data.quantiles:
                offset = params[lead_time][quantile]
                if offset is None:
                    logging.info("No params available for item: %s, lead time: %s, quantile: %s. Keeping original predictions.", data.item_id, lead_time, quantile)
                    conformalized_predictions = np.array(df[quantile])
                else:
                    conformalized_predictions = np.array(df[quantile] + offset)
                adjusted_predictions.append(conformalized_predictions)
            adjusted_predictions = np.column_stack(adjusted_predictions)

            results_lt[lead_time] = HorizonForecast(lead_time=lead_time, predictions=torch.tensor(adjusted_predictions))

        return TimeSeriesForecast(item_id=data.item_id, lead_time_forecasts=results_lt, data=data.data, freq=data.freq, quantiles=data.quantiles)
