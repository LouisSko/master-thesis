import numpy as np
from src.core.base import AbstractPostprocessor
from src.core.timeseries_evaluation import ForecastCollection, TimeSeriesForecast, HorizonForecast, TARGET
import torch
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")


class PostprocessorEQC(AbstractPostprocessor):
    """
    EmpiricalQuantileCalibrator.

    This postprocessor adjusts quantile regression outputs by computing empirical offsets to improve quantile coverage.
    """

    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir)
        self.conf_thresholds = {}
        self.ignore_first_n_train_entries = 0

    def fit(self, data: ForecastCollection):
        """
        Fits the calibrator to the prediction data by computing empirical quantile offsets.

        The method calculates how much each predicted quantile needs to be shifted
        to achieve the desired empirical coverage on the calibration data.

        Parameters:
        -----------
        data : PredictionLeadTimes
            Predictions used for calibration.
        """
        data = deepcopy(data)

        for item_id in tqdm(data.get_item_ids(), desc="Fitting EQC Postprocessor for each time series (item)"):
            self.conf_thresholds = {}
            item = data.get_time_series_forecast(item_id)
            for lead_time in item.get_lead_times():
                self.conf_thresholds[lead_time] = {}
                # TODO: could be made more efficient by accessing the predictions directly
                df = item.to_dataframe(lead_time).iloc[self.ignore_first_n_train_entries :].dropna().copy()

                if len(df) == 0:
                    logging.info("No calibration data available for item_id: %s, lead time: %s.", item_id, lead_time)
                    for q in item.quantiles:
                        self.conf_thresholds[lead_time][q] = None
                    continue

                for q in item.quantiles:
                    scores = df[TARGET] - df[q]
                    self.conf_thresholds[lead_time][q] = np.quantile(scores, q=q)

            self.save_model(model=self.conf_thresholds, item_id=item_id)

    def postprocess(self, data: ForecastCollection) -> ForecastCollection:
        """
        Applies the empirical quantile offsets to adjust predictions.

        Parameters:
        -----------
        data : ForecastCollection
            Prediction data to be postprocessed, containing raw quantile predictions.

        Returns:
        --------
        ForecastCollection
            A new `ForecastCollection` object with calibrated quantile predictions.
        """
        data = deepcopy(data)

        results_item_ids = {}

        for item_id in tqdm(data.get_item_ids(), desc="Updating Forecasts using EQC Postprocessor."):
            results_lt = {}
            item = data.get_time_series_forecast(item_id)
            self.conf_thresholds = self.load_model(item_id=item_id)
            for lead_time in item.get_lead_times():

                #### specific code #####
                df = item.to_dataframe(lead_time)  # TODO: could be made more efficient by accessing the predictions directly
                adjusted_predictions = []
                for quantile in item.quantiles:
                    offset = self.conf_thresholds[lead_time][quantile]
                    if offset is None:
                        logging.info("No params available for item: %s, lead time: %s, quantile: %s. Keeping original predictions.", item_id, lead_time, quantile)
                        conformalized_predictions = np.array(df[quantile])
                    else:
                        conformalized_predictions = np.array(df[quantile] + offset)
                    adjusted_predictions.append(conformalized_predictions)
                adjusted_predictions = np.column_stack(adjusted_predictions)
                #### specific code #####

                results_lt[lead_time] = HorizonForecast(lead_time=lead_time, predictions=torch.tensor(adjusted_predictions))

            results_item_ids[item_id] = TimeSeriesForecast(item_id=item_id, lead_time_forecasts=results_lt, data=item.data, freq=item.freq, quantiles=item.quantiles)

        return ForecastCollection(item_ids=results_item_ids)
