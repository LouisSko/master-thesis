import numpy as np
from src.core.base import AbstractPostprocessor
from src.core.timeseries_evaluation import PredictionLeadTimes, PredictionLeadTime
import torch
from typing import Dict
from copy import deepcopy


class PostprocessorEQC(AbstractPostprocessor):
    """
    EmpiricalQuantileCalibrator.

    This postprocessor adjusts quantile regression outputs by computing empirical offsets to improve quantile coverage.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conf_thresholds_lt = {}

    def _empirical_quantile_offset(self, predictions: PredictionLeadTime) -> Dict[int, Dict[float, float]]:
        """
        Computes empirical quantile offsets for each item ID and quantile.

        The method calculates how much each predicted quantile needs to be shifted
        to achieve the desired empirical coverage on the calibration data.

        Parameters:
        -----------
        predictions : PredictionLeadTime
            Object containing quantile predictions and true target values for a specific lead time.

        Returns:
        --------
        Dict[int, Dict[float, float]]
            A nested dictionary where the outer key is the item ID, the inner key is the quantile,
            and the value is the computed empirical offset for that quantile.
        """

        conformalized_thresholds = {}
        predictions = deepcopy(predictions)

        for item_id in predictions.data.item_ids:
            conformalized_thresholds[item_id] = {}

            df = predictions.to_dataframe(item_ids=[item_id]).copy()
            df = df.dropna()

            for q in predictions.quantiles:
                scores = df["target"] - df[q]
                conformalized_thresholds[item_id][q] = np.quantile(scores, q=q)

                # q_preds_conf = q_pred + threshold
                # scores = y_true-q_preds_conf
                # coverage = (scores>0).mean()

        return conformalized_thresholds

    def fit(self, data: PredictionLeadTimes):
        """
        Fits the calibrator to the prediction data by computing empirical quantile offsets.

        Parameters:
        -----------
        data : PredictionLeadTimes
            Predictions used for calibration.
        """

        data = deepcopy(data)
        for lt, preds in data.results.items():
            self.conf_thresholds_lt[lt] = self._empirical_quantile_offset(preds)

    def postprocess(self, data: PredictionLeadTimes) -> PredictionLeadTimes:
        """
        Applies the empirical quantile offsets to adjust predictions.

        Parameters:
        -----------
        data : PredictionLeadTimes
            Prediction data to be postprocessed, containing raw quantile predictions.

        Returns:
        --------
        PredictionLeadTimes
            A new `PredictionLeadTimes` object with calibrated quantile predictions.
        """

        data = deepcopy(data)
        postprocessing_results = {}

        # iterate over different lead times
        for lead_time, preds in data.results.items():
            prediction_lead_time = []

            # iterate over item ids (different time series)
            for item_id in preds.data.item_ids:
                predictions_item_id = []
                test_data = preds.to_dataframe(item_ids=[item_id])

                # iterate over quantiles
                for i, quantile in enumerate(preds.quantiles):
                    conformalized_predictions = np.array(test_data[quantile] + self.conf_thresholds_lt[lead_time][item_id][quantile])

                    predictions_item_id.append(conformalized_predictions)
                prediction_lead_time.append(predictions_item_id)
            combined_prediction_data = np.hstack(prediction_lead_time).T

            postprocessing_results[lead_time] = PredictionLeadTime(
                lead_time=lead_time, predictions=torch.tensor(combined_prediction_data), quantiles=preds.quantiles, freq=preds.freq, data=preds.data
            )

        return PredictionLeadTimes(results=postprocessing_results)
