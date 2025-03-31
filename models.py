from tqdm import tqdm
from data_preparation import ChronosBacktestingDataset
from timeseries import PredictionLeadTimes, PredictionLeadTime
import torch
from chronos import BaseChronosPipeline
from autogluon.timeseries import TimeSeriesDataFrame
from torch.utils.data import DataLoader
from typing import List
import pandas as pd


def predict_chronos(pipeline: BaseChronosPipeline, data: TimeSeriesDataFrame, lead_times: List = [1, 2, 3], freq: pd.Timedelta = pd.Timedelta("1h")) -> PredictionLeadTimes:
    """Make zero shot predictions with chronos bolt model"""

    prediction_length = max(lead_times)

    if prediction_length > 64:
        raise ValueError("Maximum lead time is 64")

    ds = ChronosBacktestingDataset(data, context_length=2048)
    dl = DataLoader(ds, batch_size=64)

    results = {ld: None for ld in lead_times}
    forecasts = []

    for batch in tqdm(dl):
        forecast = pipeline.predict(context=batch, prediction_length=prediction_length)
        forecasts.append(forecast)

    forecasts = torch.vstack(forecasts)

    for lt in lead_times:
        results[lt] = PredictionLeadTime(lead_time=lt, predictions=forecasts[..., lt - 1], freq=freq, data=data)

    return PredictionLeadTimes(results=results)
