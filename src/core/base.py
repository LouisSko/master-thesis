from abc import ABC, abstractmethod
from typing import List
import pandas as pd
from src.core.timeseries import PredictionLeadTimes
from autogluon.timeseries import TimeSeriesDataFrame


class AbstractPredictor(ABC):
    def __init__(self, lead_times: List[int] = [1, 2, 3], freq: pd.Timedelta = pd.Timedelta("1h")) -> None:

        self.lead_times = lead_times
        self.freq = freq

    @abstractmethod
    def fit(self, data: TimeSeriesDataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, data: TimeSeriesDataFrame, predict_only_last_timestep: bool = False) -> PredictionLeadTimes:
        pass


class AbstractPostprocessor(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def fit(self, data: PredictionLeadTimes) -> None:
        pass

    @abstractmethod
    def postprocess(self, data: PredictionLeadTimes) -> PredictionLeadTimes:
        pass
