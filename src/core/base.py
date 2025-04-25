from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
from src.core.timeseries_evaluation import PredictionLeadTimes, TabularDataFrame
from autogluon.timeseries import TimeSeriesDataFrame
from typing import Dict, List, Optional, Type, Union


class AbstractPredictor(ABC):
    def __init__(self, lead_times: List[int] = [1, 2, 3], freq: pd.Timedelta = pd.Timedelta("1h")) -> None:

        self.lead_times = lead_times
        self.freq = freq

    @abstractmethod
    def fit(self, data_train: TimeSeriesDataFrame, dat_val: Optional[TimeSeriesDataFrame] = None) -> None:
        pass

    @abstractmethod
    def predict(self, data: TimeSeriesDataFrame, previous_context_data: Optional[TimeSeriesDataFrame] = None, predict_only_last_timestep: bool = False) -> PredictionLeadTimes:
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


class AbstractPipeline(ABC):
    def __init__(
        self,
        model: Type[AbstractPredictor],
        model_kwargs: Dict,
        data: Union[TimeSeriesDataFrame, TabularDataFrame],
        postprocessor: Optional[Type[AbstractPostprocessor]] = None,
    ):
        self.model = model
        self.model_kwargs = model_kwargs
        self.data = data
        self.postprocessor = postprocessor

    @abstractmethod
    def backtest(self) -> PredictionLeadTimes:
        raise NotImplementedError

    @abstractmethod
    def inference(self) -> PredictionLeadTimes:
        raise NotImplementedError
