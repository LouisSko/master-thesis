from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
from src.core.timeseries_evaluation import PredictionLeadTimes, TabularDataFrame
from autogluon.timeseries import TimeSeriesDataFrame
from typing import Dict, List, Optional, Type, Union, Literal
from datetime import datetime


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
        """Initialize the pipeline with model, data, and optional postprocessor.

        Parameters
        ----------
        model : Type[AbstractPredictor]
            The predictor model class to use
        model_kwargs : Dict
            Keyword arguments to pass to the model constructor
        data : Union[TimeSeriesDataFrame, TabularDataFrame]
            The dataset to use for training and prediction
        postprocessor : Optional[Type[AbstractPostprocessor]], default=None
            Optional postprocessor class for refining predictions
        """

        self.model = model
        self.model_kwargs = model_kwargs
        self.data = data
        self.postprocessor = postprocessor

    @abstractmethod
    def backtest(
        self,
        test_start_date: pd.Timestamp,
        end_date: Optional[pd.Timestamp] = None,
        rolling_window_eval: bool = False,
        train_window_size: Optional[pd.DateOffset] = None,
        val_window_size: Optional[pd.DateOffset] = None,
        test_window_size: Optional[pd.DateOffset] = None,
        train: bool = False,
        calibration_based_on: Optional[Union[Literal["val", "train", "train_val"], pd.DateOffset]] = None,
        output_dir: Optional[str] = None,
    ) -> PredictionLeadTimes:
        """Perform backtesting of the model.

        Parameters
        ----------
        test_start_date : pd.Timestamp
            The starting date for the test dataset
        end_date : Optional[pd.Timestamp], default=None
            The ending date for the test dataset, defaults to the last date in the data
        rolling_window_eval : bool, default=False
            Whether to use rolling window approach for backtesting
        train_window_size : Optional[pd.DateOffset], default=None
            The size of the training window
        val_window_size : Optional[pd.DateOffset], default=None
            The size of the validation window
        test_window_size : Optional[pd.DateOffset], default=None
            The size of the test window
        train : bool, default=False
            Whether to train the model before prediction
        calibration_based_on : Optional[Union[Literal["val", "train", "train_val"], pd.DateOffset]], default=None
            Which dataset to use for calibration
        output_dir : Optional[str], default=None
            Base name for saving prediction files and config

        Returns
        -------
        PredictionLeadTimes
            The predictions for each lead time
        """

        raise NotImplementedError

    @abstractmethod
    def inference(self) -> PredictionLeadTimes:
        raise NotImplementedError

    def get_config(
        self,
        backtest_params: Dict,
        additional_config_info: Optional[Dict] = None,
    ) -> Dict:
        """Save configuration information to a JSON file.

        Parameters
        ----------
        prediction_file_name : str
            Base filename for the config file
        backtest_params : Dict
            Dictionary containing the backtest parameters
        additional_config_info : Optional[Dict], default=None
            Optional additional information to include in the config file

        Returns
        -------

        Dict
            A dictionary containing the configuration
        """

        # Create base config with model information
        config = {
            "model": self.model.__name__,
            "model_kwargs": self.model_kwargs,
            "postprocessor": self.postprocessor.__name__ if self.postprocessor is not None else None,
            "timestamp": datetime.now().isoformat(),
        }

        # Add backtest parameters
        config.update(backtest_params)

        # Add any additional information
        if additional_config_info:
            config.update(additional_config_info)

        return config
