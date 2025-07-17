from autogluon.timeseries import TimeSeriesDataFrame
from typing import List, Optional
from src.core.base import AbstractPredictor
import logging
from src.core.timeseries_evaluation import ForecastCollection
from src.core.utils import set_global_seed
from pydantic import Field
from pathlib import Path
from torch.utils.data import DataLoader
from src.predictors.chronos import BaseTimeSeriesDataset
from tqdm.auto import tqdm
from src.core.timeseries_evaluation import ForecastCollection
import torch
import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")
set_global_seed()


class TiRex(AbstractPredictor):
    """
    Wrapper around TiRex predictor from https://github.com/NX-AI/tirex

    TiRex must be set up as a separate microservice on a machine with CUDA installed.
    It only runs on GPU and performs zero-shot forecasting (no fitting required).
    Link to setting up TiRex as a microservice: https://github.com/LouisSko/tirex-microservice

    Parameters
    ----------
    quantiles : list of float, optional
        List of quantiles to predict. Defaults to [0.1, 0.2, ..., 0.9].
    lead_times : list of int, optional
        List of lead times (forecast horizons) to predict. Defaults to [1, 2, 3].
    output_dir : Path or None, optional
        Directory to save the fitted model. Defaults to None.
    tirex_service_url : str
        endpoint of the tirex microservice. e.g. "http://localhost:8000".
        Look at Readme for instructions for setting this up.
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        lead_times: List[int] = Field(default_factory=lambda: [1, 2, 3]),
        output_dir: Optional[Path] = None,
        tirex_service_url: str = "http://localhost:8000",
    ) -> None:
        super().__init__(lead_times, output_dir)

        self.quantiles = quantiles
        self.predictor = None
        self.context_length = 2048
        self.tirex_service_url = tirex_service_url

        self.health_endpoint = f"{self.tirex_service_url}/health"
        self.forecast_endpoint = f"{self.tirex_service_url}/tirex-forecast"
        # Check if the microservice is alive
        try:
            resp = requests.get(self.health_endpoint, timeout=10)
            resp.raise_for_status()
            logging.info("TiRex microservice is available.")
        except requests.RequestException as e:
            logging.error("TiRex microservice is not reachable. Make sure it is running.")
            raise RuntimeError("TiRex microservice unavailable.") from e

    def _fit(self, data_train: TimeSeriesDataFrame, data_val: Optional[TimeSeriesDataFrame] = None) -> None:
        """
        TiRex is a zero-shot model and the current implementation does not support fine-tuning.
        """
        logging.info("No fitting required. TiRex is a zero-shot forecaster and implementation does not support fine tuning currently.")

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        previous_context_data: Optional[TimeSeriesDataFrame] = None,
        rolling: bool = False,
        window_step: int = 1,
    ) -> ForecastCollection:
        """
        Generates forecasts for each time series by calling the TiRex microservice.

        This method can perform either:
        - Single-shot prediction (predicting from the most recent context window), or
        - Rolling backtesting (sliding a window across the time series to predict at each time point).

        Parameters
        ----------
        data : TimeSeriesDataFrame
            The time series data to forecast.
        previous_context_data : Optional[TimeSeriesDataFrame], default=None
            Optional preceding time series data for extending the context window.
        rolling : bool, default=False
            If True, performs rolling evaluation across all available time steps.
            If False, predicts only from the latest observation.
        window_step : int, default=1
            The number of time steps to move the sliding prediction window forward between each prediction.

        Returns
        -------
        ForecastCollection
            A nested dictionary mapping each item_id to lead time forecasts.
        """
        logging.info("Starting TiRex forecasting.")

        # Combine context data if given
        if previous_context_data is not None:
            # skip_first: Dict[item_id -> how many prepended rows], used for dataset indexing
            data_merged, skip_first = self._merge_data(data, previous_context_data, self.context_length)
        else:
            data_merged = data
            skip_first = None

        ds = BaseTimeSeriesDataset(
            data_merged,
            self.context_length,
            window_step,
            skip_first,
            rolling=rolling,
        )

        dl = DataLoader(ds, batch_size=256)

        all_forecast_chunks = []

        for batch in tqdm(dl, desc="Forecasting with TiRex"):
            try:
                arr = batch.numpy()
                byte_data = arr.tobytes()

                logging.debug(f"Sending request to TiRex microservice. Batch shape: {arr.shape}")

                response = requests.post(
                    self.forecast_endpoint,
                    data=byte_data,
                    headers={
                        "x-batch-size": str(arr.shape[0]),
                        "x-context-length": str(arr.shape[1]),
                        "x-dtype": str(arr.dtype),
                        "x-prediction-length": str(self.prediction_length),
                    },
                    timeout=180,
                )

                response.raise_for_status()
                prediction = self.construct_prediction(response)
                all_forecast_chunks.append(prediction)

            except requests.exceptions.RequestException as e:
                logging.error(f"Error communicating with TiRex microservice: {e}")
                raise RuntimeError("Failed to get predictions from TiRex service.") from e

            except Exception as e:
                logging.error(f"Unexpected error during prediction: {e}")
                raise

        forecasts_tensor = torch.cat(all_forecast_chunks, dim=0)

        if forecasts_tensor.shape[0] != len(ds):
            logging.error(f"Row count mismatch: predictions={forecasts_tensor.shape[0]} vs dataset={len(ds)}")
            raise ValueError("Mismatch between prediction rows and dataset rows.")

        # If rolling, output data covers all input rows
        if rolling:
            output_data = data
        else:
            # Only the most recent timestep per series
            output_data = data.slice_by_timestep(start_index=-1)

        return ds.to_forecast_collection(
            predictions=forecasts_tensor,
            lead_times=self.lead_times,
            output_data=output_data,
        )

    def construct_prediction(self, response) -> torch.Tensor:
        """
        Converts the HTTP response from the TiRex microservice to a torch.Tensor.

        Parameters
        ----------
        response : requests.Response
            The HTTP response object returned by the microservice.

        Returns
        -------
        torch.Tensor
            The reconstructed tensor containing forecasts.
        """
        try:
            out_batch = int(response.headers["X-batch_size"])
            out_pred_len = int(response.headers["X-prediction_length"])
            out_num_quantiles = int(response.headers["X-num_quantiles"])
            out_dtype = np.dtype(response.headers["X-dtype"])

            logging.debug(f"Received tensor with shape=({out_batch}, {out_pred_len}, {out_num_quantiles}), dtype={out_dtype}")

            out_np = np.frombuffer(response.content, dtype=out_dtype).reshape((out_batch, out_pred_len, out_num_quantiles)).swapaxes(1, 2).copy()
            return torch.from_numpy(out_np)

        except KeyError as e:
            logging.error(f"Missing expected header in response: {e}")
            raise RuntimeError(f"Response missing header: {e}") from e
        except Exception as e:
            logging.error(f"Failed to reconstruct prediction tensor: {e}")
            raise
