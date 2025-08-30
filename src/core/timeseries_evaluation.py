"""This Module provides utilities for probabilistic time series forecasting, including data structures, evaluation, and visualization tools."""

import os
from typing import List, Optional, Dict, Tuple, Union, Callable, Literal
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import scoringrules as sr
from scipy.interpolate import interp1d
from pydantic import BaseModel, Field, field_validator, PrivateAttr
from autogluon.timeseries import TimeSeriesDataFrame
from gluonts.model.forecast import QuantileForecast
import math
import matplotlib.dates as mdates
import matplotlib as mpl
from pathlib import Path
import joblib
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import logging
from tqdm import tqdm
from numpy.typing import NDArray
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import json
from hashlib import blake2b
import tempfile


PIPELINE_CONFIG_FILE_NAME = "pipeline_config.json"
PREDICTIONS_FILENAME = "predictions.joblib"
BACKTEST_CONFIG_FILENAME = "backtest_config.json"
EVAL_CONFIG_FILENAME = "eval_config.json"
DIR_BACKTESTS = "backtest"
DIR_MODELS = "models"
DIR_POSTPROCESSORS = "postprocessors"
ITEMID = "item_id"
TIMESTAMP = "timestamp"
TARGET = "target"


CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / "metrics_cache"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")


class TabularDataFrame(pd.DataFrame):
    def __init__(self, data: pd.DataFrame, *args, **kwargs):

        self._validate_multi_index_data_frame(data)
        self._validate_columns(data)
        super().__init__(data=data, *args, **kwargs)

    @property
    def item_ids(self) -> pd.Index:
        return self.index.unique(level=ITEMID)

    @classmethod
    def _validate_multi_index_data_frame(cls, data: pd.DataFrame):
        """Validate a multi-index pd.DataFrame can be converted to TabularDataFrame"""

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pd.DataFrame, got {type(data)}")
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError(f"data must have pd.MultiIndex, got {type(data.index)}")
        if not pd.api.types.is_datetime64_dtype(data.index.dtypes[TIMESTAMP]):
            raise ValueError(f"for {TIMESTAMP}, the only pandas dtype allowed is `datetime64`.")
        if not data.index.names == (f"{ITEMID}", f"{TIMESTAMP}"):
            raise ValueError(f"data must have index names as ('{ITEMID}', '{TIMESTAMP}'), got {data.index.names}")
        item_id_index = data.index.get_level_values(level=ITEMID)
        if not (pd.api.types.is_integer_dtype(item_id_index) or pd.api.types.is_string_dtype(item_id_index)):
            raise ValueError(f"all entries in index `{ITEMID}` must be of integer or string dtype")

    @classmethod
    def _validate_columns(cls, data: pd.DataFrame):

        if "target" not in data.columns:
            raise ValueError(f"data must contain a column '{TARGET}'")

    def split_by_time(self, cutoff_time: pd.Timestamp) -> Tuple["TabularDataFrame", "TabularDataFrame"]:
        """Split dataframe to two different ``TabularDataFrame`` s before and after a certain ``cutoff_time``.

        Parameters
        ----------
        cutoff_time: pd.Timestamp
            The time to split the current data frame into two data frames.

        Returns
        -------
        data_before: TabularDataFrame
            Data frame containing time series before the ``cutoff_time`` (exclude ``cutoff_time``).
        data_after: TabularDataFrame
            Data frame containing time series after the ``cutoff_time`` (include ``cutoff_time``).
        """

        nanosecond_before_cutoff = cutoff_time - pd.Timedelta(nanoseconds=1)
        data_before = self.loc[(slice(None), slice(None, nanosecond_before_cutoff)), :]
        data_after = self.loc[(slice(None), slice(cutoff_time, None)), :]
        before = TabularDataFrame(data_before)
        after = TabularDataFrame(data_after)
        return before, after

    def __deepcopy__(self, memo):
        copied = self.copy(deep=True)
        return self.__class__(copied)


class HorizonForecast(BaseModel):
    """
    Stores quantile forecasts for a single time series (item id) and a single forecasting horizon.

    Attributes:
        lead_time (int): Forecast lead time in hours.
        timestamps (List[pd.Timestamp]): List of timestamps corresponding to the forecast.
        predictions (torch.Tensor): Tensor of shape [num_samples, num_quantiles] containing forecasted quantiles.
        quantiles (List[float]): List of quantile levels (default: [0.1, ..., 0.9]).
        freq (pd.Timedelta): Time frequency of the data.
        target (Optional[torch.Tensor]): True target values (optional).
    """

    lead_time: int
    predictions: torch.Tensor  # Shape [num_samples, num_quantiles]

    class Config:
        arbitrary_types_allowed = True

    @field_validator("predictions")
    @classmethod
    def check_predictions(cls, pred: torch.Tensor) -> torch.Tensor:
        pred = pred.contiguous() if not pred.is_contiguous() else pred
        pred = pred.sort(dim=1)[0]  # avoid quantile crossing. TODO: potentially shouldn't be done silently
        return pred


class TimeSeriesForecast(BaseModel):
    item_id: int
    lead_time_forecasts: Dict[int, HorizonForecast]  # {lead_time: HorizonForecast}
    data: TimeSeriesDataFrame
    quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    freq: Union[pd.Timedelta, pd.DateOffset]
    forecast_mask: NDArray[np.bool_]
    _cache_dir: Optional[Path] = PrivateAttr(default=None)
    _mem_cache: Dict[str, np.ndarray] = PrivateAttr(default_factory=dict)  # key=f"{kind}:{sig}"
    _cache_sig: str = PrivateAttr(default="")

    class Config:
        arbitrary_types_allowed = True

        # ---------- cache control ----------

    def set_cache_dir(self, path: Union[str, Path]) -> None:
        """Enable disk caching in a directory and reset mem cache."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._cache_dir = p
        self._mem_cache = {}
        self._cache_sig = self._pack_signature()

    def clear_cache_dir(self) -> None:
        """Disable disk caching (memory cache stays)."""
        self._cache_dir = None
        self._mem_cache = {}
        self._cache_sig = ""

    # ---------- signature ----------
    def _pack_signature(self) -> str:
        """
        Digest that changes if predictions/targets/quantiles/mask/freq/item change.
        One signature for all packs (crps, quantile_scores, hits/coverage).
        """
        y_pred, y_true = self.get_aligned_predictions_and_targets()  # y_pred: (T,H,Q), y_true: (T,H)
        h = blake2b(digest_size=16)
        h.update(np.asarray(self.quantiles, dtype=float).tobytes())
        h.update(np.asarray(self.forecast_mask, dtype=bool).tobytes())
        h.update(str(self.freq).encode())
        h.update(np.asarray(self.item_id, dtype=np.int64).tobytes())
        # content
        h.update(y_pred.shape.__repr__().encode())
        h.update(y_pred.tobytes())
        h.update(y_true.shape.__repr__().encode())
        h.update(y_true.tobytes())
        return h.hexdigest()

    # ---------- file helpers ----------
    def _pack_file(self, kind: str) -> Optional[Path]:
        """
        File path for a given pack kind ("crps", "quantile_scores", "hits").
        Stored as .npz with a single array under the same key name.
        """
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{kind}_item{self.item_id}_{self._cache_sig}.npz"

    # ---------- generic loader/builder ----------
    def _load_or_build_pack(self, kind: str, builder: Callable[[], np.ndarray]) -> np.ndarray:
        """
        Loads a pack from memory/disk if available; otherwise builds it via `builder`,
        stores to disk (if cache dir is set), and memoizes in-memory.

        kind ∈ {"crps", "quantile_scores", "hits"}
        """
        # lazy init for legacy objects (keep names stable for users migrating)
        if not (hasattr(self, "_mem_cache") and hasattr(self, "_cache_dir") and hasattr(self, "_cache_sig")):
            # default: no disk caching; just compute signature for mem cache
            self._cache_dir = None
            self._mem_cache = {}
            self._cache_sig = self._pack_signature()

        key = f"{kind}:{self._cache_sig}"
        if key in self._mem_cache:
            return self._mem_cache[key]

        fpath = self._pack_file(kind)
        if fpath is not None and fpath.exists():
            with np.load(fpath) as z:
                arr = z[kind]
            self._mem_cache[key] = arr
            return arr

        # build
        arr = builder()

        # persist atomically
        if fpath is not None:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=fpath.parent, delete=False, suffix=".npz") as tmp:
                np.savez_compressed(tmp.name, **{kind: arr})
                tmp_name = tmp.name
            os.replace(tmp_name, fpath)  # atomic on POSIX

        self._mem_cache[key] = arr
        return arr

    # ---------- concrete pack builders ----------
    def _build_crps_pack(self) -> np.ndarray:
        """
        Returns (T, H) CRPS matrix for all horizons.
        """
        y_pred, y_true = self.get_aligned_predictions_and_targets()  # (T,H,Q), (T,H)
        T, H, Q = y_pred.shape
        crps_mat = np.full((T, H), np.nan, dtype=float)

        for h in range(H):
            yh = y_true[:, h]
            ph = y_pred[:, h, :]
            crps_vec = sr.crps_quantile(yh, ph, self.quantiles, backend="numpy")  # (T,)
            crps_mat[:, h] = crps_vec
        return crps_mat

    def _build_quantile_scores_pack(self) -> np.ndarray:
        """
        Returns (T, H, Q) pinball loss per timestamp, horizon, quantile.
        """
        y_pred, y_true = self.get_aligned_predictions_and_targets()  # (T,H,Q), (T,H)
        T, H, Q = y_pred.shape
        out = np.full((T, H, Q), np.nan, dtype=float)

        # scoringrules.quantile_score expects (N,) y and (N,) preds for one q, or vectorized column-wise
        for h in range(H):
            yh = y_true[:, h]  # (T,)
            ph = y_pred[:, h, :]  # (T,Q)
            # compute pinball for all q columns
            # vectorized: for each q, sr.quantile_score(y, preds[:,q], q)
            for qi, q in enumerate(self.quantiles):
                out[:, h, qi] = sr.quantile_score(yh, ph[:, qi], q)
        return out

    def _build_hits_pack(self) -> np.ndarray:
        """
        Returns (T, H, Q) boolean indicators of coverage: 1{ y_true <= q̂ }.
        This lets us compute empirical coverage rates quickly by averaging over time.
        """
        y_pred, y_true = self.get_aligned_predictions_and_targets()  # (T,H,Q), (T,H)
        # broadcast y_true (T,H,1) against y_pred (T,H,Q)
        hits = (y_true[..., None] <= y_pred).astype(float)  # (T,H,Q), dtype=bool
        mask = np.isnan(y_true)
        hits[mask] = np.nan
        return hits

    def get_lead_times(self) -> List[int]:
        return list(self.lead_time_forecasts.keys())

    def get_lead_time_forecast(self, lead_time: int) -> HorizonForecast:
        return self.lead_time_forecasts[lead_time]

    def get_all_lead_time_forecast(self) -> List[HorizonForecast]:
        return self.lead_time_forecasts

    def add_lead_time_forecast(self, lead_time: int, prediction: HorizonForecast) -> None:
        self.lead_time_forecasts[lead_time] = prediction

    def to_dataframe(self, forecast_horizon: int) -> pd.DataFrame:
        """
        Converts the prediction data into a Pandas DataFrame and merges it with the actual target values.

        Args:
            item_ids (Optional[List[int]]): dataframe of the item ids to retrieve.

        Returns:
            pd.DataFrame: DataFrame with timestamps, predicted quantiles, and corresponding target values.
        """

        horizon_fc = self.get_lead_time_forecast(forecast_horizon)
        result = pd.DataFrame(horizon_fc.predictions, index=self.data[self.forecast_mask].index, columns=self.quantiles)

        # add prediction date information
        result["prediction_date"] = result.index.get_level_values(TIMESTAMP) + self.freq * horizon_fc.lead_time

        # Reset index to turn MultiIndex into columns
        result_reset = result.reset_index()
        data_reset = self.data.reset_index()

        # add the target information.
        merged = result_reset.merge(data_reset, left_on=[ITEMID, "prediction_date"], right_on=[ITEMID, TIMESTAMP], how="left", suffixes=["", "_remove"])

        # add actual values
        data_reset = data_reset.rename(columns={TARGET: "current"})
        merged = merged.merge(data_reset, on=[ITEMID, TIMESTAMP])

        # if merged[TARGET].isna().all():
        #     raise ValueError("target column is nan. Frequency (freq) might not be specified correctly.")

        # remove unused columns
        merged = merged.drop(columns=[col for col in merged.columns if "_remove" in str(col) or "feature" in str(col)], errors="ignore")

        # restore original multi index
        merged.set_index(result.index.names, inplace=True)

        return merged

    def get_aligned_predictions_and_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns aligned predicted quantiles (y_pred) and targets (y_true) for all lead times.

        Returns
        -------
        y_pred : np.ndarray
            Array of shape (T, H, Q) where H is the number of lead times and Q the number of quantiles.
        y_true : np.ndarray
            Array of shape (T, H) aligned to predictions, with NaNs where target is not available.
        """
        # Stack forecasted quantiles → shape (T, H, Q)
        y_pred = np.stack([fc.predictions for fc in self.lead_time_forecasts.values()]).swapaxes(0, 1)
        T, H, Q = y_pred.shape

        # Align targets for all horizons: (T, H)
        y_true_series = self.data["target"].values
        y_true_series = np.roll(y_true_series, -1)
        y_true_series[-1] = np.nan  # last obs has no 1-step-ahead truth

        pad = np.full(H - 1, np.nan)
        y_true_padded = np.concatenate([y_true_series, pad])
        y_true = np.lib.stride_tricks.sliding_window_view(y_true_padded, window_shape=H)  # (T, H)

        # Apply forecast mask and burn-in
        y_true = y_true[self.forecast_mask]

        # Drop last row due to lack of final ground truth
        y_true = y_true[:-1]
        y_pred = y_pred[:-1]

        return y_pred, y_true

    def to_AutogluonFormat(self, idx: int = -1) -> TimeSeriesDataFrame:
        """Converts a TimeSeriesForecast into a TimeSeriesDataFrame compatible with Autogluon"""
        preds = []
        for lt, horizon_fc in self.lead_time_forecasts.items():
            preds.append(horizon_fc.predictions[idx].unsqueeze(0))
        preds = torch.cat(preds, dim=0)

        latest_ts = self.data.index.get_level_values(1)[idx]
        freq = pd.tseries.frequencies.to_offset(self.freq)
        timestamps = pd.date_range(start=latest_ts + freq, periods=lt, freq=freq)

        df = pd.DataFrame(preds, columns=list(map(str, self.quantiles)))
        df["timestamp"] = timestamps
        df["item_id"] = self.item_id
        df = df.set_index(["item_id", "timestamp"])
        if "0.5" in df.columns:
            df.insert(0, column="mean", value=df["0.5"])

        return TimeSeriesDataFrame(df)

    def to_QuantileForecast(self, idx: int = -1) -> QuantileForecast:
        """Converts a TimeSeriesForecast to a GluonTS QuantileForecast"""
        preds = []
        for lt, horizon_fc in self.lead_time_forecasts.items():
            preds.append(horizon_fc.predictions[idx].unsqueeze(0))
        preds = torch.cat(preds, dim=0).swapaxes(0, 1)
        freq = pd.tseries.frequencies.to_offset(self.freq)
        latest_ts = self.data.index.get_level_values(1)[idx] + freq

        return QuantileForecast(
            forecast_arrays=preds,
            start_date=latest_ts.to_period(freq),
            forecast_keys=list(map(str, self.quantiles)),
            item_id=self.item_id,
        )

    # ---------- public metrics using the packs ----------
    def get_crps(self, forecast_horizon: int, mean_time: bool = True) -> np.ndarray:
        """
        Selects the requested horizon from the all-horizon CRPS pack.
        Returns time-mean if mean_time else the per-timestamp vector.
        """
        lead_times_sorted = sorted(self.lead_time_forecasts.keys())
        try:
            col = lead_times_sorted.index(forecast_horizon)
        except ValueError:
            raise KeyError(f"Lead time {forecast_horizon} not found for item {self.item_id}")

        crps_mat = self._load_or_build_pack("crps", self._build_crps_pack)  # (T, H)
        crps_vec = crps_mat[:, col]

        if mean_time:
            return np.array([np.nanmean(crps_vec)])
        else:
            return crps_vec

    def get_quantile_score(self, forecast_horizon: int, mean_time: bool = True) -> Union[pd.Series, pd.DataFrame]:
        """
        Returns pinball losses by quantile for one horizon.
        If mean_time=True → pd.Series of mean loss per quantile.
        Else → pd.DataFrame indexed by time with columns=self.quantiles.

        # TODO: to_dataframe is not efficient
        """
        lead_times_sorted = sorted(self.lead_time_forecasts.keys())
        try:
            col = lead_times_sorted.index(forecast_horizon)
        except ValueError:
            raise KeyError(f"Lead time {forecast_horizon} not found for item {self.item_id}")

        qs_pack = self._load_or_build_pack("quantile_scores", self._build_quantile_scores_pack)  # (T,H,Q)

        # recover the timestamp index that matches get_aligned_predictions_and_targets() slicing
        # easiest is to reuse to_dataframe() to get the same index; we only need the index shape
        # df_idx = self.to_dataframe(forecast_horizon).index[:-1]  # aligned with y_pred[:-1]
        df_idx = self.data.index[self.forecast_mask][:-1]

        mat = qs_pack[:, col, :]  # (T, Q)
        if mean_time:
            return pd.Series(np.nanmean(mat, axis=0), index=self.quantiles)
        else:
            return pd.DataFrame(mat, index=df_idx, columns=self.quantiles)

    def get_empirical_coverage_rates(self, forecast_horizon: int) -> Dict[float, float]:
        """
        Uses the hits pack (T,H,Q) and averages over time to get empirical coverage.
        """
        lead_times_sorted = sorted(self.lead_time_forecasts.keys())
        try:
            col = lead_times_sorted.index(forecast_horizon)
        except ValueError:
            raise KeyError(f"Lead time {forecast_horizon} not found for item {self.item_id}")

        hits_pack = self._load_or_build_pack("hits", self._build_hits_pack)  # (T,H,Q), bool
        hits = hits_pack[:, col, :].astype(float)  # (T, Q)

        cov = np.nanmean(hits, axis=0)  # (Q,)

        return {q: float(c) for q, c in zip(self.quantiles, cov)}

    def get_pit_values(self, forecast_horizon: int) -> np.ndarray:
        """
        Computes the Probability Integral Transform (PIT) values for calibration analysis.

        PIT = F(y), where F(y) is the interp_func, which is the CDF approximated based on the quantiles.

        Parameters:
            data (pd.DataFrame): DataFrame containing actual target values.

        Returns:
            np.ndarray: Array of PIT values, where PIT values should follow a uniform [0,1] distribution.
        """
        df = self.to_dataframe(forecast_horizon).dropna()
        targets = df[TARGET].to_numpy()  # Shape [num_samples]
        predictions = df[self.quantiles].to_numpy()

        # Compute PIT values for each target
        pit_values = []
        for i in range(len(targets)):
            interp_func = interp1d(predictions[i], self.quantiles, bounds_error=False, fill_value=(0, 1))
            pit_values.append(interp_func(targets[i]))

        return np.array(pit_values)

    def get_pit_histogram(self, forecast_horizon: int) -> None:
        """
        Plots a histogram of Probability Integral Transform (PIT) values to assess forecast calibration.

        Parameters:
            data (pd.DataFrame): DataFrame containing actual target values.

        Returns:
            None: Displays the histogram plot.
        """
        pit_values = self.get_pit_values(forecast_horizon)
        bins = len(self.quantiles) + 1

        plt.hist(pit_values, bins=bins, range=(0, 1), density=False, alpha=0.7, edgecolor="black")
        plt.axhline(len(pit_values) / bins, color="red", linestyle="dashed", label="Uniform(0,1) reference")
        plt.xlabel("PIT Values")
        plt.ylabel("Frequency of Occurrences")
        plt.title("PIT Histogram")
        plt.legend()
        plt.show()

    def get_reliability_diagram(self, forecast_horizon: int) -> None:

        empirical_coverage_rates = self.get_empirical_coverage_rates(forecast_horizon)

        quantile_levels = sorted(empirical_coverage_rates.keys())
        empirical_coverages = [empirical_coverage_rates[q] for q in quantile_levels]

        # Create the plot
        plt.figure(figsize=(8, 8))
        plt.plot(quantile_levels, empirical_coverages, "o-", label="Empirical Coverage", markersize=8)
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")  # Diagonal line
        plt.xlabel("Nominal Quantile Level")
        plt.ylabel("Empirical Coverage")
        plt.title("Reliability Diagram for Quantile Forecasts")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_forecasts(self, start: Optional[Union[int, pd.Timestamp]] = None, context_length: int = 100) -> None:
        """
        Plot past data, true future values, and quantile forecasts for a given starting point.

        Parameters
        ----------
        start : Optional[Union[int, pd.Timestamp]]
            - If int: Index into the time series to start the forecast from.
            - If pd.Timestamp: Timestamp to start the forecast from. Must exist in the time series index.
            - If None: Defaults to the last available index.

        context_length : int
            Number of historical data points to include in the plot before the forecast start.

        Returns
        -------
        None
            Displays a matplotlib plot showing:
            - Past true values
            - Future true values
            - Predicted quantiles (median and shaded interval)
        """

        def _add_prediction_intervals(
            ax: plt.Axes,
            selected_predictions: pd.DataFrame,
            intervals: list[tuple[float, float]],
            base_color: str = "orange",
            lw_outer: float = 0.5,
            lw_inner: float = 0.5,
        ) -> None:
            """
            Shade multiple predictive intervals on an existing axes *and*
            draw boundary lines around each band.

            Parameters
            ----------
            ax
                Matplotlib axes to draw on.
            selected_predictions
                DataFrame with quantile columns.
            intervals
                List of (lower_q, upper_q) tuples, drawn in the given order.
            base_color
                A matplotlib-compatible color for both fill and lines.
            lw_outer / lw_inner
                Line-widths for the widest and narrowest bands, interpolated in-between.
            """
            n = len(intervals)
            # Draw widest interval first so narrower ones appear on top
            for i, (ql, qu) in enumerate(intervals):
                if ql not in selected_predictions or qu not in selected_predictions:
                    continue  # silently skip if quantiles are missing
                alpha = 0.2 + 0.15 * (n - 1 - i)

                ax.fill_between(
                    selected_predictions.index,
                    selected_predictions[ql],
                    selected_predictions[qu],
                    color=base_color,
                    alpha=alpha,
                    label=f"{int(qu*100)}–{int(ql*100)} % interval",
                )

                # line-width interpolates from outer- to inner-band values
                lw = lw_outer + (lw_inner - lw_outer) * (n - 1 - i) / (n - 1)
                ax.plot(selected_predictions.index, selected_predictions[ql], color=base_color, alpha=alpha + 0.1, linewidth=lw)

                ax.plot(selected_predictions.index, selected_predictions[qu], color=base_color, alpha=alpha + 0.1, linewidth=lw)

        timestamps = self.data.index.get_level_values("timestamp")

        if isinstance(start, pd.Timestamp):
            if start not in timestamps:
                raise ValueError(f"Timestamp {start} not found in data index.")
            start_idx = timestamps.get_loc(start)
        elif isinstance(start, int):
            if start < 0:
                start_idx = start % len(self.data)  # handle negative indexing
            elif start >= len(self.data):
                start_idx = len(self.data) - 1
            else:
                start_idx = start
        else:
            start_idx = len(self.data) - 1

        # get the corrected timestamps. Relevant if forecast_mask has some false values
        start_date = timestamps[start_idx]
        forecasted_ts = timestamps[self.forecast_mask]
        start_idx_preds = forecasted_ts.get_indexer([start_date], method="backfill")[0]
        corrected_start_date = forecasted_ts[start_idx_preds]
        corrected_start_idx = timestamps.get_indexer([corrected_start_date])[0]

        # Collect prediction tensor
        preds = torch.stack([hf.predictions for hf in self.lead_time_forecasts.values()], dim=1)  # shape: [num_samples, num_lead_times, num_quantiles]

        # Past context
        historic_start_idx = max(0, corrected_start_idx - context_length) if corrected_start_idx >= 0 else corrected_start_idx - context_length
        past = self.data.iloc[historic_start_idx:corrected_start_idx].reset_index(level=0, drop=True)

        # Future truth Horizon: up to max lead-time
        max_lt = max(self.get_lead_times())
        future = self.data.iloc[corrected_start_idx : corrected_start_idx + max_lt].reset_index(level=0, drop=True)

        # Align forecast tensor with corresponding timestamp index
        freq_offset = pd.tseries.frequencies.to_offset(self.freq)
        prediction_dates = [corrected_start_date + freq_offset * lt for lt in self.get_lead_times()]
        selected_predictions = pd.DataFrame(data=preds[start_idx_preds].numpy(), columns=self.quantiles, index=prediction_dates)  # shape: [num_lead_times, num_quantiles]

        plt.figure(figsize=(12, 6))

        plt.plot(past.index, past.values, label="Past", color="black", linestyle="--")
        plt.plot(future.index, future.values, label="Future (true)", color="blue")

        if 0.5 in selected_predictions.columns:
            plt.plot(selected_predictions.index, selected_predictions[0.5].values, label="Prediction (median)", color="red")

        intervals = [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9)]
        _add_prediction_intervals(plt.gca(), selected_predictions, intervals, base_color="orange")

        plt.axvline(corrected_start_date, color="gray", linestyle=":", label="Prediction start")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title(f"Forecast for item_id={self.item_id} from {corrected_start_date}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_random_plot(self, forecast_horizon: int = 1, q_lower: float = 0.1, q_upper: float = 0.9, ts_length: int = 100) -> None:
        """Randomly plot data"""

        subset = self.to_dataframe(forecast_horizon)
        # subset = pred_df.xs(item_id, level="item_id")
        rand_start_idx = np.random.randint(0, (len(subset) - ts_length))
        subset = subset.iloc[rand_start_idx : rand_start_idx + ts_length]

        # Plot settings
        plt.figure(figsize=(15, 5))

        # Plot median prediction
        plt.plot(subset.index.get_level_values("timestamp"), subset[0.5], label="Median (50%)", color="C1", linestyle="-", linewidth=2)

        # Plot confidence intervals as shaded regions
        plt.fill_between(
            subset.index.get_level_values("timestamp"),
            subset[q_lower],
            subset[q_upper],
            color="C1",
            alpha=0.2,
            label=f"{(q_upper-q_lower) * 100:.0f}% Prediction Interval ({q_upper*100:.0f}%-{q_lower*100:.0f}%)",
        )

        # Plot target values
        plt.plot(subset.index.get_level_values("timestamp"), subset["target"], label="Actual Target", color="C0", linestyle="-", linewidth=2)

        # Formatting
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.title(f"Prediction Intervals – item_id: {self.item_id} and lead time: {forecast_horizon}", fontsize=16)
        plt.legend(fontsize=8)

        # Improve x-axis tick formatting
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically space ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # Format as 'YYYY-MM-DD HH:MM'

        plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Optional: Light grid

        plt.show()


class ForecastCollection(BaseModel):
    item_ids: Dict[int, TimeSeriesForecast]  # item_id -> TimeSeriesForecast
    inference_time_seconds: Optional[float] = None  # time to obtain the predictions in [s]

    class Config:
        arbitrary_types_allowed = True

    def set_cache_dir(self, root: Union[str, Path]) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for item_id in self.get_item_ids():
            self.get_time_series_forecast(item_id).set_cache_dir(root)

    def clear_cache_dir(self) -> None:
        for item_id in self.get_item_ids():
            self.get_time_series_forecast(item_id).clear_cache_dir()

    def get_item_ids(self) -> List[int]:
        return list(self.item_ids.keys())

    def get_lead_times(self, item_id: Optional[int] = None) -> List[int]:
        if item_id is not None:
            return self.item_ids[item_id].get_lead_times()
        return sorted({lt for item in self.item_ids.values() for lt in item.get_lead_times()})

    def get_time_series_forecast(self, item_id: int) -> TimeSeriesForecast:
        return self.item_ids[item_id]

    def get_all_time_series_forecast(self) -> List[TimeSeriesForecast]:
        return self.item_ids

    def add_time_series_forecast(self, forecast: TimeSeriesForecast) -> None:
        self.item_ids[forecast.item_id] = forecast

    def to_AutogluonFormat(self, idx: int = -1) -> TimeSeriesDataFrame:
        """Converts a TimeSeriesForecast into a TimeSeriesDataFrame compatible with Autogluon"""
        preds = []
        for item_id in self.get_item_ids():
            preds.append(self.get_time_series_forecast(item_id).to_AutogluonFormat(idx))
        df = pd.concat(preds).sort_index()

        return TimeSeriesDataFrame(df)

    def to_QuantileForecast(self, idx: int = -1) -> List[QuantileForecast]:
        """Converts a TimeSeriesForecast into a TimeSeriesDataFrame compatible with Autogluon"""
        preds = []
        for item_id in self.get_item_ids():
            preds.append(self.get_time_series_forecast(item_id).to_QuantileForecast(idx))
        return preds

    def plot_forecasts(
        self,
        start: Optional[Union[int, pd.Timestamp]] = None,
        context_length: int = 100,
        max_historical_context: int = 1000,
        max_forecast_steps: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_cols: int = 3,
        sharex: bool = False,
        sharey: bool = False,
        show_xy_labels: bool = True,
        show_legend: bool = True,
        title_prefix: str = "",
        legend_position: str = "below",
        font_sizes: Optional[Dict[str, int]] = None,
        tight_margins: bool = False,
        margin_padding: float = 0.05,
        show_history_overview: bool = False,
        history_overview_height: float = 0.3,
        save_path: Optional[str] = None,
        dpi: int = 300,
    ) -> None:
        """
        Plot all TimeSeriesForecast objects in this collection using the plot_multiple_forecasts function.

        Parameters
        ----------
        start : Optional[Union[int, pd.Timestamp]]
            - If int: Index into the time series to start the forecast from.
            - If pd.Timestamp: Timestamp to start the forecast from. Must exist in the time series index.
            - If None: Defaults to the last available index for each forecast.

        context_length : int
            Number of historical data points to include in the plot before the forecast start.

        max_historical_context : int
            Maximum number of historical data points to show in the history overview subplot.
            This controls how far back the overview looks from the forecast start point.
            Default: 1000. Use larger values for longer historical context.

        max_forecast_steps : Optional[int]
            Maximum number of forecast steps to plot. If None, plots all available forecast steps.
            If specified, limits the number of future time steps displayed in the forecast plots.
            Useful for focusing on short-term predictions or reducing visual clutter.

        figsize : Optional[Tuple[int, int]]
            Figure size as (width, height). If provided, overrides width and height parameters.

        width : Optional[int]
            Figure width in inches. Used only if figsize is None.

        height : Optional[int]
            Figure height in inches. Used only if figsize is None.

        max_cols : int
            Maximum number of columns in the grid layout.

        sharex : bool
            Whether to share x-axis across subplots.

        sharey : bool
            Whether to share y-axis across subplots.

        show_xy_labels : bool
            Whether to show x- and y-labels on subplots.

        show_legend : bool
            Whether to show a single legend.

        title_prefix : str
            Prefix for subplot titles.

        legend_position : str
            Position of the legend: "below" (below the plots) or "right" (to the right of the plots).

        font_sizes : Optional[Dict[str, int]]
            Dictionary controlling font sizes for various text elements. If None, default sizes are used.
            Available keys: 'title', 'subtitle', 'xlabel', 'ylabel', 'legend', 'tick_labels', 'grid_labels'.

        tight_margins : bool
            Whether to use tight margins around the data. If True, reduces padding between plot borders and data.

        margin_padding : float
            Padding factor for margins when tight_margins=True. Smaller values (0.01-0.05) create tighter plots,
            larger values (0.1-0.2) create more spacious plots. Default: 0.05.

        show_history_overview : bool
            Whether to add a full-length historical overview subplot at the top showing the complete time series.
            This provides context for the zoomed-in forecast plots below.

        history_overview_height : float
            Height ratio for the history overview subplot relative to the total figure height.
            Range: 0.1 to 0.5. Default: 0.3 (30% of total height).

        save_path : Optional[str]
            If provided, save the plot to this path.

        dpi : int
            DPI for saving the plot.

        Returns
        -------
        None
            Displays a matplotlib figure with subplots showing forecasts for each TimeSeriesForecast object.
        """
        plot_multiple_forecasts(
            forecasts_dict=self.item_ids,
            start=start,
            context_length=context_length,
            max_historical_context=max_historical_context,
            max_forecast_steps=max_forecast_steps,
            figsize=figsize,
            width=width,
            height=height,
            max_cols=max_cols,
            sharex=sharex,
            sharey=sharey,
            show_xy_labels=show_xy_labels,
            show_legend=show_legend,
            title_prefix=title_prefix,
            legend_position=legend_position,
            font_sizes=font_sizes,
            tight_margins=tight_margins,
            margin_padding=margin_padding,
            show_history_overview=show_history_overview,
            history_overview_height=history_overview_height,
            save_path=save_path,
            dpi=dpi,
        )

    def get_crps(
        self,
        item_ids: Optional[List[int]] = None,
        lead_times: Optional[List[int]] = None,
        mean_time: bool = False,
        mean_item_ids: bool = False,
        mean_lead_times: bool = False,
        decimal_places: Optional[int] = None,
        weighting: Literal["micro", "macro"] = "micro",
    ) -> pd.DataFrame:
        """
        Aggregates CRPS with flexible outputs:

        Flags:
        - mean_time:     reduce over time (T)
        - mean_item_ids: reduce over items (I)
        - mean_lead_times: reduce over lead_time (L)

        When BOTH mean_time=True and mean_item_ids=True:
        - weighting="micro": every valid (item, time) gets equal weight (nanmean over I&T).
        - weighting="macro": average within each item over time, then equal-weight over items.

        Returns:
        Depending on flags, either:
            * per-lead rows (or per-lead columns) or
            * per-item tables, or
            * per-timestamp tables (when keeping time), or
            * a single scalar column 'Mean CRPS'.
        """
        item_ids = item_ids or self.get_item_ids()

        # --- discover leads per item and choose final lead list ---
        per_item_leads = {}
        for item_id in item_ids:
            it = self.get_time_series_forecast(item_id)
            per_item_leads[item_id] = sorted(it.lead_time_forecasts.keys())

        if lead_times is None:
            all_leads = sorted({lt for ls in per_item_leads.values() for lt in ls})
        else:
            all_leads = [lt for lt in lead_times if any(lt in ls for ls in per_item_leads.values())]

        if not all_leads:
            return pd.DataFrame()

        # --- build (I, T_max, L) CRPS array and (I, T_max) timestamp array ---
        item_mats = []  # list[(T_i, L)]
        item_ts = []  # list[pd.DatetimeIndex length T_i]
        valid_items = []
        T_max = 0
        for item_id in item_ids:
            it = self.get_time_series_forecast(item_id)
            crps_pack = it._load_or_build_pack("crps", it._build_crps_pack)  # (T, H_item)
            it_leads_sorted = sorted(it.lead_time_forecasts.keys())
            lead_to_col = {lt: j for j, lt in enumerate(it_leads_sorted)}

            # timestamps aligned to pack (forecast_masked, drop last row)
            ts = it.data.index.get_level_values("timestamp")[it.forecast_mask][:-1]
            if len(ts) == 0:
                continue

            # (T_i, L) with NaN for missing leads for this item
            M = np.full((len(ts), len(all_leads)), np.nan, dtype=float)
            any_col = False

            for j, lt in enumerate(all_leads):
                if lt in lead_to_col:
                    M[:, j] = crps_pack[:, lead_to_col[lt]]
                    any_col = True
            if not any_col:
                continue

            item_mats.append(M)
            item_ts.append(pd.DatetimeIndex(ts))
            valid_items.append(item_id)
            T_max = max(T_max, M.shape[0])

        if not item_mats:
            return pd.DataFrame()

        I = len(item_mats)
        L = len(all_leads)

        # pad to common T_max
        crps_arr = np.full((I, T_max, L), np.nan, dtype=float)
        ts_arr = np.full((I, T_max), np.datetime64("NaT"), dtype="datetime64[ns]")

        for i, (M, ts) in enumerate(zip(item_mats, item_ts)):
            Ti = M.shape[0]
            crps_arr[i, :Ti, :] = M
            ts_arr[i, :Ti] = ts.values

        # get rid of the latest incomplete entries
        # crps_arr = crps_arr[:, :-L, :]
        # ts_arr = ts_arr[:, :-L]

        # Helper to finalize (round)
        def _done(df: pd.DataFrame) -> pd.DataFrame:
            if decimal_places is not None:
                df = df.round(decimal_places)
            return df

        # --- reduce according to flags ---

        # TODO: for all methods add a "micro" and "macro" average.
        # If macro we should be able to specify the aggregation scheme

        # Case A: reduce over time & items
        if mean_time and mean_item_ids:
            if not mean_lead_times:
                # per-lead row
                if weighting == "micro":
                    vals = np.nanmean(crps_arr, axis=(0, 1))  # (L,)
                    out = pd.DataFrame([vals], index=["micro"], columns=all_leads)
                else:
                    per_item_lead = np.nanmean(crps_arr, axis=1)  # (I, L) mean over T
                    vals = np.nanmean(per_item_lead, axis=0)  # (L,)
                    out = pd.DataFrame([vals], index=["macro"], columns=all_leads)
                return _done(out)
            else:
                if weighting == "micro":
                    val = float(np.nanmean(crps_arr))
                    out = pd.DataFrame({"Mean CRPS": [val]}, index=["micro"])
                else:
                    per_item_scalar = np.nanmean(np.nanmean(crps_arr, axis=1), axis=1)  # (I,)
                    val = float(np.nanmean(per_item_scalar))
                    out = pd.DataFrame({"Mean CRPS": [val]}, index=["macro"])
                return _done(out)

        # Case B: reduce over time only (keep items)
        if mean_time and not mean_item_ids:
            per_item_lead = np.nanmean(crps_arr, axis=1)  # (I, L)
            out = pd.DataFrame(per_item_lead, index=valid_items, columns=all_leads)
            out.index.name = "item_id"
            if mean_lead_times:
                out = pd.DataFrame({"Mean CRPS": np.nanmean(per_item_lead, axis=1)}, index=valid_items)
                out.index.name = "item_id"
            return _done(out)

        # Case C: reduce over items only (keep time)
        if not mean_time and mean_item_ids:
            # We must aggregate by actual timestamps (inner-join across items by time)
            # Build long table from arrays and groupby timestamp (per lead).
            frames = []
            for j, lt in enumerate(all_leads):
                vals = crps_arr[:, :, j].ravel()
                tss = ts_arr.ravel()
                mask = np.isfinite(vals) & (tss.astype("datetime64[ns]") == tss)  # not NaT
                if not mask.any():
                    continue
                dfj = pd.DataFrame({"timestamp": tss[mask], "crps": vals[mask]})
                dfj = dfj.groupby("timestamp", as_index=True)["crps"].mean()
                dfj.name = lt
                frames.append(dfj)
            if not frames:
                return pd.DataFrame()
            out = pd.concat(frames, axis=1).sort_index()  # index = timestamp, cols = leads
            if mean_lead_times:
                out = pd.DataFrame({"Mean CRPS": out.mean(axis=1)})
            return _done(out)

        # Case D: keep both items and time (no reduction over I nor T)
        # Return a MultiIndex frame: index=(item_id, timestamp), columns=lead_time
        rows = []
        idx_items = []
        idx_timestamps = []
        for i, item_id in enumerate(valid_items):
            Ti = np.count_nonzero(~np.isnat(ts_arr[i]))
            if Ti == 0:
                continue
            df_i = pd.DataFrame(crps_arr[i, :Ti, :], columns=all_leads)
            rows.append(df_i)
            idx_items.extend([item_id] * Ti)
            idx_timestamps.extend(pd.to_datetime(ts_arr[i, :Ti]))
        if not rows:
            return pd.DataFrame()
        out = pd.concat(rows, axis=0)
        out.index = pd.MultiIndex.from_arrays([idx_items, idx_timestamps], names=["item_id", "timestamp"])
        out = out.sort_index()
        if mean_lead_times:
            out = pd.DataFrame({"Mean CRPS": out.mean(axis=1)}, index=out.index)
        return _done(out)

    def get_empirical_coverage_rates(
        self, item_ids: Optional[List[int]] = None, lead_times: Optional[List[int]] = None, mean_lead_times: bool = False, decimal_places: Optional[int] = None
    ) -> pd.DataFrame:

        item_ids = item_ids or self.get_item_ids()
        if lead_times is None:
            lead_times = self.get_lead_times()

        rates = {lt: [] for lt in lead_times}

        for item_id in item_ids:
            item = self.get_time_series_forecast(item_id)
            for lt in lead_times:
                if lt in item.lead_time_forecasts:
                    val = item.get_empirical_coverage_rates(lt)
                    rates[lt].append(pd.Series(val))

        coverage_df = pd.DataFrame({lt: pd.concat(rates[lt], axis=1).mean(axis=1) for lt in lead_times if rates[lt]})

        if mean_lead_times:
            coverage_df = pd.DataFrame(coverage_df.mean(axis=1), columns=["Empirical coverage rates averaged over all lead times"])
        else:
            coverage_df.loc[:, "Empirical coverage rates averaged over all lead times"] = coverage_df.mean(axis=1)

        coverage_df.index.name = "quantile"

        if decimal_places:
            return coverage_df.round(decimal_places)
        return coverage_df

    def get_quantile_scores(
        self, item_ids: Optional[List[int]] = None, lead_times: Optional[List[int]] = None, mean_lead_times: bool = False, decimal_places: Optional[int] = None
    ) -> pd.DataFrame:
        item_ids = item_ids or self.get_item_ids()
        if lead_times is None:
            lead_times = self.get_lead_times()

        scores = {}

        for lt in lead_times:
            values = []
            for item_id in item_ids:
                val = self.get_time_series_forecast(item_id).get_quantile_score(lt, mean_time=True)
                values.append(val)
            if values:
                scores[lt] = pd.DataFrame(values).mean(axis=0)

        df = pd.DataFrame(scores)
        if mean_lead_times:
            df = pd.DataFrame(df.mean(axis=1), columns=["QS averaged over all lead times"])
        else:
            df.loc[:, "QS averaged over all lead times"] = df.mean(axis=1)

        df.loc["Mean (CRPS/2)", :] = df.mean()
        df.index.name = "quantile"

        if decimal_places:
            return df.round(decimal_places)
        return df

    def get_pit_values(self, lead_times: Optional[List[int]] = None, item_ids: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        item_ids = item_ids or self.get_item_ids()
        if lead_times is None:
            lead_times = self.get_lead_times()
        result = {}
        for lt in lead_times:
            values = []
            for item_id in item_ids:
                item = self.get_time_series_forecast(item_id)
                if lt in item.lead_time_forecasts:
                    values.append(item.get_pit_values(lt))
            if values:
                result[lt] = np.concatenate(values)
        return result

    def plot_pit_histogram(self, lead_times: Optional[List[int]] = None, overlay: bool = False, item_ids: Optional[List[int]] = None) -> None:
        if lead_times is None:
            lead_times = self.get_lead_times()
        pit_data = self.get_pit_values(lead_times=lead_times, item_ids=item_ids)

        if item_ids:
            first_item = item_ids[0]
        else:
            first_item = self.get_item_ids()[0]

        bins = len(self.get_time_series_forecast(first_item).quantiles) + 1

        if overlay:
            plt.figure(figsize=(10, 6))
            for lt, pit_values in pit_data.items():
                plt.hist(pit_values, bins=bins, alpha=0.5, label=f"Lead time {lt}")
            plt.axhline(len(pit_values) / bins, color="red", linestyle="dashed", label="Uniform(0,1) reference")
            plt.xlabel("PIT Values")
            plt.ylabel("Observed Frequency")
            plt.title("PIT Histogram Across Lead Times")
            plt.legend()
            plt.show()

        else:
            num_plots = len(pit_data)
            cols = math.ceil(np.sqrt(num_plots))
            rows = (num_plots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
            axes = axes.flatten() if num_plots > 1 else [axes]
            for ax, (lt, pit_values) in zip(axes, pit_data.items()):
                ax.hist(pit_values, bins=bins, range=(0, 1), density=False, alpha=0.7, edgecolor="black")
                ax.axhline(len(pit_values) / bins, color="red", linestyle="dashed", label="Uniform(0,1) reference")
                ax.set_xlabel("PIT Values")
                ax.set_ylabel("Observed Frequency")
                ax.set_title(f"PIT Histogram (Lead Time {lt})")
                ax.legend()
            plt.tight_layout()
            plt.show()

    def plot_reliability_diagram(
        self,
        lead_times: Optional[List[int]] = None,
        overlay: bool = True,
        item_ids: Optional[List[int]] = None,
        show_individual_lead_times: bool = False,
        mean_lead_times: bool = True,
    ) -> None:
        """
        Plot reliability diagrams (empirical coverage vs nominal quantile)
        for probabilistic forecasts across lead times.

        Parameters
        ----------
        lead_times : list[int], optional
            Lead times to include. Defaults to all available.
        overlay : bool, default False
            If True, plot all requested content in a single axis.
            If False, produce small multiples (one panel per lead) plus an optional
            average panel if show_avg=True & show_individual_lead_times=False OR append at end if both.
        item_ids : list[int], optional
            Restrict to these item IDs. Defaults to all items.
        show_individual_lead_times : bool, default True
            Plot individual lead-time curves/panels.
        mean_lead_times : bool, default False
            Plot pooled/average curve (overlay) or panel (facets).
        """
        lead_times = lead_times or self.get_lead_times()
        if not lead_times:
            raise ValueError("No lead times available to plot.")

        # --- collect per-item coverages by lead ---
        per_lead = {lt: [] for lt in lead_times}
        for item_id in self.get_item_ids():
            if item_ids and item_id not in item_ids:
                continue
            item = self.get_time_series_forecast(item_id)
            for lt in lead_times:
                if lt in item.lead_time_forecasts:
                    val = item.get_empirical_coverage_rates(lt)  # dict {alpha: cov}
                    per_lead[lt].append(pd.Series(val))

        # Macro per-lead curves
        emp_per_lead = {}
        for lt, ser_list in per_lead.items():
            if ser_list:
                emp_per_lead[lt] = pd.concat(ser_list, axis=1).mean(axis=1)

        # Average across leads
        emp_avg = None
        if mean_lead_times:
            # need aligned series; drop missing leads
            all_series = [s for s in emp_per_lead.values() if s is not None]
            if all_series:
                # stack as columns, simple mean
                emp_avg = pd.concat(all_series, axis=1).mean(axis=1)

        # ---- plotting ----
        if overlay:
            plt.figure(figsize=(8, 8))
            if show_individual_lead_times:
                for lt, emp in emp_per_lead.items():
                    if emp is not None:
                        plt.plot(emp.index, emp.values, "o-", label=f"Lead {lt}")
            if mean_lead_times and emp_avg is not None:
                # Use a distinct style
                plt.plot(emp_avg.index, emp_avg.values, "s--", linewidth=2, label="Average across Lead Times")
            plt.plot([0, 1], [0, 1], "k--", label="Perfect")
            plt.xlabel("Nominal Quantile Level")
            plt.ylabel("Empirical Coverage")
            ttl_bits = []
            if show_individual_lead_times:
                ttl_bits.append("Leads")
            if mean_lead_times:
                ttl_bits.append("Avg")
            plt.title("Reliability Diagram (" + "+".join(ttl_bits) + ")")
            plt.legend()
            plt.grid(True)
            plt.show()
            return

        # Faceted layout
        # Decide how many panels: one per lead if show_individual_lead_times; plus one avg panel if show_avg
        panel_leads = list(emp_per_lead.keys()) if show_individual_lead_times else []
        if mean_lead_times:
            panel_leads.append("__AVG__")
        num_plots = len(panel_leads)
        cols = math.ceil(np.sqrt(num_plots))
        rows = (num_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = axes.flatten() if num_plots > 1 else [axes]

        for ax, key in zip(axes, panel_leads):
            if key == "__AVG__":
                emp = emp_avg
                label = "Average across Lead Times"
            else:
                emp = emp_per_lead.get(key)
                label = f"Lead {key}"
            if emp is None:
                ax.set_visible(False)
                continue
            ax.plot(emp.index, emp.values, "o-", label=label)
            ax.plot([0, 1], [0, 1], "k--", label="Perfect")
            ax.set_xlabel("Nominal Quantile Level")
            ax.set_ylabel("Empirical Coverage")
            ax.set_xticks(emp.index)
            ax.set_title(f"Reliability: {label}")
            ax.legend()

        # Hide any extra unused axes
        for ax in axes[num_plots:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

    def save(self, file_path: Path) -> None:
        joblib.dump(self, file_path)
        logging.info("Saved prediction collection to %s", file_path)

    @classmethod
    def load(cls, file_path: Path) -> "ForecastCollection":
        obj = joblib.load(file_path)
        if not isinstance(obj, cls):
            raise ValueError("Loaded object is not a ForecastCollection")
        obj.set_cache_dir(CACHE_DIR)

        return obj


def get_quantile_scores(
    predictions: Dict[str, Union["ForecastCollection", Path]],
    lead_times: Optional[List[int]] = None,
    item_ids: Optional[List[int]] = None,
    reference_predictions: Optional[str] = None,
    decimal_places: Optional[int] = None,
    sort: bool = True,
) -> pd.DataFrame:
    """
    Computes quantile scores for different prediction sources,
    averaged across the specified lead times, and optionally normalizes them
    using a reference prediction.

    Parameters
    -----------
    predictions : Dict[str, Union[ForecastCollection, Path]]
        Dictionary mapping keys to either:
        - Preloaded ForecastCollection objects, or
        - File paths (Path or str) to joblib files that contain ForecastCollection objects.
     lead_times : Optional[List[int]], default=None
        List of lead times to filter CRPS scores. If None, all lead times are used.
    item_ids : Optional[List[int]], default=None
        List of item IDs to include in the CRPS computation. If None, all item IDs are used.
    reference_predictions : Optional[str], default=None
        Key of a prediction set to be used as a reference for normalization.
        If provided, all CRPS values will be divided by the CRPS values from this prediction.
    decimal_places : Optional[int], default=None
        Number of decimal places to round numerical values to. If None, no rounding is applied.
    sort : bool, default=True
        If True, columns will be sorted by mean CRPS in ascending order.

    Returns
    --------
    pd.DataFrame
        A DataFrame where columns represent different prediction sources and rows represent
        quantile score values averaged across the selected lead times. If normalization is applied,
        the scores are expressed as a ratio to the reference prediction.
    """

    scores_dict = {}

    for key, value in predictions.items():
        if isinstance(value, ForecastCollection):
            pred = value
        elif isinstance(value, (str, Path)):
            pred = ForecastCollection.load(value)
        else:
            raise TypeError(f"Unsupported prediction type for key '{key}': {type(value)}")

        q_scores_df = pred.get_quantile_scores(lead_times=lead_times, mean_lead_times=True, item_ids=item_ids)

        scores_dict[key] = q_scores_df.squeeze()  # Convert single-row DataFrame to Series

    scores = pd.DataFrame(scores_dict)

    scores.columns = [key for key in predictions.keys()]

    if reference_predictions:
        scores = scores.apply(lambda x: x / x[reference_predictions], axis=1)

    if sort:
        scores = scores.T.sort_values(by="Mean (CRPS/2)", axis=0).T

    if decimal_places:
        return scores.round(decimal_places)
    return scores


def get_empirical_coverage_rates(
    predictions: Dict[str, Union["ForecastCollection", Path]],
    lead_times: Optional[List[int]] = None,
    item_ids: Optional[List[int]] = None,
    decimal_places: Optional[int] = None,
) -> pd.DataFrame:
    """Computes empirical coverage rates for different prediction sources,
    averaged across the specified lead times.

    Parameters
    -----------
    predictions : Dict[str, Union[ForecastCollection, Path]]
        Dictionary mapping keys to either:
        - Preloaded ForecastCollection objects, or
        - File paths (Path or str) to joblib files that contain ForecastCollection objects.
    lead_times : Optional[List[int]], default=None
        List of lead times to filter CRPS scores. If None, all lead times are used.
    item_ids : Optional[List[int]], default=None
        List of item IDs to include in the CRPS computation. If None, all item IDs are used.
    decimal_places : Optional[int], default=None
        Number of decimal places to round numerical values to. If None, no rounding is applied.

    Returns
    --------
    pd.DataFrame
        A DataFrame where each column corresponds to a prediction source,
        and values represent the empirical coverage rates averaged over the specified lead times.
    """
    scores_dict = {}

    for key, value in predictions.items():
        if isinstance(value, ForecastCollection):
            pred = value
        elif isinstance(value, (str, Path)):
            pred = ForecastCollection.load(value)
        else:
            raise TypeError(f"Unsupported prediction type for key '{key}': {type(value)}")

        cov_rate_df = pred.get_empirical_coverage_rates(lead_times=lead_times, mean_lead_times=True, item_ids=item_ids)

        scores_dict[key] = cov_rate_df.squeeze()  # Convert single-row DataFrame to Series

    scores = pd.DataFrame(scores_dict)

    scores.columns = [key for key in predictions.keys()]

    if decimal_places:
        return scores.round(decimal_places)
    return scores


def _compute_crps_one(
    key: str,
    value: Union["ForecastCollection", str, Path],
    lead_times: Optional[List[int]],
    mean_lead_times: bool,
    item_ids: Optional[List[int]],
    reference_key: Optional[str],
) -> Tuple[str, pd.Series, bool]:
    """
    Compute CRPS for a single prediction entry (preloaded ForecastCollection or path).
    Returns (key, series, is_reference).
    """
    pred = ForecastCollection.load(value) if isinstance(value, (str, Path)) else value

    crps_df = pred.get_crps(
        lead_times=lead_times,
        mean_lead_times=mean_lead_times,
        mean_time=True,
        mean_item_ids=True,
        item_ids=item_ids,
        decimal_places=None,
    )

    series = crps_df.squeeze()  # Series indexed by lead times or a scalar if mean_lead_times
    is_reference = reference_key is not None and key == reference_key
    return key, series, is_reference


def get_crps_scores(
    predictions: Dict[str, Union["ForecastCollection", Path]],
    lead_times: Optional[List[int]] = None,
    mean_lead_times: bool = False,
    item_ids: Optional[List[int]] = None,
    reference_predictions: Optional[str] = None,
    add_mean: Optional[bool] = True,
    decimal_places: Optional[int] = None,
    sort: bool = True,
    n_jobs: int = -1,  # parallelism: use all cores by default
    backend: str = "loky",  # processes (safer for CPU-bound work)
    prefer: Optional[str] = "processes",
) -> pd.DataFrame:
    """
    Parallelized CRPS computation for picklable ForecastCollections (and/or paths to them).
    """
    keys_in_order = list(predictions.keys())

    tasks = (delayed(_compute_crps_one)(key, predictions[key], lead_times, mean_lead_times, item_ids, reference_predictions) for key in keys_in_order)

    results: List[Tuple[str, pd.Series, bool]] = []
    with tqdm_joblib(tqdm(total=len(keys_in_order), desc="Compute CRPS (parallel)")):
        results = Parallel(n_jobs=n_jobs, backend=backend, prefer=prefer)(tasks)

    # Reassemble in original order
    series_by_key: Dict[str, pd.Series] = {}
    reference_scores: Optional[pd.Series] = None
    for key, s, is_ref in results:
        series_by_key[key] = s
        if is_ref:
            reference_scores = s.copy()

    # Safety for reference
    if reference_predictions and reference_scores is None:
        raise ValueError(f"Reference prediction '{reference_predictions}' not found.")

    # Build scores table
    if mean_lead_times:
        scores = pd.DataFrame(series_by_key, index=["Mean CRPS"])
        add_mean = False
    else:
        scores = pd.DataFrame(series_by_key)
        scores.index.name = "lead times"

    # # Normalize by reference if requested
    # if reference_predictions:
    #     scores = scores.div(reference_scores.squeeze(), axis=0)

    # Optional mean row
    if add_mean:
        scores.loc["Mean CRPS", :] = scores.mean(axis=0)

    # Sort columns by mean CRPS if requested
    if sort and "Mean CRPS" in scores.index:
        scores = scores.T.sort_values(by="Mean CRPS", axis=0).T
    else:
        # keep original column order
        scores = scores.reindex(columns=keys_in_order)

    if reference_predictions:
        scores = scores.div(scores[reference_predictions].squeeze(), axis=0)

    # Rounding
    if decimal_places is not None:
        scores = scores.round(decimal_places)

    return scores


def plot_crps(
    predictions: Dict[str, Union["ForecastCollection", Path]],
    selected_keys: Optional[List] = None,
    lead_times: Optional[List[int]] = None,
    item_ids: Optional[List[int]] = None,
    rolling_window_eval: Optional[int] = None,
    reference_predictions: Optional[str] = None,
) -> None:
    """Plots the mean CRPS (Continuous Ranked Probability Score) over time for different prediction sources.

    Parameters
    -----------
    predictions : Dict[str, Union[ForecastCollection, Path]]
        Dictionary mapping keys to either:
        - Preloaded ForecastCollection objects, or
        - File paths (Path or str) to joblib files that contain ForecastCollection objects.
    selected_keys: Optional[List], default=None
        List of ForecastCollection objects which sould be considered. If None, all ForecastCollections are displayed.
    lead_times : Optional[List[int]], default=None
        List of lead times to filter CRPS scores. If None, all lead times are used.
    item_ids : Optional[List[int]], default=None
        List of item IDs to include in the CRPS computation. If None, all item IDs are used.
    rolling_window_eval : Optional[int], default=None
        Window size for computing the rolling mean of CRPS scores. If None, no smoothing is applied.
    reference_predictions : Optional[str], default=None
        Key of a prediction set to be used as a reference for normalization.
        If provided, all CRPS values will be divided by the CRPS values from this prediction.

    Returns
    --------
    None
    """

    if selected_keys:
        predictions = {key: value for key, value in predictions.items() if key in selected_keys}

    scores_dict = {}

    for key, value in predictions.items():
        if isinstance(value, ForecastCollection):
            pred = value
        elif isinstance(value, (str, Path)):
            pred = ForecastCollection.load(value)
        else:
            raise TypeError(f"Unsupported prediction type for key '{key}': {type(value)}")

        crps_df = pred.get_crps(
            lead_times=lead_times,
            mean_lead_times=True,
            mean_time=False,
            mean_item_ids=True,
            item_ids=item_ids,
            decimal_places=None,
        )

        scores_dict[key] = crps_df.squeeze()  # Convert single-row DataFrame to Series

    df = pd.DataFrame(scores_dict)

    if rolling_window_eval:
        df = df.rolling(window=rolling_window_eval).mean()

    if reference_predictions:
        df = df.apply(lambda x: x / x[reference_predictions], axis=1)

    # Plotting with matplotlib
    plt.figure(figsize=(12, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col, linewidth=2)

    plt.title("Mean CRPS Over Time", fontsize=16)
    plt.xlabel("Timestamp", fontsize=12)
    plt.ylabel("CRPS" if not reference_predictions else "Relative CRPS", fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(title="Prediction Source", fontsize=10)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.show()


def plot_crps_across_lead_times(
    predictions: Dict[str, Union["ForecastCollection", Path]],
    selected_keys: Optional[List] = None,
    item_ids: Optional[List[int]] = None,
    reference_predictions: Optional[str] = None,
    lead_times: Optional[List[int]] = None,
) -> None:
    """Plots the mean CRPS score across various forecast lead times.

    Parameters
    -----------
    predictions : Dict[str, Union[ForecastCollection, Path]]
        Dictionary mapping keys to either:
        - Preloaded ForecastCollection objects, or
        - File paths (Path or str) to joblib files that contain ForecastCollection objects.
    selected_keys: Optional[List], default=None
        List of ForecastCollection objects which sould be considered. If None, all ForecastCollections are displayed.
    item_ids : Optional[List[int]], default=None
        List of item IDs to include in the CRPS computation. If None, all item IDs are used.
    reference_predictions : Optional[str], default=None
        Key of a prediction set to be used as a reference for normalization.
        If provided, all CRPS values will be divided by the CRPS values from this prediction.

    Returns
    --------
    None
    """

    if selected_keys:
        predictions = {key: value for key, value in predictions.items() if key in selected_keys}

    df = get_crps_scores(predictions, item_ids=item_ids, lead_times=lead_times, reference_predictions=reference_predictions, add_mean=False, decimal_places=None)
    df.rename(columns={reference_predictions: f"{reference_predictions} (Reference)"}, inplace=True)
    ax = df.plot(figsize=(12, 8), legend=True)
    # ax.set_title("CRPS Scores Comparison across Forecasting Lead Times", fontsize=16)
    ax.set_ylabel("Relative Mean CRPS Score", fontsize=14)
    ax.set_xlabel("Lead Times", fontsize=14)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


def get_crps_by_period(
    predictions: Dict[str, Union["ForecastCollection", Path]],
    date_splits: List[pd.Timestamp],
    lead_times: Optional[List[int]] = None,
    item_ids: Optional[List[int]] = None,
    reference_predictions: Optional[str] = None,
    decimal_places: Optional[int] = None,
) -> pd.DataFrame:
    """Computes the mean CRPS (Continuous Ranked Probability Score) over time periods
    defined by timestamp splits for different prediction sources.

    Parameters
    -----------
    predictions : Dict[str, Union[ForecastCollection, Path]]
        Dictionary mapping keys to either:
        - Preloaded ForecastCollection objects, or
        - File paths (Path or str) to joblib files that contain ForecastCollection objects.
    date_splits : List[pd.Timestamp]
        List of timestamps to split the CRPS data into time-based segments.
        The function calculates mean CRPS in each period between these dates.
    lead_times : Optional[List[int]], default=None
        List of lead times to filter CRPS scores. If None, all lead times are used.
    item_ids : Optional[List[int]], default=None
        List of item IDs to include in the CRPS computation. If None, all item IDs are used.
    reference_predictions : Optional[str], default=None
        Key of a prediction set to be used as a reference for normalization.
        If provided, all CRPS values will be divided by the CRPS values from this prediction.
    decimal_places : Optional[int], default=None
        Number of decimal places to round numerical values to. If None, no rounding is applied.

    Returns
    --------
    pd.DataFrame
        A DataFrame with the mean CRPS values for each time segment (e.g., "2023-01-01_to_2023-06-01"),
        optionally normalized by a reference prediction. Each column corresponds to a prediction key.
    """

    scores_dict = {}

    for key, value in predictions.items():
        if isinstance(value, ForecastCollection):
            pred = value
        elif isinstance(value, (str, Path)):
            pred = ForecastCollection.load(value)
        else:
            raise TypeError(f"Unsupported prediction type for key '{key}': {type(value)}")

        crps_df = pred.get_crps(
            lead_times=lead_times,
            mean_lead_times=True,
            mean_time=False,
            mean_item_ids=False,
            item_ids=item_ids,
            decimal_places=None,
        )

        scores_dict[key] = crps_df.squeeze()  # Convert single-row DataFrame to Series

    df = pd.DataFrame(scores_dict)

    df = df.reset_index()

    results: dict[str, pd.Series] = {}

    date_splits = sorted(date_splits)

    first_date = df[TIMESTAMP].min()
    last_date = df[TIMESTAMP].max()
    date_splits.append(last_date)

    for d in date_splits:
        if not isinstance(d, pd.Timestamp):
            raise TypeError(f"All date_splits must be pd.Timestamp, got {type(d)}")

        subset_df = df[(df[TIMESTAMP] > first_date) & (df[TIMESTAMP] <= d)]

        # after = df[df["timestamp"] >= d]
        results[f"{first_date.strftime("%d-%m-%Y")}_to_{d.strftime("%d-%m-%Y")}"] = subset_df.drop(columns=[ITEMID, TIMESTAMP]).mean()
        first_date = d

    results = pd.DataFrame(results).T

    if reference_predictions:
        results = results.apply(lambda x: x / x[reference_predictions], axis=1)

    if decimal_places:
        return results.round(decimal_places)
    return results


def diebold_mariano_test(loss1: Union[pd.Series, np.ndarray], loss2: Union[pd.Series, np.ndarray], maxlags: int = 8) -> Tuple[float, float]:
    """
    Perform the Diebold-Mariano test for equal predictive accuracy.

    The test evaluates whether the predictive performance of two forecasting models
    is significantly different, using a loss differential series and Newey-West
    adjusted standard errors.

    Parameters
    ----------
    loss1 : Union[pd.Series, np.ndarray]
        The loss series (e.g., CRPS) for the first model.
    loss2 : Union[pd.Series, np.ndarray]
        The loss series for the second model.
    maxlags : int, default=8
        The maximum lag to use in computing Newey-West HAC standard errors.

    Returns
    -------
    Tuple[float, float]
        A tuple containing:
        - t-statistic of the mean loss differential
        - p-value of the test statistic

    Interpretation
    --------------
    Positive t-statistic:
        The first forecast (`loss1`) is more accurate than the second forecast (`loss2`).
    Negative t-statistic:
        The second forecast (`loss2`) is more accurate than the first forecast (`loss1`).
    """
    # 1. Compute loss differential
    d = np.array(loss2 - loss1)
    d = d[~np.isnan(d)]

    # Regress d on a constant
    X = np.ones((len(d), 1))
    ols = sm.OLS(d, X).fit()

    # Compute Newey-West standard errors with lag K
    nw_results = ols.get_robustcov_results(cov_type="HAC", maxlags=maxlags)

    # Print summary to see the SE
    # print(nw_results.summary())

    # Extract the standard error of the intercept
    # se_mean = nw_results.bse[0]
    # tvalues = np.mean(d) / se_mean

    return nw_results.tvalues.item(), nw_results.pvalues.item()


def get_pairwise_diebold_mariano_test(
    predictions: Dict[str, Union["ForecastCollection", Path]],
    lead_times: Optional[List[int]] = None,
    item_ids: Optional[List[int]] = None,
    maxlags: int = 8,
    decimal_places: Optional[int] = None,
    reduce_matrix: bool = False,
    sort: bool = True,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute pairwise Diebold-Mariano test statistics between multiple forecast models.

    For each unique pair of models, this function:
    1. Loads the CRPS values.
    2. Computes the loss differentials.
    3. Performs the Diebold-Mariano test.
    4. Stores the resulting t-statistics and p-values in DataFrames.

    Parameters
    ----------
    predictions : Dict[str, Union[ForecastCollection, Path]]
        Dictionary mapping model names to either:
        - Preloaded ForecastCollection objects, or
        - Paths to joblib files containing ForecastCollection objects.
    lead_times : Optional[List[int]], default=None
        List of lead times to include in the CRPS computation. If None, all lead times are used.
    item_ids : Optional[List[int]], default=None
        List of item IDs to include in the CRPS computation. If None, all item IDs are used.
    maxlags : int, default=8
        Maximum lag to use for Newey-West HAC standard errors in the Diebold-Mariano test.
    decimal_places : Optional[int], default=None
        Number of decimal places to round the results to. If None, no rounding is applied.
    reduce_matrix : bool, default=False,
        Whether to only show non redundant information by only computing the upper diagonal of the matrix.
    sort : bool, default=True,
        Whether to sort the results
    top_k : int, optional
        top k results to display. defaults to all ForecastCollections are shown

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Two DataFrames:
        - t-statistics of the Diebold-Mariano tests (upper triangle only)
        - p-values corresponding to the t-statistics
        Rows and columns correspond to model names.

    Interpretation
    --------------
    In the returned tables:
    - A **positive t-statistic** means that the row model's forecasts are **more accurate**
      than the column model's forecasts.
    - A **negative t-statistic** means that the column model's forecasts are **more accurate**
      than the row model's forecasts.
    - The p-value indicates the statistical significance of the difference in accuracy.
    """
    scores_dict = {}

    for key, value in tqdm(predictions.items(), desc="Compute CRPS score"):
        if isinstance(value, ForecastCollection):
            pred = value
        elif isinstance(value, (str, Path)):
            pred = ForecastCollection.load(value)
        else:
            raise TypeError(f"Unsupported prediction type for key '{key}': {type(value)}")

        crps_df = pred.get_crps(
            lead_times=lead_times,
            mean_lead_times=True,
            mean_time=False,
            mean_item_ids=True,
            item_ids=item_ids,
            decimal_places=None,
        )

        scores_dict[key] = crps_df.squeeze()  # Convert single-row DataFrame to Series

    # macro weighting scheme
    if sort:
        keys = sorted(scores_dict, key=lambda k: scores_dict[k].mean())
    else:
        keys = list(scores_dict.keys())
    if top_k:
        keys = keys[:top_k] if sort else sorted(scores_dict, key=lambda k: scores_dict[k].mean())[:top_k]

    if reduce_matrix:
        dm_tval = pd.DataFrame(index=keys[:-1], columns=keys[1:], dtype=float)
        dm_pval = pd.DataFrame(index=keys[:-1], columns=keys[1:], dtype=float)

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):  # Upper triangle only, no diagonal
                t_val, p_val = diebold_mariano_test(scores_dict[keys[i]], scores_dict[keys[j]], maxlags)
                dm_tval.loc[keys[i], keys[j]] = t_val
                dm_pval.loc[keys[i], keys[j]] = p_val
    else:
        dm_tval = pd.DataFrame(index=keys, columns=keys, dtype=float)
        dm_pval = pd.DataFrame(index=keys, columns=keys, dtype=float)

        for i in range(len(keys)):
            for j in range(len(keys)):  # Upper triangle only, no diagonal
                t_val, p_val = diebold_mariano_test(scores_dict[keys[i]], scores_dict[keys[j]], maxlags)
                dm_tval.loc[keys[i], keys[j]] = t_val
                dm_pval.loc[keys[i], keys[j]] = p_val

    if decimal_places is not None:
        dm_tval = dm_tval.round(decimal_places)
        dm_pval = dm_pval.round(decimal_places)

    return dm_tval, dm_pval


def plot_crps_barh(
    predictions: Dict[str, Union["ForecastCollection", Path]],
    lead_times: Union[List[int], List[List[int]]] = [list(range(1, 4))],
    reference_predictions: Optional[str] = None,
    groups: Optional[Dict[str, List[str]]] = None,
    group_colors: Optional[Dict[str, str]] = None,
    default_color: str = "#7f7f7f",
    # NEW:
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    panel_size: Tuple[float, float] = (8.0, 6.0),  # (width, height) per panel
    title_fontsize: int = 16,
    xlabel_fontsize: int = 14,
    tick_fontsize: int = 12,
    barlabel_fontsize: int = 10,
    legend_fontsize: int = 12,
) -> None:
    """
    Plot one or more horizontal bar charts of Mean (or Relative) CRPS scores
    for specified lead times, using a common x-axis range across subplots.

    Grid control:
      - Pass nrows/ncols to force a layout (e.g., 1x3, 2x2, 3x3).
      - Leave both None to auto-arrange into a near-square grid.
    """

    # Normalize lead_times → list of lists
    if isinstance(lead_times[0], int):
        lead_times = [lead_times]
    n_subplots = len(lead_times)

    # --- First pass: compute all CRPS data & gather global min/max -------------
    crps_frames = []  # list of (lead_time_set, df_sorted)
    global_min = float("inf")
    global_max = float("-inf")

    for lead_time_set in lead_times:
        crps_results_mean = get_crps_scores(
            predictions,
            lead_times=lead_time_set,
            reference_predictions=reference_predictions,
            mean_lead_times=True,
            add_mean=True,
            sort=True,
        )

        df = crps_results_mean.loc["Mean CRPS"].to_frame().reset_index()
        df.columns = ["Model", "Mean CRPS"]
        df = df.reset_index(drop=True)
        crps_frames.append((lead_time_set, df))

        vmin = df["Mean CRPS"].min()
        vmax = df["Mean CRPS"].max()
        global_min = min(global_min, vmin)
        global_max = max(global_max, vmax)

    # margin
    span = global_max - global_min
    if span == 0:
        span = abs(global_max) if global_max != 0 else 1.0
    pad = span * 0.15
    xlo = 0 if global_min >= 0 else (global_min - pad)
    xhi = global_max + pad

    # --- Grid layout (auto or specified) --------------------------------------
    def _auto_grid(n: int) -> Tuple[int, int]:
        # near-square grid
        c = int(math.ceil(math.sqrt(n)))
        r = int(math.ceil(n / c))
        return r, c

    if nrows is None and ncols is None:
        nrows, ncols = _auto_grid(n_subplots)
    elif nrows is None:
        nrows = int(math.ceil(n_subplots / ncols))
    elif ncols is None:
        ncols = int(math.ceil(n_subplots / nrows))
    # If still too few slots, expand rows
    if nrows * ncols < n_subplots:
        nrows = int(math.ceil(n_subplots / ncols))

    fig_width = panel_size[0] * ncols
    fig_height = panel_size[1] * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex=True,  # share x across all panels
    )
    axes_flat = axes.ravel()

    # choose label text depending on relative vs absolute
    x_label = "Relative CRPS" if reference_predictions else "Mean CRPS"

    # Prepare legend colors (even if no bars match, legend still shows all groups)
    legend_labels = []
    legend_colors = []
    if groups is not None:
        if group_colors is None:
            # generate a palette if not provided
            try:
                palette = sns.color_palette("tab10", n_colors=len(groups))
                auto_colors = {g: mpl.colors.to_hex(palette[i]) for i, g in enumerate(groups.keys())}
            except Exception:
                # fallback if seaborn not available
                cmap = plt.cm.get_cmap("tab10", len(groups))
                auto_colors = {g: mpl.colors.to_hex(cmap(i)) for i, g in enumerate(groups.keys())}
            group_colors = auto_colors

        # keep insertion order from `groups`
        for g in group_colors.keys():
            legend_labels.append(g)
            legend_colors.append(group_colors.get(g, default_color))
        # Add an "Other" bucket
        # legend_labels.append("Other")
        # legend_colors.append(default_color)

    # --- Plot loop ------------------------------------------------------------
    for ax, payload in zip(axes_flat, crps_frames):
        lead_time_set, df = payload

        # Colors
        if groups is not None and group_colors is not None:
            colors: List[str] = []
            for model in df["Model"]:
                for group_name, substrings in groups.items():
                    if any(s.lower() in model.lower() for s in substrings):
                        colors.append(group_colors.get(group_name, default_color))
                        break
                else:
                    colors.append(default_color)
        else:
            colors = sns.color_palette("deep", n_colors=len(df))

        bars = ax.barh(df["Model"], df["Mean CRPS"], height=0.6, color=colors)

        # Annotation offset: a small fraction of axis span
        offset = (xhi - xlo) * 0.005
        for bar in bars:
            width = bar.get_width()
            xpos = width + offset if width >= 0 else width - offset
            ax.text(
                xpos,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}",
                va="center",
                ha="left" if width >= 0 else "right",
                fontsize=barlabel_fontsize,
            )

        ax.set_xlim(xlo, xhi)
        title = (
            f"Relative Mean CRPS (Lead Times {min(lead_time_set)}–{max(lead_time_set)})"
            if reference_predictions
            else f"Mean CRPS (Lead Times {min(lead_time_set)}–{max(lead_time_set)})"
        )
        ax.set_xlabel(x_label, fontsize=xlabel_fontsize)
        ax.set_title(title, fontsize=title_fontsize, pad=20)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    # --- Figure-level legend at the bottom -----------------------------------
    if groups is not None:
        import matplotlib.patches as mpatches

        handles = [mpatches.Patch(color=c, label=l) for l, c in zip(legend_labels, legend_colors)]

        fig.subplots_adjust(bottom=0.2)
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 6),
            frameon=False,
            bbox_to_anchor=(0.5, -0.03),
            fontsize=legend_fontsize,
        )

    plt.tight_layout(h_pad=5)
    plt.show()


def plot_pairwise_diebold_mariano_test(
    predictions: Dict[str, Union["ForecastCollection", Path]],
    lead_times: Optional[Union[List[int], List[List[int]]]] = None,
    item_ids: Optional[List[int]] = None,
    maxlags: int = 480,
    decimal_places: Optional[int] = None,
    reduce_matrix: bool = False,
    sort: bool = True,
    figsize_per_panel: float = 5.0,
    top_k: Optional[int] = None,
    # -------------------- FONT CONTROLS --------------------
    fontsize_suptitle: int = 25,
    fontsize_title: int = 15,
    fontsize_annot: int = 13,
    fontsize_cbar_label: int = 15,
    fontsize_cbar_ticks: int = 15,
    fontsize_xtick: int = 13,
    fontsize_ytick: int = 13,
    font_scale: Optional[float] = None,  # multiply all sizes (e.g., 0.9 to shrink)
):
    """
    Plot pairwise Diebold-Mariano t-values across models.

    Parameters
    ----------
    predictions : dict[str, ForecastCollection|Path]
    lead_times : list[int] or list[list[int]], optional
        - list[int] -> single heatmap (original behavior)
        - list[list[int]] -> multiple heatmaps, auto-gridded.
    item_ids : list[int], optional
    maxlags : int
    decimal_places : int, optional
    reduce_matrix : bool, default False
        If True, plot reduced matrix (e.g., lower triangle) and suppress diag grey + shared colorbar.
    sort : bool
    figsize_per_panel : float
        Size (inches) allocated per panel (square); figure size scales with grid.
    top_k : int, optional
        top k results to display. defaults to all ForecastCollections are shown

    Fonts
    -----
    fontsize_suptitle : int
    fontsize_title : int
    fontsize_annot : int
    fontsize_cbar_label : int
    fontsize_cbar_ticks : int
    fontsize_xtick : int
    fontsize_ytick : int
    font_scale : float, optional
        Multiply all font sizes by this factor.
    """

    # -------------------- apply font_scale if given --------------------
    if font_scale is not None:
        fontsize_suptitle = int(round(fontsize_suptitle * font_scale))
        fontsize_title = int(round(fontsize_title * font_scale))
        fontsize_annot = int(round(fontsize_annot * font_scale))
        fontsize_cbar_label = int(round(fontsize_cbar_label * font_scale))
        fontsize_cbar_ticks = int(round(fontsize_cbar_ticks * font_scale))
        fontsize_xtick = int(round(fontsize_xtick * font_scale))
        fontsize_ytick = int(round(fontsize_ytick * font_scale))

    # ------------------------------------------------------------------
    # Detect single vs multi-panel input
    # ------------------------------------------------------------------
    is_multi = lead_times is not None and isinstance(lead_times, (list, tuple)) and len(lead_times) > 0 and isinstance(lead_times[0], (list, tuple, np.ndarray))

    if not is_multi:
        lead_groups = [lead_times]  # may still be None -> handled below
    else:
        lead_groups = list(lead_times)

    n_panels = len(lead_groups)

    # If no lead_times supplied, use all leads from first prediction
    if lead_times is None:
        first_entry = list(predictions.keys())[0]
        if isinstance(predictions[first_entry], ForecastCollection):
            pred = first_entry
        elif isinstance(first_entry, (str, Path)):
            pred = ForecastCollection.load(first_entry)
        else:
            raise TypeError(f"Unsupported prediction type for key '{first_entry}': {type(predictions[first_entry])}")
        full_leads = pred.get_lead_times()
        lead_groups = [full_leads]
        n_panels = 1
        is_multi = False

    # ------------------------------------------------------------------
    # Auto grid size
    # ------------------------------------------------------------------
    def _auto_grid(n):
        """Return nrows, ncols following rule: <=3 -> 1 row; else near-square."""
        if n <= 3:
            return 1, n
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        return nrows, ncols

    nrows, ncols = _auto_grid(n_panels)

    # ------------------------------------------------------------------
    # Shared color scale
    # ------------------------------------------------------------------
    vmin, vmax = -3, 3

    deep_colors = sns.color_palette("deep")
    neg_color = deep_colors[1]
    pos_color = deep_colors[0]
    custom_cmap = LinearSegmentedColormap.from_list("CustomDeep", [neg_color, "white", pos_color], N=256)

    # ------------------------------------------------------------------
    # Figure + axes layout
    # ------------------------------------------------------------------
    if not reduce_matrix:
        fig_w = figsize_per_panel * ncols + 0.8
        fig_h = figsize_per_panel * nrows
        fig = plt.figure(figsize=(fig_w, fig_h))
        from matplotlib import gridspec

        gs = fig.add_gridspec(
            nrows=nrows,
            ncols=ncols + 1,  # extra column for cbar
            width_ratios=[1] * ncols + [0.04],
            height_ratios=[1] * nrows,
            wspace=0.1,
            hspace=0.1,
        )
        axes = []
        for r in range(nrows):
            row_axes = []
            for c in range(ncols):
                row_axes.append(fig.add_subplot(gs[r, c]))
            axes.append(row_axes)
        cbar_ax = fig.add_subplot(gs[:, -1])  # span all rows
    else:
        fig_w = figsize_per_panel * ncols
        fig_h = figsize_per_panel * nrows
        fig, ax_grid = plt.subplots(
            nrows,
            ncols,
            figsize=(fig_w, fig_h),
            squeeze=False,
        )
        axes = ax_grid.tolist()
        cbar_ax = None

    # ------------------------------------------------------------------
    # Helper: draw one panel
    # ------------------------------------------------------------------
    def _panel(ax, leads, show_cbar=False, cbar_ax=None, show_y=True, show_x=True):
        tvalues, pvalues = get_pairwise_diebold_mariano_test(
            predictions,
            lead_times=leads,
            item_ids=item_ids,
            maxlags=maxlags,
            decimal_places=decimal_places,
            reduce_matrix=reduce_matrix,
            sort=sort,
            top_k=top_k,
        )

        # Truncate long names
        # tvalues = tvalues.rename(columns={n: n[:20] for n in tvalues.columns})
        # pvalues = pvalues.rename(columns={n: n[:20] for n in pvalues.columns})
        # tvalues.index = [i[:20] for i in tvalues.index]
        # pvalues.index = [i[:20] for i in pvalues.index]

        assert tvalues.shape == pvalues.shape
        assert tvalues.index.equals(pvalues.index) and tvalues.columns.equals(pvalues.columns)

        def stars(p):
            if pd.isna(p):
                return ""
            if p <= 0.01:
                return "***"
            if p <= 0.05:
                return "**"
            if p <= 0.10:
                return "*"
            return ""

        t_str = tvalues.applymap(lambda x: f"{x}" if pd.notna(x) else "")
        s_str = pvalues.applymap(stars)
        annot = (t_str + "\n" + s_str).where(~pvalues.isna(), "")

        mask = pvalues.isna().to_numpy()

        hm = sns.heatmap(
            tvalues,
            mask=mask,
            cmap=custom_cmap,
            vmin=vmin,
            vmax=vmax,
            center=0,
            annot=annot,
            fmt="",
            annot_kws={"size": fontsize_annot},
            cbar=show_cbar,
            cbar_ax=cbar_ax,
            cbar_kws=(
                None
                if not show_cbar
                else dict(
                    label="DM t-value",
                    ticks=[vmin, -2, -1, 0, 1, 2, vmax],
                )
            ),
            square=True,
            linewidths=0.01,
            linecolor="lightgrey",
            ax=ax,
        )

        # Colorbar styling
        if show_cbar:
            cbar = hm.collections[0].colorbar
            cbar.set_ticklabels(["≤-3", "-2", "-1", "0", "1", "2", "≥3"])
            cbar.ax.set_ylabel("DM t-value", fontsize=fontsize_cbar_label)
            cbar.ax.tick_params(labelsize=fontsize_cbar_ticks)

        # Panel title
        ax.set_title(f"Lead Time {min(leads)}–{max(leads)}", pad=10, fontsize=fontsize_title)

        # Tick labels
        # X ticks (rotate & size)
        # Center ticks under cells
        ax.set_xticks(np.arange(tvalues.shape[1]) + 0.5)
        ax.set_xticklabels(
            tvalues.columns,
            rotation=45,
            ha="right",
            va="bottom",
            rotation_mode="anchor",
        )
        # Push tick labels further away
        ax.tick_params(axis="x", which="major", pad=15)

        for lab in ax.get_xticklabels():
            lab.set_fontsize(fontsize_xtick)

        # Y ticks (size)
        for lab in ax.get_yticklabels():
            lab.set_fontsize(fontsize_ytick)

        # Grey the diagonal only for full matrix
        if not reduce_matrix:
            n = tvalues.shape[0]
            diag_face = "0.85"
            for k in range(n):
                ax.add_patch(
                    plt.Rectangle(
                        (k, k),
                        1,
                        1,
                        facecolor=diag_face,
                        edgecolor=diag_face,
                        linewidth=0,
                        zorder=3,
                    )
                )

        # Hide axes if requested
        if not show_y:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False, labelright=False)
        if not show_x:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False, labeltop=False)

        return hm

    # ------------------------------------------------------------------
    # Draw panels into grid
    # ------------------------------------------------------------------
    heatmaps = []
    panel_idx = 0
    for r in range(nrows):
        for c in range(ncols):
            if panel_idx >= n_panels:
                axes[r][c].axis("off")
                continue
            leads = lead_groups[panel_idx]
            show_cbar = (panel_idx == n_panels - 1) and (cbar_ax is not None)
            hm = _panel(
                axes[r][c],
                leads,
                show_cbar=show_cbar,
                cbar_ax=cbar_ax if show_cbar else None,
                show_y=(c == 0),
                show_x=(r == (nrows - 1)),
            )
            heatmaps.append(hm)
            panel_idx += 1

    # Super-title
    if n_panels > 1:
        fig.suptitle("Diebold-Mariano t-values by Lead Time Group", y=0.95, fontsize=fontsize_suptitle)

    plt.show()
    return fig, axes, heatmaps


def plot_reliability_diagram(
    collections: Dict[str, "ForecastCollection"],
    lead_times: Optional[Union[List[int], List[List[int]]]] = None,
    overlay: bool = True,
    item_ids: Optional[List[int]] = None,
    show_individual_lead_times: bool = False,
    mean_lead_times: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    font_sizes: Optional[Dict[str, int]] = None,
) -> None:
    """Plot reliability diagrams (empirical coverage vs nominal quantile) for *multiple* ForecastCollection objects.

    Supports three display modes:

    1. **Single-overlay (default)**: If ``overlay=True`` and ``lead_times`` is a *flat* list (or ``None``), all
       collections are plotted together in one axis. Optionally show individual lead lines and/or an equal-weight
       macro-mean across the selected lead times.
    2. **Multi-panel overlay**: If ``overlay=True`` *and* ``lead_times`` is a *list of lead-time groups* (list of lists),
       one subplot is created per group; *within* each subplot all collections are overlaid for the group's lead set.
       A single shared legend is placed to the **right** of the grid (as you requested).
    3. **Faceted by collection**: If ``overlay=False`` (regardless of lead-time grouping), create one subplot per
       collection (original behavior). Lead-time grouping is ignored in this mode; pass a flat list of leads to control
       the subset used in each panel.

    Parameters
    ----------
    collections : dict[str, ForecastCollection]
        Mapping from a short name/label to a ForecastCollection instance. Appears in legends / subplot titles.
    lead_times : list[int] | list[list[int]], optional
        Flat list → single group.
        List of lists → multi-panel overlay with one subplot per group.
        ``None`` → use the union of *all* available lead times across collections (single group).
    overlay : bool, default True
        Overlay across *collections* (single axis or multi-panel grouping). If False, facet by collection.
    item_ids : list[int], optional
        Restrict to these item IDs (applied independently per collection). Missing IDs are ignored.
    show_individual_lead_times : bool, default False
        Plot per-lead curves (within whichever axes the mode dictates).
    mean_lead_times : bool, default True
        Plot macro-mean curve across the selected lead_times (within panel) using equal-weight mean across leads.
    """
    if not collections:
        raise ValueError("No ForecastCollection objects supplied.")

    # Defaults
    if font_sizes is None:
        font_sizes = {}
    label_fs = font_sizes.get("labels", 18)
    tick_fs = font_sizes.get("ticks", 18)
    title_fs = font_sizes.get("titles", 18)
    legend_fs = font_sizes.get("legend", 18)
    sup_fs = font_sizes.get("suptitle", 24)

    # ------------------------------------------------------------------
    # Utilities (DRY helpers)
    # ------------------------------------------------------------------
    def _get_all_leads() -> List[int]:
        leads = set()
        for fc in collections.values():
            if isinstance(fc, Path):
                fc = ForecastCollection.load(fc)
            leads.update(fc.get_lead_times())
        return sorted(leads)

    def _collect_coverages(fc: "ForecastCollection", leads: List[int]):
        """Return dict: {lead_time: [Series per item]}.

        Each Series indexed by nominal quantile (floats).
        """
        out = {lt: [] for lt in leads}
        if isinstance(fc, Path):
            fc = ForecastCollection.load(fc)

        for item_id in fc.get_item_ids():
            if item_ids and item_id not in item_ids:
                continue
            item = fc.get_time_series_forecast(item_id)
            for lt in leads:
                if lt in item.lead_time_forecasts:
                    val = item.get_empirical_coverage_rates(lt)  # {alpha: cov}
                    out[lt].append(pd.Series(val))
        return out

    def _macro_mean(series_list: List[pd.Series]):
        if not series_list:
            return None
        return pd.concat(series_list, axis=1).mean(axis=1)

    def _aggregate_fc(fc: "ForecastCollection", leads: List[int]):
        """Gather per-item coverages, per-lead macro means, and optional mean across leads.

        Returns (emp_per_lead: dict[int, Series], emp_avg: Series|None).
        """
        per_lead = _collect_coverages(fc, leads)
        emp_per_lead = {}
        for lt, ser_list in per_lead.items():
            if ser_list:
                emp_per_lead[lt] = _macro_mean(ser_list)
        emp_avg = None
        if mean_lead_times:
            all_ser = [s for s in emp_per_lead.values() if s is not None]
            if all_ser:
                emp_avg = pd.concat(all_ser, axis=1).mean(axis=1)
        return emp_per_lead, emp_avg

    def _infer_quantile_levels(emp_per_lead, emp_avg):
        if emp_avg is not None:
            return emp_avg.index.to_list()
        for s in emp_per_lead.values():
            if s is not None:
                return s.index.to_list()
        return []

    def _format_lead_title(leads: List[int]) -> str:
        if not leads:
            return ""
        leads = sorted(leads)
        contiguous = all(b - a == 1 for a, b in zip(leads[:-1], leads[1:]))
        if contiguous:
            return f"Lead Times {leads[0]}–{leads[-1]}"
        if len(leads) <= 6:
            return "Lead Times " + ", ".join(str(x) for x in leads)
        return f"{len(leads)} Leads"

    def _format_ax(ax, quant_levels, *, add_labels=True):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if quant_levels:
            ax.set_xticks(quant_levels)
            ax.set_xticklabels([f"{q:.1f}" for q in quant_levels], fontsize=tick_fs)
            ax.set_yticks(quant_levels)
            ax.set_yticklabels([f"{q:.1f}" for q in quant_levels], fontsize=tick_fs)
        if add_labels:
            ax.set_xlabel("Nominal Quantile Level", fontsize=label_fs)
            ax.set_ylabel("Empirical Coverage", fontsize=label_fs)
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect Calibration")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.set_aspect("equal", adjustable="box")

    for k, fc in collections.items():
        if isinstance(fc, str):
            collections[k] = Path(str)
    # ------------------------------------------------------------------
    # Determine lead_times / lead_time_groups
    # ------------------------------------------------------------------
    all_leads = _get_all_leads()
    if lead_times is None:
        lead_times = all_leads

    # Normalize to groups.
    # lead_time_groups: List[List[int]]
    if lead_times and isinstance(lead_times[0], (list, tuple, np.ndarray)):
        lead_time_groups = [sorted(set(map(int, grp))) for grp in lead_times]
    else:
        lead_time_groups = [sorted(set(map(int, lead_times)))]

    # Validate groups
    if not any(grp for grp in lead_time_groups):
        raise ValueError("No lead times available across supplied collections.")

    # ------------------------------------------------------------------
    # Helper to precompute aggregations *for a given lead group*
    # ------------------------------------------------------------------
    def _precompute_for_group(leads_for_group: List[int]):
        agg = {}
        for name, fc in collections.items():
            emp_per_lead, emp_avg = _aggregate_fc(fc, leads_for_group)
            agg[name] = (emp_per_lead, emp_avg)
        return agg

    palette = sns.color_palette("deep", n_colors=len(collections))
    color_by_collection = {name: palette[i] for i, name in enumerate(collections)}

    # ------------------------------------------------------------------
    # MULTI-PANEL OVERLAY MODE (list-of-lists)
    # ------------------------------------------------------------------
    if overlay and len(lead_time_groups) > 1:
        n_panels = len(lead_time_groups)

        # Grid heuristic
        if n_panels <= 3:
            rows, cols = 1, n_panels
        elif n_panels == 4:
            rows, cols = 2, 2
        else:
            cols = math.ceil(np.sqrt(n_panels))
            rows = math.ceil(n_panels / cols)

        # Use user-specified figsize if given, otherwise scale
        if figsize is None:
            fig_width = cols * 8
            fig_height = rows * 8
            figsize = (fig_width, fig_height)

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False, constrained_layout=True)
        axes = axes.ravel()

        legend_handles = {}
        for ax, leads_for_group in zip(axes, lead_time_groups):
            agg = _precompute_for_group(leads_for_group)
            global_quant_levels = []
            for _, (emp_per_lead, emp_avg) in agg.items():
                ql = _infer_quantile_levels(emp_per_lead, emp_avg)
                if ql:
                    global_quant_levels = ql
                    break

            for name, (emp_per_lead, emp_avg) in agg.items():
                base_c = color_by_collection[name]
                if show_individual_lead_times:
                    for lt, emp in emp_per_lead.items():
                        if emp is None:
                            continue
                        (ln,) = ax.plot(
                            emp.index,
                            emp.values,
                            marker="o",
                            linestyle="--",
                            label=f"{name} LT {lt}",
                            alpha=0.8,
                        )
                        legend_handles.setdefault(ln.get_label(), ln)
                if mean_lead_times and emp_avg is not None:
                    (ln,) = ax.plot(
                        emp_avg.index,
                        emp_avg.values,
                        marker="s",
                        linestyle="-",
                        linewidth=2,
                        label=f"{name}",
                        color=base_c,
                    )
                    legend_handles.setdefault(ln.get_label(), ln)

            _format_ax(ax, global_quant_levels, add_labels=False)
            ax.set_title(_format_lead_title(leads_for_group), fontsize=title_fs)

        for ax in axes[n_panels:]:
            ax.set_visible(False)

        fig.supxlabel("Nominal Quantile Level", fontsize=label_fs)
        fig.supylabel("Empirical Coverage", fontsize=label_fs)

        ttl_bits = []
        if show_individual_lead_times:
            ttl_bits.append("Leads")
        if mean_lead_times:
            ttl_bits.append("Avg")
        suffix = " + ".join(ttl_bits) if ttl_bits else ""
        fig.suptitle(f"Grouped Lead Times ({suffix})", fontsize=sup_fs)

        handles = list(legend_handles.values())
        labels = [h.get_label() for h in handles]
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            borderaxespad=0.0,
            frameon=False,
            fontsize=legend_fs,
        )

        fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, hspace=0.2, wspace=0.2)
        plt.show()
        return

    # ------------------------------------------------------------------
    # SINGLE-PANEL OVERLAY (flat list)
    # ------------------------------------------------------------------
    leads_for_overlay = lead_time_groups[0]

    # Precompute once
    agg = {}
    for name, fc in collections.items():
        emp_per_lead, emp_avg = _aggregate_fc(fc, leads_for_overlay)
        agg[name] = (emp_per_lead, emp_avg)

    # Global quantile grid (assume shared; fallback to first non-empty series found)
    global_quant_levels = []
    for _, (emp_per_lead, emp_avg) in agg.items():
        ql = _infer_quantile_levels(emp_per_lead, emp_avg)
        if ql:
            global_quant_levels = ql
            break

    if overlay:  # single panel overlay
        fig, ax = plt.subplots(figsize=(8, 8))
        for name, (emp_per_lead, emp_avg) in agg.items():
            base_c = color_by_collection[name]
            if show_individual_lead_times:
                for lt, emp in emp_per_lead.items():
                    if emp is None:
                        continue
                    ax.plot(
                        emp.index,
                        emp.values,
                        marker="o",
                        linestyle="--",
                        label=f"{name} Lead Time {lt}",
                        alpha=0.8,
                    )
            if mean_lead_times and emp_avg is not None:
                ax.plot(
                    emp_avg.index,
                    emp_avg.values,
                    marker="s",
                    linestyle="-",
                    linewidth=2,
                    label=f"{name}",
                    color=base_c,
                )

        _format_ax(ax, global_quant_levels)
        ttl_bits = []
        if show_individual_lead_times:
            ttl_bits.append("Leads")
        if mean_lead_times:
            ttl_bits.append("Avg")
        lt_title = _format_lead_title(leads_for_overlay)
        suffix = " + ".join(ttl_bits) if ttl_bits else ""
        title = lt_title if not suffix else f"{lt_title} ({suffix})"
        ax.set_title(title)
        ax.legend()
        plt.show()
        return

    # ------------------------------------------------------------------
    # Faceted mode: one subplot per collection
    # ------------------------------------------------------------------
    n = len(collections)
    cols = math.ceil(np.sqrt(n))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (name, (emp_per_lead, emp_avg)) in zip(axes, agg.items()):
        base_c = color_by_collection[name]
        if show_individual_lead_times:
            for lt, emp in emp_per_lead.items():
                if emp is not None:
                    ax.plot(
                        emp.index,
                        emp.values,
                        marker="o",
                        linestyle="--",
                        label=f"Lead Time {lt}",
                        alpha=0.8,
                    )
        if mean_lead_times and emp_avg is not None:
            ax.plot(
                emp_avg.index,
                emp_avg.values,
                marker="s",
                linestyle="-",
                linewidth=2,
                label="Avg",
                color=base_c,
            )

        _format_ax(ax, global_quant_levels, add_labels=False)
        ax.set_title(name)
        ax.legend(fontsize="small")

    # Hide any unused axes
    for ax in axes[len(collections) :]:
        ax.set_visible(False)

    # Set common x/y labels on outer figure
    fig.supxlabel("Nominal Quantile Level")
    fig.supylabel("Empirical Coverage")

    # Single suptitle reflecting lead selection & content flags
    ttl_bits = []
    if show_individual_lead_times:
        ttl_bits.append("Leads")
    if mean_lead_times:
        ttl_bits.append("Avg")
    lt_title = _format_lead_title(leads_for_overlay)
    suffix = ' + ".join(ttl_bits) if ttl_bits else "'
    suptitle = lt_title if not suffix else f"{lt_title} ({suffix})"
    fig.suptitle(suptitle)

    plt.tight_layout()
    plt.show()


def _coerce_paths(x: Union[List[Union[Path, str]], Path, str, None]) -> Optional[List[Path]]:
    """Normalize single path, iterable of paths, or None -> list[Path] | None."""
    if x is None:
        return None
    if isinstance(x, (str, Path)):
        return [Path(x)]
    return [Path(p) for p in x]


def _collect_files(
    *,
    files: Optional[List[Path]],
    dirs: Optional[List[Path]],
    required_name: Optional[str] = None,
    required_suffix: Optional[str] = None,
    recursive: bool = True,
) -> List[Path]:
    """
    Collect candidate files from explicit list and/or directories.

    Parameters
    ----------
    required_name : str, optional
        Exact filename to match (e.g., 'predictions.joblib').
    required_suffix : str, optional
        File suffix (e.g., '.joblib'); ignored if required_name given.
    recursive : bool
        Recurse into directories if True.

    Returns
    -------
    list[Path]
    """
    out: List[Path] = []

    # Explicit files
    if files:
        for f in files:
            if not f.is_file():
                logging.warning(f"Skipping non-file path: `{f}`")
                continue
            if required_name and f.name != required_name:
                logging.warning(f"Skipping `{f}` (name != {required_name})")
                continue
            if required_suffix and f.suffix != required_suffix:
                logging.warning(f"Skipping `{f}` (suffix != {required_suffix})")
                continue
            out.append(f)

    # Directories
    if dirs:
        for d in dirs:
            if not d.is_dir():
                logging.warning(f"Skipping non-directory path: `{d}`")
                continue
            if recursive:
                if required_name:
                    found = list(d.rglob(required_name))
                elif required_suffix:
                    found = [p for p in d.rglob(f"*{required_suffix}") if p.is_file()]
                else:
                    found = [p for p in d.rglob("*") if p.is_file()]
            else:
                if required_name:
                    found = [d / required_name] if (d / required_name).is_file() else []
                elif required_suffix:
                    found = [p for p in d.glob(f"*{required_suffix}") if p.is_file()]
                else:
                    found = [p for p in d.glob("*") if p.is_file()]
            out.extend(found)

    return out


def _strip_tokens(parts: List[str], tokens: set) -> List[str]:
    """Return a new list with any token removed."""
    return [p for p in parts if p not in tokens]


def _remove_duplicates(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def _build_key(filepath: Path, *, n_files: int, common_path: Path, strip_tokens: set) -> str:
    """
    Build a stable experiment/model key consistent across loaders.

    Rules (mirrors + fixes your original load_predictions logic):
    - If multiple files: key = underscore-joined parent path *relative to common root*.
    - If single file: use deepest non-generic parent directory (skip strip_tokens).
    - If nothing remains, fall back to file stem.

    Examples
    --------
    results/.../chronos-zero-shot-prediction_length/backtest/backtest_config.json
      -> 'chronos-zero-shot-prediction_length'
    """
    if n_files > 1:
        parts = list(filepath.relative_to(common_path).parent.parts)
    else:
        # full absolute parent chain; pick deepest non-generic piece
        parts = list(filepath.parent.parts)
        parts = _strip_tokens(parts, strip_tokens)
        parts = parts[-1:]  # keep last surviving part

    parts = _strip_tokens(parts, strip_tokens)
    parts = _remove_duplicates(parts)
    if not parts:
        return filepath.stem
    return "-".join(parts)


def load_predictions(
    prediction_dirs: Union[List[Union[Path, str]], Path, str, None] = None,
    prediction_files: Union[List[Union[Path, str]], Path, str, None] = None,
    load: bool = True,
) -> Dict[str, Union[Path, "ForecastCollection"]]:
    """
    Load saved prediction files from specified files or recursively from directories.

    You can either:
    - Provide a list of prediction files to load, or
    - Provide one or more directories. The function will recursively search for 'predictions.joblib' files inside them.

    If directories are used, the key for each loaded prediction will be constructed as 'parentfolder_filename'
    to make them distinguishable.

    Parameters
    ----------
    prediction_dirs : str, Path, or list of str/Path, optional
        One or multiple directories to search for prediction files.
    prediction_files : str, Path, or list of str/Path, optional
        Specific prediction files to load directly.
    load : bool, default = True
        Whether to load predictions into memory or only paths. Use lazy loading in case of large file sizes.
    Returns
    -------
    Dict[str, Union[Path, Union[Path, "ForecastCollection"]]
        A dictionary mapping generated keys to Paths or loaded prediction objects.
    """
    all_predictions: Dict[str, "ForecastCollection"] = {}
    files_list = _coerce_paths(prediction_files)
    dirs_list = _coerce_paths(prediction_dirs)

    if not files_list and not dirs_list:
        raise ValueError("Either prediction_files or prediction_dirs must be provided.")

    all_file_paths = _collect_files(
        files=files_list,
        dirs=dirs_list,
        required_name=PREDICTIONS_FILENAME,
        required_suffix=".joblib",  # redundancy safe
        recursive=True,
    )

    if not all_file_paths:
        logging.warning("No prediction files were found.")
        return all_predictions

    common_path = Path(os.path.commonpath(all_file_paths))
    n_files = len(all_file_paths)
    logging.info(f"Common path identified: {common_path}")

    for filepath in all_file_paths:
        key = _build_key(
            filepath,
            n_files=n_files,
            common_path=common_path,
            strip_tokens={DIR_BACKTESTS, DIR_MODELS, DIR_POSTPROCESSORS},
        )
        if load:
            print(filepath)
            all_predictions[key] = ForecastCollection.load(filepath)
            logging.info(f"Loaded prediction file: `{filepath}` as key: {key}")
        else:
            all_predictions[key] = filepath  # type: ignore[assignment]
            logging.info(f"Found prediction file: `{filepath}` as key: {key}")

    if all_predictions:
        formatted_keys = "\n      - " + "\n      - ".join(all_predictions.keys())
        logging.info("Finished loading predictions. \n \n  Loaded keys:%s", formatted_keys)
    else:
        logging.warning("No prediction files were loaded.")

    return all_predictions


def load_execution_times(
    execution_dirs: Union[List[Union[Path, str]], Path, str, None] = None,
    execution_files: Union[List[Union[Path, str]], Path, str, None] = None,
    round_ndigits: Optional[int] = 2,
    fillna_value: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load execution_time metadata from one or more backtest_config.json files and return a DataFrame.

    Parameters
    ----------
    execution_dirs : str | Path | list[str|Path], optional
        One or more directories to search recursively for backtest_config.json files.
    execution_files : str | Path | list[str|Path], optional
        Specific backtest_config.json files to load directly.
    round_ndigits : int, optional
        If provided, round numeric columns to this many decimals.
    fillna_value : float, optional
        If provided, fill NaNs with this value (e.g., 0.0). Leave as None to keep NaNs.

    Returns
    -------
    pd.DataFrame
        MultiIndexed DataFrame (level 0 = key matching load_predictions, level 1 = predictor/postprocessor name).
    """
    files_list = _coerce_paths(execution_files)
    dirs_list = _coerce_paths(execution_dirs)

    if not files_list and not dirs_list:
        raise ValueError("Either execution_files or execution_dirs must be provided.")

    all_file_paths = _collect_files(
        files=files_list,
        dirs=dirs_list,
        required_name=EVAL_CONFIG_FILENAME,
        recursive=True,
    )

    if not all_file_paths:
        logging.warning("No execution_time config files were found.")
        return pd.DataFrame()

    common_path = Path(os.path.commonpath(all_file_paths))
    n_files = len(all_file_paths)

    records = []
    for filepath in all_file_paths:
        key = _build_key(
            filepath,
            n_files=n_files,
            common_path=common_path,
            strip_tokens={DIR_BACKTESTS, DIR_MODELS, DIR_POSTPROCESSORS},
        )

        try:
            with open(filepath, "r") as f:
                cfg: dict = json.load(f)
        except Exception as err:
            logging.error(f"Failed to read JSON `{filepath}`: {err}")
            continue

        exec_dict: dict = cfg.get("execution_time")

        if exec_dict is None:
            logging.warning(f"`execution_time` missing in `{filepath}`; skipping.")
            continue

        # Optionally. could also be done relative / update total_train_time
        if "execution_time_predictor" in exec_dict:
            predictor_results = exec_dict["execution_time_predictor"]
            exec_dict["predictor_train_time"] = predictor_results["predictor_train_time"]
            exec_dict["predictor_inference_time"] = predictor_results["predictor_inference_time"]
            # if predictor_results["predictor_train_time"] is not None:
            #    exec_dict["total_train_time"] += predictor_results["predictor_train_time"]

        exec_dict.pop("execution_time_predictor", None)
        exec_dict.pop("postprocessor_name", None)
        exec_dict.pop("predictor_name", None)
        # exec_dict = _prepare_execution_time(exec_dict)
        if not exec_dict:
            logging.info(f"No execution_time rows extracted for `{filepath}`; skipping.")
            continue

        exec_dict["__key__"] = key
        records.append(exec_dict)

        logging.info(f"Loaded execution_time from `{filepath}` as key: {key}")

    if not records:
        logging.warning("No execution_time data extracted from files.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    df = df.set_index(["__key__"]).sort_index()
    df.index = df.index.set_names(["name"])

    # Optional fill/round
    if fillna_value is not None:
        df = df.fillna(fillna_value)
    if round_ndigits is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].round(round_ndigits)

    return df


def plot_multiple_forecasts(
    forecasts_dict: Dict[int, "TimeSeriesForecast"],
    start: Optional[Union[int, pd.Timestamp]] = None,
    context_length: int = 100,
    max_historical_context: int = 1000,
    max_forecast_steps: Optional[int] = None,
    complete_data: Optional["TimeSeriesDataFrame"] = None,  # kept for API compatibility
    figsize: Optional[Tuple[int, int]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    max_cols: int = 3,
    sharex: bool = False,
    sharey: bool = False,
    show_xy_labels: bool = True,
    show_legend: bool = True,
    title_prefix: str = "",
    legend_position: str = "below",  # "below" or "right"
    font_sizes: Optional[Dict[str, int]] = None,
    tight_margins: bool = False,
    margin_padding: float = 0.05,
    show_history_overview: bool = False,
    history_overview_height: float = 0.3,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Plot multiple TimeSeriesForecast objects in a grid layout with subplots.

    Parameters
    ----------
    forecasts_dict : Dict[int, TimeSeriesForecast]
        Dictionary mapping item_id to TimeSeriesForecast objects.

    start : Optional[Union[int, pd.Timestamp]]
        - If int: Index into the time series to start the forecast from.
        - If pd.Timestamp: Timestamp to start the forecast from. Must exist in the time series index.
        - If None: Defaults to the last available index for each forecast.

    context_length : int
        Number of historical data points to include in the plot before the forecast start.

    max_historical_context : int
        Maximum number of historical data points to show in the history overview subplot.
        This controls how far back the overview looks from the forecast start point.
        Default: 1000. Use larger values for longer historical context.

    max_forecast_steps : Optional[int]
        Maximum number of forecast steps to plot. If None, plots all available forecast steps.
        If specified, limits the number of future time steps displayed in the forecast plots.
        Useful for focusing on short-term predictions or reducing visual clutter.

    figsize : Optional[Tuple[int, int]]
        Figure size as (width, height). If provided, overrides width and height parameters.

    width : Optional[int]
        Figure width in inches. Used only if figsize is None.

    height : Optional[int]
        Figure height in inches. Used only if figsize is None.

    max_cols : int
        Maximum number of columns in the grid layout.

    sharex : bool
        Whether to share x-axis across subplots.

    sharey : bool
        Whether to share y-axis across subplots.

    show_xy_labels : bool
        Whether to show x- and y-labels on subplots.

    show_legend : bool
        Whether to show legends on subplots.

    title_prefix : str
        Prefix for subplot titles.

    legend_position : str
        Position of the legend: "below" (below the plots) or "right" (to the right of the plots).

    font_sizes : Optional[Dict[str, int]]
        Dictionary controlling font sizes for various text elements. If None, default sizes are used.
        Available keys: 'title', 'subtitle', 'xlabel', 'ylabel', 'legend', 'tick_labels', 'grid_labels'.
        Example: {'title': 16, 'subtitle': 14, 'xlabel': 12, 'ylabel': 12, 'legend': 10, 'tick_labels': 10}.

    tight_margins : bool
        Whether to use tight margins around the data. If True, reduces padding between plot borders and data.

    margin_padding : float
        Padding factor for margins when tight_margins=True. Smaller values (0.01-0.05) create tighter plots,
        larger values (0.1-0.2) create more spacious plots. Default: 0.05.

    show_history_overview : bool
        Whether to add a full-length historical overview subplot at the top showing the complete time series.
        This provides context for the zoomed-in forecast plots below.

    history_overview_height : float
        Height ratio for the history overview subplot relative to the total figure height.
        Range: 0.1 to 0.5. Default: 0.3 (30% of total height).

    save_path : Optional[str]
        If provided, save the plot to this path.

    dpi : int
        DPI for saving the plot.

    Returns
    -------
    None
        Displays a matplotlib figure with subplots showing forecasts for each TimeSeriesForecast object.
    """

    if isinstance(start, int):
        start: pd.Timestamp = next(iter(forecasts_dict.values())).data.index.get_level_values(TIMESTAMP)[start]

    if not forecasts_dict:
        raise ValueError("forecasts_dict cannot be empty")

    # ---------- small helpers (reduce repetition) ----------

    def merge_font_sizes(user_fs: Optional[Dict[str, int]]) -> Dict[str, int]:
        base = {"title": 16, "subtitle": 14, "xlabel": 12, "ylabel": 12, "legend": 10, "tick_labels": 10, "grid_labels": 10}
        if user_fs:
            base.update(user_fs)
        return base

    def grid_dims(n: int, max_cols_: int) -> Tuple[int, int]:
        cols = min(max_cols_, n)
        rows = (n + cols - 1) // cols
        return rows, cols

    def resolve_figsize(n_rows: int, n_cols: int, add_overview: bool) -> Tuple[int, int]:
        """Return (fig_w, fig_h) applying width/height/figsize rules once."""
        if figsize is not None:
            w, h = figsize
        else:
            w = width if width is not None else 6 * n_cols
            h = height if height is not None else 4 * n_rows
        if add_overview:
            h = h * (1 + history_overview_height)
        return (w, h)

    def determine_start_index(start_, data_length: int, timestamps=None) -> int:
        if isinstance(start_, pd.Timestamp):
            if timestamps is not None and start_ in timestamps:
                return timestamps.get_loc(start_)
            return data_length - 1
        if isinstance(start_, int):
            if start_ < 0:
                return start_ % data_length
            return min(start_, data_length - 1)
        return data_length - 1

    def corrected_start(timestamps: pd.Index, forecast_mask) -> Tuple[pd.Timestamp, int]:
        """Return (corrected_start_date, corrected_start_idx) using global `start`."""
        start_idx = determine_start_index(start, len(timestamps), timestamps)
        start_date = timestamps[start_idx]
        forecasted_ts = timestamps[forecast_mask]
        start_idx_preds = forecasted_ts.get_indexer([start_date], method="backfill")[0]
        corr_start_date = forecasted_ts[start_idx_preds]
        corr_start_idx = timestamps.get_indexer([corr_start_date])[0]
        return corr_start_date, corr_start_idx

    def format_axes(ax: plt.Axes, fs: Dict[str, int], show_labels: bool, tight: bool):
        if show_labels:
            ax.set_xlabel("Date", fontsize=fs["xlabel"])
            ax.set_ylabel("Value", fontsize=fs["ylabel"])
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=fs["tick_labels"])
        plt.setp(ax.yaxis.get_majorticklabels(), fontsize=fs["tick_labels"])
        if tight:
            ax.margins(y=margin_padding, x=0)

    def add_prediction_intervals(
        ax: plt.Axes,
        selected_predictions: pd.DataFrame,
        intervals: List[Tuple[float, float]],
        base_color: str = "orange",
        lw_outer: float = 0.1,
        lw_inner: float = 0.1,
    ) -> None:
        n = len(intervals)
        for i, (ql, qu) in enumerate(intervals):
            if ql not in selected_predictions.columns or qu not in selected_predictions.columns:
                continue
            alpha = 0.2 + 0.15 * (n - 1 - i)
            ax.fill_between(
                selected_predictions.index,
                selected_predictions[ql],
                selected_predictions[qu],
                color=base_color,
                alpha=alpha,
                label=f"{int(qu*100)}–{int(ql*100)} % interval",
            )
            lw = lw_outer + (lw_inner - lw_outer) * (n - 1 - i) / max(n - 1, 1)
            ax.plot(selected_predictions.index, selected_predictions[ql], color=base_color, alpha=alpha + 0.1, linewidth=lw)
            ax.plot(selected_predictions.index, selected_predictions[qu], color=base_color, alpha=alpha + 0.1, linewidth=lw)

    def build_selected_predictions(forecast, corr_start_date: pd.Timestamp, start_idx_preds: int, max_lt: int) -> pd.DataFrame:
        preds = torch.stack([hf.predictions for hf in forecast.lead_time_forecasts.values()], dim=1)
        freq_offset = pd.tseries.frequencies.to_offset(forecast.freq)
        pred_dates = [corr_start_date + freq_offset * lt for lt in range(1, max_lt + 1)]
        return pd.DataFrame(
            data=preds[start_idx_preds, :max_lt, :].numpy(),
            columns=forecast.quantiles,
            index=pred_dates,
        )

    def extend_historic_data(forecast, complete_data: Optional["TimeSeriesDataFrame"] = None) -> Tuple[pd.DataFrame, pd.Index, np.ndarray]:
        """Get the appropriate data for individual subplots, using complete_data if available."""
        if complete_data is not None:
            item_id = forecast.data.item_ids[0]
            id_complete_data = complete_data.loc[[item_id]]
            full_data = id_complete_data.reset_index(level=0, drop=True)
            ts = full_data.index
            forecast_mask = forecast.forecast_mask
            missing_vals = len(full_data) - len(forecast_mask)
            mask_pad = np.full(missing_vals, False)
            forecast_mask = np.concatenate((mask_pad, forecast_mask), axis=0)
        else:
            full_data = forecast.data.reset_index(level=0, drop=True)
            ts = full_data.index
            forecast_mask = forecast.forecast_mask

        return full_data, ts, forecast_mask

    def history_overview_subplot(fig_, n_rows: int, n_cols: int, first_forecast, fs: Dict[str, int]) -> Tuple[plt.Axes, pd.Index, pd.Timestamp, int]:

        if complete_data is not None:
            full_data, ts, forecast_mask = extend_historic_data(
                first_forecast,
                complete_data,
            )

        else:
            full_data = first_forecast.data.reset_index(level=0, drop=True)
            ts = full_data.index
            forecast_mask = first_forecast.forecast_mask

        corr_start_date, corr_start_idx = corrected_start(ts, forecast_mask)

        history_start_idx = max(0, corr_start_idx - max_historical_context)
        history_data = full_data.iloc[history_start_idx : corr_start_idx + 1]

        ax_hist = plt.subplot2grid((n_rows, n_cols), (0, 0), colspan=n_cols, fig=fig_)

        ax_hist.plot(history_data.index, history_data["target"], color="black", linestyle="--", linewidth=1)

        # Highlight area if start specified
        if start is not None:
            forecast_start = full_data.index[corr_start_idx]
            forecast_end = full_data.index[min(corr_start_idx + context_length, len(full_data) - 1)]
            ax_hist.axvspan(forecast_start, forecast_end, alpha=0.1, color="gray", label="_nolegend_")
            ax_hist.axvline(forecast_start, color="black", linestyle=":", alpha=0.7, label="Prediction start")

        ax_hist.set_title("Historical Overview", fontsize=fs["subtitle"])
        # ax_hist.set_xlabel("Time", fontsize=fs["xlabel"])
        # ax_hist.set_ylabel("Value", fontsize=fs["ylabel"])
        ax_hist.grid(True, alpha=0.3)
        if tight_margins:
            ax_hist.margins(y=margin_padding, x=0)

        return ax_hist, ts, corr_start_date, corr_start_idx

    def gather_legend(fig_, axes_for_legend: List[plt.Axes], fs: Dict[str, int]):
        handles, labels = [], []
        for ax_ in axes_for_legend:
            h, l = ax_.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in labels:
                    handles.append(hh)
                    labels.append(ll)
        if not handles:
            return
        if legend_position == "below":
            fig_.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=min(4, len(handles)), fontsize=fs["legend"], frameon=True)
        elif legend_position == "right":
            fig_.legend(handles, labels, loc="center left", bbox_to_anchor=(1.05, 0.5), ncol=1, fontsize=fs["legend"], frameon=True)

    # ---------- compute layout / sizes once ----------

    font_sizes = merge_font_sizes(font_sizes)
    num_forecasts = len(forecasts_dict)
    n_rows, n_cols = grid_dims(num_forecasts, max_cols)

    if show_history_overview:
        n_rows += 1

    fig_w, fig_h = resolve_figsize(n_rows, n_cols, show_history_overview)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharey=sharey, sharex=sharex, squeeze=False, constrained_layout=True)

    axes_flat = axes.flatten()

    # ---------- optional history overview ----------

    if show_history_overview:
        first_fc = next(iter(forecasts_dict.values()))
        hist_ax, first_ts, corr_start_date_first, corr_start_idx_first = history_overview_subplot(fig, n_rows, n_cols, first_fc, font_sizes)
        forecast_axes = axes[1:].flatten() if n_rows > 1 else axes_flat

        for i in range(0, n_cols):
            axes.flatten()[i].set_visible(False)

    else:
        hist_ax = None
        forecast_axes = axes_flat

    # ---------- plot individual forecasts ----------

    for idx, (item_id, forecast) in enumerate(forecasts_dict.items()):
        if idx >= len(forecast_axes):
            break
        ax = forecast_axes[idx]

        # try:
        # Get the appropriate data (using complete_data if available)
        full_data, timestamps, forecast_mask = extend_historic_data(forecast, complete_data)
        corr_start_date, corr_start_idx = corrected_start(timestamps, forecast_mask)

        # context (past) - now can use extended context if complete_data is available
        historic_start_idx = max(0, corr_start_idx - context_length)
        past = full_data.iloc[historic_start_idx:corr_start_idx]

        # horizon (future)
        max_lt = max(forecast.get_lead_times())
        if max_forecast_steps is not None:
            max_lt = min(max_lt, max_forecast_steps)
        future = full_data.iloc[corr_start_idx : corr_start_idx + max_lt]

        # predictions aligned to dates
        forecasted_ts = timestamps[forecast_mask]
        start_idx_preds = forecasted_ts.get_indexer([corr_start_date])[0]
        selected_predictions = build_selected_predictions(forecast, corr_start_date, start_idx_preds, max_lt)

        # plots
        ax.plot(past.index, past["target"], label="Past", color="black", linestyle="--", linewidth=1)
        ax.plot(future.index, future["target"], label="Future (true)", color="blue", linewidth=1.5)

        if 0.5 in selected_predictions.columns:
            ax.plot(selected_predictions.index, selected_predictions[0.5].values, label="Prediction (median)", color="red", linewidth=1.5)

        ax.axvline(corr_start_date, color="black", linestyle=":", label="Prediction start")
        forecast_end = selected_predictions.index[-1]
        ax.axvspan(corr_start_date, forecast_end, alpha=0.1, color="gray", label="_nolegend_")

        add_prediction_intervals(
            ax,
            selected_predictions,
            intervals=[(0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9)],
            base_color="orange",
        )

        ax.set_title(f"{item_id}", fontsize=font_sizes["subtitle"])
        format_axes(ax, font_sizes, show_xy_labels, tight_margins)

    # hide unused subplots
    for idx in range(num_forecasts, len(forecast_axes)):
        forecast_axes[idx].set_visible(False)

    # unified legend
    if show_legend:
        legend_axes = [hist_ax] if (hist_ax is not None) else []
        # add the first visible forecast axis (enough to capture handles)
        for ax in forecast_axes[:num_forecasts]:
            if ax.get_visible():
                legend_axes.append(ax)
                break
        gather_legend(fig, legend_axes, font_sizes)

    # title
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=font_sizes["title"], y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()
