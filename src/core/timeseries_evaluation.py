"""This Module provides utilities for probabilistic time series forecasting, including data structures, evaluation, and visualization tools."""

from typing import List, Optional, Dict, Tuple, Union
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import scoringrules as sr
from scipy.interpolate import interp1d
from pydantic import BaseModel, Field, field_validator
from autogluon.timeseries import TimeSeriesDataFrame
import math
import matplotlib.dates as mdates
from pathlib import Path
import joblib
import logging
import os

DIR_BACKTESTS = "backtest"
DIR_MODELS = "models"
DIR_POSTPROCESSORS = "postprocessors"
ITEMID = "item_id"
TIMESTAMP = "timestamp"
TARGET = "target"

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
    quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    freq: Union[pd.Timedelta, pd.DateOffset]
    data: Union[TimeSeriesDataFrame, TabularDataFrame]

    class Config:
        arbitrary_types_allowed = True

    @field_validator("predictions")
    @classmethod
    def check_predictions(cls, pred: torch.Tensor) -> torch.Tensor:
        pred = pred.contiguous() if not pred.is_contiguous() else pred
        pred = pred.sort(dim=1)[0]  # avoid quantile crossing. TODO: potentially shouldn't be done silently
        return pred

    def to_dataframe(self, item_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Converts the prediction data into a Pandas DataFrame and merges it with the actual target values.

        Args:
            item_ids (Optional[List[int]]): dataframe of the item ids to retrieve.

        Returns:
            pd.DataFrame: DataFrame with timestamps, predicted quantiles, and corresponding target values.
        """

        result = pd.DataFrame(self.predictions, index=self.data.index, columns=self.quantiles)

        result["prediction_date"] = result.index.get_level_values("timestamp") + self.freq * self.lead_time

        # Reset index to turn MultiIndex into columns
        result_reset = result.reset_index()
        data_reset = self.data.reset_index()

        # add the target information.
        if isinstance(self.data, TimeSeriesDataFrame):
            merged = result_reset.merge(
                data_reset, left_on=["item_id", "prediction_date"], right_on=["item_id", "timestamp"], how="left", suffixes=["", "_remove"]  # From result  # From preds.data
            )

        # add the target information. For tabular data frame, the target is already aligned
        elif isinstance(self.data, TabularDataFrame):
            merged = result_reset.merge(
                data_reset, left_on=["item_id", "timestamp"], right_on=["item_id", "timestamp"], how="left", suffixes=["", "_remove"]  # From result  # From preds.data
            )

        # remove unused columns
        merged = merged.drop(columns=[col for col in merged.columns if "_remove" in str(col) or "feature" in str(col)], errors="ignore")

        # restore original multi index
        merged.set_index(result.index.names, inplace=True)

        # get only the specified item id
        if item_ids is not None:
            merged = merged.loc[item_ids].copy()

        return merged

    def get_crps(self, item_ids: Optional[List[int]] = None, mean_time: bool = True) -> np.ndarray:
        """
        Computes the Continuous Ranked Probability Score (CRPS) for the forecast.

        Args:
            data (pd.DataFrame): DataFrame containing actual target values.

        Returns:
            float: The mean CRPS score across all samples.
        """
        data = self.to_dataframe(item_ids=item_ids)
        crps = sr.crps_quantile(data["target"].to_numpy(), data[self.quantiles].to_numpy(), self.quantiles)

        if mean_time:
            return np.array([crps[~np.isnan(crps)].mean()])
        else:
            return crps

    def get_quantile_score(self, item_ids: Optional[List[int]] = None, mean_time: bool = True) -> Union[pd.DataFrame, pd.Series]:
        """
        Computes the average quantile score (pinball loss) for the forecast.

        Args:
            data (pd.DataFrame): DataFrame containing actual target values.

        Returns:
            pd.Series: A Series with the mean pinball loss for each quantile.
        """
        data = self.to_dataframe(item_ids=item_ids)
        quantile_scores = np.column_stack([sr.quantile_score(data["target"].to_numpy(), data[q].to_numpy(), q) for q in self.quantiles])

        quantile_scores = pd.DataFrame(quantile_scores, columns=self.quantiles, index=data.index)
        if mean_time:
            # quantile_scores = pd.DataFrame(quantile_scores.mean(), columns=["Mean QS"])
            # quantile_scores.index.name = "quantile"
            return quantile_scores.mean()
        else:
            return quantile_scores

    def get_pit_values(self, item_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        Computes the Probability Integral Transform (PIT) values for calibration analysis.

        PIT = F(y), where F(y) is the interp_func, which is the CDF approximated based on the quantiles.

        Parameters:
            data (pd.DataFrame): DataFrame containing actual target values.

        Returns:
            np.ndarray: Array of PIT values, where PIT values should follow a uniform [0,1] distribution.
        """
        targets = self.to_dataframe(item_ids=item_ids).dropna()["target"].to_numpy()  # Shape [num_samples]

        # Compute PIT values for each target
        pit_values = []
        for i in range(len(targets)):
            interp_func = interp1d(self.predictions[i], self.quantiles, bounds_error=False, fill_value=(0, 1))
            pit_values.append(interp_func(targets[i]))

        return np.array(pit_values)

    def get_pit_histogram(self, item_ids: Optional[List[int]] = None) -> None:
        """
        Plots a histogram of Probability Integral Transform (PIT) values to assess forecast calibration.

        Parameters:
            data (pd.DataFrame): DataFrame containing actual target values.

        Returns:
            None: Displays the histogram plot.
        """
        pit_values = self.get_pit_values(item_ids=item_ids)
        bins = len(self.quantiles)

        plt.hist(pit_values, bins=bins, range=(0, 1), density=False, alpha=0.7, edgecolor="black")
        plt.axhline(len(pit_values) / bins, color="red", linestyle="dashed", label="Uniform(0,1) reference")
        plt.xlabel("PIT Values")
        plt.ylabel("Frequency of Occurrences")
        plt.title("PIT Histogram")
        plt.legend()
        plt.show()

    def get_empirical_coverage_rates(self, item_ids: Optional[List[int]] = None) -> Dict[float, float]:

        results = self.to_dataframe(item_ids=item_ids).dropna()
        empirical_coverage_rates = {q: (results[q] >= results["target"]).mean() for q in self.quantiles}

        return empirical_coverage_rates

    def get_reliability_diagram(self, item_ids: Optional[List[int]] = None) -> None:

        empirical_coverage_rates = self.get_empirical_coverage_rates(item_ids=item_ids)

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

    def get_random_plot(self, item_id: int = 0, q_lower: float = 0.1, q_upper: float = 0.9, ts_length: int = 100) -> None:
        """Randomly plot data"""

        subset = self.to_dataframe(item_ids=[item_id])
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
        plt.title(f"Prediction Intervals – item_id: {item_id} and lead time: {self.lead_time}", fontsize=16)
        plt.legend(fontsize=8)

        # Improve x-axis tick formatting
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically space ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # Format as 'YYYY-MM-DD HH:MM'

        plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Optional: Light grid

        plt.show()


class TimeSeriesForecast(BaseModel):
    item_id: int
    lead_time_forecasts: Dict[int, HorizonForecast]  # {lead_time: HorizonForecast}

    def get_lead_times(self) -> List[int]:
        return list(self.lead_time_forecasts.keys())

    def get_quantiles(self) -> List[float]:
        return sorted({item.quantiles for item in self.lead_time_forecasts.values()})

    def get_lead_time_forecast(self, lead_time: int) -> HorizonForecast:
        return self.lead_time_forecasts[lead_time]

    def get_all_lead_time_forecast(self) -> List[HorizonForecast]:
        return self.lead_time_forecasts

    def add_lead_time_forecast(self, lead_time: int, prediction: HorizonForecast) -> None:
        self.lead_time_forecasts[lead_time] = prediction


class ForecastCollection(BaseModel):
    items: Dict[int, TimeSeriesForecast]  # item_id -> TimeSeriesForecast

    class Config:
        arbitrary_types_allowed = True

    def get_item_ids(self) -> List[int]:
        return list(self.items.keys())

    def get_lead_times(self, item_id: Optional[int] = None) -> List[int]:
        if item_id is not None:
            return self.items[item_id].get_lead_times()
        return sorted({lt for item in self.items.values() for lt in item.get_lead_times()})

    def get_time_series_forecast(self, item_id: int) -> TimeSeriesForecast:
        return self.items[item_id]

    def get_all_time_series_forecast(self) -> List[TimeSeriesForecast]:
        return self.items

    def add_time_series_forecast(self, forecast: TimeSeriesForecast) -> None:
        self.items[forecast.item_id] = forecast

    def get_crps(
        self,
        item_ids: Optional[List[int]] = None,
        lead_times: Optional[List[int]] = None,
        mean_time: bool = True,
        mean_item_ids: bool = False,
        mean_lead_times: bool = False,
    ) -> pd.DataFrame:
        item_ids = item_ids or self.get_item_ids()
        lead_times = lead_times or self.get_lead_times()

        all_scores = []
        for item_id in item_ids:
            scores = []
            item = self.get_time_series_forecast(item_id)
            for lt in lead_times:
                if lt in item.lead_time_forecasts:
                    crps = item.get_lead_time_forecast(lt).get_crps(item_ids=[item_id], mean_time=mean_time)
                    scores.append(crps)

            scores = np.vstack(scores).T
            idx = [item_id] if mean_time else item.get_lead_time_forecast(lt).to_dataframe().index  # TODO: make this nicer
            all_scores.append(pd.DataFrame(scores, index=idx, columns=lead_times))

        crps_scores = pd.concat(all_scores)

        if mean_lead_times:
            crps_scores = pd.DataFrame(crps_scores.mean(axis=1), columns=["Mean CRPS"])
        if mean_item_ids:
            if mean_time:
                crps_scores = pd.DataFrame(crps_scores.mean(axis=0), columns=["Mean CRPS"]).T
            else:
                crps_scores = crps_scores.groupby(level=TIMESTAMP).mean()
        if mean_time:
            crps_scores.index.name = ITEMID

        # if add_mean:
        #     if not mean_time:
        #         crps_scores.loc["Mean CRPS", :] = crps_scores.mean(axis=0)
        #     if not mean_lead_times:
        #         crps_scores.loc[:, "Mean CRPS"] = crps_scores.mean(axis=1)

        return crps_scores.round(3)

    def get_empirical_coverage_rates(self, item_ids: Optional[List[int]] = None, lead_times: Optional[List[int]] = None, mean_lead_times: bool = False) -> pd.DataFrame:
        item_ids = item_ids or self.get_item_ids()
        lead_times = lead_times or self.get_lead_times()

        rates = {lt: [] for lt in lead_times}

        for item_id in item_ids:
            item = self.get_time_series_forecast(item_id)
            for lt in lead_times:
                if lt in item.lead_time_forecasts:
                    val = item.get_lead_time_forecast(lt).get_empirical_coverage_rates()
                    rates[lt].append(pd.Series(val))

        coverage_df = pd.DataFrame({lt: pd.concat(rates[lt], axis=1).mean(axis=1) for lt in lead_times if rates[lt]})

        if mean_lead_times:
            coverage_df = pd.DataFrame(coverage_df.mean(axis=1), columns=["Empirical coverage rates averaged over all lead times"])
        else:
            coverage_df.loc[:, "Empirical coverage rates averaged over all lead times"] = coverage_df.mean(axis=1)

        coverage_df.index.name = "quantile"
        return coverage_df.round(3)

    def get_quantile_scores(self, item_ids: Optional[List[int]] = None, lead_times: Optional[List[int]] = None, mean_lead_times: bool = False) -> pd.DataFrame:
        item_ids = item_ids or self.get_item_ids()
        lead_times = lead_times or self.get_lead_times()

        scores = {}
        for lt in lead_times:
            values = []
            for item_id in item_ids:
                val = self.get_time_series_forecast(item_id).get_lead_time_forecast(lt).get_quantile_score(item_ids=[item_id], mean_time=True)
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
        return df.round(3)

    def get_pit_values(self, lead_times: Optional[List[int]] = None, item_ids: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        item_ids = item_ids or self.get_item_ids()
        lead_times = lead_times or self.get_lead_times()
        result = {}
        for lt in lead_times:
            values = []
            for item_id in item_ids:
                item = self.get_time_series_forecast(item_id)
                if lt in item.lead_time_forecasts:
                    values.append(item.get_lead_time_forecast(lt).get_pit_values(item_ids=[item_id]))
            if values:
                result[lt] = np.concatenate(values)
        return result

    def get_pit_histogram(self, lead_times: Optional[List[int]] = None, overlay: bool = False, item_ids: Optional[List[int]] = None) -> None:
        lead_times = lead_times or self.get_lead_times()
        pit_data = self.get_pit_values(lead_times=lead_times, item_ids=item_ids)

        if item_ids:
            first_item = item_ids[0]
        else:
            first_item = self.items.keys()[0]
        bins = self.get_time_series_forecast(first_item).get_quantiles()
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

    def get_reliability_diagram(self, lead_times: Optional[List[int]] = None, overlay: bool = False, item_ids: Optional[List[int]] = None) -> None:
        lead_times = lead_times or self.get_lead_times()

        if overlay:
            plt.figure(figsize=(8, 8))
            for lt in lead_times:
                values = []
                for item_id in self.get_item_ids():
                    if item_ids and item_id not in item_ids:
                        continue
                    item = self.get_time_series_forecast(item_id)
                    if lt in item.lead_time_forecasts:
                        val = item.get_lead_time_forecast(lt).get_empirical_coverage_rates(item_ids=[item_id])
                        values.append(pd.Series(val))
                if values:
                    emp = pd.concat(values, axis=1).mean(axis=1)
                    plt.plot(emp.index, emp.values, "o-", label=f"Lead time {lt}")
            plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
            plt.xlabel("Nominal Quantile Level")
            plt.ylabel("Empirical Coverage")
            plt.title("Reliability Diagram for Quantile Forecasts")
            plt.legend()
            plt.grid(True)
            plt.show()

        else:
            num_plots = len(lead_times)
            cols = math.ceil(np.sqrt(num_plots))
            rows = (num_plots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
            axes = axes.flatten() if num_plots > 1 else [axes]
            for ax, lt in zip(axes, lead_times):
                values = []
                for item_id in self.get_item_ids():
                    if item_ids and item_id not in item_ids:
                        continue
                    item = self.get_time_series_forecast(item_id)
                    if lt in item.lead_time_forecasts:
                        val = item.get_lead_time_forecast(lt).get_empirical_coverage_rates(item_ids=[item_id])
                        values.append(pd.Series(val))
                if values:
                    emp = pd.concat(values, axis=1).mean(axis=1)
                    ax.plot(emp.index, emp.values, "o-", label=f"Lead time {lt}")
                    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
                    ax.set_xlabel("Nominal Quantile Level")
                    ax.set_ylabel("Empirical Coverage")
                    ax.set_xticks(emp.index)
                    ax.set_title("Reliability Diagram")
                    ax.legend()
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
        return obj


def get_quantile_scores(
    predictions: Dict[str, ForecastCollection],
    lead_times: Optional[List[int]] = None,
    item_ids: Optional[List[int]] = None,
    reference_predictions: Optional[str] = None,
) -> pd.DataFrame:
    """
    Computes quantile scores for different prediction sources,
    averaged across the specified lead times, and optionally normalizes them
    using a reference prediction.

    Parameters
    -----------
    predictions : Dict[str, ForecastCollection]
        Dictionary of prediction objects, where each value provides a `get_crps` method
        that returns a DataFrame with CRPS values indexed by ['item_id', 'timestamp'].
    lead_times : Optional[List[int]], default=None
        List of lead times to filter CRPS scores. If None, all lead times are used.
    item_ids : Optional[List[int]], default=None
        List of item IDs to include in the CRPS computation. If None, all item IDs are used.
    reference_predictions : Optional[str], default=None
        Key of a prediction set to be used as a reference for normalization.
        If provided, all CRPS values will be divided by the CRPS values from this prediction.

    Returns
    --------
    pd.DataFrame
        A DataFrame where columns represent different prediction sources and rows represent
        quantile score values averaged across the selected lead times. If normalization is applied,
        the scores are expressed as a ratio to the reference prediction.
    """

    scores = pd.concat([pred.get_quantile_scores(lead_times=lead_times, mean_lead_times=True, item_ids=item_ids) for pred in predictions.values()], axis=1)
    scores.columns = [key for key in predictions.keys()]

    if reference_predictions:
        scores = scores.apply(lambda x: x / x[reference_predictions], axis=1)
    return scores.round(3)


def get_empirical_coverage_rates(predictions: Dict[str, ForecastCollection], lead_times: Optional[List[int]] = None, item_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """Computes empirical coverage rates for different prediction sources,
    averaged across the specified lead times.

    Parameters
    -----------
    predictions : Dict[str, ForecastCollection]
        Dictionary of prediction objects, where each value provides a `get_crps` method
        that returns a DataFrame with CRPS values indexed by ['item_id', 'timestamp'].
    lead_times : Optional[List[int]], default=None
        List of lead times to filter CRPS scores. If None, all lead times are used.
    item_ids : Optional[List[int]], default=None
        List of item IDs to include in the CRPS computation. If None, all item IDs are used.

    Returns
    --------
    pd.DataFrame
        A DataFrame where each column corresponds to a prediction source,
        and values represent the empirical coverage rates averaged over the specified lead times.
    """

    scores = pd.concat([pred.get_empirical_coverage_rates(lead_times=lead_times, mean_lead_times=True, item_ids=item_ids) for pred in predictions.values()], axis=1)
    scores.columns = [key for key in predictions.keys()]

    return scores.round(3)


def get_crps_scores(
    predictions: Dict[str, ForecastCollection],
    lead_times: Optional[List[int]] = None,
    mean_lead_times: bool = False,
    item_ids: Optional[List[int]] = None,
    reference_predictions: Optional[str] = None,
) -> pd.DataFrame:
    """Computes and returns CRPS (Continuous Ranked Probability Score) values
    averaged across specified lead times and optionally normalized by a reference prediction.

    Parameters
    -----------
    predictions : Dict[str, ForecastCollection]
        Dictionary of prediction objects, where each value provides a `get_crps` method
        that returns a DataFrame with CRPS values indexed by ['item_id', 'timestamp'].
    lead_times : Optional[List[int]], default=None
        List of lead times to filter CRPS scores. If None, all lead times are used.
    item_ids : Optional[List[int]], default=None
        List of item IDs to include in the CRPS computation. If None, all item IDs are used.
    reference_predictions : Optional[str], default=None
        Key of a prediction set to be used as a reference for normalization.
        If provided, all CRPS values will be divided by the CRPS values from this prediction.

    Returns
    --------
    pd.DataFrame
        A DataFrame with CRPS values (or normalized CRPS) for each prediction source.
        Rows represent lead times (or a single row if mean_lead_times=True), and columns represent prediction sources.
    """

    scores = pd.concat(
        [pred.get_crps(lead_times=lead_times, mean_lead_times=mean_lead_times, mean_time=True, mean_item_ids=True, item_ids=item_ids) for pred in predictions.values()], axis=0
    ).T
    scores.columns = [key for key in predictions.keys()]
    scores.index.name = "lead times"
    if reference_predictions:
        scores = scores.apply(lambda x: x / x[reference_predictions], axis=1)

    scores.loc["Mean CRPS", :] = scores.mean(axis=0)
    return scores.round(3)


def plot_crps(
    predictions: Dict[str, ForecastCollection],
    selected_keys: Optional[List] = None,
    lead_times: Optional[List[int]] = None,
    item_ids: Optional[List[int]] = None,
    rolling_window_eval: Optional[int] = None,
    reference_predictions: Optional[str] = None,
) -> None:
    """Plots the mean CRPS (Continuous Ranked Probability Score) over time for different prediction sources.

    Parameters
    -----------
    predictions : Dict[str, ForecastCollection]
        Dictionary of prediction objects, where each value provides a `get_crps` method
        that returns a DataFrame with CRPS values indexed by ['item_id', 'timestamp'].
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

    # Compute CRPS DataFrame
    df = pd.concat([pred.get_crps(lead_times=lead_times, mean_lead_times=True, mean_time=False, item_ids=item_ids) for pred in predictions.values()], axis=1)

    df.columns = list(predictions.keys())

    if reference_predictions:
        df = df.apply(lambda x: x / x[reference_predictions], axis=1)

    df = df.reset_index(level=0, drop=True).groupby("timestamp").mean()

    if rolling_window_eval:
        df = df.rolling(window=rolling_window_eval).mean()

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


def get_crps_by_period(
    predictions: Dict[str, ForecastCollection],
    date_splits: List[pd.Timestamp],
    lead_times: Optional[List[int]] = None,
    item_ids: Optional[List[int]] = None,
    reference_predictions: Optional[str] = None,
) -> pd.DataFrame:
    """Computes the mean CRPS (Continuous Ranked Probability Score) over time periods
    defined by timestamp splits for different prediction sources.

    Parameters
    -----------
    predictions : Dict[str, ForecastCollection]
        Dictionary of prediction objects, where each value provides a `get_crps` method
        that returns a DataFrame with CRPS values indexed by ['item_id', 'timestamp'].
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

    Returns
    --------
    pd.DataFrame
        A DataFrame with the mean CRPS values for each time segment (e.g., "2023-01-01_to_2023-06-01"),
        optionally normalized by a reference prediction. Each column corresponds to a prediction key.
    """

    df = pd.concat([pred.get_crps(lead_times=lead_times, mean_lead_times=True, mean_time=False, mean_item_ids=False, item_ids=item_ids) for pred in predictions.values()], axis=1)

    df.columns = [key for key in predictions.keys()]

    df = df.reset_index()

    results: dict[str, pd.Series] = {}

    date_splits = sorted(date_splits)

    first_date = df["timestamp"].min()
    last_date = df["timestamp"].max()
    date_splits.append(last_date)

    for d in date_splits:
        if not isinstance(d, pd.Timestamp):
            raise TypeError(f"All date_splits must be pd.Timestamp, got {type(d)}")

        subset_df = df[(df["timestamp"] > first_date) & (df["timestamp"] <= d)]

        # after = df[df["timestamp"] >= d]
        results[f"{first_date.strftime("%d-%m-%Y")}_to_{d.strftime("%d-%m-%Y")}"] = subset_df.drop(columns=["item_id", "timestamp"]).mean()
        first_date = d

    results = pd.DataFrame(results).T

    if reference_predictions:
        results = results.apply(lambda x: x / x[reference_predictions], axis=1)

    return results.round(3)


def load_predictions(
    prediction_dirs: Union[List[Union[Path, str]], Path, str, None] = None,
    prediction_files: Union[List[Union[Path, str]], Path, str, None] = None,
) -> Dict[str, ForecastCollection]:
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

    Returns
    -------
    Dict[str, ForecastCollection]
        A dictionary mapping generated keys to loaded prediction objects.
    """

    all_predictions = {}
    all_file_paths: List[Path] = []

    # Collect the prediction files from either the provided list or directories
    if prediction_files:
        logging.info("Loading predictions from provided files...")

        if isinstance(prediction_files, (str, Path)):
            prediction_files = [prediction_files]

        for file in prediction_files:
            file = Path(file)
            if file.is_file() and file.suffix == ".joblib":
                all_file_paths.append(file)

    elif prediction_dirs:
        logging.info("Loading predictions by searching in provided directories...")

        if isinstance(prediction_dirs, (str, Path)):
            prediction_dirs = [prediction_dirs]

        for prediction_dir in prediction_dirs:
            prediction_dir = Path(prediction_dir)
            if not prediction_dir.is_dir():
                logging.warning(f"Skipping non-directory path: `{prediction_dir}`")
                continue

            for filepath in prediction_dir.rglob("predictions.joblib"):
                all_file_paths.append(filepath)

    else:
        raise ValueError("Either prediction_files or prediction_dirs must be provided.")

    # If we found any joblib files, process them
    if all_file_paths:
        # Find the common path prefix
        common_path = Path(os.path.commonpath(all_file_paths))
        logging.info(f"Common path identified: {common_path}")

        for filepath in all_file_paths:
            # Remove the common path from the filename and the generic prediction.joblib at the end
            if len(all_file_paths) > 1:
                relevant_dirs = list(filepath.relative_to(common_path).parent.parts)
            else:
                relevant_dirs = [filepath.parent.parts[-1]]
            for d in relevant_dirs:
                if d in [DIR_BACKTESTS, DIR_MODELS, DIR_POSTPROCESSORS]:
                    relevant_dirs.remove(d)

            key = "_".join(relevant_dirs)
            all_predictions[key] = joblib.load(filepath)
            logging.info(f"Loaded prediction file: `{filepath}` as key: {key}")
    else:
        logging.warning("No prediction files were found.")

    if all_predictions:
        formatted_keys = "\n      - " + "\n      - ".join(all_predictions.keys())
        logging.info("Finished loading predictions. \n \n  Loaded keys:%s", formatted_keys)
    else:
        logging.warning("No prediction files were loaded.")

    return all_predictions
