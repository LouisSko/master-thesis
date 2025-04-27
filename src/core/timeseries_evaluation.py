"""This Module provides utilities for probabilistic time series forecasting, including data structures, evaluation, and visualization tools."""

from typing import List, Optional, Dict, Tuple, Union
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import scoringrules as sr
from scipy.interpolate import interp1d
from pydantic import BaseModel, Field
from autogluon.timeseries import TimeSeriesDataFrame
import math
import matplotlib.dates as mdates
from pathlib import Path
import joblib
import logging

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


class PredictionLeadTime(BaseModel):
    """
    A class for handling probabilistic forecasts at multiple quantiles.

    Attributes:
        lead_time (int): Forecast lead time in hours.
        item_id (int): Identifier for the item being forecasted.
        timestamps (List[pd.Timestamp]): List of timestamps corresponding to the forecast.
        predictions (torch.Tensor): Tensor of shape [num_samples, num_quantiles] containing forecasted quantiles.
        quantiles (List[float]): List of quantile levels (default: [0.1, ..., 0.9]).
        freq (pd.Timedelta): Time frequency of the data.
        target (Optional[torch.Tensor]): True target values (optional).
    """

    lead_time: int
    predictions: torch.Tensor  # Shape [num_samples, num_quantiles]
    quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    freq: pd.Timedelta
    target: Optional[torch.Tensor] = None
    data: Union[TimeSeriesDataFrame, TabularDataFrame]

    class Config:
        arbitrary_types_allowed = True

    def to_dataframe(self, item_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Converts the prediction data into a Pandas DataFrame and merges it with the actual target values.

        Args:
            item_ids (Optional[List[int]]): dataframe of the item ids to retrieve.

        Returns:
            pd.DataFrame: DataFrame with timestamps, predicted quantiles, and corresponding target values.
        """

        result = pd.DataFrame(self.predictions, index=self.data.index, columns=self.quantiles)
        result["prediction_date"] = result.index.get_level_values("timestamp") + self.lead_time * self.freq

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
        targets = self.to_dataframe(item_ids=item_ids)["target"].to_numpy()  # Shape [num_samples]

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

        results = self.to_dataframe(item_ids=item_ids)
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


class PredictionLeadTimes(BaseModel):
    """
    A container class for multiple PredictionLeadTime instances, providing aggregate metrics and plots.

    Attributes:
        results (Dict[int, PredictionLeadTime]): Dictionary mapping lead times to PredictionLeadTime instances.
    """

    results: Dict[int, PredictionLeadTime]

    class Config:
        arbitrary_types_allowed = True

    def save(self, file_path: Path) -> None:
        """Save predictions to files"""
        joblib.dump(self, file_path)
        logging.info("Saved prediction file (PredictionLeadTimes) to %s", file_path)

    def get_crps(self, lead_times: Optional[List[int]] = None, mean_lead_times: bool = False, mean_time: bool = False, item_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Computes CRPS for selected lead times."""
        lead_times = lead_times or list(self.results.keys())
        crps_scores = {lt: self.results[lt].get_crps(item_ids, mean_time=mean_time) for lt in lead_times}
        crps_scores = pd.DataFrame(data=crps_scores, index=["mean"] if mean_time else self.results[lead_times[0]].to_dataframe(item_ids=item_ids).index)  # TODO: make this nicer

        if mean_lead_times:
            crps_scores = pd.DataFrame(crps_scores.mean(axis=1), columns=["Mean CRPS"])

        return crps_scores.round(2)

    def get_quantile_scores(self, lead_times: Optional[List[int]] = None, mean_lead_times: bool = False, item_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Computes quantile scores for selected lead times."""
        lead_times = lead_times or list(self.results.keys())
        quantile_scores = {lt: self.results[lt].get_quantile_score(item_ids, mean_time=True) for lt in lead_times}

        quantile_scores = pd.DataFrame(quantile_scores)
        mean_scores = quantile_scores.mean(axis=1)

        if mean_lead_times:
            quantile_scores = pd.DataFrame(mean_scores, columns=["QS averaged over all lead times"])
        else:
            quantile_scores.loc[:, "QS averaged over all lead times"] = mean_scores

        quantile_scores.loc["Mean (CRPS/2)", :] = quantile_scores.mean()
        quantile_scores.index.name = "quantile"
        return quantile_scores.round(2)

    def get_empirical_coverage_rates(self, lead_times: Optional[List[int]] = None, mean_lead_times: bool = False, item_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Computes empirical coverage rates for selected lead times."""
        lead_times = lead_times or list(self.results.keys())

        coverage_rates = pd.DataFrame({lt: self.results[lt].get_empirical_coverage_rates(item_ids) for lt in lead_times})

        if mean_lead_times:
            coverage_rates = pd.DataFrame(coverage_rates).mean(axis=1)
            coverage_rates = pd.DataFrame(coverage_rates, columns=["Empirical coverage rates averaged over all lead times"])
        else:
            coverage_rates.loc[:, "Empirical coverage rates averaged over all lead times"] = coverage_rates.mean(axis=1)

        coverage_rates.index.name = "quantile"
        return coverage_rates.round(2)

    def get_pit_values(self, lead_times: Optional[List[int]] = None, item_ids: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """Computes PIT values for selected lead times."""
        lead_times = lead_times or list(self.results.keys())
        return {lt: self.results[lt].get_pit_values(item_ids) for lt in lead_times}

    def get_pit_histogram(self, lead_times: Optional[List[int]] = None, overlay: bool = False, item_ids: Optional[List[int]] = None) -> None:
        """Plots PIT histograms for selected lead times."""
        lead_times = lead_times or list(self.results.keys())

        if overlay:
            plt.figure(figsize=(10, 6))
            bins = len(next(iter(self.results.values())).quantiles)

            for lt in lead_times:
                pit_values = self.results[lt].get_pit_values(item_ids)
                plt.hist(pit_values, bins=bins, alpha=0.5, label=f"Lead time {lt}")

            plt.axhline(len(pit_values) / bins, color="red", linestyle="dashed", label="Uniform(0,1) reference")
            plt.xlabel("PIT Values")
            plt.ylabel("Observed Frequency")
            plt.title("PIT Histogram Across Lead Times")
            plt.legend()
            plt.show()

        else:
            num_plots = len(lead_times)
            cols = math.ceil(np.sqrt(num_plots))
            rows = (num_plots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
            axes = axes.flatten() if num_plots > 1 else [axes]
            for ax, lt in zip(axes, lead_times):
                pred = self.results[lt]
                pit_values = pred.get_pit_values()
                bins = len(pred.quantiles)
                ax.hist(pit_values, bins=bins, range=(0, 1), density=False, alpha=0.7, edgecolor="black")
                ax.axhline(len(pit_values) / bins, color="red", linestyle="dashed", label="Uniform(0,1) reference")
                ax.set_xlabel("PIT Values")
                ax.set_ylabel("Observed Frequency")
                ax.set_title(f"PIT Histogram (Lead Time {lt})")
                ax.legend()
            plt.tight_layout()
            plt.show()

    def get_reliability_diagram(self, lead_times: Optional[List[int]] = None, overlay: bool = False, item_ids: Optional[List[int]] = None) -> None:
        """Plots reliability diagrams for selected lead times."""
        lead_times = lead_times or list(self.results.keys())
        plt.figure(figsize=(8, 8))

        if overlay:
            for lt in lead_times:
                pred = self.results[lt]
                empirical_coverage_rates = pred.get_empirical_coverage_rates(item_ids)
                quantile_levels = sorted(empirical_coverage_rates.keys())
                empirical_coverages = [empirical_coverage_rates[q] for q in quantile_levels]
                plt.plot(quantile_levels, empirical_coverages, "o-", label=f"Lead time {lt}")

            plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
            plt.xticks(quantile_levels)
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
                pred = self.results[lt]
                empirical_coverage_rates = pred.get_empirical_coverage_rates(item_ids)
                quantile_levels = sorted(empirical_coverage_rates.keys())
                empirical_coverages = [empirical_coverage_rates[q] for q in quantile_levels]
                ax.plot(quantile_levels, empirical_coverages, "o-", label=f"Lead time {lt}")
                ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
                ax.set_xlabel("Nominal Quantile Level")
                ax.set_ylabel("Empirical Coverage")
                ax.set_xticks(quantile_levels)
                ax.set_title("Reliability Diagram for Quantile Forecasts")
                ax.legend()
            plt.tight_layout()
            plt.show()


def get_quantile_scores(
    predictions: Dict[str, PredictionLeadTimes],
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
    predictions : Dict[str, PredictionLeadTimes]
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
    return scores


def get_empirical_coverage_rates(predictions: Dict[str, PredictionLeadTimes], lead_times: Optional[List[int]] = None, item_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """Computes empirical coverage rates for different prediction sources,
    averaged across the specified lead times.

    Parameters
    -----------
    predictions : Dict[str, PredictionLeadTimes]
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

    return scores


def get_crps_scores(
    predictions: Dict[str, PredictionLeadTimes],
    lead_times: Optional[List[int]] = None,
    mean_lead_times: bool = False,
    item_ids: Optional[List[int]] = None,
    reference_predictions: Optional[str] = None,
) -> pd.DataFrame:
    """Computes and returns CRPS (Continuous Ranked Probability Score) values
    averaged across specified lead times and optionally normalized by a reference prediction.

    Parameters
    -----------
    predictions : Dict[str, PredictionLeadTimes]
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

    scores = pd.concat([pred.get_crps(lead_times=lead_times, mean_lead_times=mean_lead_times, mean_time=True, item_ids=item_ids) for pred in predictions.values()], axis=0).T
    scores.columns = [key for key in predictions.keys()]
    scores.index.name = "lead times"
    if reference_predictions:
        scores = scores.apply(lambda x: x / x[reference_predictions], axis=1)
    return scores


def plot_crps(
    predictions: Dict[str, PredictionLeadTimes],
    lead_times: Optional[List[int]] = None,
    item_ids: Optional[List[int]] = None,
    rolling_window_eval: Optional[int] = None,
    reference_predictions: Optional[str] = None,
) -> None:
    """Plots the mean CRPS (Continuous Ranked Probability Score) over time for different prediction sources.

    Parameters
    -----------
    predictions : Dict[str, PredictionLeadTimes]
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
    predictions: Dict[str, PredictionLeadTimes],
    lead_times: Optional[List[int]] = None,
    item_ids: Optional[List[int]] = None,
    reference_predictions: Optional[str] = None,
    date_splits: Optional[List[pd.Timestamp]] = None,
) -> pd.DataFrame:
    """Computes the mean CRPS (Continuous Ranked Probability Score) over time periods
    defined by timestamp splits for different prediction sources.

    Parameters
    -----------
    predictions : Dict[str, PredictionLeadTimes]
        Dictionary of prediction objects, where each value provides a `get_crps` method
        that returns a DataFrame with CRPS values indexed by ['item_id', 'timestamp'].
    lead_times : Optional[List[int]], default=None
        List of lead times to filter CRPS scores. If None, all lead times are used.
    item_ids : Optional[List[int]], default=None
        List of item IDs to include in the CRPS computation. If None, all item IDs are used.
    reference_predictions : Optional[str], default=None
        Key of a prediction set to be used as a reference for normalization.
        If provided, all CRPS values will be divided by the CRPS values from this prediction.
    date_splits : Optional[List[pd.Timestamp]], default=None
        List of timestamps to split the CRPS data into time-based segments.
        The function calculates mean CRPS in each period between these dates.

    Returns
    --------
    pd.DataFrame
        A DataFrame with the mean CRPS values for each time segment (e.g., "2023-01-01_to_2023-06-01"),
        optionally normalized by a reference prediction. Each column corresponds to a prediction key.
    """

    df = pd.concat([pred.get_crps(lead_times=lead_times, mean_lead_times=True, mean_time=False, item_ids=item_ids) for pred in predictions.values()], axis=1)
    df.columns = [key for key in predictions.keys()]

    if reference_predictions:
        df = df.apply(lambda x: x / x[reference_predictions], axis=1)

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
        results[f"{first_date}_to_{d}"] = subset_df.drop(columns=["item_id", "timestamp"]).mean()
        first_date = d

    return pd.DataFrame(results).T


def load_predictions(
    prediction_dirs: Union[List[Union[Path, str]], Path, str, None] = None,
    prediction_files: Union[List[Union[Path, str]], Path, str, None] = None,
) -> Dict[str, object]:
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
    Dict[str, object]
        A dictionary mapping generated keys to loaded prediction objects.
    """

    all_predictions = {}

    if prediction_files:
        logging.info("Loading predictions from provided file list...")

        if isinstance(prediction_files, (str, Path)):
            prediction_files = [prediction_files]

        for file in prediction_files:
            file = Path(file)
            if file.is_file() and file.suffix == ".joblib":
                key = file.stem
                all_predictions[key] = joblib.load(file)
                logging.info(f"Loaded prediction file: {file} as key: {key}")
            else:
                logging.warning(f"Skipping invalid or non-joblib file: {file}")

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
                parent_name = filepath.parent.name
                key = f"{parent_name}_{filepath.stem}"
                all_predictions[key] = joblib.load(filepath)
                logging.info(f"Loaded prediction file: `{filepath}` as key: {key}")

    else:
        raise ValueError("Either prediction_files or prediction_dirs must be provided.")

    if all_predictions:
        formatted_keys = "\n      - " + "\n      - ".join(all_predictions.keys())
        logging.info("Finished loading predictions. \n \n  Loaded keys:%s", formatted_keys)
    else:
        logging.warning("No prediction files were loaded.")

    return all_predictions
