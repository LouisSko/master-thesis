from typing import List, Optional, Dict
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import scoringrules as sr
from scipy.interpolate import interp1d
from pydantic import BaseModel, Field
from autogluon.timeseries import TimeSeriesDataFrame
import math

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
    data: TimeSeriesDataFrame

    class Config:
        arbitrary_types_allowed = True

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the prediction data into a Pandas DataFrame and merges it with the actual target values.

        Args:
            data (pd.DataFrame): DataFrame containing actual target values.

        Returns:
            pd.DataFrame: DataFrame with timestamps, predicted quantiles, and corresponding target values.
        """

        result = pd.DataFrame(self.predictions, index=self.data.index, columns=self.quantiles)
        result["prediction_date"] = result.index.get_level_values("timestamp") + self.lead_time * self.freq

        # Reset index to turn MultiIndex into columns
        result_reset = result.reset_index()
        data_reset = self.data.reset_index()

        # Perform the merge
        merged = result_reset.merge(
            data_reset, left_on=["item_id", "prediction_date"], right_on=["item_id", "timestamp"], how="left", suffixes=["", "_remove"]  # From result  # From preds.data
        )

        merged = merged.drop(columns=["timestamp_remove"])

        merged.set_index(result.index.names, inplace=True)

        # Restore the original MultiIndex
        return merged

    def get_crps(self) -> float:
        """
        Computes the Continuous Ranked Probability Score (CRPS) for the forecast.

        Args:
            data (pd.DataFrame): DataFrame containing actual target values.

        Returns:
            float: The mean CRPS score across all samples.
        """
        data = self.to_dataframe()
        crps = sr.crps_quantile(data["target"].to_numpy(), data[self.quantiles].to_numpy(), self.quantiles)
        return crps[~np.isnan(crps)].mean()

    def get_quantile_score(self) -> pd.DataFrame:
        """
        Computes the average quantile score (pinball loss) for the forecast.

        Args:
            data (pd.DataFrame): DataFrame containing actual target values.

        Returns:
            pd.DataFrame: A DataFrame with the mean pinball loss for each quantile.
        """
        data = self.to_dataframe()
        quantile_scores = np.column_stack([sr.quantile_score(data["target"].to_numpy(), data[q].to_numpy(), q) for q in self.quantiles])

        return pd.DataFrame(quantile_scores, columns=self.quantiles).mean()

    def get_pit_values(self) -> np.ndarray:
        """
        Computes the Probability Integral Transform (PIT) values for calibration analysis.

        PIT = F(y), where F(y) is the interp_func, which is the CDF approximated based on the quantiles.

        Args:
            data (pd.DataFrame): DataFrame containing actual target values.

        Returns:
            np.ndarray: Array of PIT values, where PIT values should follow a uniform [0,1] distribution.
        """
        targets = self.to_dataframe()["target"].to_numpy()  # Shape [num_samples]

        # Compute PIT values for each target
        pit_values = []
        for i in range(len(targets)):
            interp_func = interp1d(self.predictions[i], self.quantiles, bounds_error=False, fill_value=(0, 1))
            pit_values.append(interp_func(targets[i]))

        return np.array(pit_values)

    def get_pit_histogram(self) -> None:
        """
        Plots a histogram of Probability Integral Transform (PIT) values to assess forecast calibration.

        Args:
            data (pd.DataFrame): DataFrame containing actual target values.

        Returns:
            None: Displays the histogram plot.
        """
        pit_values = self.get_pit_values()
        bins = len(self.quantiles)

        plt.hist(pit_values, bins=bins, range=(0, 1), density=False, alpha=0.7, edgecolor="black")
        plt.axhline(len(pit_values) / bins, color="red", linestyle="dashed", label="Uniform(0,1) reference")
        plt.xlabel("PIT Values")
        plt.ylabel("Frequency of Occurrences")
        plt.title("PIT Histogram")
        plt.legend()
        plt.show()

    def get_empirical_coverage_rates(self) -> Dict[float, float]:

        results = self.to_dataframe()
        empirical_coverage_rates = {q: (results[q] >= results["target"]).mean() for q in self.quantiles}

        return empirical_coverage_rates

    def get_reliability_diagram(self) -> None:

        empirical_coverage_rates = self.get_empirical_coverage_rates()

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


# class PredictionLeadTimes(BaseModel):
#     """
#     A container class for multiple PredictionLeadTime instances.

#     Attributes:
#         results (Dict[int, PredictionLeadTime]): Dictionary mapping lead times to PredictionLeadTime instances.
#     """

#     results: Dict[int, PredictionLeadTime]

#     class Config:
#         arbitrary_types_allowed = True


class PredictionLeadTimes(BaseModel):
    """
    A container class for multiple PredictionLeadTime instances, providing aggregate metrics and plots.

    Attributes:
        results (Dict[int, PredictionLeadTime]): Dictionary mapping lead times to PredictionLeadTime instances.
    """

    results: Dict[int, PredictionLeadTime]

    class Config:
        arbitrary_types_allowed = True

    def get_crps(self, lead_times: Optional[List[int]] = None) -> Dict[int, float]:
        """Computes CRPS for selected lead times."""
        lead_times = lead_times or list(self.results.keys())
        return {lt: self.results[lt].get_crps() for lt in lead_times}

    def get_quantile_scores(self, lead_times: Optional[List[int]] = None) -> pd.DataFrame:
        """Computes quantile scores for selected lead times."""
        lead_times = lead_times or list(self.results.keys())
        return pd.DataFrame({lt: self.results[lt].get_quantile_score() for lt in lead_times})

    def get_pit_values(self, lead_times: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """Computes PIT values for selected lead times."""
        lead_times = lead_times or list(self.results.keys())
        return {lt: self.results[lt].get_pit_values() for lt in lead_times}

    def get_pit_histogram(self, lead_times: Optional[List[int]] = None, overlay: bool = False) -> None:
        """Plots PIT histograms for selected lead times."""
        lead_times = lead_times or list(self.results.keys())
        
        if overlay:
            plt.figure(figsize=(10, 6))
            bins = len(next(iter(self.results.values())).quantiles)

            for lt in lead_times:
                pit_values = self.results[lt].get_pit_values()
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


    def get_empirical_coverage_rates(self, lead_times: Optional[List[int]] = None) -> pd.DataFrame:
        """Computes empirical coverage rates for selected lead times."""
        lead_times = lead_times or list(self.results.keys())
        return pd.DataFrame({lt: self.results[lt].get_empirical_coverage_rates() for lt in lead_times})

    def get_reliability_diagram(self, lead_times: Optional[List[int]] = None, overlay: bool = False) -> None:
        """Plots reliability diagrams for selected lead times."""
        lead_times = lead_times or list(self.results.keys())
        plt.figure(figsize=(8, 8))

        if overlay:
            for lt in lead_times:
                pred = self.results[lt]
                empirical_coverage_rates = pred.get_empirical_coverage_rates()
                quantile_levels = sorted(empirical_coverage_rates.keys())
                empirical_coverages = [empirical_coverage_rates[q] for q in quantile_levels]
                plt.plot(quantile_levels, empirical_coverages, "o-", label=f"Lead time {lt}")

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
                pred = self.results[lt]
                empirical_coverage_rates = pred.get_empirical_coverage_rates()
                quantile_levels = sorted(empirical_coverage_rates.keys())
                empirical_coverages = [empirical_coverage_rates[q] for q in quantile_levels]
                ax.plot(quantile_levels, empirical_coverages, "o-", label=f"Lead time {lt}")
                ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
                ax.set_xlabel("Nominal Quantile Level")
                ax.set_ylabel("Empirical Coverage")
                ax.set_title("Reliability Diagram for Quantile Forecasts")
                ax.legend()
            plt.tight_layout()
            plt.show()