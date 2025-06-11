from src.data.preprocessor import read_smard_data, read_exchange_rates_data
from src.pipeline.pipeline import ForecastingPipeline
from src.predictors.benchmarks import RollingSeasonalQuantilePredictor, RandomWalkBenchmark
from pathlib import Path
import argparse
import eval_constants
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline for selected dataset.")
    parser.add_argument("--dataset", type=str, required=True, choices=["day_ahead_prices", "electricity_consumption", "exchange_rates"], help="Dataset to evaluate")
    args = parser.parse_args()

    lead_times = eval_constants.lead_times
    quantiles = eval_constants.quantiles
    test_start_date = eval_constants.test_start_date
    postprocessors = eval_constants.postprocessors
    postprocessor_kwargs = eval_constants.postprocessor_kwargs

    if args.dataset == "day_ahead_prices":
        data, mapping, freq = read_smard_data(
            file_paths=["data/day_ahead_prices/Day-ahead_prices_201501010000_202001010000_Hour.csv", "data/day_ahead_prices/Day-ahead_prices_202001010000_202506120000_Hour.csv"],
            selected_time_series=[
                "Belgium [€/MWh] Original resolutions",
                "Denmark 1 [€/MWh] Original resolutions",
                "Denmark 2 [€/MWh] Original resolutions",
                "France [€/MWh] Original resolutions",
                "Netherlands [€/MWh] Original resolutions",
                "Norway 2 [€/MWh] Original resolutions",
                "Sweden 4 [€/MWh] Original resolutions",
                "Switzerland [€/MWh] Original resolutions",
                "Czech Republic [€/MWh] Original resolutions",
                "Slovenia [€/MWh] Original resolutions",
                "Hungary [€/MWh] Original resolutions",
            ],
            freq=pd.Timedelta("1h"),
        )

        # Naive Rolling Seasonal quantile predictions
        pipeline = ForecastingPipeline(
            model=RollingSeasonalQuantilePredictor,
            model_kwargs={"quantiles": quantiles, "lead_times": lead_times, "freq": freq},
            postprocessors=postprocessors,
            postprocessor_kwargs=postprocessor_kwargs,
            output_dir=eval_constants.output_dir_day_ahead_prices / "seasonal_rolling",
        )

        results = pipeline.backtest(
            test_start_date=test_start_date,
            data=data,
            rolling_window_eval=False,
            train=True,
            val_window_size=None,
            train_window_size=None,
            test_window_size=None,
            calibration_based_on="train",
            save_results=True,
        )

    elif args.dataset == "electricity_consumption":
        data, mapping, freq = read_smard_data(
            file_paths=[
                "data/electricity_consumption/Actual_consumption_201501010000_202001010000_Quarterhour.csv",
                "data/electricity_consumption/Actual_consumption_202001010000_202506120000_Quarterhour.csv",
            ],
            selected_time_series=["grid load [MWh] Original resolutions", "Residual load [MWh] Original resolutions"],
            freq=pd.Timedelta("15 min"),
        )


        # Naive Rolling Seasonal quantile predictions
        pipeline = ForecastingPipeline(
            model=RollingSeasonalQuantilePredictor,
            model_kwargs={"quantiles": quantiles, "lead_times": lead_times, "freq": freq},
            postprocessors=postprocessors,
            postprocessor_kwargs=postprocessor_kwargs,
            output_dir=eval_constants.output_dir_electricity_consumption / "seasonal_rolling",
        )

        results = pipeline.backtest(
            test_start_date=test_start_date,
            data=data,
            rolling_window_eval=False,
            train=True,
            val_window_size=None,
            train_window_size=None,
            test_window_size=None,
            calibration_based_on="train",
            save_results=True,
        )

    elif args.dataset == "exchange_rates":
        data, mapping, freq = read_exchange_rates_data(files_dir="data/exchange_rates/")

        # Naive rolling quantile predictions
        pipeline = ForecastingPipeline(
            model=RandomWalkBenchmark,
            model_kwargs={"quantiles": quantiles, "lead_times": lead_times, "freq": freq},
            postprocessors=postprocessors,
            postprocessor_kwargs=postprocessor_kwargs,
            output_dir=eval_constants.output_dir_exchange_rates / "random_walk",
        )

        results = pipeline.backtest(
            test_start_date=test_start_date,
            data=data,
            rolling_window_eval=False,
            train=True,
            val_window_size=None,
            train_window_size=None,
            test_window_size=None,
            calibration_based_on="train",
            save_results=True,
        )

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()
