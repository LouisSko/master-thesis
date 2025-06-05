from src.data.preprocessor import read_smard_data, read_exchange_rates_data
from src.pipeline.pipeline import ForecastingPipeline
from src.predictors.benchmarks import RollingSeasonalQuantilePredictor, RandomWalkBenchmark
from pathlib import Path
import argparse
import eval_constants


def main():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline for selected dataset.")
    parser.add_argument("--dataset", type=str, required=True, choices=["wholesale_prices", "energy_consumption", "exchange_rates"], help="Dataset to evaluate")
    args = parser.parse_args()

    lead_times = eval_constants.lead_times
    quantiles = eval_constants.quantiles
    test_start_date = eval_constants.test_start_date
    postprocessors = eval_constants.postprocessors
    postprocessor_kwargs = eval_constants.postprocessor_kwargs

    if args.dataset == "wholesale_prices":
        data, mapping, freq = read_smard_data(
            file_paths=["data/Gro_handelspreise_201501010000_202101010000_Stunde.csv", "data/Gro_handelspreise_202101010000_202504240000_Stunde.csv"],
            selected_time_series=None,
        )
        output_dir = Path("./results/wholesale_prices/pipeline/")

        # Naive Rolling Seasonal quantile predictions
        pipeline = ForecastingPipeline(
            model=RollingSeasonalQuantilePredictor,
            model_kwargs={"quantiles": quantiles, "lead_times": lead_times, "freq": freq},
            postprocessors=postprocessors,
            postprocessor_kwargs=postprocessor_kwargs,
            output_dir=output_dir / "seasonal_rolling",
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

    elif args.dataset == "energy_consumption":
        data, mapping, freq = read_smard_data(
            file_paths=["data/Realisierter_Stromverbrauch_201501010000_202101010000_Stunde.csv", "data/Realisierter_Stromverbrauch_202101010000_202504240000_Stunde.csv"],
            selected_time_series=["Netzlast [MWh] Berechnete Auflösungen", "Residuallast [MWh] Berechnete Auflösungen"],
        )
        output_dir = Path("./results/energy_consumption/pipeline/")

        # Naive Rolling Seasonal quantile predictions
        pipeline = ForecastingPipeline(
            model=RollingSeasonalQuantilePredictor,
            model_kwargs={"quantiles": quantiles, "lead_times": lead_times, "freq": freq},
            postprocessors=postprocessors,
            postprocessor_kwargs=postprocessor_kwargs,
            output_dir=output_dir / "seasonal_rolling",
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
        output_dir = Path("./results/exchange_rates/pipeline/")

        # Naive rolling quantile predictions
        pipeline = ForecastingPipeline(
            model=RandomWalkBenchmark,
            model_kwargs={"quantiles": quantiles, "lead_times": lead_times, "freq": freq},
            postprocessors=postprocessors,
            postprocessor_kwargs=postprocessor_kwargs,
            output_dir=output_dir / "random_walk",
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
