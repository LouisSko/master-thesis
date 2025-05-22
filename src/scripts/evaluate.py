from src.data.preprocessor import read_smard_data, read_exchange_rates_data
from src.postprocessors.mle import PostprocessorMLE
from src.postprocessors.qr import PostprocessorQR
from src.postprocessors.eqc import PostprocessorEQC
from src.pipeline.pipeline import ForecastingPipeline
from src.predictors.chronos import Chronos
from autogluon.timeseries import TimeSeriesDataFrame
from src.predictors.benchmarks import RollingQuantilePredictor, RollingSeasonalQuantilePredictor
import numpy as np
from pathlib import Path
import pandas as pd
from typing import List, Union
import argparse


def evaluate(data: TimeSeriesDataFrame, freq: Union[pd.Timedelta, pd.DateOffset], lead_times: List[int], quantiles: List[float], test_start_date: pd.Timestamp, output_dir: Path):

    # chronos zero shot results
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={"pretrained_model_name_or_path": "amazon/chronos-bolt-tiny", "device_map": "mps", "lead_times": lead_times, "freq": freq},
        postprocessors=[PostprocessorMLE, PostprocessorQR, PostprocessorEQC],
        output_dir=output_dir / "chronos-bolt-zero-shot",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=False,
        val_window_size=None,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on="train",
        save_results=True,
    )

    # chronos zero shot results with sampling
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={"pretrained_model_name_or_path": "amazon/chronos-bolt-tiny", "sampling": True, "device_map": "mps", "lead_times": lead_times, "freq": freq},
        # postprocessors=[PostprocessorMLE, PostprocessorQR, PostprocessorEQC],
        output_dir=output_dir / "chronos-bolt-zero-shot-sampling",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=False,
        val_window_size=None,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on=None,
        save_results=True,
    )

    # chronos full fine tuning
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": "amazon/chronos-bolt-tiny",
            "device_map": "mps",
            "lead_times": lead_times,
            "freq": freq,
            "finetuning_type": "full",
            "finetuning_hp_search": False,
        },
        postprocessors=[PostprocessorMLE, PostprocessorQR, PostprocessorEQC],
        output_dir=output_dir / "chronos-bolt-finetuned-full",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=pd.DateOffset(years=3),
        train_window_size=None,
        test_window_size=None,
        calibration_based_on="val",
        save_results=True,
    )

    # chronos last layer fine tuning
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": "amazon/chronos-bolt-tiny",
            "device_map": "mps",
            "lead_times": lead_times,
            "freq": freq,
            "finetuning_type": "last_layer",
            "finetuning_hp_search": False,
        },
        postprocessors=[PostprocessorMLE, PostprocessorQR, PostprocessorEQC],
        output_dir=output_dir / "chronos-bolt-finetuned-last-layer",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=pd.DateOffset(years=3),
        train_window_size=None,
        test_window_size=None,
        calibration_based_on="val",
        save_results=True,
    )

    # chronos LoRa tuning
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": "amazon/chronos-bolt-tiny",
            "device_map": "mps",
            "lead_times": lead_times,
            "freq": freq,
            "finetuning_type": "LoRa",
            "finetuning_hp_search": False,
        },
        postprocessors=[PostprocessorMLE, PostprocessorQR, PostprocessorEQC],
        output_dir=output_dir / "chronos-bolt-finetuned-lora",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=pd.DateOffset(years=3),
        train_window_size=None,
        test_window_size=None,
        calibration_based_on="val",
        save_results=True,
    )

    # Naive Rolling Seasonal quantile predictions
    pipeline = ForecastingPipeline(
        model=RollingSeasonalQuantilePredictor,
        model_kwargs={"quantiles": quantiles, "lead_times": lead_times, "freq": freq},
        postprocessors=[PostprocessorQR, PostprocessorMLE, PostprocessorEQC],
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

    # Naive rolling quantile predictions
    pipeline = ForecastingPipeline(
        model=RollingQuantilePredictor,
        model_kwargs={"quantiles": quantiles, "lead_times": lead_times, "freq": freq},
        postprocessors=[PostprocessorQR, PostprocessorMLE, PostprocessorEQC],
        output_dir=output_dir / "naive_rolling",
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


def main():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline for selected dataset.")
    parser.add_argument("--dataset", type=str, required=True, choices=["grosshandelspreise", "stromverbrauch", "exchange_rates"], help="Dataset to evaluate")

    args = parser.parse_args()

    if args.dataset == "grosshandelspreise":
        data, mapping, freq = read_smard_data(
            file_paths=["data/Gro_handelspreise_201501010000_202101010000_Stunde.csv", "data/Gro_handelspreise_202101010000_202504240000_Stunde.csv"],
            selected_time_series=None,
        )
        output_dir = Path("./results/grosshandelspreise/pipeline/")

    elif args.dataset == "stromverbrauch":
        data, mapping, freq = read_smard_data(
            file_paths=["data/Realisierter_Stromverbrauch_201501010000_202101010000_Stunde.csv", "data/Realisierter_Stromverbrauch_202101010000_202504240000_Stunde.csv"],
            selected_time_series=["Netzlast [MWh] Berechnete Auflösungen", "Residuallast [MWh] Berechnete Auflösungen"],
        )
        output_dir = Path("./results/stromverbrauch/pipeline/")

    elif args.dataset == "exchange_rates":
        data, mapping, freq = read_exchange_rates_data(files_dir="data/exchange_rates/")
        output_dir = Path("./results/exchange_rates/pipeline/")

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    lead_times = np.arange(1, 192 + 1).tolist()
    quantiles = np.round(np.arange(0.1, 1, 0.1), 1).tolist()

    test_start_date = pd.Timestamp("2023-01-01")

    evaluate(data, freq, lead_times, quantiles, test_start_date, output_dir)


if __name__ == "__main__":
    main()
