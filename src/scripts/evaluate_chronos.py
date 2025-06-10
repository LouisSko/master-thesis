from src.data.preprocessor import read_smard_data, read_exchange_rates_data
from src.pipeline.pipeline import ForecastingPipeline
from src.predictors.chronos import Chronos
from autogluon.timeseries import TimeSeriesDataFrame
from pathlib import Path
import pandas as pd
from typing import Union
import argparse
import eval_constants
import torch


def evaluate(
    data: TimeSeriesDataFrame,
    freq: Union[pd.Timedelta, pd.DateOffset],
    val_window_size: pd.DateOffset,
    model_name: str,  # like chronos-t5-tiny or chronos-bolt-tiny
    output_dir: Path,
):

    lead_times = eval_constants.lead_times
    quantiles = eval_constants.quantiles
    test_start_date = eval_constants.test_start_date
    postprocessors = eval_constants.postprocessors
    postprocessor_kwargs = eval_constants.postprocessor_kwargs

    if torch.cuda.is_available():
        device_map = "cuda"
    elif torch.mps.is_available():
        device_map = "mps"
    else:
        device_map = "cpu"

    # chronos zero shot results
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={"pretrained_model_name_or_path": f"amazon/{model_name}", "device_map": device_map, "lead_times": lead_times, "freq": freq},
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
        output_dir=output_dir / f"{model_name}-zero-shot",
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

    del pipeline
    del results

    # chronos full fine tuning
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/{model_name}",
            "device_map": device_map,
            "lead_times": lead_times,
            "freq": freq,
            "finetuning_type": "full",
            "finetuning_hp_search": False,
        },
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
        output_dir=output_dir / f"{model_name}-finetuned-full",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=val_window_size,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on="val",
        save_results=True,
    )

    del pipeline
    del results

    # chronos last layer fine tuning
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/{model_name}",
            "device_map": device_map,
            "lead_times": lead_times,
            "freq": freq,
            "finetuning_type": "last_layer",
            "finetuning_hp_search": False,
        },
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
        output_dir=output_dir / f"{model_name}-finetuned-last-layer",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=val_window_size,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on="val",
        save_results=True,
    )

    del pipeline
    del results

    # chronos LoRa tuning
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/{model_name}",
            "device_map": device_map,
            "lead_times": lead_times,
            "freq": freq,
            "finetuning_type": "LoRa",
            "finetuning_hp_search": False,
        },
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
        output_dir=output_dir / f"{model_name}-finetuned-lora",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=val_window_size,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on="val",
        save_results=True,
    )

    del pipeline
    del results


def main():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline for selected dataset.")
    parser.add_argument("--dataset", type=str, required=True, choices=["wholesale_prices", "electricity_consumption", "exchange_rates"], help="Dataset to evaluate")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["chronos-bolt-tiny", "chronos-bolt-small", "chronos-bolt-base", "chronos-t5-tiny", "chronos-t5-small", "chronos-t5-base"],
        help="Model used for predictions",
    )

    args = parser.parse_args()

    if args.dataset == "wholesale_prices":
        val_window_size = pd.DateOffset(years=1)
        data, mapping, freq = read_smard_data(
            file_paths=["data/Gro_handelspreise_201501010000_202101010000_Stunde.csv", "data/Gro_handelspreise_202101010000_202504240000_Stunde.csv"],
            selected_time_series=None,
        )
        output_dir = Path("./results/wholesale_prices/pipeline/")

    elif args.dataset == "electricity_consumption":
        val_window_size = pd.DateOffset(years=1)
        data, mapping, freq = read_smard_data(
            file_paths=["data/Realisierter_Stromverbrauch_201501010000_202101010000_Stunde.csv", "data/Realisierter_Stromverbrauch_202101010000_202504240000_Stunde.csv"],
            selected_time_series=["Netzlast [MWh] Berechnete Auflösungen", "Residuallast [MWh] Berechnete Auflösungen"],
        )
        output_dir = Path("./results/electricity_consumption/pipeline/")

    elif args.dataset == "exchange_rates":
        val_window_size = pd.DateOffset(years=5)
        data, mapping, freq = read_exchange_rates_data(files_dir="data/exchange_rates/")
        output_dir = Path("./results/exchange_rates/pipeline/")

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    evaluate(data, freq, val_window_size, args.model_name, output_dir)


if __name__ == "__main__":
    main()
