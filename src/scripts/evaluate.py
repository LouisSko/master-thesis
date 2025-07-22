from src.pipeline.pipeline import ForecastingPipeline
from src.predictors.chronos import Chronos
from src.predictors.tirex import TiRex
from src.predictors.benchmarks import SeasonalNaive, RandomWalk
import eval_constants
import torch
from src.predictors.autogluon_wrapper import SeasonalNaive_Ag, PatchTST_Ag, TiDE_Ag
import argparse

# 3 dataset to chose from
DAP = "day_ahead_prices"
EC = "electricity_consumption"
ER = "exchange_rates"


def evaluate():

    # specify chronos variant
    chronos_variant = "tiny"

    parser = argparse.ArgumentParser(description="Run evaluation pipeline for selected dataset.")
    parser.add_argument("--dataset", type=str, required=True, choices=[DAP, EC, ER], help="Dataset to evaluate")

    args = parser.parse_args()

    if args.dataset == EC:
        dataset_config = eval_constants.get_electricity_consumption_config()
    elif args.dataset == DAP:
        dataset_config = eval_constants.get_day_ahead_prices_config()
    elif args.dataset == ER:
        dataset_config = eval_constants.get_exchange_rate_config()

    # common settings
    lead_times = eval_constants.lead_times
    quantiles = eval_constants.quantiles
    test_start_date = eval_constants.test_start_date
    postprocessors = eval_constants.postprocessors
    postprocessor_kwargs = eval_constants.postprocessor_kwargs

    # dataset specific:
    freq = dataset_config["freq"]
    val_window_size = dataset_config["val_window_size"]
    test_window_step = dataset_config["test_window_step"]
    calibration_window_step = dataset_config["calibration_window_step"]
    output_dir = dataset_config["output_dir"]
    seasonal_period = dataset_config["seasonal_period"]
    data = dataset_config["data"]

    if torch.cuda.is_available():
        device_map = "cuda"
    elif torch.mps.is_available():
        device_map = "mps"
    else:
        device_map = "cpu"

    # ------------------------ Chronos-Bolt ------------------------

    # chronos bolt zero shot results
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/chronos-bolt-{chronos_variant}",
            "device_map": device_map,
            "lead_times": lead_times,
            "name": f"Chronos-Bolt-{chronos_variant}",
        },
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
        freq=freq,
        output_dir=output_dir / f"Chronos-Bolt-{chronos_variant}",
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
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
    )

    del pipeline
    del results

    # chronos bolt full fine tuning
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/chronos-bolt-{chronos_variant}",
            "device_map": device_map,
            "lead_times": lead_times,
            "finetuning_type": "full",
            "finetuning_hp_search": False,
            "finetuning_warmup_new_neurons": True,
            "finetuning_adjust_pretrained_prediction_length": True,
            "name": f"Chronos-Bolt-{chronos_variant}-FT_Full",
        },
        postprocessors=None,
        postprocessor_kwargs=None,
        freq=freq,
        output_dir=output_dir / f"Chronos-Bolt-{chronos_variant}-FT_Full",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=val_window_size,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on=None,
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
    )

    # chronos-bolt last layer fine tuning
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/chronos-bolt-{chronos_variant}",
            "device_map": device_map,
            "lead_times": lead_times,
            "finetuning_type": "last_layer",
            "finetuning_hp_search": False,
            "finetuning_warmup_new_neurons": True,
            "finetuning_adjust_pretrained_prediction_length": True,
            "name": f"Chronos-Bolt-{chronos_variant}-FT_Last",
        },
        postprocessors=None,
        postprocessor_kwargs=None,
        freq=freq,
        output_dir=output_dir / f"Chronos-Bolt-{chronos_variant}-FT_Last",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=val_window_size,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on=None,
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
    )

    del pipeline
    del results

    # chronos-bolt lora fine tuning
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/chronos-bolt-{chronos_variant}",
            "device_map": device_map,
            "lead_times": lead_times,
            "finetuning_type": "LoRA",
            "finetuning_hp_search": False,
            "finetuning_warmup_new_neurons": True,
            "finetuning_adjust_pretrained_prediction_length": True,
            "name": f"Chronos-Bolt-{chronos_variant}-FT_LoRA",
        },
        postprocessors=None,
        postprocessor_kwargs=None,
        freq=freq,
        output_dir=output_dir / f"Chronos-Bolt-{chronos_variant}-FT_LoRA",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=val_window_size,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on=None,
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
    )

    del pipeline
    del results

    # ------------------------ Chronos-T5 ------------------------

    # chronos-t5 zero-shot
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/chronos-t5-{chronos_variant}",
            "device_map": device_map,
            "lead_times": lead_times,
            "name": f"Chronos-T5-{chronos_variant}",
        },
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
        freq=freq,
        output_dir=output_dir / f"Chronos-T5-{chronos_variant}",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=False,
        val_window_size=val_window_size,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on="train",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
    )

    del pipeline
    del results

    # chronos-t5 full fine tuning
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/chronos-t5-{chronos_variant}",
            "device_map": device_map,
            "lead_times": lead_times,
            "finetuning_type": "full",
            "finetuning_hp_search": False,
            "finetuning_warmup_new_neurons": False,
            "finetuning_adjust_pretrained_prediction_length": True,
            "name": f"Chronos-T5-{chronos_variant}-FT_Full",
        },
        postprocessors=None,
        postprocessor_kwargs=None,
        freq=freq,
        output_dir=output_dir / f"Chronos-T5-{chronos_variant}-FT_Full",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=val_window_size,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on=None,
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
    )

    del pipeline
    del results

    # chronos-t5 last layer fine tuning
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/chronos-t5-{chronos_variant}",
            "device_map": device_map,
            "lead_times": lead_times,
            "finetuning_type": "last_layer",
            "finetuning_hp_search": False,
            "finetuning_warmup_new_neurons": False,
            "finetuning_adjust_pretrained_prediction_length": True,
            "name": f"Chronos-T5-{chronos_variant}-FT_Last",
        },
        postprocessors=None,
        postprocessor_kwargs=None,
        freq=freq,
        output_dir=output_dir / f"Chronos-T5-{chronos_variant}-FT_Last",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=val_window_size,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on=None,
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
    )

    del pipeline
    del results

    # chronos-t5 LoRA fine tuning
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/chronos-t5-{chronos_variant}",
            "device_map": device_map,
            "lead_times": lead_times,
            "finetuning_type": "last_layer",
            "finetuning_hp_search": False,
            "finetuning_warmup_new_neurons": False,
            "finetuning_adjust_pretrained_prediction_length": True,
            "name": f"Chronos-T5-{chronos_variant}-FT_LoRA",
        },
        postprocessors=None,
        postprocessor_kwargs=None,
        freq=freq,
        output_dir=output_dir / f"Chronos-T5-{chronos_variant}-FT_LoRA",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=val_window_size,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on=None,
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
    )

    del pipeline
    del results

    # ------------------------ Benchmarks ------------------------

    # RandomWalk - self implemented
    pipeline = ForecastingPipeline(
        model=RandomWalk,
        model_kwargs={"quantiles": quantiles, "lead_times": lead_times, "name": "RandomWalk"},
        postprocessors=None,
        postprocessor_kwargs=None,
        output_dir=output_dir / "RandomWalk",
        freq=freq,
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=val_window_size,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on=None,
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
    )

    del pipeline
    del results

    # SeasonalNaive Benchmark - self implemented
    pipeline = ForecastingPipeline(
        model=SeasonalNaive,
        model_kwargs={"quantiles": quantiles, "lead_times": lead_times, "freq": freq, "name": "SeasonalNaive"},
        postprocessors=None,
        postprocessor_kwargs=None,
        output_dir=output_dir / "SeasonalNaive",
        freq=freq,
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=val_window_size,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on=None,
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
    )

    del pipeline
    del results

    # seasonal naive model autogluon
    pipeline = ForecastingPipeline(
        model=SeasonalNaive_Ag,
        model_kwargs={"lead_times": lead_times, "freq": freq, "seasonal_period": seasonal_period, "name": "SeasonalNaiveAg"},
        postprocessors=None,
        freq=freq,
        output_dir=output_dir / "SeasonalNaiveAg",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=val_window_size,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on=None,
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
    )

    del pipeline
    del results

    # PatchTST autogluon
    pipeline = ForecastingPipeline(
        model=PatchTST_Ag,
        model_kwargs={"lead_times": lead_times, "freq": freq, "name": "PatchTST"},
        postprocessors=None,
        freq=freq,
        output_dir=output_dir / "PatchTST",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=val_window_size,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on=None,
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
    )

    del pipeline
    del results

    # TiDE autogluon
    pipeline = ForecastingPipeline(
        model=TiDE_Ag,
        model_kwargs={"lead_times": lead_times, "freq": freq, "name": "TiDE"},
        postprocessors=None,
        freq=freq,
        output_dir=output_dir / "TiDE",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=val_window_size,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on=None,
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
    )

    del pipeline
    del results

    # tirex zero shot results
    pipeline = ForecastingPipeline(
        model=TiRex,
        model_kwargs={
            "lead_times": lead_times,
            "tirex_service_url": "http://localhost:8000",
            "name": "TiRex",
        },
        postprocessors=None,
        postprocessor_kwargs=None,
        freq=freq,
        output_dir=output_dir / "TiRex",
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
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
    )

    del pipeline
    del results


if __name__ == "__main__":
    evaluate()
