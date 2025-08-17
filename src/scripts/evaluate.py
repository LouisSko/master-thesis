from src.pipeline.pipeline import ForecastingPipeline
from src.predictors.chronos import Chronos
from src.predictors.tirex import TiRex
from src.predictors.benchmarks import SeasonalNaive, RandomWalk, OnlineRandomWalk
import eval_constants
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    auto_calibration = "auto" if eval_constants.auto_calibration else None
    auto_determine_val_set = eval_constants.auto_determine_val_set

    # dataset specific:
    freq = dataset_config["freq"]
    val_window_size = dataset_config["val_window_size"]
    calibration_window_step = dataset_config["calibration_window_step"]
    test_window_step = dataset_config["test_window_step"]
    output_dir = dataset_config["output_dir"]
    seasonal_period = dataset_config["seasonal_period"]
    data = dataset_config["data"]

    if torch.cuda.is_available():
        device_map = "cuda"
    elif torch.mps.is_available():
        device_map = "mps"
    else:
        device_map = "cpu"

    # torch.cuda.set_device(1)

    # ------------------------ Chronos-Bolt ------------------------

    # following models are evaluated:

    # Chronos-Bolt-*
    # Chronos-Bolt-*-PP_Offset
    # Chronos-Bolt-*-PP_QuantReg
    # Chronos-Bolt-*-PP_Gauss

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
        calibration_based_on=auto_calibration or "train",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
        auto_determine_val_set=auto_determine_val_set,
    )

    del pipeline
    del results

    # following models are evaluated:

    # Chronos-Bolt-*-FT_LongOut
    # Chronos-Bolt-*-FT_LongOut-PP_Offset
    # Chronos-Bolt-*-FT_LongOut-PP_QuantReg
    # Chronos-Bolt-*-FT_LongOut-PP_Gauss

    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/chronos-bolt-{chronos_variant}",
            "device_map": device_map,
            "lead_times": lead_times,
            "finetuning_schedule": ["full"],
            "finetuning_hp_search": False,
            "finetuning_warmup_new_neurons": True,
            "finetuning_adjust_pretrained_prediction_length": True,
            "name": f"Chronos-Bolt-{chronos_variant}-FT_LongOut",
        },
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
        freq=freq,
        output_dir=output_dir / f"Chronos-Bolt-{chronos_variant}-FT_LongOut",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=val_window_size,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on=auto_calibration or "val",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
        auto_determine_val_set=auto_determine_val_set,
    )

    # following models are evaluated:

    # Chronos-Bolt-*-FT_Last
    # Chronos-Bolt-*-FT_Last-PP_Offset
    # Chronos-Bolt-*-FT_Last-PP_QuantReg
    # Chronos-Bolt-*-FT_Last-PP_Gauss

    # chronos-bolt last layer fine tuning
    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/chronos-bolt-{chronos_variant}",
            "device_map": device_map,
            "lead_times": lead_times,
            "finetuning_schedule": ["last_layer"],
            "finetuning_hp_search": False,
            "finetuning_warmup_new_neurons": False,
            "finetuning_adjust_pretrained_prediction_length": False,
            "name": f"Chronos-Bolt-{chronos_variant}-FT_Last",
        },
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
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
        calibration_based_on=auto_calibration or "val",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
        auto_determine_val_set=auto_determine_val_set,
    )

    del pipeline
    del results

    # following models are evaluated:

    # Chronos-Bolt-*-FT_LoRA
    # Chronos-Bolt-*-FT_LoRA-PP_Offset
    # Chronos-Bolt-*-FT_LoRA-PP_QuantReg
    # Chronos-Bolt-*-FT_LoRA-PP_Gauss

    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/chronos-bolt-{chronos_variant}",
            "device_map": device_map,
            "lead_times": lead_times,
            "finetuning_schedule": ["lora"],
            "finetuning_hp_search": False,
            "finetuning_warmup_new_neurons": False,
            "finetuning_adjust_pretrained_prediction_length": False,
            "name": f"Chronos-Bolt-{chronos_variant}-FT_LoRA",
        },
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
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
        calibration_based_on=auto_calibration or "val",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
        auto_determine_val_set=auto_determine_val_set,
    )

    del pipeline
    del results

    # following models are evaluated:

    # Chronos-Bolt-*-FT_Full
    # Chronos-Bolt-*-FT_Full-PP_Offset
    # Chronos-Bolt-*-FT_Full-PP_QuantReg
    # Chronos-Bolt-*-FT_Full-PP_Gauss

    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/chronos-bolt-{chronos_variant}",
            "device_map": device_map,
            "lead_times": lead_times,
            "finetuning_schedule": ["full"],
            "finetuning_hp_search": False,
            "finetuning_warmup_new_neurons": False,
            "finetuning_adjust_pretrained_prediction_length": False,
            "name": f"Chronos-Bolt-{chronos_variant}-FT_Full",
        },
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
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
        calibration_based_on=auto_calibration or "val",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
        auto_determine_val_set=auto_determine_val_set,
    )

    del pipeline
    del results

    # ------------------------ Chronos-T5 ------------------------

    # following models are evaluated:

    # Chronos-T5-*
    # Chronos-T5-*-PP_Offset
    # Chronos-T5-*-PP_QuantReg
    # Chronos-T5-*-PP_Gauss

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
        calibration_based_on=auto_calibration or "train",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
        auto_determine_val_set=auto_determine_val_set,
    )

    del pipeline
    del results

    # ------- trained on adjusted prediction length ---------

    # following models are evaluated: 
    # Chronos-T5-*-FT_LongOut
    # Chronos-T5-*-FT_LongOut-PP_Offset
    # Chronos-T5-*-FT_LongOut-PP_QuantReg
    # Chronos-T5-*-FT_LongOut-PP_Gauss

    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/chronos-t5-{chronos_variant}",
            "device_map": device_map,
            "lead_times": lead_times,
            "finetuning_schedule": ["full"],
            "finetuning_hp_search": False,
            "finetuning_warmup_new_neurons": False,
            "finetuning_adjust_pretrained_prediction_length": True,
            "name": f"Chronos-T5-{chronos_variant}-FT_LongOut",
        },
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
        freq=freq,
        output_dir=output_dir / f"Chronos-T5-{chronos_variant}-FT_LongOut",
    )

    results = pipeline.backtest(
        data=data,
        test_start_date=test_start_date,
        rolling_window_eval=False,
        train=True,
        val_window_size=val_window_size,
        train_window_size=None,
        test_window_size=None,
        calibration_based_on=auto_calibration or "val",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
        auto_determine_val_set=auto_determine_val_set,
    )

    del pipeline
    del results

    # ------- trained on original 64 steps, not extended head ---------

    # following models are evaluated: 
    # Chronos-T5-*-FT_Full
    # Chronos-T5-*-FT_Full-PP_Offset
    # Chronos-T5-*-FT_Full-PP_QuantReg
    # Chronos-T5-*-FT_Full-PP_Gauss

    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/chronos-t5-{chronos_variant}",
            "device_map": device_map,
            "lead_times": lead_times,
            "finetuning_schedule": ["full"],
            "finetuning_hp_search": False,
            "finetuning_warmup_new_neurons": False,
            "finetuning_adjust_pretrained_prediction_length": False,
            "name": f"Chronos-T5-{chronos_variant}-FT_Full",
        },
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
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
        calibration_based_on=auto_calibration or "val",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
        auto_determine_val_set=auto_determine_val_set,
    )

    del pipeline
    del results

    # following models are evaluated: 
    # Chronos-T5-*-FT_Last
    # Chronos-T5-*-FT_Last-PP_Offset
    # Chronos-T5-*-FT_Last-PP_QuantReg
    # Chronos-T5-*-FT_Last-PP_Gauss

    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/chronos-t5-{chronos_variant}",
            "device_map": device_map,
            "lead_times": lead_times,
            "finetuning_schedule": ["last_layer"],
            "finetuning_hp_search": False,
            "finetuning_warmup_new_neurons": False,
            "finetuning_adjust_pretrained_prediction_length": False,
            "name": f"Chronos-T5-{chronos_variant}-FT_Last",
        },
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
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
        calibration_based_on=auto_calibration or "val",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
        auto_determine_val_set=auto_determine_val_set,
    )

    del pipeline
    del results

    # following models are evaluated: 
    # Chronos-T5-*-FT_LoRA
    # Chronos-T5-*-FT_LoRA-PP_Offset
    # Chronos-T5-*-FT_LoRA-PP_QuantReg
    # Chronos-T5-*-FT_LoRA-PP_Gauss

    pipeline = ForecastingPipeline(
        model=Chronos,
        model_kwargs={
            "pretrained_model_name_or_path": f"amazon/chronos-t5-{chronos_variant}",
            "device_map": device_map,
            "lead_times": lead_times,
            "finetuning_schedule": ["lora"],
            "finetuning_hp_search": False,
            "finetuning_warmup_new_neurons": False,
            "finetuning_adjust_pretrained_prediction_length": False,
            "name": f"Chronos-T5-{chronos_variant}-FT_LoRA",
        },
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
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
        calibration_based_on=auto_calibration or "val",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
        auto_determine_val_set=auto_determine_val_set,
    )

    del pipeline
    del results

    # ------------------------ RandomWalk ------------------------

    # following models are evaluated:
    # RandomWalk
    # RandomWalk-PP_Offset
    # RandomWalk-PP_QuantReg
    # RandomWalk-PP_Gauss
    pipeline = ForecastingPipeline(
        model=RandomWalk,
        model_kwargs={"quantiles": quantiles, "lead_times": lead_times, "name": "RandomWalk"},
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
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
        calibration_based_on=auto_calibration or "train_val",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
        auto_determine_val_set=False,
    )

    del pipeline
    del results

    # ------------------------ OnlineRandomWalk ------------------------

    # following models are evaluated:
    # OnlineRandomWalk
    # OnlineRandomWalk-PP_Offset
    # OnlineRandomWalk-PP_QuantReg
    # OnlineRandomWalk-PP_Gauss
    pipeline = ForecastingPipeline(
        model=OnlineRandomWalk,
        model_kwargs={"quantiles": quantiles, "lead_times": lead_times, "name": "OnlineRandomWalk"},
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
        output_dir=output_dir / "OnlineRandomWalk",
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
        calibration_based_on=auto_calibration or "train_val",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
        auto_determine_val_set=False,
    )

    del pipeline
    del results

    # ------------------------ SeasonalNaive ------------------------

    # following models are evaluated:
    # SeasonalNaive
    # SeasonalNaive-PP_Offset
    # SeasonalNaive-PP_QuantReg
    # SeasonalNaive-PP_Gauss
    pipeline = ForecastingPipeline(
        model=SeasonalNaive,
        model_kwargs={"quantiles": quantiles, "lead_times": lead_times, "freq": freq, "name": "SeasonalNaive"},
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
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
        calibration_based_on=auto_calibration or "train_val",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
        auto_determine_val_set=False,
    )

    del pipeline
    del results

    # ------------------------ SeasonalNaiveAg ------------------------

    # following models are evaluated:
    # SeasonalNaiveAg
    # SeasonalNaiveAg-PP_Offset
    # SeasonalNaiveAg-PP_QuantReg
    # SeasonalNaiveAg-PP_Gauss
    pipeline = ForecastingPipeline(
        model=SeasonalNaive_Ag,
        model_kwargs={"lead_times": lead_times, "freq": freq, "seasonal_period": seasonal_period, "name": "SeasonalNaiveAg"},
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
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
        calibration_based_on=auto_calibration or "train_val",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
        auto_determine_val_set=False,
    )

    del pipeline
    del results

    # ------------------------ PatchTST ------------------------

    # following models are evaluated:
    # PatchTST
    # PatchTST-PP_Offset
    # PatchTST-PP_QuantReg
    # PatchTST-PP_Gauss
    pipeline = ForecastingPipeline(
        model=PatchTST_Ag,
        model_kwargs={"lead_times": lead_times, "freq": freq, "name": "PatchTST"},
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
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
        calibration_based_on=auto_calibration or "train_val",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
        auto_determine_val_set=False,
    )

    del pipeline
    del results

    # ------------------------ TiDE ------------------------

    # following models are evaluated:
    # TiDE
    # TiDE-PP_Offset
    # TiDE-PP_QuantReg
    # TiDE-PP_Gauss
    pipeline = ForecastingPipeline(
        model=TiDE_Ag,
        model_kwargs={"lead_times": lead_times, "freq": freq, "name": "TiDE"},
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
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
        calibration_based_on=auto_calibration or "train_val",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
        auto_determine_val_set=False,
    )   

    del pipeline
    del results

    # ------------------------ TiRex ------------------------

    # following models are evaluated:
    # TiRex
    # TiRex-PP_Offset
    # TiRex-PP_QuantReg
    # TiRex-PP_Gauss

    pipeline = ForecastingPipeline(
        model=TiRex,
        model_kwargs={
            "lead_times": lead_times,
            "tirex_service_url": "http://localhost:8000",
            "name": "TiRex",
        },
        postprocessors=postprocessors,
        postprocessor_kwargs=postprocessor_kwargs,
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
        calibration_based_on=auto_calibration or "train",
        save_results=True,
        test_window_step=test_window_step,
        calibration_window_step=calibration_window_step,
        auto_determine_val_set=False,
    )

    del pipeline
    del results


if __name__ == "__main__":
    evaluate()
