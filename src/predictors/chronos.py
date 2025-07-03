from tqdm import tqdm
import torch
from chronos import BaseChronosPipeline
from chronos.chronos_bolt import ChronosBoltPipeline
from chronos.chronos import ChronosPipeline, ChronosTokenizer
from autogluon.timeseries import TimeSeriesDataFrame
from torch.utils.data import DataLoader
from typing import Callable, List, Optional, Dict, Any, Literal, Union
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from src.core.base import AbstractPredictor
from src.core.timeseries_evaluation import TARGET, ITEMID, TIMESTAMP
import logging
from src.core.timeseries_evaluation import ForecastCollection, TimeSeriesForecast, HorizonForecast
from optuna.trial import Trial
from pathlib import Path
from transformers.trainer import TrainingArguments
from peft import PeftModel
import json
from transformers.trainer import Trainer
from transformers import PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainerCallback, EarlyStoppingCallback, TrainerState, TrainerControl
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")


class ChronosLoraConfig(LoraConfig):
    """Chronos t5 lora configuration"""

    def __init__(self, prediction_length=None, **kwargs):
        super().__init__(**kwargs)
        self.prediction_length = prediction_length

    def to_dict(self):
        base = LoraConfig.to_dict(self)
        base["prediction_length"] = self.prediction_length
        return base


class ChronosInferenceDataset(Dataset):
    """A dataset for inference with time series data.

    This dataset extracts fixed-length context windows from time series data
    for inference tasks.

    Args:
        target_df (TimeSeriesDataFrame): The time series data containing target values.
        context_length (int): The number of time steps to use as context.
        target_column (str, optional): The column name containing the target values. Defaults to "target".
    """

    def __init__(
        self,
        target_df: TimeSeriesDataFrame,
        context_length: int,
        target_column: str = "target",
    ):
        assert context_length > 0, "context_length must be greater than 0"
        self.context_length = context_length
        self.target_array = target_df[target_column].to_numpy(dtype=np.float32)
        self.freq = target_df.freq

        # Store pointer to start:end of each time series
        cum_sizes = target_df.num_timesteps_per_item().values.cumsum()
        self.indptr = np.append(0, cum_sizes).astype(np.int32)

    def __len__(self):
        """Returns the number of time series in the dataset."""
        return len(self.indptr) - 1

    def _get_context(self, a: np.ndarray, pad_value=np.nan):
        """Extracts the context window, padding with a specified value if needed."""
        a = a[-self.context_length :]
        pad_size = self.context_length - len(a)
        if pad_size > 0:
            pad = np.full(shape=(pad_size,), fill_value=pad_value)
            a = np.concatenate((pad, a))
        return a

    def __getitem__(self, idx) -> np.ndarray:
        """Retrieves the context window for the given index."""
        start_idx = self.indptr[idx]
        end_idx = self.indptr[idx + 1]
        return self._get_context(self.target_array[start_idx:end_idx])


class ChronosBacktestingDataset(Dataset):
    """A dataset for backtesting with time series data.

    This dataset extracts historical context windows for backtesting purposes.

    Args:
        data (TimeSeriesDataFrame): The time series data containing target values.
        context_length (int): The number of time steps to use as context.
        target_column (str, optional): The column name containing the target values. Defaults to "target".
    """

    def __init__(
        self,
        data: TimeSeriesDataFrame,
        context_length: int,
        target_column: str = "target",
        return_target: bool = False,
        prediction_length: Optional[int] = None,
        tokenizer: Optional["ChronosTokenizer"] = None,
    ):
        assert context_length > 0, "context_length must be greater than 0"
        self.context_length = context_length

        self.return_target = return_target
        self.prediction_length = prediction_length
        self.tokenizer = tokenizer

        if self.return_target:
            if self.prediction_length is None:
                raise ValueError("prediction_length needs to be specified if return target is set to true.")
            # when target should be returned, the dataset is used for training/evaluation and we should reorder based on timestamps
            data = data.sort_values(by=[ITEMID, TIMESTAMP])

        self.target_array = data[target_column].to_numpy(dtype=np.float32)
        self.freq = data.freq
        self.item_ids = pd.factorize(data.index.get_level_values(ITEMID))[0]
        cum_sizes = data.num_timesteps_per_item().values.cumsum()
        self.indptr = np.append(0, cum_sizes).astype(np.int32)
        self.item_ids_mask = {item_id: self.item_ids == item_id for item_id in np.unique(self.item_ids)}

    def __len__(self):
        """Returns the total number of time steps in the dataset."""
        return len(self.target_array)

    def _get_context(self, a: np.ndarray, pad_value=np.nan):
        """Extracts the context window, padding with a specified value if needed."""
        a = a[-self.context_length :]
        pad_size = self.context_length - len(a)
        if pad_size > 0:
            pad = np.full(shape=(pad_size,), fill_value=pad_value)
            a = np.concatenate((pad, a))
        return a.astype(np.float32)

    def _get_future_targets(self, a: np.ndarray, pad_value=np.nan):
        """Extracts the future targets, padding with a specified value if needed."""
        a = a[: self.prediction_length]
        pad_size = self.prediction_length - len(a)
        if pad_size > 0:
            pad = np.full(shape=(pad_size,), fill_value=pad_value)
            a = np.concatenate((a, pad))
        return a.astype(np.float32)

    def to_chronos_format(self, context: np.ndarray, future_target: np.ndarray):

        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(torch.tensor(context).unsqueeze(0))
        labels, labels_mask = self.tokenizer.label_input_transform(torch.tensor(future_target).unsqueeze(0), scale)
        labels[labels_mask == 0] = -100

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    def to_chronos_bolt_format(self, context: np.ndarray, future_target: np.ndarray):
        return {"context": context, "target": future_target}

    def __getitem__(self, idx) -> np.ndarray:
        """Retrieves the context window for the given index within its corresponding time series."""

        item_id = self.item_ids[idx]
        item_id_start_idx = self.indptr[item_id]
        # idx in the target_array controlled for the item_id_start_idx
        start_idx = idx - item_id_start_idx
        # get array of corresponding item id
        target_sub_array = self.target_array[self.item_ids_mask[item_id]]
        context = self._get_context(target_sub_array[: start_idx + 1])

        if self.return_target:
            future_target = self._get_future_targets(target_sub_array[start_idx + 1 :])

            if self.tokenizer is not None:
                return self.to_chronos_format(context, future_target)
            else:
                return self.to_chronos_bolt_format(context, future_target)

        return context


class Chronos(AbstractPredictor):
    """
    Chronos time series predictor using a pretrained Chronos pipeline (from HuggingFace).

    This class implements prediction logic based on a fixed context window and multiple lead times.

    Parameters
    ----------
    pretrained_model_name_or_path : str or Path, optional
        Name or path of the chronos model. Defaults to "amazon/chronos-bolt-tiny".
    device_map : str, optional
        Device to run inference on, e.g., "cpu", "cuda", or "mps". Defaults to "mps".
    context_length : int, optional
        Number of timesteps used as context for prediction. Defaults to 2048.
    lead_times : List[int], optional
        List of prediction steps ahead (lead times). Defaults to [1, 2, 3].
    sampling: bool, optional
        Whether to sample multiple trajectories. Defaults to False.
    freq : pd.Timedelta, optional
        Frequency of the time series data. Defaults to 1 hour.
    finetuning_type : {"full", "last_layer", "LoRa"}, optional
        Type of fine-tuning to apply. Defaults to "full".
    finetuning_adjust_pretrained_prediction_length : bool, defaults to True
        Whether the original pretrained prediction length should be overwritten.
        This involves changing the number of output neurons in case of chronos-bolt. Defaults to true.
    finetuning_warmup_new_neurons : bool, defaults to True
        Whether to first fine tune the model on the newely added neurons.
        Only relevant if finetuning_adjust_pretrained_prediction_length is set to True
    finetuning_hp_search : bool, optional
        Whether to perform hyperparameter search during fine-tuning. Defaults to False.
    finetuning_hp_search_trials : int, optional
        Number of trials for hyperparameter search. Defaults to 10.
    output_dir : Path, optional
        Directory to store results. Defaults to Path("./models/").
    """

    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path] = "amazon/chronos-bolt-tiny",
        device_map: str = "mps",
        context_length: int = 2048,
        lead_times: List[int] = [1, 2, 3],
        sampling: bool = False,
        freq: Union[pd.Timedelta, pd.DateOffset] = pd.Timedelta("1h"),
        finetuning_type: Literal["full", "last_layer", "LoRa"] = "full",
        finetuning_adjust_pretrained_prediction_length: bool = True,
        finetuning_hp_search: Optional[bool] = False,
        finetuning_hp_search_trials: Optional[int] = 10,
        finetuning_warmup_new_neurons: bool = True,
        output_dir: Optional[Path] = Path("./models/"),
    ) -> None:
        super().__init__(lead_times, freq, output_dir)
        self.context_length = context_length
        self.prediction_length = max(self.lead_times)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.base_model_name = None
        self.device_map = device_map
        self.finetuning_type = finetuning_type
        self.finetuning_adjust_pretrained_prediction_length = finetuning_adjust_pretrained_prediction_length
        self.finetuning_hp_search = finetuning_hp_search
        self.finetuning_hp_search_trials = finetuning_hp_search_trials
        self.lora = False
        self.finetuning_warmup_new_neurons = finetuning_warmup_new_neurons

        self.quantiles = np.arange(0.1, 1, 0.1).round(1)
        # if self.prediction_length > 64:
        #    logging.error("Maximum supported lead time is 64 currently.")
        #    raise ValueError("Maximum supported lead time is 64 currently.")

        self.pipeline = self._pipeline_init(self.pretrained_model_name_or_path)

        if isinstance(self.pipeline, ChronosBoltPipeline):
            if self.context_length > 2048:
                logging.info("Context length detected of: %s. Adapt context length to maximum of 2048", self.context_length)
                self.context_length = 2048
            if sampling:
                logging.info("Sampling is turned on. Chronos-bolt will sample multiple trajectories to compute quantiles for prediction lengths >64.")
            self.sampling = sampling
        elif isinstance(self.pipeline, ChronosPipeline):
            if self.context_length > 512:
                logging.info("Context length detected of: %s. Adapt context length to maximum of 512", self.context_length)
                self.context_length = 512
            if sampling:
                logging.warning("Sampling does not need to be explicitly enabled for chronos-t5. It uses sampling by default.")
            self.sampling = False
        else:
            raise ValueError("Unknown base_model_name: %s. Either needs to contain 'chronos-t5' or 'chronos-bolt'.", self.base_model_name)

    def _pipeline_init(self, pretrained_model_name_or_path: Union[str, Path]) -> BaseChronosPipeline:
        """Creates and returns an instance of the Chronos pipeline."""

        logging.info("Loading Chronos pipeline from model: %s", pretrained_model_name_or_path)

        # add lora weights if adapter_config exists in directory
        if (Path(pretrained_model_name_or_path) / "adapter_config.json").exists():
            logging.info(f"Found LoRa configuration in {pretrained_model_name_or_path}.")

            with open(Path(pretrained_model_name_or_path) / "adapter_config.json", "r") as f:
                adapter_config: dict = json.load(f)

            base_model_name = adapter_config.get("base_model_name_or_path")
            logging.info("Base model name: %s", base_model_name)

            logging.info("Initializing Chronos pipeline with model: %s", base_model_name)
            pipeline = BaseChronosPipeline.from_pretrained(base_model_name, device_map=self.device_map)

            # update prediction length of base model based on lora configuration
            # TODO: this is a hack. it produces a warning, that there is an unexpected keyword argument. Should get fixed
            if isinstance(pipeline, ChronosPipeline):
                pred_length = adapter_config.get("prediction_length")
                logging.info("Setting prediction length of chronos-t5 to %s based on LoRa configuration.", pred_length)
                pipeline.inner_model.config.prediction_length = pred_length
                pipeline.inner_model.config.chronos_config["prediction_length"] = pred_length
                pipeline.model.config.prediction_length = pred_length

            # Apply LoRA adapters
            pipeline.inner_model = PeftModel.from_pretrained(pipeline.inner_model, pretrained_model_name_or_path, is_trainable=False)
            self.lora = True
            logging.info("LoRa adapters applied successfully.")

        else:
            logging.info("Initializing Chronos pipeline with model: %s", pretrained_model_name_or_path)
            pipeline = BaseChronosPipeline.from_pretrained(pretrained_model_name_or_path, device_map=self.device_map)

        # pipeline = resize_chronos_bolt_output_layers(pipeline, self.prediction_length)

        return pipeline

    def _fit(self, data_train: TimeSeriesDataFrame, data_val: Optional[TimeSeriesDataFrame] = None) -> None:
        """
        Finetuning chronos model

        Parameters
        ----------
            data_train (TimeSeriesDataFrame): Training data (not used).
            data_val (TimeSeriesDataFrame): Evaluation data (optional).
        """
        def _build_model(
            source: Union[str, Path],  # name or ckpt dir
            mode: Literal["full", "last_layer", "LoRa", "new_rows"],
        ) -> PreTrainedModel:
            """Helper that creates a pipeline (optionally from a checkpoint) and prepares it according to `mode`"""

            pipe = self._pipeline_init(source)

            # freeze
            for p in pipe.inner_model.parameters():
                p.requires_grad = False

            # resize head if requested (only for Bolt)
            if isinstance(pipe, ChronosBoltPipeline) and self.finetuning_adjust_pretrained_prediction_length:
                unfreeze_new = mode == "new_rows"
                pipe = resize_chronos_bolt_output_layers(pipe, self.prediction_length, unfreeze_new_neurons=unfreeze_new)

            # unfreeze
            if mode == "full":
                for p in pipe.inner_model.parameters():
                    p.requires_grad = True

            elif mode == "last_layer":
                if isinstance(pipe, ChronosPipeline):
                    for p in pipe.inner_model.lm_head.parameters():
                        p.requires_grad = True
                else:  # Bolt
                    head = pipe.inner_model.output_patch_embedding
                    for m in (head.output_layer, head.residual_layer):
                        for p in m.parameters():
                            p.requires_grad = True

            elif mode == "new_rows":
                # nothing extra to do – resize_chronos_bolt_output_layers already
                # attached the gradient mask and left requires_grad=True
                pass

            elif mode == "LoRa":
                # attach LoRA adapters (all original params stay frozen)
                if isinstance(pipe, ChronosPipeline):
                    lcfg = ChronosLoraConfig(
                        prediction_length=self.prediction_length,
                        r=8,
                        lora_alpha=8,
                        lora_dropout=0.0,
                        target_modules=["q", "k", "v"],
                        bias="none",
                        task_type=TaskType.SEQ_2_SEQ_LM,
                    )
                else:
                    lcfg = LoraConfig(
                        task_type=None,
                        inference_mode=False,
                        r=8,
                        lora_alpha=8,
                        lora_dropout=0.0,
                        target_modules=["q", "k", "v"],
                    )
                pipe.inner_model = get_peft_model(pipe.inner_model, lcfg)

            else:
                raise ValueError(f"unknown mode {mode}")

            print_trainable_params(pipe.inner_model)
            return pipe.inner_model

        # convenience wrappers for Trainer
        def init_full():
            return _build_model(self.pretrained_model_name_or_path, "full")

        def init_last():
            return _build_model(self.pretrained_model_name_or_path, "last_layer")

        def init_lora():
            return _build_model(self.pretrained_model_name_or_path, "LoRa")

        def init_new_rows():
            return _build_model(self.pretrained_model_name_or_path, "new_rows")

        model_inits = {"full": init_full, "last_layer": init_last, "LoRa": init_lora}

        # Ensure config.prediction_length is up-to-date for T5 (no head resize)
        if self.finetuning_adjust_pretrained_prediction_length:
            prediction_length = self.prediction_length
            self.pipeline.model.config.prediction_length = prediction_length
            self.pipeline.inner_model.config.chronos_config["prediction_length"] = prediction_length
        else:
            prediction_length = self.pipeline.inner_model.config.chronos_config["prediction_length"]  # standard

        logging.info("Prediction length will be set to %s during training.", self.prediction_length)

        # 1) optional warm-up
        warm_ckpt: Optional[Path] = None

        if isinstance(self.pipeline, ChronosBoltPipeline) and self.finetuning_adjust_pretrained_prediction_length and self.finetuning_warmup_new_neurons:
            warm_dir = self.output_dir / "warmup-new-neurons"
            logging.info(">>> Warm-up: training only new output neurons …")

            fine_tune(
                model_init=init_new_rows,
                data_train=data_train,
                data_val=data_val,
                output_dir=warm_dir,
                hp_tuning=False,
                context_length=self.context_length,
                prediction_length=prediction_length,
                tokenizer=getattr(self.pipeline, "tokenizer", None),
                specific_train_kwargs={"learning_rate": 1e-4, "num_train_epochs": 1, "warmup_ratio": 0.1},
            )
            warm_ckpt = warm_dir / "fine-tuned-ckpt"
            logging.info("Warm-up finished, best checkpoint at %s", warm_ckpt)

        # 2) main fine-tune
        final_dir = self.output_dir / f"finetuned-{self.finetuning_type}"
        logging.info(">>> Main fine-tuning (%s) …", self.finetuning_type)

        if warm_ckpt is not None:
            model_init_main = lambda: _build_model(warm_ckpt, self.finetuning_type)
        else:
            model_init_main = model_inits[self.finetuning_type]

        fine_tune(
            model_init=model_init_main,
            data_train=data_train,
            data_val=data_val,
            output_dir=final_dir,
            hp_tuning=self.finetuning_hp_search,
            n_trials=self.finetuning_hp_search_trials,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            tokenizer=getattr(self.pipeline, "tokenizer", None),
            specific_train_kwargs={"num_train_epochs": 3},
        )

        # reload final model
        self.pipeline = self._pipeline_init(final_dir / "fine-tuned-ckpt")
        logging.info("Two-stage fine-tuning complete – model reloaded.")

    def predict(
        self,
        data: TimeSeriesDataFrame,
        previous_context_data: Optional[TimeSeriesDataFrame] = None,
        predict_only_last_timestep: bool = False,
    ) -> ForecastCollection:
        """Predicts future values for the given time series data using a pretrained Chronos model.

        Parameters:
            data (TimeSeriesDataFrame): The target data to forecast.
            predict_only_last_timestep (bool): Whether to forecast only the last timestep of each series.
            previous_context_data (Optional[TimeSeriesDataFrame]): Optional preceding data to provide context.

        Returns:
            PredictionCollection: A nested dict structure holding item_id -> lead_time -> TimeSeriesForecast.
        """

        if previous_context_data is not None:
            data_merged = self._merge_data(data, previous_context_data, self.context_length)
        else:
            data_merged = data

        if predict_only_last_timestep:
            ds = ChronosInferenceDataset(data_merged, self.context_length)
            data = data.slice_by_timestep(start_index=-1)
        else:
            ds = ChronosBacktestingDataset(data_merged, self.context_length)

        dl = DataLoader(ds, batch_size=16)

        forecasts = []
        for batch in tqdm(dl, desc="Predicting using Chronos"):
            # TODO: make this nicer by using chronos predict_quantiles function directly
            # explicit sampling is done only for chronos-bolt. chronos-t5 does that by default behaviour
            if self.sampling:
                forecast = self.pipeline.predict_sampling(context=batch, prediction_length=self.prediction_length)
            else:
                forecast = self.pipeline.predict(context=batch, prediction_length=self.prediction_length)

                # chronos-t5 forecast output shape: [batch_size, num_trajectories, prediction_length]
                if isinstance(self.pipeline, ChronosPipeline):
                    forecast = torch.quantile(forecast, q=torch.tensor(self.quantiles, dtype=forecast.dtype), dim=1).swapaxes(1, 0)

            forecasts.append(forecast)
        forecasts = torch.vstack(forecasts)  #  output shape: [batch_size, quantiles, prediction_length]

        if not predict_only_last_timestep:
            mask = data_merged.index.isin(data.index)
            forecasts = forecasts[mask, ...]

        ts_forecast: Dict[int, TimeSeriesForecast] = {}

        for item_id in data.item_ids:
            lt_forcast: Dict[int, HorizonForecast] = {}
            item_mask = data.index.get_level_values("item_id") == item_id
            for lt in self.lead_times:
                lt_forcast[lt] = HorizonForecast(
                    lead_time=lt,
                    predictions=forecasts[item_mask, :, lt - 1],
                )
            ts_forecast[item_id] = TimeSeriesForecast(item_id=item_id, lead_time_forecasts=lt_forcast, data=data.loc[item_mask].copy(), freq=self.freq)

        return ForecastCollection(item_ids=ts_forecast)


############## Functions for fine tuning and hp optimization ##############


class BestCheckpointCallback(TrainerCallback):
    """Callback to save best model checkpoint during hyperparameter search."""

    def __init__(self, metric_name="eval_loss", greater_is_better=False):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        if self.metric_name not in metrics:
            return

        metric_value = metrics[self.metric_name]

        # Check if current checkpoint is better than previous best
        is_better = (self.greater_is_better and metric_value > self.best_metric) or (not self.greater_is_better and metric_value < self.best_metric)

        if is_better:
            self.best_metric = metric_value
            self.best_checkpoint = state.global_step

    def on_train_begin(self, args, state, control, **kwargs):
        self.best_checkpoint = None
        self.best_metric = np.inf if not self.greater_is_better else -np.inf

    def get_best_metric(self):
        return self.best_metric

    def get_best_checkpoint(self):
        return self.best_checkpoint


# TODO: implement a trainer class for that
def fine_tune(
    model_init: Callable[[], PreTrainedModel],
    data_train: TimeSeriesDataFrame,
    data_val: Optional[TimeSeriesDataFrame] = None,
    output_dir: Union[str, Path] = Path("./models/test-finetuning/"),
    hp_tuning: bool = False,
    n_trials: Optional[int] = None,
    context_length: int = 2048,
    prediction_length: int = 64,
    tokenizer: Optional["ChronosTokenizer"] = None,
    specific_train_kwargs: Dict = {},
):
    """
    Fine-tune a Chronos Bolt model (or other Hugging Face PreTrainedModel) on time series data.

    Parameters
    ----------
    model_init : Callable[[], PreTrainedModel]
        A function that returns a fresh instance of the model to fine-tune.
    data_train : TimeSeriesDataFrame
        Training data in Chronos-compatible format.
    data_val : Optional[TimeSeriesDataFrame], default=None
        Validation data. Required if `hp_tuning` is True or if evaluation during training is desired.
    output_dir : Union[str, Path], default=Path("./models/test-full-finetuning/")
        Path to save the model and optionally intermediate checkpoints.
    hp_tuning : bool, default=False
        Whether to perform hyperparameter tuning using Optuna.
    n_trials : Optional[int], default=None
        Number of Optuna trials. Required if `hp_tuning` is True.
    context_length : int, default=2048
        Context length the model sees during training.
    prediction_length : Optional[int], default=None
        Number of timestamps the model is required to predict in the future.
    specific_train_kwargs : Dict, default={},
        Additional training arguments to include. Override default values
    """

    def create_callbacks():
        callbacks = [BestCheckpointCallback()]
        if data_val is not None:
            patience = 3
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
            logging.info("Validation data is available, setting early_stopping_patience=%s", patience)
        return callbacks

    logging.info("Preparing training dataset...")
    train_dataset = ChronosBacktestingDataset(
        data=data_train,
        context_length=context_length,
        target_column=TARGET,
        return_target=True,
        prediction_length=prediction_length,
        tokenizer=tokenizer,
    )

    eval_dataset = None
    if data_val is not None:
        logging.info("Preparing validation dataset...")
        eval_dataset = ChronosBacktestingDataset(
            data=data_val,
            context_length=context_length,
            target_column=TARGET,
            return_target=True,
            prediction_length=prediction_length,
            tokenizer=tokenizer,
        )

    # Create separate directory for final training
    final_training_path = output_dir / "training"
    final_training_path.mkdir(exist_ok=True, parents=True)

    # add specific train args for chronos bolt
    if tokenizer is None:
        specific_train_kwargs.update({"label_names": [TARGET]})

    # Create args for final training with best hyperparameters
    fine_tune_trainer_kwargs = build_train_args(
        base_path=final_training_path,
        eval_during_ft=data_val is not None,
        save_checkpoints=True,
        pipeline_kwargs=specific_train_kwargs,
    )

    logging.info("Training results are going to be logged in tensorboard.")
    logging.info(f"Run `tensorboard --logdir {output_dir}` in the terminal to start.")

    if hp_tuning:
        if data_val is None:
            logging.error("Validation data is required for hyperparameter tuning.")
            raise ValueError("Validation data is required for hyperparameter tuning.")
        if n_trials is None:
            logging.error("n_trials must be specified when hp_tuning is enabled.")
            raise ValueError("n_trials must be specified when hp_tuning is enabled.")

        # Create separate path for hyperparameter tuning logs
        hp_tuning_path = output_dir / "hp_tuning"
        hp_tuning_path.mkdir(exist_ok=True, parents=True)

        # Args for hyperparameter tuning phase
        hp_tuning_args = build_train_args(
            base_path=hp_tuning_path,
            eval_during_ft=data_val is not None,
            save_checkpoints=True,
            pipeline_kwargs=specific_train_kwargs,
        )

        logging.info("Starting hyperparameter tuning with optuna (%s trials)...", n_trials)
        logging.debug(f"Hyperparameter tuning args: {hp_tuning_args}")

        hp_trainer = Trainer(
            model_init=model_init,
            args=hp_tuning_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=create_callbacks(),
        )

        # Perform hyperparameter tuning with Optuna
        best_hp_args = tune_hp_optuna(hp_trainer, hp_space_optuna, n_trials=n_trials)

        # Apply best hyperparameters to final training args
        for key, value in best_hp_args.__dict__.items():
            if not key.startswith("_") and not key.endswith("dir"):
                setattr(fine_tune_trainer_kwargs, key, value)

        logging.info("Hyperparameter tuning completed.")

    logging.info("Final training hyperparameters:")
    logging.info(fine_tune_trainer_kwargs)

    trainer = Trainer(
        model_init=model_init,
        args=fine_tune_trainer_kwargs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=create_callbacks(),
    )

    # save training args
    with open(final_training_path / "trainer_args.json", "w") as f:
        f.write(fine_tune_trainer_kwargs.to_json_string())

    logging.info("Starting final training process...")
    trainer.train()

    # Save the fine-tuned model to the specified output directory
    final_model_path = output_dir / "fine-tuned-ckpt"
    trainer.model.save_pretrained(final_model_path)
    logging.info("Saved fine-tuned model to %s.", final_model_path)


def tune_hp_optuna(trainer: Trainer, hp_space_optuna: Dict[str, Any], n_trials: int = 10):
    """Run hyperparameter search with optuna using the best checkpoint for each trial"""

    def custom_compute_objective(metrics):
        """Use the best metric seen during this trial, not the last one. Otherwise optuna will always pick the last model instead of the best checkpoint.

        It gets that based on the `BestCheckpointCallback` callback's best metric"""

        best_checkpoint_callback = None
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, BestCheckpointCallback):
                best_checkpoint_callback = callback
                break

        return best_checkpoint_callback.get_best_metric()

    best_run = trainer.hyperparameter_search(
        direction="minimize",
        hp_space=hp_space_optuna,
        compute_objective=custom_compute_objective,
        n_trials=n_trials,
        backend="optuna",
    )

    # Look at best run
    logging.info("Best configuration: %s", best_run)

    # Update training args with best hyperparameters
    for key, value in best_run.hyperparameters.items():
        setattr(trainer.args, key, value)

    return trainer.args


def hp_space_optuna(trial: Trial):
    """Define search space for hyperparameter search"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-2, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64, 128]),
        # "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
        # "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
    }


def build_train_args(
    *,
    base_path: Path,
    save_checkpoints: bool = True,
    eval_during_ft: bool = True,
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
) -> TrainingArguments:
    """
    Construct `transformers.TrainingArguments` from defaults + pipeline_kwargs
    """

    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    epochs = 3
    eval_ratio = 0.1 / epochs
    logging_steps = 0.05 / epochs
    log_dir = base_path / "logs"

    fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 0)

    defaults = dict(
        output_dir=base_path,
        overwrite_output_dir=False,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        weight_decay=0.0,
        optim="adamw_torch_fused",
        logging_dir=log_dir,
        logging_strategy="steps",
        logging_steps=logging_steps,
        disable_tqdm=True,
        num_train_epochs=epochs,
        gradient_accumulation_steps=1,
        dataloader_num_workers=4,
        tf32=False,
        fp16=fp16,
        report_to="tensorboard",
        prediction_loss_only=True,
        save_strategy="steps" if save_checkpoints else "no",
        save_steps=eval_ratio if save_checkpoints else None,
        save_only_model=True,
        save_total_limit=5,
        eval_strategy="steps" if eval_during_ft else "no",
        eval_steps=eval_ratio if eval_during_ft else None,
        eval_on_start=eval_during_ft,
        load_best_model_at_end=eval_during_ft,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        use_cpu=False,
    )

    merged = {**defaults, **pipeline_kwargs}
    return TrainingArguments(**merged)


def check_model_parameters(chronos: Chronos, model_name: str = "amazon/chronos-bolt-tiny"):
    """Helper function to verify, if weights have changed."""

    chronos_copy = Chronos(model_name=model_name, device_map="mps", lead_times=np.arange(1, 65), freq=pd.Timedelta("1h"))

    for (name1, p1), (name2, p2) in zip(chronos_copy.pipeline.inner_model.named_parameters(), chronos.pipeline.inner_model.named_parameters()):
        if not torch.equal(p1, p2):
            logging.info(f"Parameter %s has changed!", name1)


def print_trainable_params(model: PreTrainedModel) -> None:
    """Computes fraction of trainable params for a PreTrainedModel"""

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    fraction_trainable_params = trainable_params / total_params

    fraction_trainable_params = np.round(fraction_trainable_params * 100, 2)

    print(f"trainable params: {trainable_params} || all params: {total_params} || trainable%: {fraction_trainable_params}")


def _resize_proj(old_linear: nn.Linear, num_quantiles: int, old_H: int, new_H: int, unfreeze_new_neurons: bool) -> nn.Linear:
    """
    Resize a projection that is laid out (Q · H, in_dim) in row-major order.

    We first view it as (Q, H, in_dim) or (Q, H) for bias, manipulate the
    horizon axis, then restore the flattened (Q · H, …) shape.
    """
    in_dim = old_linear.in_features
    device = old_linear.weight.device
    dtype = old_linear.weight.dtype
    has_bias = old_linear.bias is not None

    # split into (Q, H, in_dim)
    W = old_linear.weight.data.view(num_quantiles, old_H, in_dim)
    b = old_linear.bias.data.view(num_quantiles, old_H) if has_bias else None

    # build new tensors
    if new_H == old_H:  # nothing to do
        new_W = W
        new_b = b
    elif new_H < old_H:  # truncate
        new_W = W[:, :new_H]
        new_b = b[:, :new_H] if has_bias else None
    else:  # extend — repeat last horizon
        reps = new_H - old_H
        extra_W = W[:, -1:].expand(-1, reps, -1)
        new_W = torch.cat([W, extra_W], dim=1)

        if has_bias:
            extra_b = b[:, -1:].expand(-1, reps)
            new_b = torch.cat([b, extra_b], dim=1)
        else:
            new_b = None

    # flatten back & create new Linear
    new_linear = nn.Linear(in_dim, num_quantiles * new_H, bias=has_bias).to(device, dtype)
    new_linear.weight.data.copy_(new_W.reshape(num_quantiles * new_H, in_dim))
    if has_bias:
        new_linear.bias.data.copy_(new_b.reshape(num_quantiles * new_H))

    # gradient mask: only horizons ≥ old_H
    if unfreeze_new_neurons and new_H > old_H:
        # shape (Q, H) -> True for newly-added horizons
        mask_2d = torch.zeros(num_quantiles, new_H, dtype=dtype, device=device)
        mask_2d[:, old_H:] = 1.0
        flat_mask = mask_2d.reshape(-1)  # (Q·H,)

        def mask_grad_weight(grad):
            return grad * flat_mask.unsqueeze(1)  # broadcast to (Q·H, in_dim)

        def mask_grad_bias(grad):
            return grad * flat_mask

        new_linear.weight.register_hook(mask_grad_weight)
        if has_bias:
            new_linear.bias.register_hook(mask_grad_bias)

    return new_linear


def resize_chronos_bolt_output_layers(pipeline, new_prediction_length: int, unfreeze_new_neurons: bool = False):
    """
    In-place resize of Chronos-Bolt’s output head so that the first
    `old_prediction_length` horizons stay identical.

    Parameters
    ----------
    pipeline : ChronosBoltPipeline
    new_prediction_length : int
        Desired forecast horizon (H).  Can be >, <, or == the current one.
    unfreeze_new_neurons : bool
        Whether to explicitly unfreeze the weights and biases for the added neurons, Defaults to False.
    """
    model = pipeline.inner_model
    rb = model.output_patch_embedding  # ResidualBlock

    old_H = model.chronos_config.prediction_length
    if new_prediction_length == old_H:
        logging.info("Prediction length unchanged (%d); nothing to do.", old_H)
        return pipeline

    num_quantiles = len(model.chronos_config.quantiles)

    # 1) resize output_layer  (d_ff → Q · H)
    # 2) resize residual_layer(d_model → Q · H)
    rb.output_layer = _resize_proj(
        rb.output_layer,
        num_quantiles,
        old_H,
        new_prediction_length,
        unfreeze_new_neurons,
    )
    rb.residual_layer = _resize_proj(
        rb.residual_layer,
        num_quantiles,
        old_H,
        new_prediction_length,
        unfreeze_new_neurons,
    )

    # layer-norm (if enabled)
    if getattr(rb, "use_layer_norm", False):
        rb.layer_norm = nn.modules.normalization.T5LayerNorm(
            num_quantiles * new_prediction_length,
            eps=rb.layer_norm.variance_epsilon,
        ).to(rb.layer_norm.weight.device, rb.layer_norm.weight.dtype)

    # Update every place where the horizon length lives in the config
    model.chronos_config.prediction_length = new_prediction_length
    model.config.chronos_config["prediction_length"] = new_prediction_length
    # Some checkpoints also duplicate it here:
    if hasattr(model.config, "prediction_length"):
        model.config.prediction_length = new_prediction_length

    logging.info(
        "Resized Chronos-Bolt head from %d to %d steps (%d → %d parameters each).",
        old_H,
        new_prediction_length,
        num_quantiles * old_H,
        num_quantiles * new_prediction_length,
    )
    return pipeline
