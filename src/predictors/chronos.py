from tqdm import tqdm
import torch
from chronos import BaseChronosPipeline
from autogluon.timeseries import TimeSeriesDataFrame
from torch.utils.data import DataLoader
from typing import Callable, List, Optional, Dict, Any
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from src.core.base import AbstractPredictor
from src.core.timeseries_evaluation import PredictionLeadTimes, PredictionLeadTime
from optuna.trial import Trial
from pathlib import Path
from transformers.trainer import TrainingArguments
from peft import PeftModel
import json
from transformers.trainer import Trainer
from transformers import PreTrainedModel
from peft import get_peft_model, LoraConfig
from transformers import TrainerCallback, EarlyStoppingCallback, TrainerState, TrainerControl


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
    ):
        assert context_length > 0, "context_length must be greater than 0"
        self.context_length = context_length

        self.return_target = return_target
        self.prediction_length = prediction_length

        if self.return_target:
            if self.prediction_length is None:
                raise ValueError("prediction_length needs to be specified if return target is set to true.")
            # when target should be returned, the dataset is used for training/evaluation and we should reorder based on timestamps
            # data = data.sort_values(by=["timestamp","item_id"])

        self.target_array = data[target_column].to_numpy(dtype=np.float32)
        self.freq = data.freq
        self.item_ids = data.index.get_level_values("item_id").to_numpy()
        cum_sizes = data.num_timesteps_per_item().values.cumsum()
        self.indptr = np.append(0, cum_sizes).astype(np.int32)
        self.item_ids_mask = {item_id: self.item_ids == item_id for item_id in self.item_ids}

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

    def __getitem__(self, idx) -> np.ndarray:
        """Retrieves the context window for the given index within its corresponding time series."""
        item_id = self.item_ids[idx]
        item_id_start_idx = self.indptr[item_id]
        start_idx = idx - item_id_start_idx
        # get array of corresponding item id
        target_sub_array = self.target_array[self.item_ids_mask[item_id]]
        context = self._get_context(target_sub_array[: start_idx + 1])

        if self.return_target:
            future_target = self._get_future_targets(target_sub_array[start_idx + 1 :])
            return {"context": context, "target": future_target}

        return context


class Chronos(AbstractPredictor):
    """Chronos time series predictor using a pretrained Chronos pipeline (from HuggingFace).
    This class implements prediction logic based on a fixed context window and multiple lead times.

    Parameters:
        model_name (str): Name or path of the pretrained Chronos model.
        device_map (str): Device to run inference on, e.g., "cpu", "cuda", or "mps".
        context_length (int): Number of timesteps used as context for prediction.
        lead_times (List[int]): List of prediction steps ahead (lead times).
        freq (pd.Timedelta): Frequency of the time series data.
        lora (bool): Whether to load the model with LoRA adapters. If True, `model_name`
            should point to a directory containing LoRA adapter weights and config.
            The base model will be automatically determined from the adapter config.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-bolt-tiny",
        device_map: str = "mps",
        context_length: int = 2048,
        lead_times: List[int] = [1, 2, 3],
        freq: pd.Timedelta = pd.Timedelta("1h"),
        lora: bool = False,
    ) -> None:
        super().__init__(lead_times, freq)
        self.context_length = context_length
        self.prediction_length = max(self.lead_times)

        if self.prediction_length > 64:
            raise ValueError("Maximum supported lead time is 64 currently.")

        print("Load chronos pipeline.")

        # add lora weights if specified
        if lora:
            print("Load LoRa weights.")

            # Load configs for sanity check
            with open(Path(model_name) / "adapter_config.json", "r") as f:
                adapter_config = json.load(f)

            base_model_name = adapter_config.get("base_model_name_or_path")

            self.pipeline = BaseChronosPipeline.from_pretrained(
                base_model_name,
                device_map=device_map,
            )

            # Apply LoRA adapters
            self.pipeline.inner_model = PeftModel.from_pretrained(self.pipeline.inner_model, model_name, is_trainable=False)

        else:
            self.pipeline = BaseChronosPipeline.from_pretrained(
                model_name,
                device_map=device_map,
            )

    def fit(self, data: TimeSeriesDataFrame) -> None:
        """Placeholder method for training. Not implemented, as Chronos is a pretrained model. Implement Finetuning in the future

        Parameters:
            data (TimeSeriesDataFrame): Training data (not used).

        Raises:
            NotImplementedError: Always raised since training is not supported.
        """

        raise NotImplementedError

    def _merge_data(self, data: TimeSeriesDataFrame, previous_context_data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Merges previous context data with the new prediction data, ensuring time continuity and context length.

        Parameters:
            data (TimeSeriesDataFrame): New data for which predictions will be made.
            previous_context_data (TimeSeriesDataFrame): Historical context data preceding `data`.

        Returns:
            TimeSeriesDataFrame: Combined and sorted dataframe used as input for prediction.

        Raises:
            ValueError: If any time series are not continuous between the two datasets.
        """

        shared_ids = data.item_ids.intersection(previous_context_data.item_ids)

        for item_id in shared_ids:
            prev_series = previous_context_data.loc[item_id]
            curr_series = data.loc[item_id]

            if len(prev_series) == 0 or len(curr_series) == 0:
                continue

            last_prev_time = prev_series.index[-1]
            first_curr_time = curr_series.index[0]

            expected_next_time = last_prev_time + self.freq
            if expected_next_time != first_curr_time:
                raise ValueError(f"Data for item_id '{item_id}' is not consecutive. " f"Expected {expected_next_time}, got {first_curr_time}.")

        previous_context_data = previous_context_data.loc[data.item_ids]
        previous_context_data = previous_context_data.groupby("item_id").tail(self.context_length)
        data_merged = pd.concat([previous_context_data, data]).sort_index()

        return TimeSeriesDataFrame(data_merged)

    def predict(
        self,
        data: TimeSeriesDataFrame,
        predict_only_last_timestep: bool = False,
        previous_context_data: Optional[TimeSeriesDataFrame] = None,
    ) -> PredictionLeadTimes:
        """Predicts future values for the given time series data using a pretrained Chronos model.

        Parameters:
            data (TimeSeriesDataFrame): The target data to forecast.
            predict_only_last_timestep (bool): Whether to forecast only the last timestep of each series.
            previous_context_data (Optional[TimeSeriesDataFrame]): Optional preceding data to provide context.

        Returns:
            PredictionLeadTimes: A dictionary-like object containing lead-time-specific predictions.
        """

        # Add previous context to test data if available
        if previous_context_data is not None:
            data_merged = self._merge_data(data, previous_context_data)
        else:
            data_merged = data

        if predict_only_last_timestep:
            ds = ChronosInferenceDataset(data_merged, self.context_length)
            data = data.slice_by_timestep(start_index=-1)
        else:
            # Make predictions for all timestamps of the test data
            ds = ChronosBacktestingDataset(data_merged, self.context_length)

        dl = DataLoader(ds, batch_size=64)

        results = {ld: None for ld in self.lead_times}
        forecasts = []

        for batch in tqdm(dl, desc="Predicting"):
            forecast = self.pipeline.predict(context=batch, prediction_length=self.prediction_length)
            forecasts.append(forecast)

        forecasts = torch.vstack(forecasts)

        if predict_only_last_timestep is False:
            # Mask to ensure we only return forecasts for the actual prediction set
            mask = data_merged.index.isin(data.index)
            forecasts = forecasts[mask, ...]

        for lt in self.lead_times:
            results[lt] = PredictionLeadTime(lead_time=lt, predictions=forecasts[..., lt - 1], freq=self.freq, data=data)

        return PredictionLeadTimes(results=results)


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


def fine_tune(
    model_init: Callable[[], PreTrainedModel],
    data_train: TimeSeriesDataFrame,
    data_val: Optional[TimeSeriesDataFrame] = None,
    path: Path = Path("./models/test-full-finetuning/"),
    hp_tuning: bool = False,
    n_trials: Optional[int] = None,
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
    path : Path, default=Path("./models/test-full-finetuning/")
        Path to save the trained model and optionally intermediate checkpoints.
    hp_tuning : bool, default=False
        Whether to perform hyperparameter tuning using Optuna.
    n_trials : Optional[int], default=None
        Number of Optuna trials. Required if `hp_tuning` is True.
    """

    def create_callbacks():
        callbacks = [BestCheckpointCallback()]
        if data_val is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=5))
        return callbacks

    context_length = 2048
    prediction_length = 64
    target_column = "target"

    # train_dataset = ChronosFineTuningDataset(
    #     target_df=data_train,
    #     target_column=target_column,
    #     context_length=context_length,
    #     prediction_length=prediction_length,
    #     tokenizer=None,
    #     mode="training",
    # ).shuffle(True)

    eval_dataset = None
    # if data_val is not None:
    #     eval_dataset = ChronosFineTuningDataset(
    #         target_df=data_val,
    #         target_column=target_column,
    #         context_length=context_length,
    #         prediction_length=prediction_length,
    #         tokenizer=None,
    #         mode="validation",
    # )

    train_dataset = ChronosBacktestingDataset(
        data=data_train,
        context_length=context_length,
        target_column=target_column,
        return_target=True,
        prediction_length=prediction_length,
    )
    eval_dataset = ChronosBacktestingDataset(
        data=data_val,
        context_length=context_length,
        target_column=target_column,
        return_target=True,
        prediction_length=prediction_length,
    )

    # Create separate directory for final training
    final_training_path = path / "training"
    final_training_path.mkdir(exist_ok=True, parents=True)

    # Create args for final training with best hyperparameters
    fine_tune_trainer_kwargs = create_trainer_kwargs(path=final_training_path, eval_during_fine_tune=data_val is not None, save_checkpoints=True)

    if hp_tuning:
        if data_val is None:
            raise ValueError("Validation data is required for hyperparameter tuning.")
        if n_trials is None:
            raise ValueError("n_trials must be specified when hp_tuning is enabled.")

        # Create separate path for hyperparameter tuning logs
        hp_tuning_path = path / "hp_tuning"
        hp_tuning_path.mkdir(exist_ok=True, parents=True)

        # Args for hyperparameter tuning phase
        hp_tuning_args = create_trainer_kwargs(path=hp_tuning_path, eval_during_fine_tune=data_val is not None, save_checkpoints=True)

        print("HP tuning hyperparameters:")
        print(50 * "-")
        print(hp_tuning_args)
        print(50 * "-")

        hp_trainer = Trainer(
            model_init=model_init,
            args=hp_tuning_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=create_callbacks(),
        )

        print("Tuning hyperparameters...")
        best_hp_args = tune_hp_optuna(hp_trainer, hp_space_optuna, n_trials=n_trials)

        # Apply best hyperparameters to final training args
        for key, value in best_hp_args.__dict__.items():
            if not key.startswith("_") and not key.endswith("dir"):
                setattr(fine_tune_trainer_kwargs, key, value)

    print("Final training hyperparameters:")
    print(50 * "-")
    print(fine_tune_trainer_kwargs)
    print(50 * "-")

    trainer = Trainer(
        model_init=model_init,
        args=fine_tune_trainer_kwargs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=create_callbacks(),
    )

    trainer.train()

    final_model_path = path / "fine-tuned-ckpt"
    trainer.model.save_pretrained(final_model_path)
    print(f"Saved model to {final_model_path}.")


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
    print("Best configuration: ", best_run)

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


def compute_objective(metrics):
    """Objective to minimize"""

    return metrics["eval_loss"]


def model_init_full_tuning() -> PreTrainedModel:
    """Initialize a fresh Chronos model where all parameters are trainable"""

    chronos_fresh = Chronos(model_name="amazon/chronos-bolt-small", device_map="mps", lead_times=np.arange(1, 65), freq=pd.Timedelta("1h"))

    return chronos_fresh.pipeline.inner_model


def model_init_last_layer_tuning() -> PreTrainedModel:
    """Initialize a fresh Chronos model where only last layer is trainable."""

    chronos_fresh = Chronos(model_name="amazon/chronos-bolt-tiny", device_map="mps", lead_times=np.arange(1, 65), freq=pd.Timedelta("1h"))

    # whether to only tune the last layer
    for param in chronos_fresh.pipeline.inner_model.parameters():
        param.requires_grad = False

    # train only last residual block (incl. output layer)
    for param in chronos_fresh.pipeline.inner_model.output_patch_embedding.parameters():
        param.requires_grad = True

    return chronos_fresh.pipeline.inner_model


def model_init_lora() -> PreTrainedModel:
    """Initialize a fresh Chronos model with LoRA"""

    # Initialize the Chronos model
    chronos_fresh = Chronos(model_name="amazon/chronos-bolt-tiny", device_map="mps", lead_times=np.arange(1, 65), freq=pd.Timedelta("1h"))

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=None,
        inference_mode=False,
        r=8,  # LoRA rank
        lora_alpha=8,  # Scaling factor
        lora_dropout=0.0,  # Dropout rate
        target_modules=[
            # Self Attention (inside SelfAttention and EncDecAttention)
            "q",
            "k",
            "v",
            "o",  #
            # Feedforward (inside DenseReluDense blocks (Feed Forward))
            "wi",
            "wo",
            # output_patch_embedding and input_patch_embedding blocks
            "hidden_layer",
            "output_layer",
            "residual_layer",
        ],
    )

    # Apply LoRA to the base model
    lora_model = get_peft_model(chronos_fresh.pipeline.inner_model, lora_config)

    lora_model.print_trainable_parameters()

    return lora_model


def create_trainer_kwargs(path: str = Path("./models/test/"), eval_during_fine_tune: bool = True, save_checkpoints: bool = True):
    """Define the training arguments"""

    save_eval_steps = 100
    logging_steps = 100
    target_column = "target"
    dir = "transformers_logs"

    # create TrainingArguments
    fine_tune_trainer_kwargs = {
        "output_dir": path / dir,
        "overwrite_output_dir": False,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "learning_rate": 1e-05,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.1,
        "weight_decay": 0.0,
        "optim": "adamw_torch_fused",
        "logging_dir": Path(path) / dir,
        "logging_strategy": "steps",
        "logging_steps": logging_steps,
        "disable_tqdm": True,
        # "max_steps": 500,
        "num_train_epochs": 1,
        "gradient_accumulation_steps": 1,
        "dataloader_num_workers": 0,
        "tf32": False,
        "report_to": "tensorboard",
        "prediction_loss_only": True,
        "save_strategy": "steps" if save_checkpoints else "no",
        "save_steps": save_eval_steps if save_checkpoints else None,
        "save_only_model": True,
        "save_total_limit": 5,
        "evaluation_strategy": "steps" if eval_during_fine_tune else "no",
        "eval_steps": save_eval_steps if eval_during_fine_tune else None,
        "eval_on_start": True if eval_during_fine_tune else False,
        "load_best_model_at_end": True if eval_during_fine_tune else False,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "use_cpu": False,
        "label_names": [target_column],
    }

    return TrainingArguments(**fine_tune_trainer_kwargs)


def check_model_parameters(chronos: Chronos, model_name: str = "amazon/chronos-bolt-tiny"):
    """Helper function to verify, if weights have changed."""

    chronos_copy = Chronos(model_name=model_name, device_map="mps", lead_times=np.arange(1, 65), freq=pd.Timedelta("1h"))

    for (name1, p1), (name2, p2) in zip(chronos_copy.pipeline.inner_model.named_parameters(), chronos.pipeline.inner_model.named_parameters()):
        if not torch.equal(p1, p2):
            print(f"Parameter {name1} has changed!")
