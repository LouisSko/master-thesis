from tqdm.auto import tqdm
import torch
from chronos import BaseChronosPipeline
from chronos.chronos_bolt import ChronosBoltPipeline
from chronos.chronos import ChronosPipeline, ChronosTokenizer
from autogluon.timeseries import TimeSeriesDataFrame
from torch.utils.data import DataLoader
from typing import Callable, List, Optional, Dict, Any, Literal, Union, Iterable
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from src.core.base import AbstractPredictor
from src.core.utils import set_global_seed
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
set_global_seed()


class ChronosLoraConfig(LoraConfig):
    """Chronos t5 lora configuration"""

    def __init__(self, prediction_length=None, **kwargs):
        super().__init__(**kwargs)
        self.prediction_length = prediction_length

    def to_dict(self):
        base = LoraConfig.to_dict(self)
        base["prediction_length"] = self.prediction_length
        return base


class BaseTimeSeriesDataset(Dataset):
    """
    A dataset for rolling backtesting and inference with time series data.

    Extracts multiple historical context windows (with optional targets).
    Also provides helper to assemble forecasts.
    """

    def __init__(
        self,
        data: TimeSeriesDataFrame,
        context_length: int,
        window_step: int = 1,
        skip_first_n_samples: Optional[Dict[int, int]] = None,
        target_column: str = "target",
        return_target: bool = False,
        prediction_length: Optional[int] = None,
        tokenizer: Optional["ChronosTokenizer"] = None,
        rolling: bool = False,
    ):
        assert context_length > 0, "context_length must be greater than 0"
        assert window_step > 0, "window_step must be greater than 0"

        self.context_length = context_length
        self.window_step = window_step
        self.return_target = return_target
        self.prediction_length = prediction_length
        self.tokenizer = tokenizer
        self.skip_first_n_samples = skip_first_n_samples
        self.rolling = rolling

        if self.return_target and self.prediction_length is None:
            raise ValueError("prediction_length must be set when return_target=True")

        if self.return_target:
            data = data.sort_values([ITEMID, TIMESTAMP])

        self.target_array = data[target_column].to_numpy(np.float32)
        self.item_ids = pd.factorize(data.index.get_level_values(ITEMID))[0]
        self.timestamps = data.index.get_level_values(TIMESTAMP)

        cum_sizes = data.num_timesteps_per_item().cumsum()
        self.indptr = np.append(0, cum_sizes)
        self.item_ids_mask = {item_id: self.item_ids == item_id for item_id in np.unique(self.item_ids)}

        self._compute_valid_indices(skip_first_n_samples)

        if self.rolling:
            self._compute_valid_indices(skip_first_n_samples)
        else:
            # only last observation per series
            self.valid_idx = self._compute_latest_indices()

    def _compute_latest_indices(self):
        indices = []
        for item_id in np.unique(self.item_ids):
            mask = self.item_ids_mask[item_id]
            idx = np.where(mask)[0][-1]
            indices.append(idx)
        return np.array(indices)

    def _compute_valid_indices(self, skip_first_n_samples):
        self.valid_idx = []
        self.ranges = {}  # item_id -> (start, stop)

        pointer = 0
        for item_id in np.unique(self.item_ids):
            mask = self.item_ids == item_id
            series_len = mask.sum()
            offset = self.indptr[item_id]
            start = skip_first_n_samples.get(item_id, 0) if skip_first_n_samples else 0
            end = series_len - 1

            idxs = offset + np.arange(start, end + 1, self.window_step)
            self.valid_idx.extend(idxs)

            self.ranges[item_id] = (pointer, pointer + len(idxs))
            pointer += len(idxs)

        self.valid_idx = np.array(self.valid_idx)

    def __len__(self):
        """Returns the total number of time steps in the dataset."""
        return len(self.valid_idx)

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

        real_idx = self.valid_idx[idx]
        item_id = self.item_ids[real_idx]
        item_start = self.indptr[item_id]
        pos_in_series = real_idx - item_start

        series = self.target_array[self.item_ids_mask[item_id]]
        # get series of corresponding item id
        series = self.target_array[self.item_ids_mask[item_id]]
        context = self._get_context(series[: pos_in_series + 1])

        if self.return_target:
            future_target = self._get_future_targets(series[pos_in_series + 1 :])

            if self.tokenizer is not None:
                return self.to_chronos_format(context, future_target)
            else:
                return self.to_chronos_bolt_format(context, future_target)

        return context

    @property
    def pred_index(self):
        return pd.MultiIndex.from_arrays(
            [self.item_ids[self.valid_idx], self.timestamps[self.valid_idx]],
            names=[ITEMID, TIMESTAMP],
        )

    @property
    def valid_item_ids(self):
        return self.item_ids[self.valid_idx]

    @property
    def valid_timestamps(self):
        return pd.to_datetime(self.timestamps[self.valid_idx], unit="s")

    def to_forecast_collection(self, predictions: torch.Tensor, lead_times: List[int], output_data: TimeSeriesDataFrame) -> ForecastCollection:
        """
        Assemble a ForecastCollection given model predictions.

        predictions is a tensor [N x num_quantiles x prediction length]
        """

        freq = pd.tseries.frequencies.to_offset(output_data.freq)

        preds_df = pd.DataFrame(
            {
                "item_id": self.item_ids[self.valid_idx],
                "timestamp": self.timestamps[self.valid_idx],
            }
        )

        assert len(preds_df) == predictions.shape[0]

        forecasts = {}
        for item_id, group in preds_df.groupby("item_id"):
            s, e = group.index.min(), group.index.max() + 1
            preds = predictions[s:e]
            timestamps = group["timestamp"]

            mask = output_data.loc[[item_id]].index.get_level_values(TIMESTAMP).isin(timestamps)

            lt_forecasts = {lt: HorizonForecast(lead_time=lt, predictions=preds[..., lt - 1]) for lt in lead_times}

            forecasts[item_id] = TimeSeriesForecast(
                item_id=item_id,
                lead_time_forecasts=lt_forecasts,
                data=output_data.loc[[item_id]],
                freq=freq,
                forecast_mask=mask,
            )

        return ForecastCollection(item_ids=forecasts)


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
    lead_times : Optional[Iterable[int]], default=None
        An iterable of integers specifying the forecast lead times.
        If None, defaults to [1, 2, 3].
    sampling: bool, optional
        Whether to sample multiple trajectories. Defaults to False.
    finetuning_type : {"full", "last_layer", "LoRA"}, optional
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
    name : str, optional
        Name of the model, defaults to the class name
    """

    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path] = "amazon/chronos-bolt-tiny",
        device_map: str = "mps",
        context_length: int = 2048,
        lead_times: Optional[Iterable[int]] = None,
        sampling: bool = False,
        finetuning_type: Literal["full", "last_layer", "LoRA"] = "full",
        finetuning_adjust_pretrained_prediction_length: bool = True,
        finetuning_hp_search: Optional[bool] = False,
        finetuning_hp_search_trials: Optional[int] = 10,
        finetuning_warmup_new_neurons: bool = True,
        output_dir: Optional[Path] = Path("./models/"),
        name: Optional[str] = None,
    ) -> None:
        super().__init__(lead_times=lead_times, name=name, output_dir=output_dir)
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
            logging.info(f"Found LoRA configuration in {pretrained_model_name_or_path}.")

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
                logging.info("Setting prediction length of chronos-t5 to %s based on LoRA configuration.", pred_length)
                pipeline.inner_model.config.prediction_length = pred_length
                pipeline.inner_model.config.chronos_config["prediction_length"] = pred_length
                pipeline.model.config.prediction_length = pred_length

            # Apply LoRA adapters
            pipeline.inner_model = PeftModel.from_pretrained(pipeline.inner_model, pretrained_model_name_or_path, is_trainable=False)
            self.lora = True
            logging.info("LoRA adapters applied successfully.")

        else:
            logging.info("Initializing Chronos pipeline with model: %s", pretrained_model_name_or_path)
            pipeline = BaseChronosPipeline.from_pretrained(pretrained_model_name_or_path, device_map=self.device_map)

        # pipeline = resize_chronos_bolt_output_layers(pipeline, self.prediction_length)

        return pipeline

    def _fit(
        self,
        data_train: TimeSeriesDataFrame,
        data_val: Optional[TimeSeriesDataFrame] = None,
        train_window_step: int = 1,
        val_window_step: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Finetuning chronos model

        Parameters
        ----------
        data_train : TimeSeriesDataFrame
            Training data used to construct rolling windows for model training.
        data_val : Optional[TimeSeriesDataFrame], default=None
            Optional validation data used for early stopping and evaluation.
        train_window_step : int, default=1
            Number of time steps to shift the rolling window between training samples.
            A higher value reduces overlap between windows and the number of training examples.
            Recommended to use all available data (i.e., set to 1), since early stopping is used to prevent overfitting.
        val_window_step : Optional[int], default=None
            Number of time steps to shift the rolling window between validation samples.
            A higher value reduces overlap between validation windows and the number of validation examples.
            If not specified, defaults to `prediction_length`.
        """

        def _build_model(
            source: Union[str, Path],  # name or ckpt dir
            mode: Literal["full", "last_layer", "LoRA", "new_rows"],
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

            elif mode == "LoRA":
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
            return _build_model(self.pretrained_model_name_or_path, "LoRA")

        def init_new_rows():
            return _build_model(self.pretrained_model_name_or_path, "new_rows")

        model_inits = {"full": init_full, "last_layer": init_last, "LoRA": init_lora}

        # here a warmup is not necessary, since output neurons are already trained
        if self.pipeline.inner_model.config.chronos_config["prediction_length"] >= self.prediction_length:
            self.finetuning_warmup_new_neurons = False
            logging.info(
                "Warmup training of new output neurons gets disabled since prediction length fo %s is not greater than the configured prediction length of %s",
                self.prediction_length,
                self.pipeline.inner_model.config.chronos_config["prediction_length"],
            )
        # Ensure config.prediction_length is up-to-date for T5 (no head resize)
        if self.finetuning_adjust_pretrained_prediction_length:
            prediction_length = self.prediction_length
            self.pipeline.model.config.prediction_length = prediction_length  # TODO: check if I need this
            self.pipeline.inner_model.config.chronos_config["prediction_length"] = prediction_length
        else:
            prediction_length = self.pipeline.inner_model.config.chronos_config["prediction_length"]  # standard

        logging.info("Prediction length will be set to %s during training.", prediction_length)

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
                train_window_step=train_window_step,
                val_window_step=val_window_step,
                tokenizer=getattr(self.pipeline, "tokenizer", None),
                specific_train_kwargs={"learning_rate": 1e-3, "num_train_epochs": 10, "warmup_ratio": 0.0, "lr_scheduler_type": "constant"},
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
            train_window_step=train_window_step,
            val_window_step=val_window_step,
            tokenizer=getattr(self.pipeline, "tokenizer", None),
            specific_train_kwargs={"num_train_epochs": 3},
        )

        # reload final model
        self.pipeline = self._pipeline_init(final_dir / "fine-tuned-ckpt")
        logging.info("Two-stage fine-tuning complete – model reloaded.")

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        previous_context_data: Optional[TimeSeriesDataFrame] = None,
        rolling: bool = False,
        window_step: int = 1,
    ) -> ForecastCollection:
        """
        Generates forecasts for each time series.

        This method can perform either:
        - *single-shot prediction* (predicting from the most recent context window), or
        - *rolling backtesting* (sliding a window across the time series to predict at each time point).

        Parameters
        ----------
        data : TimeSeriesDataFrame
            The time series data to forecast.
        previous_context_data : Optional[TimeSeriesDataFrame], default=None
            Optional preceding time series data for extending the context window.
        rolling : bool, default=False
            If True, performs rolling evaluation across all available time steps.
            If False, predicts only from the latest observation.
        window_step : int, default=1
            The number of time steps to move the sliding (rolling) prediction window forward between each prediction.
            This controls how densely forecasts are generated across time. A smaller value creates more overlapping
            forecasts, while a larger value skips more observations between windows.
            The rolling procedure is applied independently to each time series in the dataset.

        Returns
        -------
        ForecastCollection
            A nested dictionary mapping each item_id to lead time forecasts.
        """
        # Combine context data if given
        if previous_context_data is not None:
            # skip_first: Dict[item_id -> how many prepended rows], used for dataset indexing
            data_merged, skip_first = self._merge_data(data, previous_context_data, self.context_length)
        else:
            data_merged = data
            skip_first = None

        # Choose the appropriate dataset for single-shot or rolling prediction
        ds = BaseTimeSeriesDataset(
            data_merged,
            self.context_length,
            window_step,
            skip_first,
            rolling=rolling,
        )

        dl = DataLoader(ds, batch_size=128)

        forecasts = []

        # Iterate batches and generate predictions
        for batch in tqdm(dl, desc="Predicting using Chronos"):
            if self.sampling:
                forecast = self.pipeline.predict_sampling(
                    context=batch,
                    prediction_length=self.prediction_length,
                )
            else:
                forecast = self.pipeline.predict(
                    context=batch,
                    prediction_length=self.prediction_length,
                )
                if isinstance(self.pipeline, ChronosPipeline):
                    # Convert trajectories to quantiles
                    forecast = torch.quantile(
                        forecast,
                        q=torch.tensor(self.quantiles, dtype=forecast.dtype),
                        dim=1,
                    ).swapaxes(1, 0)

            forecasts.append(forecast)

        # Concatenate batches
        forecasts = torch.vstack(forecasts)
        assert forecasts.shape[0] == len(ds), "row count mismatch"

        # If rolling, output data covers all input rows
        if rolling:
            output_data = data
        else:
            # Only the most recent timestep per series
            output_data = data.slice_by_timestep(start_index=-1)

        collection = ds.to_forecast_collection(predictions=forecasts, lead_times=self.lead_times, output_data=output_data)

        return collection


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
    train_window_step: int = 1,
    val_window_step: Optional[int] = None,
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
    train_window_step : int, default=1.
        The stride on the train dataset. Recommended to use all data (=1) since early stopping is used.
    val_window_step : int, default=None.
        The stride on the validation dataset. Defaults to prediction length.
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

    val_window_step = val_window_step or prediction_length
    logging.info("Setting train stride: %s, validation stride: %s", train_window_step, val_window_step)
    logging.info("Preparing training dataset...")
    train_dataset = BaseTimeSeriesDataset(
        data=data_train,
        context_length=context_length,
        window_step=train_window_step,
        target_column=TARGET,
        return_target=True,
        skip_first_n_samples=None,# {item_id: 512 for item_id in data_train.item_ids},
        prediction_length=prediction_length,
        tokenizer=tokenizer,
        rolling=True,
    )

    eval_dataset = None
    if data_val is not None:
        logging.info("Preparing validation dataset...")
        data_val = pd.concat([data_train, data_val]).sort_index()
        skip_first_n_samples = (data_train.num_timesteps_per_item() - 1).to_dict()
        eval_dataset = BaseTimeSeriesDataset(
            data=data_val,
            context_length=context_length,
            window_step=val_window_step,
            target_column=TARGET,
            skip_first_n_samples=skip_first_n_samples,
            return_target=True,
            prediction_length=prediction_length,
            tokenizer=tokenizer,
            rolling=True,
        )
        if len(eval_dataset) == 0:
            raise ValueError("No samples in evaluation dataset.")

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

    num_train_epochs = pipeline_kwargs.get("num_train_epochs", 3)
    eval_ratio = 0.1 / num_train_epochs
    logging_steps = 0.05 / num_train_epochs
    log_dir = base_path / "logs"

    fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 0)

    defaults = dict(
        output_dir=base_path,
        overwrite_output_dir=False,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        auto_find_batch_size=True,
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        weight_decay=0.0,
        optim="adamw_torch_fused",
        logging_dir=log_dir,
        logging_strategy="steps",
        logging_steps=logging_steps,
        disable_tqdm=True,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=1,
        dataloader_num_workers=4,
        seed=42,
        # data_seed=42,
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

    chronos_copy = Chronos(model_name=model_name, device_map="mps", lead_times=np.arange(1, 65))

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
