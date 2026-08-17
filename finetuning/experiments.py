from .preprocessing import clean_dataset_optimized
from datasets import load_dataset
from .models import load_model, get_model_name

import json
import os
import shutil
import gc
import torch
from .metrics import (
    compute_metrics,
    batched_predict_metrics_trainer,
    CustomTrainer,
    WholeEpochCounter,
)
from model import MultiTaskRoberta

from datasets import DatasetDict
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from typing import Optional
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    patience: int = 5
    num_epochs: int = 15
    save_model: bool = False
    seed: int = 42
    batch_size: Optional[int] = None


@dataclass
class DatasetConfig:
    custom_dataset: Optional[str] = None


ALLSIDES_BASE_MEDIA_SPLIT = "upasanachatterjee/AllSides-media-split"
ALLSIDES_BASE_RANDOM_SPLIT = "upasanachatterjee/AllSides-random-split"
_BATCH_SIZE = 64
_GRAD_ACCUMULATION = 4
_EFFECTIVE_BATCH_SIZE = _BATCH_SIZE * _GRAD_ACCUMULATION
_NUM_WORKERS = 8
_EVAL_BATCH_SIZE = 64
_MAX_LENGTH = 512
_NUM_PROC = 24


def cleanup():
    if os.path.isdir("test_trainer"):
        shutil.rmtree("test_trainer")
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()


def get_cleaned_dataset(dataset, tokenizer, model):
    """Clean and tokenize all three splits, undersampling only the training one."""
    # Balancing test or validation would change what the reported numbers mean.
    use_bias_keys = isinstance(model, MultiTaskRoberta)

    def prepare(split: str, skip_undersampling: bool):
        return clean_dataset_optimized(
            dataset[split],
            tokenizer=tokenizer,
            num_proc=_NUM_PROC,
            max_length=_MAX_LENGTH,
            skip_undersampling=skip_undersampling,
            use_bias_keys=use_bias_keys,
        )

    return (
        prepare("train", False),
        prepare("test", True),
        prepare("validation", True),
    )


def make_experiment_name(model_name: str) -> str:
    """Directory/file stem for one run. The 'baseline_trunc' segment is kept so
    the names line up with results produced before theme-conditioned and
    whole-article variants were dropped."""
    return f"{model_name}_baseline_trunc"


def run_single(
    model_name: str,
    model,
    tokenizer,
    ds: DatasetDict,
    loc: str,
    experiment_config: Optional[ExperimentConfig] = None,
):
    """Run a single experiment with given configurations."""
    if experiment_config is None:
        experiment_config = ExperimentConfig()

    train, test, validation = get_cleaned_dataset(ds, tokenizer, model)

    name = make_experiment_name(model_name)
    print(f"Training {name}")
    metrics_test = train_model(
        model, train, test, validation, f"{loc}/{name}", experiment_config
    )

    add_row_counts(
        metrics_test, {"train": train, "test": test, "validation": validation}
    )

    with open(f"{loc}/{name}_test_metrics.json", "w") as f:
        json.dump(metrics_test, f, indent=2)
    return metrics_test


def _load_dataset_by_config(dataset_config: DatasetConfig) -> DatasetDict:
    """Load one of the two AllSides splits, which already carry bias/text/id."""
    # Neither ships a validation split; test stands in, and only drives early stopping.
    ds = load_dataset(dataset_config.custom_dataset or ALLSIDES_BASE_MEDIA_SPLIT)
    if not ds.get("validation"):
        ds["validation"] = ds["test"]
    return ds


def run_experiment(
    model,
    loc: str,
    dataset_config: Optional[DatasetConfig] = None,
    experiment_config: Optional[ExperimentConfig] = None,
    model_name: Optional[str] = None,
) -> dict:
    """Fine-tune one hub name or checkpoint path on one bias dataset, writing its test
    metrics into `loc` and returning them.
    """
    # `model_name` names the file: every non-hub path would otherwise be "custom".
    if dataset_config is None:
        dataset_config = DatasetConfig()
    if experiment_config is None:
        experiment_config = ExperimentConfig()

    set_seed(experiment_config.seed)
    model_name = model_name or get_model_name(model)
    cleanup()

    if dataset_config.custom_dataset:
        print(f"performing custom dataset action: {dataset_config.custom_dataset}")
    ds = _load_dataset_by_config(dataset_config)
    tokenizer, model = load_model(model)

    return run_single(model_name, model, tokenizer, ds, loc, experiment_config)


def make_training_args(
    output_dir: str = "test_trainer",
    num_epochs: int = 15,
    learning_rate: float = 1e-5,
    batch_size_override: Optional[int] = None,
    seed: int = 42,
) -> TrainingArguments:
    """Shared HF TrainingArguments for every finetuning run in this study, with
    accumulation derived so any batch size gives the same effective batch.
    """
    batch_size = batch_size_override or _BATCH_SIZE
    return TrainingArguments(
        output_dir=output_dir,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        gradient_checkpointing=True,
        fp16=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, _EFFECTIVE_BATCH_SIZE // batch_size),
        dataloader_num_workers=_NUM_WORKERS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        warmup_ratio=0.06,
        save_safetensors=False,
        disable_tqdm=False,
        seed=seed,
        remove_unused_columns=False,
    )


def count_unique_ids(dataset) -> int:
    """Number of distinct article ids, i.e. rows before any per-id aggregation."""
    return len(dataset.to_pandas()["ID"].unique()) if dataset else 0


def add_row_counts(metrics: dict, datasets: dict) -> None:
    """Add row counts to metrics dictionary."""
    for split_name, dataset in datasets.items():
        metrics[f"{split_name}_rows"] = count_unique_ids(dataset)


def training_args_to_dict(training_args: TrainingArguments, patience: int) -> dict:
    """Convert training arguments to dictionary with additional params."""
    return {
        key: getattr(training_args, key)
        for key in [
            "output_dir",
            "learning_rate",
            "num_train_epochs",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "weight_decay",
            "warmup_ratio",
            "fp16",
            # Recorded so a metrics JSON identifies its own seed, not just its directory.
            "seed",
        ]
    } | {"patience": patience}


def make_trainer(
    model,
    training_args: TrainingArguments,
    train_ds,
    eval_ds,
    compute_fn,
    patience: int = 5,
) -> Trainer:
    print("patience=", patience)
    print("compute_fn=", compute_fn)

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=patience),
        WholeEpochCounter(),
    ]

    return CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_fn,
        callbacks=callbacks,
    )


def evaluate_and_cleanup(trainer: Trainer, test_ds):
    """Score the held-out split, then free the GPU before the next run."""
    test_metrics = batched_predict_metrics_trainer(
        trainer, test_ds, batch_size=_EVAL_BATCH_SIZE
    )
    cleanup()
    return test_metrics


def train_model(
    model,
    train_ds,
    test_ds,
    val_ds,
    save_name: str,
    experiment_config: ExperimentConfig,
) -> dict:
    """Fine-tune on train, early-stop on validation, score on test. Returns the test
    metrics with the training arguments attached; validation only selects the model.
    """
    training_args = make_training_args(
        num_epochs=experiment_config.num_epochs,
        seed=experiment_config.seed,
        batch_size_override=experiment_config.batch_size,
    )
    print("per_device_train_batch_size=", training_args.per_device_train_batch_size)
    print("gradient_accumulation_steps=", training_args.gradient_accumulation_steps)

    trainer = make_trainer(
        model,
        training_args,
        train_ds,
        val_ds,
        compute_metrics,
        experiment_config.patience,
    )

    trainer.train()
    if experiment_config.save_model:
        trainer.save_model(f"{save_name}/finetuned_model")
    cleanup()

    test_metrics = evaluate_and_cleanup(trainer, test_ds)
    test_metrics["log_history"] = trainer.state.log_history
    test_metrics["training_args"] = training_args_to_dict(
        training_args, experiment_config.patience
    )
    return test_metrics
