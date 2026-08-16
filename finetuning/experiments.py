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


SEED = 42


@dataclass
class ExperimentConfig:
    patience: int = 5
    num_epochs: int = 15
    save_model: bool = False


@dataclass
class DatasetConfig:
    custom_dataset: Optional[str] = None


ALLSIDES_EXTENDED_MEDIA_SPLIT = "upasanachatterjee/article-bias-prediction-media-splits-updated"
ALLSIDES_EXTENDED_RANDOM_SPLIT = "upasanachatterjee/allsides_random_split_extended"
_BATCH_SIZE = 128
_GRAD_ACCUMULATION = 64
_NUM_WORKERS = 8
_EVAL_BATCH_SIZE = 128
# Articles are truncated, not chunked, so this is the whole input a model sees.
_MAX_LENGTH = 512
# Tokenization/undersampling workers for datasets.map; tuned for the training box.
_NUM_PROC = 24


def remove_int_bias_1(example):
    return example["int_bias"] != 1


def make_binary(example):
    example["int_bias"] = 0 if example["int_bias"] in [0, 2] else 1
    return example


def cleanup():
    if os.path.isdir("test_trainer"):
        shutil.rmtree("test_trainer")
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()


def get_cleaned_dataset(dataset, tokenizer, model):
    """Clean and tokenize all three splits.

    Only the training split is undersampled -- balancing test or validation
    would change what the reported numbers mean. Column names follow
    MultiTaskRoberta's `bias_*` convention when that is the model being trained.
    """
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


def load_and_rename_dataset() -> DatasetDict:
    """Default AllSides split, with columns renamed to the names used here
    ('bias'→'int_bias', 'content'→'text', 'ID'→'id')."""
    path = "upasanachatterjee/allsides_media-splits_sentiments"
    print("loading allsides_media-splits_sentiments")
    ds: DatasetDict = load_dataset(path)
    ds = (
        ds.rename_column("bias", "int_bias")
        .rename_column("content", "text")
        .rename_column("ID", "id")
    )
    return ds


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
    """Resolve `custom_dataset` to a DatasetDict with int_bias/text/id columns.

    The value is overloaded: it names either a label transform of the default
    AllSides split ("make_binary", "remove_int_bias_1") or a hub dataset to load
    instead. Splits that ship without a validation set reuse test, which is only
    used for early stopping.
    """
    if dataset_config.custom_dataset == "make_binary":
        ds = load_and_rename_dataset()
        for split in ["validation", "train", "test"]:
            ds[split] = ds[split].map(make_binary)
    elif dataset_config.custom_dataset == "remove_int_bias_1":
        ds = load_and_rename_dataset()
        for split in ["validation", "train", "test"]:
            ds[split] = ds[split].filter(remove_int_bias_1)
    elif dataset_config.custom_dataset == "mediabiasgroup/BABE":
        ds = load_dataset(dataset_config.custom_dataset)
        ds = ds.rename_column("label", "int_bias").rename_column("uuid", "id")
        ds["validation"] = ds["test"]
    elif dataset_config.custom_dataset and "upasanachatterjee" in dataset_config.custom_dataset:
        ds = load_dataset(dataset_config.custom_dataset)
        if not ds.get("validation"):
            ds["validation"] = ds["test"]
    else:
        if dataset_config.custom_dataset:
            print("unsupported action")
        ds = load_and_rename_dataset()
    return ds


def run_experiment(
    model,
    loc: str,
    dataset_config: Optional[DatasetConfig] = None,
    experiment_config: Optional[ExperimentConfig] = None,
) -> dict:
    """Fine-tune one model on one bias dataset and write its test metrics.

    `model` is a hub name or checkpoint path (see `models.load_model`); `loc` is
    the directory the metrics JSON lands in. Returns those metrics.
    """
    if dataset_config is None:
        dataset_config = DatasetConfig()
    if experiment_config is None:
        experiment_config = ExperimentConfig()

    set_seed(SEED)
    model_name = get_model_name(model)
    cleanup()

    if dataset_config.custom_dataset:
        print(f"performing custom dataset action: {dataset_config.custom_dataset}")
    ds = _load_dataset_by_config(dataset_config)
    tokenizer, model = load_model(model)

    return run_single(model_name, model, tokenizer, ds, loc, experiment_config)


def make_training_args(
    output_dir: str = "test_trainer",
    num_epochs: int = 15,
    learning_rate: float = 5e-5,
    batch_size_override: Optional[int] = None,
) -> TrainingArguments:
    """Shared HF TrainingArguments for every finetuning run in this study."""
    return TrainingArguments(
        output_dir=output_dir,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        gradient_checkpointing=True,
        fp16=True,
        per_device_train_batch_size=batch_size_override or _BATCH_SIZE,
        gradient_accumulation_steps=_GRAD_ACCUMULATION,
        dataloader_num_workers=_NUM_WORKERS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        weight_decay=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        warmup_ratio=0.06,
        save_safetensors=False,
        seed=SEED,
        remove_unused_columns=False,
    )


def ensure_validation_dataset(test_ds, val_ds):
    if val_ds and len(val_ds) > 0:
        return val_ds
    if test_ds and len(test_ds) > 0:
        print("No validation set, using test as validation")
        return test_ds
    raise ValueError("Both validation and test sets are empty")


def count_unique_ids(dataset) -> int:
    """Number of distinct article ids, i.e. rows before any per-id aggregation."""
    return len(dataset.to_pandas()["id"].unique()) if dataset else 0


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

    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]

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
    """Fine-tune on the training split, early-stopping on validation, score on test.

    Returns the test metrics, with the training arguments attached for the
    record. Validation is only used for model selection, so no metrics from it
    are reported.
    """
    training_args = make_training_args(num_epochs=experiment_config.num_epochs)
    print("per_device_train_batch_size=", training_args.per_device_train_batch_size)
    val_ds = ensure_validation_dataset(test_ds, val_ds)

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
    test_metrics["training_args"] = training_args_to_dict(
        training_args, experiment_config.patience
    )
    return test_metrics
