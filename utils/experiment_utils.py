from transformers import set_seed
from .ds_utils import clean_dataset_optimized
from datasets import load_dataset
from .model_utils import load_model, get_model_name, BERT, BART, ROBERTA, POLITICS

import json
import os
import shutil
import gc
import torch
from .trainer_utils import (
    compute_metrics,
    batched_predict_metrics_trainer,
    CustomTrainer,
)
from model import MultiTaskRoberta

from datasets import DatasetDict
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from typing import Optional
from dataclasses import dataclass

set_seed(42)


@dataclass
class ExperimentConfig:
    patience: int = 5
    num_epochs: int = 15
    save_model: bool = False


@dataclass
class DatasetConfig:
    sentiment: bool = False
    no_undersampling: bool = False
    custom_dataset: Optional[str] = None


ALLSIDES_EXTENDED_MEDIA_SPLIT = "dragonslayer631/article-bias-prediction-media-splits-updated"
ALLSIDES_EXTENDED_RANDOM_SPLIT = "dragonslayer631/allsides_random_split_extended"
_BATCH_SIZE = 64
_GRAD_ACCUMULATION = 32
_NUM_WORKERS = 8
_EVAL_BATCH_SIZE = 64


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


def get_cleaned_dataset(
    dataset,
    tokenizer,
    sentiments,
    max_length,
    no_undersampling=False,
    model=None,
):
    use_bias_keys = isinstance(model, MultiTaskRoberta)

    training_dataset = clean_dataset_optimized(
        dataset["train"],
        tokenizer=tokenizer,
        num_proc=24,
        sentiments=sentiments,
        max_length=max_length,
        skip_undersampling=no_undersampling,
        use_bias_keys=use_bias_keys,
    )
    test_dataset = clean_dataset_optimized(
        dataset["test"],
        tokenizer=tokenizer,
        num_proc=24,
        sentiments=sentiments,
        max_length=max_length,
        skip_undersampling=True,
        use_bias_keys=use_bias_keys,
    )
    validation_dataset = clean_dataset_optimized(
        dataset["validation"],
        tokenizer=tokenizer,
        num_proc=24,
        sentiments=sentiments,
        max_length=max_length,
        skip_undersampling=True,
        use_bias_keys=use_bias_keys,
    )

    return training_dataset, test_dataset, validation_dataset


def make_experiment_name(
    model_name: str, theme: Optional[str], trunc: bool, sentiment: bool
) -> str:
    text_mode = "trunc" if trunc else "batch"
    sent_mode = "sentiment" if sentiment else "no_sentiment"
    theme = theme if theme else "baseline"
    return f"{model_name}_{theme}_{text_mode}_{sent_mode}"


def load_and_rename_dataset(sentiment: bool) -> DatasetDict:
    """
    Returns a DatasetDict with columns renamed to
    {'bias'→'int_bias', 'content'→'text', 'ID'→'id'}
    and a 'validation' split set up.
    """
    path = "dragonslayer631/allsides_media-splits_sentiments"
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
    dataset_config: DatasetConfig,
    loc: str,
    experiment_config: Optional[ExperimentConfig] = None,
):
    """Run a single experiment with given configurations."""
    if experiment_config is None:
        experiment_config = ExperimentConfig()

    # 1) clean & tokenize
    train, test, validation = get_cleaned_dataset(
        ds,
        tokenizer,
        dataset_config.sentiment,
        512,
        dataset_config.no_undersampling,
        model=model,
    )

    # 2) train & evaluate
    name = make_experiment_name(model_name, None, False, dataset_config.sentiment)
    print(f"Training {name}")
    metrics_val, metrics_test = train_model(
        model, train, test, validation, f"{loc}/{name}", model_name, experiment_config
    )

    # 3) attach row counts
    add_row_counts(metrics_test, {"train": train, "test": test})
    metrics_val["validation_rows"] = count_unique_ids(validation, "validation")

    # 4) save
    metrics_filename = f"{loc}/{name}_test_metrics.json"
    with open(metrics_filename, "w") as f:
        json.dump(metrics_test, f, indent=2)
    return metrics_val, metrics_test


def _load_dataset_by_config(dataset_config: DatasetConfig) -> DatasetDict:
    if dataset_config.custom_dataset == "make_binary":
        ds = load_and_rename_dataset(dataset_config.sentiment)
        for split in ["validation", "train", "test"]:
            ds[split] = ds[split].map(make_binary)
    elif dataset_config.custom_dataset == "remove_int_bias_1":
        ds = load_and_rename_dataset(dataset_config.sentiment)
        for split in ["validation", "train", "test"]:
            ds[split] = ds[split].filter(remove_int_bias_1)
    elif dataset_config.custom_dataset == "mediabiasgroup/BABE":
        ds = load_dataset(dataset_config.custom_dataset)
        ds = ds.rename_column("label", "int_bias").rename_column("uuid", "id")
        ds["validation"] = ds["test"]
    elif dataset_config.custom_dataset and "dragonslayer631" in dataset_config.custom_dataset:
        ds = load_dataset(dataset_config.custom_dataset)
        if not ds.get("validation"):
            ds["validation"] = ds["test"]
    else:
        if dataset_config.custom_dataset:
            print("unsupported action")
        ds = load_and_rename_dataset(dataset_config.sentiment)
    return ds


def run_experiment(
    model,
    loc: str,
    dataset_config: Optional[DatasetConfig] = None,
    experiment_config: Optional[ExperimentConfig] = None,
):
    """Run experiments with configuration objects."""
    if dataset_config is None:
        dataset_config = DatasetConfig()
    if experiment_config is None:
        experiment_config = ExperimentConfig()

    model_name = get_model_name(model)
    cleanup()

    if dataset_config.custom_dataset:
        print(f"performing custom dataset action: {dataset_config.custom_dataset}")
    ds = _load_dataset_by_config(dataset_config)
    tokenizer, model = load_model(model)

    return run_single(
        model_name, model, tokenizer, ds, dataset_config, loc, experiment_config
    )


def make_training_args(
    model_name: str,
    output_dir: str = "test_trainer",
    num_epochs: int = 15,
    learning_rate: float = 5e-5,
    batch_size_override: Optional[int] = None,
) -> TrainingArguments:
    """Create training arguments with model-specific configurations."""
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
        seed=42,
        remove_unused_columns=False,
    )


def ensure_validation_dataset(test_ds, val_ds):
    if val_ds and len(val_ds) > 0:
        return val_ds
    if test_ds and len(test_ds) > 0:
        print("No validation set, using test as validation")
        return test_ds
    raise ValueError("Both validation and test sets are empty")


def count_unique_ids(dataset, split_name: str) -> int:
    """Count unique IDs in a dataset split."""
    return len(dataset.to_pandas()["id"].unique()) if dataset else 0


def add_row_counts(metrics: dict, datasets: dict) -> None:
    """Add row counts to metrics dictionary."""
    for split_name, dataset in datasets.items():
        metrics[f"{split_name}_rows"] = count_unique_ids(dataset, split_name)


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


def evaluate_and_cleanup(trainer: Trainer, test_ds, model_name: str):
    """Evaluate trainer and cleanup resources."""
    test_metrics = batched_predict_metrics_trainer(
        trainer, test_ds, batch_size=_EVAL_BATCH_SIZE
    )
    cleanup()  # your existing gc + torch.cuda.empty_cache
    return test_metrics


def train_model(
    model,
    train_ds,
    test_ds,
    val_ds,
    save_name: str,
    model_name: str,
    experiment_config: ExperimentConfig,
):
    """Train model with experiment configuration."""
    training_args = make_training_args(
        model_name, num_epochs=experiment_config.num_epochs
    )
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

    training_args_dict = training_args_to_dict(training_args, experiment_config.patience)
    test_metrics = evaluate_and_cleanup(trainer, test_ds, model_name)
    test_metrics["training_args"] = training_args_dict
    return {}, test_metrics
