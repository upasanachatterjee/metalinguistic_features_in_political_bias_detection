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
from .basil import BASIL_CLASS_WEIGHTS, BasilConfig, load_basil
from .basil import variant_name as basil_variant_name
from .mitweet import MITweetConfig, load_mitweet, variant_name
from .semeval import SemEvalConfig, load_semeval
from .semeval import variant_name as semeval_variant_name
from .basil_metrics import BasilTrainer, batched_predict_basil, compute_basil_metrics
from .mitweet_metrics import (
    RelevanceTrainer,
    batched_predict_ideology,
    batched_predict_relevance,
    compute_relevance_metrics,
)
from .semeval_metrics import (
    batched_predict_opinion,
    batched_predict_stance,
    compute_opinion_metrics,
    compute_stance_metrics,
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
from functools import partial


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
_BATCH_SIZE = 32
_GRAD_ACCUMULATION = 4
_EFFECTIVE_BATCH_SIZE = _BATCH_SIZE * _GRAD_ACCUMULATION
_NUM_WORKERS = 8
_EVAL_BATCH_SIZE = 32
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
    # Both ship a 161-row validation split now; the alias is kept for older revisions.
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
    metric_for_best_model: str = "eval_f1_macro",
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
        metric_for_best_model=metric_for_best_model,
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
    trainer_cls=CustomTrainer,
) -> Trainer:
    print("patience=", patience)
    print("compute_fn=", compute_fn)

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=patience),
        WholeEpochCounter(),
    ]

    return trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_fn,
        callbacks=callbacks,
    )


def evaluate_and_cleanup(trainer: Trainer, test_ds, predict_fn=batched_predict_metrics_trainer):
    """Score the held-out split, then free the GPU before the next run."""
    test_metrics = predict_fn(trainer, test_ds, batch_size=_EVAL_BATCH_SIZE)
    cleanup()
    return test_metrics


def save_finetuned(trainer: Trainer, save_name: str) -> str:
    """Write the best epoch's weights somewhere `models.load_model` can read back, returning
    the path.
    """
    # MultiTaskRoberta is not a PreTrainedModel, so trainer.save_model writes a bare state
    # dict that no loader can rebuild the heads from; save_checkpoint carries their sizes.
    model = trainer.accelerator.unwrap_model(trainer.model)
    if isinstance(model, MultiTaskRoberta):
        os.makedirs(save_name, exist_ok=True)
        path = f"{save_name}/finetuned_model.pt"
        model.save_checkpoint(path)
    else:
        path = f"{save_name}/finetuned_model"
        trainer.save_model(path)
    print(f"saved fine-tuned model to {path}")
    return path


def train_model(
    model,
    train_ds,
    test_ds,
    val_ds,
    save_name: str,
    experiment_config: ExperimentConfig,
    compute_fn=compute_metrics,
    predict_fn=batched_predict_metrics_trainer,
    trainer_cls=CustomTrainer,
    metric_for_best_model: str = "eval_f1_macro",
) -> dict:
    """Fine-tune on train, early-stop on validation, score on test. Returns the test
    metrics with the training arguments attached; validation only selects the model.
    """
    training_args = make_training_args(
        num_epochs=experiment_config.num_epochs,
        seed=experiment_config.seed,
        batch_size_override=experiment_config.batch_size,
        metric_for_best_model=metric_for_best_model,
    )
    print("per_device_train_batch_size=", training_args.per_device_train_batch_size)
    print("gradient_accumulation_steps=", training_args.gradient_accumulation_steps)

    trainer = make_trainer(
        model,
        training_args,
        train_ds,
        val_ds,
        compute_fn,
        experiment_config.patience,
        trainer_cls,
    )

    trainer.train()
    # load_best_model_at_end has already restored the best epoch, so this saves that one.
    saved_path = save_finetuned(trainer, save_name) if experiment_config.save_model else None
    cleanup()

    test_metrics = evaluate_and_cleanup(trainer, test_ds, predict_fn)
    # test_metrics["log_history"] = trainer.state.log_history
    test_metrics["training_args"] = training_args_to_dict(
        training_args, experiment_config.patience
    )
    if saved_path is not None:
        test_metrics["saved_model_path"] = saved_path
    return test_metrics


def _task_plumbing(task: str) -> dict:
    """The scorer, predictor, trainer class and selection metric one MITweet task needs."""
    if task == "relevance":
        return {
            "compute_fn": compute_relevance_metrics,
            "predict_fn": batched_predict_relevance,
            "trainer_cls": RelevanceTrainer,
            # MITweet picks the best relevance epoch on the flattened binary F1.
            "metric_for_best_model": "eval_f1_micro",
        }
    return {
        "compute_fn": compute_metrics,
        "predict_fn": batched_predict_ideology,
        "trainer_cls": CustomTrainer,
        "metric_for_best_model": "eval_f1_macro",
    }


def run_mitweet_experiment(
    model,
    loc: str,
    mitweet_config: Optional[MITweetConfig] = None,
    experiment_config: Optional[ExperimentConfig] = None,
    model_name: Optional[str] = None,
) -> dict:
    """Fine-tune one hub name or checkpoint path on one MITweet variant, writing its test
    metrics into `loc` and returning them.
    """
    if mitweet_config is None:
        mitweet_config = MITweetConfig()
    if experiment_config is None:
        experiment_config = ExperimentConfig()

    set_seed(experiment_config.seed)
    model_name = model_name or get_model_name(model)
    cleanup()
    os.makedirs(loc, exist_ok=True)

    tokenizer, model = load_model(model, task=mitweet_config.task)
    splits = load_mitweet(mitweet_config, tokenizer, isinstance(model, MultiTaskRoberta))

    name = make_experiment_name(model_name)
    print(f"Training {name} on {variant_name(mitweet_config)}")
    metrics_test = train_model(
        model,
        splits["train"],
        splits["test"],
        splits["validation"],
        f"{loc}/{name}",
        experiment_config,
        **_task_plumbing(mitweet_config.task),
    )

    add_row_counts(metrics_test, dict(splits))
    # Recorded so a metrics JSON identifies its own fold, not just its directory. The random
    # split has no folds, so it stays absent there rather than claiming a meaningless 0.
    if mitweet_config.split == "facet":
        metrics_test["fold"] = mitweet_config.fold
    with open(f"{loc}/{name}_test_metrics.json", "w") as f:
        json.dump(metrics_test, f, indent=2)
    return metrics_test


def run_prediction_only(
    model,
    loc: str,
    mitweet_config: Optional[MITweetConfig] = None,
    model_name: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """Score a checkpoint on one MITweet test split without training it -- the zero-shot arm
    of the AllSides-to-MITweet comparison.
    """
    if mitweet_config is None:
        mitweet_config = MITweetConfig()

    set_seed(seed)
    model_name = model_name or get_model_name(model)
    cleanup()
    os.makedirs(loc, exist_ok=True)

    tokenizer, model = load_model(model, task=mitweet_config.task)
    splits = load_mitweet(mitweet_config, tokenizer, isinstance(model, MultiTaskRoberta))
    plumbing = _task_plumbing(mitweet_config.task)

    training_args = make_training_args(
        num_epochs=1,
        seed=seed,
        batch_size_override=_EVAL_BATCH_SIZE,
        metric_for_best_model=plumbing["metric_for_best_model"],
    )
    # Nothing is trained here, so there is no best epoch to restore.
    training_args.load_best_model_at_end = False
    trainer = plumbing["trainer_cls"](
        model=model, args=training_args, compute_metrics=plumbing["compute_fn"]
    )

    name = make_experiment_name(model_name)
    print(f"Predicting {name} on {variant_name(mitweet_config)}, zero-shot")
    metrics_test = plumbing["predict_fn"](
        trainer, splits["test"], batch_size=_EVAL_BATCH_SIZE
    )

    add_row_counts(metrics_test, {"test": splits["test"]})
    # Named so the JSON says it was never trained, rather than carrying a training config.
    metrics_test["training_args"] = {"zero_shot": True, "seed": seed}
    with open(f"{loc}/{name}_test_metrics.json", "w") as f:
        json.dump(metrics_test, f, indent=2)
    cleanup()
    return metrics_test


def _semeval_plumbing(task: str) -> dict:
    """The scorer, predictor and selection metric one SemEval task needs."""
    if task == "stance":
        return {
            "compute_fn": compute_stance_metrics,
            "predict_fn": batched_predict_stance,
            # The shared task's own metric, so the best epoch is the one it would have ranked.
            "metric_for_best_model": "eval_f1_favor_against",
        }
    return {
        "compute_fn": compute_opinion_metrics,
        "predict_fn": batched_predict_opinion,
        "metric_for_best_model": "eval_f1_macro",
    }


def run_semeval_experiment(
    model,
    loc: str,
    semeval_config: Optional[SemEvalConfig] = None,
    experiment_config: Optional[ExperimentConfig] = None,
    model_name: Optional[str] = None,
) -> dict:
    """Fine-tune one hub name or checkpoint path on one SemEval-2016 Task 6 variant, writing
    its test metrics into `loc` and returning them.
    """
    if semeval_config is None:
        semeval_config = SemEvalConfig()
    if experiment_config is None:
        experiment_config = ExperimentConfig()

    set_seed(experiment_config.seed)
    model_name = model_name or get_model_name(model)
    cleanup()
    os.makedirs(loc, exist_ok=True)

    tokenizer, model = load_model(model, task=semeval_config.task)
    splits = load_semeval(semeval_config, tokenizer, isinstance(model, MultiTaskRoberta))

    name = make_experiment_name(model_name)
    print(f"Training {name} on {semeval_variant_name(semeval_config)}")
    metrics_test = train_model(
        model,
        splits["train"],
        splits["test"],
        splits["validation"],
        f"{loc}/{name}",
        experiment_config,
        **_semeval_plumbing(semeval_config.task),
    )

    add_row_counts(metrics_test, dict(splits))
    with open(f"{loc}/{name}_test_metrics.json", "w") as f:
        json.dump(metrics_test, f, indent=2)
    return metrics_test


def _basil_plumbing(task: str) -> dict:
    """The scorer, predictor, trainer class and selection metric one BASIL task needs."""
    return {
        "compute_fn": compute_basil_metrics,
        "predict_fn": batched_predict_basil,
        # partial, not a subclass per task: `make_trainer` calls trainer_cls with keywords only.
        "trainer_cls": partial(BasilTrainer, class_weights=BASIL_CLASS_WEIGHTS[task]),
        # Macro-F1 over a 94%-negative class would select an epoch that predicts nothing.
        "metric_for_best_model": "eval_f1_positive",
    }


def run_basil_experiment(
    model,
    loc: str,
    basil_config: Optional[BasilConfig] = None,
    experiment_config: Optional[ExperimentConfig] = None,
    model_name: Optional[str] = None,
) -> dict:
    """Fine-tune one hub name or checkpoint path on one BASIL task and fold, writing its test
    metrics into `loc` and returning them.
    """
    if basil_config is None:
        basil_config = BasilConfig()
    if experiment_config is None:
        experiment_config = ExperimentConfig()

    set_seed(experiment_config.seed)
    model_name = model_name or get_model_name(model)
    cleanup()
    os.makedirs(loc, exist_ok=True)

    tokenizer, model = load_model(model, task=basil_config.task)
    splits = load_basil(basil_config, tokenizer, isinstance(model, MultiTaskRoberta))

    name = make_experiment_name(model_name)
    print(f"Training {name} on {basil_variant_name(basil_config)} fold {basil_config.fold}")
    metrics_test = train_model(
        model,
        splits["train"],
        splits["test"],
        splits["validation"],
        f"{loc}/{name}",
        experiment_config,
        **_basil_plumbing(basil_config.task),
    )

    add_row_counts(metrics_test, dict(splits))
    # Recorded so a metrics JSON identifies its own fold, not just its directory.
    metrics_test["fold"] = basil_config.fold
    with open(f"{loc}/{name}_test_metrics.json", "w") as f:
        json.dump(metrics_test, f, indent=2)
    return metrics_test
