"""Metrics for the two SemEval-2016 Task 6 tasks. `f1_favor_against` is the shared task's own
metric, which averages FAVOR and AGAINST and ignores NONE, so the name is kept literal.
"""

from typing import Any, Callable, Dict, List

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer

from .metrics import (
    _compute_classification_metrics,
    _predict_chunks,
    format_confusion_matrix,
)
from .semeval import AGAINST, ALL_TARGETS, FAVOR


def _favor_against_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    """SemEval's headline metric: the mean of the FAVOR and AGAINST F1s, on a 0-1 scale."""
    # NONE is excluded by the shared task's definition, not by oversight; do not "fix" this
    # into a 3-class macro, or the number stops being comparable with the published ones.
    return float(
        f1_score(
            labels, predictions, labels=[AGAINST, FAVOR], average="macro", zero_division=0
        )
    )


def _macro_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Macro F1 over only the classes present, as `mitweet_metrics._facet_scores` does."""
    return float(
        f1_score(
            labels, predictions, average="macro", labels=sorted(set(labels.tolist())),
            zero_division=0,
        )
    )


def _target_scores(
    predictions: np.ndarray,
    labels: np.ndarray,
    targets: np.ndarray,
    scorer: Callable[[np.ndarray, np.ndarray], float],
) -> Dict[str, Any]:
    """Per-target scores and their means. `scorer` is the task's own F1 definition."""
    per_target_f1: Dict[str, float] = {}
    per_target_accuracy: Dict[str, float] = {}
    target_f1: List[float] = []
    target_accuracy: List[float] = []

    for target in range(len(ALL_TARGETS)):
        rows = np.where(targets == target)[0]
        # Targets absent from this split contribute nothing, rather than a zero.
        if rows.size == 0:
            continue
        truth, predicted = labels[rows], predictions[rows]
        score = scorer(predicted, truth)
        accuracy = accuracy_score(truth, predicted)
        target_f1.append(score)
        target_accuracy.append(float(accuracy))
        per_target_f1[f"f1_target_{target}"] = round(score * 100, 2)
        per_target_accuracy[f"accuracy_target_{target}"] = round(float(accuracy) * 100, 2)

    return {
        "target_f1_macro": round(float(np.mean(target_f1)) * 100, 2),
        "target_accuracy": round(float(np.mean(target_accuracy)) * 100, 2),
        **per_target_f1,
        **per_target_accuracy,
    }


def _pooled_scores(predictions: np.ndarray, labels: np.ndarray, stance: bool) -> Dict[str, Any]:
    """The scores computable without the target column: the 3-class ones, plus SemEval's."""
    scores = _compute_classification_metrics(predictions, labels)
    scores["accuracy"] = round(float(accuracy_score(labels, predictions)) * 100, 2)
    if stance:
        scores["f1_favor_against"] = round(_favor_against_f1(predictions, labels) * 100, 2)
    return scores


def _compute(eval_pred, stance: bool) -> Dict[str, Any]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(np.asarray(logits), axis=-1)
    labels = np.asarray(labels).astype(np.int64)
    print(format_confusion_matrix(labels, predictions))
    return _pooled_scores(predictions, labels, stance)


def compute_stance_metrics(eval_pred) -> Dict[str, Any]:
    """Per-epoch stance scorer. `f1_favor_against` is what the best epoch is selected on."""
    return _compute(eval_pred, stance=True)


def compute_opinion_metrics(eval_pred) -> Dict[str, Any]:
    """Per-epoch opinion scorer; the best epoch is selected on the 3-class `f1_macro`."""
    return _compute(eval_pred, stance=False)


def _batched_predict(
    trainer: Trainer, dataset: Dataset, batch_size: int, stance: bool
) -> Dict[str, Any]:
    """Score a SemEval test split, per target as well as pooled, plus the raw per-example lists.

    `f1_favor_against` is the shared task's own number over the whole split; `target_f1_macro`
    is the mean of the same score computed within each target.
    """
    if not dataset:
        return {}
    logits, labels, ids, targets = _predict_chunks(trainer, dataset, batch_size, "target")
    predicted = np.argmax(logits, axis=1)
    actual = np.asarray(labels).astype(np.int64)
    target_index = np.asarray([int(target) for target in targets], dtype=np.int64)

    print(format_confusion_matrix(actual, predicted))
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"{len(ids) - len(set(ids))} duplicate ids in the SemEval test set. Each row is "
            "one tweet, so ids must be unique; per-example results cannot be joined across "
            "seeds otherwise."
        )

    metrics = _pooled_scores(predicted, actual, stance)
    metrics.update(
        _target_scores(
            predicted, actual, target_index, _favor_against_f1 if stance else _macro_f1
        )
    )
    metrics["preds"] = [int(value) for value in predicted]
    metrics["labels"] = [int(value) for value in actual]
    metrics["ids"] = ids
    metrics["targets"] = [int(target) for target in target_index]
    return metrics


def batched_predict_stance(
    trainer: Trainer, dataset: Dataset, batch_size: int = 32
) -> Dict[str, Any]:
    """Score the stance test split, per-target F1s being SemEval's FAVOR/AGAINST average."""
    return _batched_predict(trainer, dataset, batch_size, stance=True)


def batched_predict_opinion(
    trainer: Trainer, dataset: Dataset, batch_size: int = 32
) -> Dict[str, Any]:
    """Score the opinion test split, per-target F1s being macro over the classes present."""
    return _batched_predict(trainer, dataset, batch_size, stance=False)
