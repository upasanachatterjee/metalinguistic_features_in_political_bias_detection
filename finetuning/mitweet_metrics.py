"""Metrics for the two MITweet tasks, reproducing the paper's definitions. Their "Micro-F1"
is sklearn-macro over the three ideology classes pooled across facets, so the names are kept literal.
"""

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import Trainer

from .metrics import (
    CustomTrainer,
    _compute_classification_metrics,
    _predict_chunks,
    _unpack,
    format_confusion_matrix,
)
from .mitweet import NUM_FACETS, RELEVANCE_POS_WEIGHT


class RelevanceTrainer(CustomTrainer):
    """CustomTrainer that computes the relevance loss itself, so MultiTaskRoberta and the HF
    baselines are trained on the identical weighted BCE rather than each head's own default.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        labels, logits, _ = _unpack(inputs, outputs)
        pos_weight = torch.tensor(
            RELEVANCE_POS_WEIGHT, device=logits.device, dtype=logits.dtype
        )
        loss = F.binary_cross_entropy_with_logits(
            logits, labels.to(logits.dtype), pos_weight=pos_weight
        )
        if not isinstance(outputs, dict):
            outputs = {"logits": logits, "loss": loss}
        else:
            outputs["logits"] = logits
            outputs["loss"] = loss
        return (loss, outputs) if return_outputs else loss


def _relevance_predictions(logits: np.ndarray) -> np.ndarray:
    """Threshold the 12 per-facet sigmoids at 0.5, as MITweet does."""
    return (1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64))) > 0.5).astype(np.int64)


def _relevance_scores(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Per-facet binary scores plus the paper's two headline numbers.

    `f1_macro` is the mean over facets (their Macro-F1); `f1_micro` flattens every cell into
    one binary vector (their Micro-F1).
    """
    per_facet_f1: Dict[str, float] = {}
    per_facet_precision: Dict[str, float] = {}
    per_facet_recall: Dict[str, float] = {}
    facet_f1: List[float] = []

    for facet in range(NUM_FACETS):
        truth, predicted = labels[:, facet], predictions[:, facet]
        # A facet with no positives here has no binary F1; MITweet drops it from the mean.
        if truth.sum() == 0:
            continue
        score = f1_score(truth, predicted, zero_division=0)
        facet_f1.append(float(score))
        per_facet_f1[f"f1_facet_{facet}"] = round(float(score) * 100, 2)
        per_facet_precision[f"precision_facet_{facet}"] = round(
            float(precision_score(truth, predicted, zero_division=0)) * 100, 2
        )
        per_facet_recall[f"recall_facet_{facet}"] = round(
            float(recall_score(truth, predicted, zero_division=0)) * 100, 2
        )

    flat_truth, flat_predicted = labels.reshape(-1), predictions.reshape(-1)
    return {
        "f1_macro": round(float(np.mean(facet_f1)) * 100, 2),
        "f1_micro": round(float(f1_score(flat_truth, flat_predicted, zero_division=0)) * 100, 2),
        "precision_micro": round(
            float(precision_score(flat_truth, flat_predicted, zero_division=0)) * 100, 2
        ),
        "recall_micro": round(
            float(recall_score(flat_truth, flat_predicted, zero_division=0)) * 100, 2
        ),
        "accuracy": round(float(accuracy_score(flat_truth, flat_predicted)) * 100, 2),
        **per_facet_f1,
        **per_facet_precision,
        **per_facet_recall,
    }


def compute_relevance_metrics(eval_pred) -> Dict[str, Any]:
    """Per-epoch relevance scorer. `f1_micro` is what MITweet selects the best epoch on."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    return _relevance_scores(_relevance_predictions(logits), np.asarray(labels).astype(np.int64))


def _facet_scores(predictions: np.ndarray, labels: np.ndarray, facets: np.ndarray) -> Dict[str, Any]:
    """Per-facet ideology scores, and the means MITweet reports as Macro-F1 and Macro-Acc."""
    per_facet_f1: Dict[str, float] = {}
    per_facet_accuracy: Dict[str, float] = {}
    facet_f1: List[float] = []
    facet_accuracy: List[float] = []

    for facet in range(NUM_FACETS):
        rows = np.where(facets == facet)[0]
        # Facets absent from this split contribute nothing, rather than a zero.
        if rows.size == 0:
            continue
        truth, predicted = labels[rows], predictions[rows]
        # Macro over only the classes this facet actually shows, as in MITweet's scorer.
        score = f1_score(
            truth, predicted, average="macro", labels=sorted(set(truth.tolist())), zero_division=0
        )
        accuracy = accuracy_score(truth, predicted)
        facet_f1.append(float(score))
        facet_accuracy.append(float(accuracy))
        per_facet_f1[f"f1_facet_{facet}"] = round(float(score) * 100, 2)
        per_facet_accuracy[f"accuracy_facet_{facet}"] = round(float(accuracy) * 100, 2)

    return {
        "facet_f1_macro": round(float(np.mean(facet_f1)) * 100, 2),
        "facet_accuracy": round(float(np.mean(facet_accuracy)) * 100, 2),
        **per_facet_f1,
        **per_facet_accuracy,
    }


def batched_predict_relevance(
    trainer: Trainer, dataset: Dataset, batch_size: int = 32
) -> Dict[str, Any]:
    """Score the relevance test split, returning the metrics plus the raw 12-vectors and ids."""
    if not dataset:
        return {}
    logits, labels, ids, _ = _predict_chunks(trainer, dataset, batch_size)
    predictions = _relevance_predictions(logits)
    labels = np.asarray(labels).astype(np.int64)

    metrics = _relevance_scores(predictions, labels)
    metrics["preds"] = predictions.tolist()
    metrics["labels"] = labels.tolist()
    metrics["ids"] = ids
    return metrics


def batched_predict_ideology(
    trainer: Trainer, dataset: Dataset, batch_size: int = 32
) -> Dict[str, Any]:
    """Score the ideology test split. Pooled metrics are MITweet's Micro-*, the `facet_*` means
    are their Macro-*.
    """
    if not dataset:
        return {}
    logits, labels, ids, facets = _predict_chunks(trainer, dataset, batch_size, "facet")
    predicted = np.argmax(logits, axis=1)
    actual = np.asarray(labels).astype(np.int64)
    facet_index = np.asarray([int(facet) for facet in facets], dtype=np.int64)

    print(format_confusion_matrix(actual, predicted))
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"{len(ids) - len(set(ids))} duplicate ids in the ideology test set. Each row is "
            "one (tweet, facet) pair, so ids must be unique; per-example results cannot be "
            "joined across seeds otherwise."
        )

    metrics = _compute_classification_metrics(predicted, actual)
    metrics["accuracy"] = round(float(accuracy_score(actual, predicted)) * 100, 2)
    metrics.update(_facet_scores(predicted, actual, facet_index))
    metrics["preds"] = [int(value) for value in predicted]
    metrics["labels"] = [int(value) for value in actual]
    metrics["ids"] = ids
    metrics["facets"] = [int(facet) for facet in facet_index]
    return metrics
