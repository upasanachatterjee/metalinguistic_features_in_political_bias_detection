"""Metrics for the two BASIL tasks. The paper reports positive-class precision/recall/F1, so
`f1_positive` is what the best epoch is selected on; macro-F1 is reported alongside it.
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

BASIL_CLASSES = (0, 1)
SOURCES = ("fox", "hpo", "nyt")


class BasilTrainer(CustomTrainer):
    """CustomTrainer that computes the weighted cross-entropy itself, so MultiTaskRoberta and
    the HF baselines are trained on the identical objective rather than each head's own default.
    """

    def __init__(self, *args, class_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        labels, logits, _ = _unpack(inputs, outputs)
        # Unweighted, a 5.6%-positive task collapses to predicting the negative class.
        weight = torch.tensor(
            self.class_weights, device=logits.device, dtype=logits.dtype
        )
        loss = F.cross_entropy(logits, labels.long(), weight=weight)
        if not isinstance(outputs, dict):
            outputs = {"logits": logits, "loss": loss}
        else:
            outputs["logits"] = logits
            outputs["loss"] = loss
        return (loss, outputs) if return_outputs else loss


def _positive_scores(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """BASIL's reported numbers: precision, recall and F1 of the biased class alone."""
    return {
        "f1_positive": round(
            float(f1_score(labels, predictions, pos_label=1, zero_division=0)) * 100, 2
        ),
        "precision_positive": round(
            float(precision_score(labels, predictions, pos_label=1, zero_division=0)) * 100, 2
        ),
        "recall_positive": round(
            float(recall_score(labels, predictions, pos_label=1, zero_division=0)) * 100, 2
        ),
    }


def _pooled_scores(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Everything computable without the source column: the paper's metric plus the 2-class ones."""
    scores = _compute_classification_metrics(predictions, labels)
    scores["accuracy"] = round(float(accuracy_score(labels, predictions)) * 100, 2)
    scores.update(_positive_scores(predictions, labels))
    return scores


def _source_scores(
    predictions: np.ndarray, labels: np.ndarray, sources: np.ndarray
) -> Dict[str, Any]:
    """Positive-class F1 within each outlet, so a model that only works on one is visible."""
    per_source: Dict[str, float] = {}
    source_f1: List[float] = []
    for source in SOURCES:
        rows = np.where(sources == source)[0]
        # An outlet absent from this split contributes nothing, rather than a zero.
        if rows.size == 0:
            continue
        score = float(
            f1_score(labels[rows], predictions[rows], pos_label=1, zero_division=0)
        )
        source_f1.append(score)
        per_source[f"f1_source_{source}"] = round(score * 100, 2)
    return {
        "source_f1_positive": round(float(np.mean(source_f1)) * 100, 2),
        **per_source,
    }


def compute_basil_metrics(eval_pred) -> Dict[str, Any]:
    """Per-epoch scorer. `f1_positive` is what the best epoch is selected on."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(np.asarray(logits), axis=-1)
    labels = np.asarray(labels).astype(np.int64)
    # print(format_confusion_matrix(labels, predictions, BASIL_CLASSES))
    return _pooled_scores(predictions, labels)


def batched_predict_basil(
    trainer: Trainer, dataset: Dataset, batch_size: int = 32
) -> Dict[str, Any]:
    """Score a BASIL test split, per outlet as well as pooled, plus the raw per-example lists."""
    if not dataset:
        return {}
    logits, labels, ids, sources = _predict_chunks(trainer, dataset, batch_size, "source")
    predicted = np.argmax(logits, axis=1)
    actual = np.asarray(labels).astype(np.int64)

    # print(format_confusion_matrix(actual, predicted, BASIL_CLASSES))
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"{len(ids) - len(set(ids))} duplicate ids in the BASIL test set. Each row is one "
            "sentence, so ids must be unique; per-example results cannot be joined across "
            "seeds otherwise."
        )

    metrics = _pooled_scores(predicted, actual)
    metrics.update(_source_scores(predicted, actual, np.asarray(sources)))
    metrics["preds"] = [int(value) for value in predicted]
    metrics["labels"] = [int(value) for value in actual]
    metrics["ids"] = ids
    metrics["sources"] = list(sources)
    return metrics
