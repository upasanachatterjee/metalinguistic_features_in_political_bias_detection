from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)
from typing import Dict, Any, List
from transformers import Trainer, TrainerCallback
from datasets import Dataset
import math
import numpy as np
import torch
import torch.nn.functional as F


def _first_present(mapping, *keys):
    """The first of `keys` present in `mapping` with a non-None value, else None."""
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _unpack(inputs, outputs):
    """Pull (labels, logits, loss) out of a batch/output pair, any of which may be
    None. Prefers MultiTaskRoberta's `relevance_*`/`bias_*` keys over the plain names.
    """
    labels = _first_present(inputs, "relevance_labels", "bias_labels", "labels")

    if isinstance(outputs, dict):
        logits = _first_present(
            outputs, "relevance_logits", "bias_logits", "logits"
        )
        loss = _first_present(outputs, "relevance_loss", "bias_loss", "loss")
    else:
        logits = getattr(outputs, "logits", outputs)
        loss = getattr(outputs, "loss", None)

    return labels, logits, loss


class CustomTrainer(Trainer):
    """Trainer accepting either MultiTaskRoberta or an HF baseline, which disagree on
    output key names; `_unpack` reconciles them and nothing else differs.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        labels, logits, loss = _unpack(inputs, outputs)

        if loss is None:
            loss = F.cross_entropy(logits, labels)
        if not isinstance(outputs, dict):
            outputs = {"logits": logits, "loss": loss}
        elif "logits" not in outputs:
            outputs["logits"] = logits
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            outputs = model(**inputs)
        labels, logits, loss = _unpack(inputs, outputs)

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)


class WholeEpochCounter(TrainerCallback):
    """Round `state.epoch` up at the epoch boundary, so each epoch gets its own row in
    the metrics table and its own `log_history` entry.
    """

    def on_epoch_end(self, args, state, control, **kwargs):
        state.epoch = math.ceil(state.epoch)


BIAS_CLASSES = (0, 1, 2)
_CORNER = "true\\pred"


def format_confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray, classes=BIAS_CLASSES
) -> str:
    """The confusion matrix as printable text, rows = true, columns = predicted."""
    matrix = confusion_matrix(labels, predictions, labels=classes)
    cell = max(len(_CORNER), *(len(str(name)) for name in classes)) + 2
    header = f"{_CORNER:>{cell}}" + "".join(
        f"{name:>{cell}}" for name in classes
    )
    rows = [
        f"{classes[index]:>{cell}}"
        + "".join(f"{count:>{cell}d}" for count in row)
        for index, row in enumerate(matrix)
    ]
    return "\n".join(["confusion matrix", header, *rows])


def _compute_classification_metrics(
    predictions: np.ndarray, labels: np.ndarray
) -> Dict[str, Any]:
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    total_accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_micro = f1_score(labels, predictions, average="micro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    precision_macro = precision_score(
        labels, predictions, average="macro", zero_division=0
    )
    recall_macro = recall_score(labels, predictions, average="macro", zero_division=0)
    precision_micro = precision_score(
        labels, predictions, average="micro", zero_division=0
    )
    recall_micro = recall_score(labels, predictions, average="micro", zero_division=0)

    per_label_accuracy: Dict[str, float] = {}
    per_label_f1: Dict[str, float] = {}
    per_label_precision: Dict[str, float] = {}
    per_label_recall: Dict[str, float] = {}
    classes = np.unique(labels)
    per_class_precision, per_class_recall, per_class_f1, _ = (
        precision_recall_fscore_support(
            labels, predictions, labels=classes, zero_division=0
        )
    )
    for idx, class_id in enumerate(classes):
        class_indices = np.where(labels == class_id)[0]
        if len(class_indices) == 0:
            continue
        class_preds = predictions[class_indices]
        class_labels = labels[class_indices]
        per_label_accuracy[f"accuracy_class_{class_id}"] = accuracy_score(
            class_labels, class_preds
        )
        per_label_f1[f"f1_class_{class_id}"] = per_class_f1[idx]
        per_label_precision[f"precision_class_{class_id}"] = per_class_precision[idx]
        per_label_recall[f"recall_class_{class_id}"] = per_class_recall[idx]

    return {
        # "accuracy": round(float(total_accuracy) * 100, 2),
        "f1_macro": round(float(f1_macro) * 100, 2),
        # "f1_micro": round(float(f1_micro) * 100, 2),
        "f1_weighted": round(float(f1_weighted) * 100, 2),
        "precision_macro": round(float(precision_macro) * 100, 2),
        "recall_macro": round(float(recall_macro) * 100, 2),
        # "precision_micro": round(float(precision_micro) * 100, 2),
        # "recall_micro": round(float(recall_micro) * 100, 2),
        # **{k: round(float(v) * 100, 2) for k, v in per_label_accuracy.items()},
        **{k: round(float(v) * 100, 2) for k, v in per_label_f1.items()},
        **{k: round(float(v) * 100, 2) for k, v in per_label_precision.items()},
        **{k: round(float(v) * 100, 2) for k, v in per_label_recall.items()},
    }


def compute_metrics(eval_pred) -> Dict[str, Any]:
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(np.asarray(logits), axis=-1)
    # print(format_confusion_matrix(np.asarray(labels), predictions))
    return _compute_classification_metrics(predictions, labels)


def batched_predict_metrics_trainer(
    trainer: Trainer, dataset: Dataset, batch_size: int = 64
) -> Dict[str, Any]:
    """Predict over the dataset in slices and score it, returning the metrics plus raw
    preds, labels and ids for error analysis.
    """
    # Sliced only to bound peak memory; one row is already one article.
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_ids: List[Any] = []

    if not dataset:
        return {}

    chunk_compute_metrics, trainer.compute_metrics = trainer.compute_metrics, None

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        chunk = dataset.select(range(start, end))
        ids = chunk["ID"]
        output = trainer.predict(chunk)

        logits = output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]

        logits = np.asarray(logits)
        labels = np.asarray(output.label_ids)
        all_logits.append(logits)
        all_labels.append(labels)
        all_ids.extend(ids)

    trainer.compute_metrics = chunk_compute_metrics

    predicted = np.argmax(np.concatenate(all_logits, axis=0), axis=1)
    actual = np.concatenate(all_labels, axis=0)
    ids = list(all_ids)

    # print(format_confusion_matrix(actual, predicted))

    if len(set(ids)) != len(ids):
        print(
            f"WARNING: {len(ids) - len(set(ids))} duplicate ids in the evaluation "
            "set. Metrics are per row, not per article; aggregate before comparing "
            "against earlier results."
        )

    metrics = _compute_classification_metrics(predicted, actual)

    metrics["preds"] = [int(p) for p in predicted]
    metrics["labels"] = [int(v) for v in actual]
    metrics["ids"] = ids

    return metrics


def _predict_chunks(trainer: Trainer, dataset: Dataset, batch_size: int, extra_column=None):
    """Predict over the dataset in slices, returning (logits, labels, ids, extra)."""
    # Sliced only to bound peak memory, as in `metrics.batched_predict_metrics_trainer`.
    chunk_compute_metrics, trainer.compute_metrics = trainer.compute_metrics, None

    all_logits, all_labels, all_ids, all_extra = [], [], [], []
    for start in range(0, len(dataset), batch_size):
        chunk = dataset.select(range(start, min(start + batch_size, len(dataset))))
        output = trainer.predict(chunk)
        logits = output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        all_logits.append(np.asarray(logits))
        all_labels.append(np.asarray(output.label_ids))
        all_ids.extend(chunk["ID"])
        if extra_column is not None:
            all_extra.extend(chunk[extra_column])

    trainer.compute_metrics = chunk_compute_metrics
    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_labels, axis=0),
        list(all_ids),
        all_extra,
    )
