from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)
from typing import Dict, Any, List, Optional
from transformers import Trainer
from datasets import Dataset
import numpy as np
import torch
import torch.nn.functional as F


class CustomTrainer(Trainer):
    """Custom Trainer that supports different loss functions."""

    def __init__(self, loss_fn: Optional[torch.nn.Module] = None, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):
        """Override compute_loss to use custom loss function if provided."""
        labels = inputs.get("bias_labels")
        outputs = model(**inputs)
        logits = outputs.get("bias_logits")
        loss = outputs.get("bias_loss")

        # if not labels:
        #     labels = inputs.get("labels")
        #     outputs = model(**inputs)
        #     logits = outputs.get("logits")
        #     loss = outputs.get("loss")

        # Only compute custom loss if not already computed by model
        if loss is None and self.loss_fn is not None:
            # Standard loss functions expect class indices
            loss = self.loss_fn(logits, labels)
        elif loss is None:
            # Use default cross-entropy loss
            loss = F.cross_entropy(logits, labels)

        # Create outputs dict with logits for evaluation
        if not isinstance(outputs, dict):
            outputs = {"logits": logits, "loss": loss}
        elif "logits" not in outputs:
            outputs["logits"] = logits

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.get("bias_labels")
        # if not labels:
        #     labels = inputs.get("labels")

        with torch.no_grad():
            outputs = model(**inputs)

            # Extract bias-specific outputs
            if isinstance(outputs, dict):
                loss = outputs.get("bias_loss")
                logits = outputs.get("bias_logits")
                # if not loss:
                #     loss = outputs.get("loss")
                #     logits = outputs.get("logits")
            else:
                loss = outputs.loss if hasattr(outputs, "loss") else None
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)


def _compute_classification_metrics(
    predictions: np.ndarray, labels: np.ndarray
) -> Dict[str, Any]:
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    if predictions.ndim != 1:
        raise ValueError(
            f"Expected predictions shape (batch_size,), got {predictions.shape}"
        )
    if labels.ndim != 1:
        raise ValueError(f"Expected labels shape (batch_size,), got {labels.shape}")
    if predictions.shape[0] != labels.shape[0]:
        raise ValueError(
            "Predictions and labels must have the same number of samples, "
            f"got {predictions.shape[0]} and {labels.shape[0]}"
        )

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
    classes = np.unique(labels)
    _, _, per_class_f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=classes, zero_division=0
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

    return {
        "accuracy": round(float(total_accuracy) * 100, 2),
        "f1_macro": round(float(f1_macro) * 100, 2),
        "f1_micro": round(float(f1_micro) * 100, 2),
        "f1_weighted": round(float(f1_weighted) * 100, 2),
        "precision_macro": round(float(precision_macro) * 100, 2),
        "recall_macro": round(float(recall_macro) * 100, 2),
        "precision_micro": round(float(precision_micro) * 100, 2),
        "recall_micro": round(float(recall_micro) * 100, 2),
        **{k: round(float(v) * 100, 2) for k, v in per_label_accuracy.items()},
        **{k: round(float(v) * 100, 2) for k, v in per_label_f1.items()},
    }


def compute_metrics(eval_pred) -> Dict[str, Any]:
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    logits = np.asarray(logits)
    labels = np.asarray(labels)

    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits shape (batch_size, num_labels), got {logits.shape}"
        )
    if labels.ndim != 1:
        raise ValueError(f"Expected labels shape (batch_size,), got {labels.shape}")

    predictions = np.argmax(logits, axis=-1)
    return _compute_classification_metrics(predictions, labels)


def batched_predict_metrics_trainer(
    trainer: Trainer, dataset: Dataset, batch_size: int = 64
) -> Dict[str, Any]:
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_ids: List[Any] = []

    if not dataset:
        return {}

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        chunk = dataset.select(range(start, end))
        ids = chunk["id"]
        output = trainer.predict(chunk)

        logits = output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]

        logits = np.asarray(logits)
        if logits.ndim != 2:
            raise ValueError(
                f"Expected logits shape (batch_size, num_labels), got {logits.shape}"
            )

        labels = np.asarray(output.label_ids)
        all_logits.append(logits)
        all_labels.append(labels)
        all_ids.extend(ids)

    # Concatenate all results
    logits = np.concatenate(all_logits, axis=0)
    predictions = np.argmax(logits, axis=1).tolist()
    labels = np.concatenate(all_labels, axis=0).tolist()
    ids = list(all_ids)

    id_to_logits_labels = {}
    for idx, pred, label in zip(ids, predictions, labels):
        if idx not in id_to_logits_labels.keys():
            id_to_logits_labels[idx] = [(pred, label)]
        else:
            id_to_logits_labels[idx].append((pred, label))

    ids = list(id_to_logits_labels.keys())
    values = [id_to_logits_labels[idx] for idx in ids]
    values = [max(set(tuples), key=tuples.count) for tuples in values]
    predictions = [int(tuples[0]) for tuples in values]
    labels = [int(tuples[1]) for tuples in values]

    metrics = _compute_classification_metrics(np.array(predictions), np.array(labels))

    metrics["preds"] = predictions
    metrics["labels"] = labels
    metrics["ids"] = ids

    return metrics
