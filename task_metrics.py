"""Evaluation metrics for the tone and theme objectives.

Pretraining loss alone is a poor read on whether these two heads learned
anything useful:

* **Tone MSE** is dominated by the target's mean offset. GDELT tone over this
  corpus has mean ~-2.7 and std ~2.8, so 47% of the MSE of a zero-predicting
  head is the offset alone -- a head that learns the corpus mean and nothing
  else scores respectably. Correlation is what separates that from a head that
  actually tracks the target.
* **Theme BCE** over 2000 mostly-rare labels is dominated by the easy negatives.
  A head that predicts all-zero scores well under 0.5 gets a respectable loss
  and zero F1.

Nothing here is wired into the optimizer; these are read-outs. The gradient
diagnostic collects predictions as it runs and reports them, and fine-tuning /
evaluation code can call the two entry points directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, f1_score

# Labels/predictions with fewer than this many finite pairs make correlation
# meaningless rather than merely noisy.
MIN_CORRELATION_SAMPLES = 3

# Prevalence bands the theme report groups labels into, as (name, low, high)
# with `low <= prevalence < high`.
PREVALENCE_GROUPS = (
    ("rare (<1%)", 0.0, 0.01),
    ("uncommon (1-10%)", 0.01, 0.10),
    ("common (>=10%)", 0.10, 1.01),
)


def _safe_correlation(fn, x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Correlation, or None when it is undefined rather than merely small.

    A constant prediction vector (common very early in training, when the head
    still outputs roughly its bias term) makes the denominator zero; scipy
    returns NaN with a warning. None says "not measurable" instead of pretending
    the correlation is 0.
    """
    if x.size < MIN_CORRELATION_SAMPLES:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    value = float(fn(x, y)[0])
    return None if not np.isfinite(value) else value


def tone_metrics(
    predictions: Sequence[float], targets: Sequence[float]
) -> Dict[str, Any]:
    """Score tone regression, in GDELT tone units throughout.

    Targets are never rescaled, so MSE/RMSE/MAE are directly comparable across
    runs and directly interpretable: an MAE of 2 means the head is off by two
    tone points, on a field whose corpus std is ~2.8.

    The correlations are the point of this function. MSE alone cannot
    distinguish a head that has learned the corpus mean (~-2.7) and nothing else
    from one that tracks the target -- the first scores respectably. Pearson and
    Spearman are scale-free and answer the question MSE gets used as a proxy for.
    The reported prediction spread makes the same failure visible directly: a
    mean-only head has a near-zero std.
    """
    preds = np.asarray(predictions, dtype=np.float64).reshape(-1)
    gold = np.asarray(targets, dtype=np.float64).reshape(-1)
    if preds.shape != gold.shape:
        raise ValueError(
            f"tone predictions {preds.shape} and targets {gold.shape} disagree."
        )
    if preds.size == 0:
        return {"count": 0}

    errors = preds - gold
    return {
        "count": int(preds.size),
        "mse": float(np.mean(errors**2)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "pearson": _safe_correlation(pearsonr, preds, gold),
        "spearman": _safe_correlation(spearmanr, preds, gold),
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
        "target_mean": float(gold.mean()),
        "target_std": float(gold.std()),
    }


def _grouped_theme_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    predicted: np.ndarray,
    prevalence: np.ndarray,
) -> List[Dict[str, Any]]:
    """Macro F1 / AP within each prevalence band.

    Aggregate macro numbers over 2000 labels are dominated by the rare tail;
    splitting them says whether the head learned the common themes and gave up
    on the rare ones, or failed uniformly.
    """
    groups: List[Dict[str, Any]] = []
    for name, low, high in PREVALENCE_GROUPS:
        columns = np.flatnonzero((prevalence >= low) & (prevalence < high))
        if columns.size == 0:
            continue
        # A label with no positives in this evaluation set has no defined AP.
        scorable = columns[labels[:, columns].sum(axis=0) > 0]
        groups.append(
            {
                "group": name,
                "num_labels": int(columns.size),
                "num_scorable_labels": int(scorable.size),
                "f1_macro": float(
                    f1_score(
                        labels[:, columns],
                        predicted[:, columns],
                        average="macro",
                        zero_division=0,
                    )
                ),
                "average_precision_macro": (
                    float(
                        average_precision_score(
                            labels[:, scorable], scores[:, scorable], average="macro"
                        )
                    )
                    if scorable.size
                    else None
                ),
            }
        )
    return groups


def theme_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    group_by_prevalence: bool = True,
) -> Dict[str, Any]:
    """Score multi-label theme prediction.

    `logits` are raw head outputs; they are squashed with a sigmoid here, so a
    `pos_weight`-trained head is scored on the same footing as an unweighted one
    (pos_weight shifts the loss, not the decision rule).

    F1 uses a fixed `threshold`; average precision is threshold-free and is the
    more honest number when almost every label is rare -- report both.
    Prevalence for the grouped breakdown is measured on the labels passed in.
    """
    scores = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    gold = np.asarray(labels).astype(np.int64)
    if scores.shape != gold.shape:
        raise ValueError(
            f"theme logits {scores.shape} and labels {gold.shape} disagree."
        )
    if scores.ndim != 2:
        raise ValueError(f"Expected (examples, labels), got {scores.shape}.")
    if scores.shape[0] == 0:
        return {"count": 0}

    predicted = (scores >= threshold).astype(np.int64)
    # Labels with no positives here have undefined AP; excluding them from the
    # macro average is the difference between "the head is bad" and "the
    # evaluation set never showed this theme".
    scorable = np.flatnonzero(gold.sum(axis=0) > 0)

    metrics: Dict[str, Any] = {
        "count": int(gold.shape[0]),
        "num_labels": int(gold.shape[1]),
        "num_scorable_labels": int(scorable.size),
        "threshold": threshold,
        "f1_micro": float(
            f1_score(gold, predicted, average="micro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(gold, predicted, average="macro", zero_division=0)
        ),
        "positives_per_example_true": float(gold.sum(axis=1).mean()),
        "positives_per_example_pred": float(predicted.sum(axis=1).mean()),
    }

    if scorable.size:
        metrics["average_precision_micro"] = float(
            average_precision_score(
                gold[:, scorable].ravel(), scores[:, scorable].ravel()
            )
        )
        metrics["average_precision_macro"] = float(
            average_precision_score(
                gold[:, scorable], scores[:, scorable], average="macro"
            )
        )
    else:
        metrics["average_precision_micro"] = None
        metrics["average_precision_macro"] = None

    if group_by_prevalence:
        prevalence = gold.mean(axis=0)
        metrics["by_prevalence"] = _grouped_theme_metrics(
            scores, gold, predicted, prevalence
        )
    return metrics


def tone_report_lines(metrics: Dict[str, Any]) -> List[str]:
    if not metrics.get("count"):
        return ["  tone: no predictions collected"]
    lines = ["  tone (GDELT tone units)"]
    for key in ("count", "mse", "rmse", "mae", "pearson", "spearman"):
        value = metrics.get(key)
        if value is None:
            formatted = "undefined"
        elif isinstance(value, int):
            formatted = f"{value:,}"
        else:
            formatted = f"{value:.4f}"
        lines.append(f"    {key:<24}{formatted:>14}")
    lines.append(
        f"    predictions mean/std {metrics['pred_mean']:.3f} / "
        f"{metrics['pred_std']:.3f}  vs targets "
        f"{metrics['target_mean']:.3f} / {metrics['target_std']:.3f}"
    )
    return lines


def theme_report_lines(metrics: Dict[str, Any]) -> List[str]:
    if not metrics.get("count"):
        return ["  themes: no predictions collected"]
    lines = [f"  themes (threshold {metrics['threshold']})"]
    for key in (
        "count",
        "num_scorable_labels",
        "f1_micro",
        "f1_macro",
        "average_precision_micro",
        "average_precision_macro",
        "positives_per_example_true",
        "positives_per_example_pred",
    ):
        value = metrics.get(key)
        if value is None:
            formatted = "undefined"
        elif isinstance(value, int):
            formatted = f"{value:,}"
        else:
            formatted = f"{value:.4f}"
        lines.append(f"    {key:<28}{formatted:>14}")

    for group in metrics.get("by_prevalence", []):
        ap = group["average_precision_macro"]
        lines.append(
            f"    {group['group']:<20}labels={group['num_labels']:>5} "
            f"f1_macro={group['f1_macro']:.4f} "
            f"ap_macro={'n/a' if ap is None else f'{ap:.4f}'}"
        )
    return lines
