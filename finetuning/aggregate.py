"""Read-outs over the per-seed fine-tuning results: mean (std) across seeds, the
majority-vote ensemble, how much the seeds disagree, and which classes get confused.

One seed is one sample, and the gaps between the tlp_* ablations are the same size as
the gaps between seeds.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# The scorer the JSONs were written with, so the ensemble stays on their x100 scale.
from .metrics import _compute_classification_metrics

# Written by `experiments.run_single`, via `make_experiment_name`.
METRICS_SUFFIX = "_baseline_trunc_test_metrics.json"

# Everything in the JSON that is not a scalar metric. `fold` is an int but indexes the data
# partition, so it would otherwise be summarised as though it were a score.
_NON_METRIC_KEYS = ("preds", "labels", "ids", "training_args", "fold")

# Per-class keys are excluded: they exist only for classes present in the labels.
DEFAULT_SUMMARY_METRICS = (
    "accuracy",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
)

_SEED_DIR_RE = re.compile(r"seed_(-?\d+)$")
_FOLD_DIR_RE = re.compile(r"(?:^|/)fold_(\d+)/")

# Labels are ordinal (0=left, 1=center, 2=right), so distance 2 is a polarity flip.
NUM_BIAS_CLASSES = 3
POLARITY_FLIP_DISTANCE = 2


@dataclass
class RunResult:
    """One metrics JSON: one model, one seed, one test split. `fold` is None for the studies
    that run a single fixed split.
    """

    model: str
    seed: int
    path: str
    metrics: Dict[str, float]
    preds: np.ndarray
    labels: np.ndarray
    ids: List[str]
    fold: Optional[int] = None


def load_result(path: str) -> RunResult:
    """Read one `*_test_metrics.json` into a RunResult. The model name comes from the
    filename stem and the seed from the `seed_N` directory; neither is in the JSON.
    """
    with open(path) as f:
        payload = json.load(f)

    filename = os.path.basename(path)
    if not filename.endswith(METRICS_SUFFIX):
        raise ValueError(f"{path} is not a {METRICS_SUFFIX} file")
    model = filename[: -len(METRICS_SUFFIX)]

    seed_dir = os.path.basename(os.path.dirname(path))
    match = _SEED_DIR_RE.match(seed_dir)
    if match is None:
        raise ValueError(
            f"{path} is not inside a seed_N directory (found {seed_dir!r}); "
            "the seed cannot be recovered, as it is not stored in the JSON"
        )

    for key in ("preds", "labels", "ids"):
        if key not in payload:
            raise ValueError(
                f"{path} has no {key!r}. It predates the per-example lists added "
                "to batched_predict_metrics_trainer and cannot be aggregated."
            )

    metrics = {
        key: value
        for key, value in payload.items()
        if key not in _NON_METRIC_KEYS and isinstance(value, (int, float))
    }

    # Written by the runner for the folded studies; otherwise recovered from the path.
    fold = payload.get("fold")
    if fold is None:
        from_path = _FOLD_DIR_RE.search(path)
        fold = int(from_path.group(1)) if from_path else None

    return RunResult(
        model=model,
        seed=int(match.group(1)),
        path=path,
        metrics=metrics,
        preds=np.asarray(payload["preds"], dtype=np.int64),
        labels=np.asarray(payload["labels"], dtype=np.int64),
        ids=[str(i) for i in payload["ids"]],
        fold=fold,
    )


def discover_results(root: str) -> List[RunResult]:
    """Load every metrics JSON under `root/seed_*/`, e.g.
    `results_undersampling/media_split`, sorted by (model, seed).
    """
    paths = sorted(glob.glob(os.path.join(root, "seed_*", f"*{METRICS_SUFFIX}")))
    results = [load_result(p) for p in paths]
    return sorted(results, key=lambda r: (r.model, r.seed))


def discover_folds(root: str) -> List[RunResult]:
    """Load every metrics JSON under `root/fold_*/seed_*/`, for the studies that run folds."""
    paths = sorted(glob.glob(os.path.join(root, "fold_*", "seed_*", f"*{METRICS_SUFFIX}")))
    # `discover_results` on a folded tree returns [] rather than failing, which reads as
    # "these runs scored nothing" instead of "you pointed at the wrong level".
    if not paths:
        raise ValueError(
            f"no {METRICS_SUFFIX} files under {root}/fold_*/seed_*/. For a single-split study "
            "use discover_results(root) instead."
        )
    results = [load_result(p) for p in paths]
    return sorted(results, key=lambda r: (r.model, r.fold if r.fold is not None else -1, r.seed))


def by_model(results: Sequence[RunResult]) -> Dict[str, List[RunResult]]:
    """Group runs by model name, each group sorted by seed."""
    grouped: Dict[str, List[RunResult]] = {}
    for result in results:
        grouped.setdefault(result.model, []).append(result)
    for runs in grouped.values():
        runs.sort(key=lambda r: r.seed)
    return grouped


def coverage(results: Sequence[RunResult]) -> pd.DataFrame:
    """Which seeds each model actually has on disk. Worth checking before any summary,
    since a model whose runs crashed gets a mean over fewer seeds without saying so.
    """
    rows = [
        {
            "model": model,
            "n_seeds": len(runs),
            "seeds": ", ".join(str(r.seed) for r in runs),
            # First axis, not `.size`: relevance labels are one 12-vector per row.
            "test_rows": len(runs[0].labels),
        }
        for model, runs in sorted(by_model(results).items())
    ]
    return pd.DataFrame(rows)


def summarize(
    results: Sequence[RunResult],
    metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Mean and sample standard deviation (ddof=1) of each metric across seeds, per
    model, on the JSONs' percentage scale.
    """
    # std is None, not NaN, for one seed, as in `task_metrics._safe_correlation`.
    wanted = tuple(metrics) if metrics is not None else DEFAULT_SUMMARY_METRICS

    rows = []
    for model, runs in sorted(by_model(results).items()):
        for metric in wanted:
            values = np.asarray(
                [r.metrics[metric] for r in runs if metric in r.metrics],
                dtype=np.float64,
            )
            if values.size == 0:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if values.size > 1 else None
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": round(mean, 2),
                    "std": None if std is None else round(std, 2),
                    "n": int(values.size),
                    "min": round(float(values.min()), 2),
                    "max": round(float(values.max()), 2),
                    "mean_std": (
                        f"{mean:.2f}" if std is None else f"{mean:.2f} ({std:.2f})"
                    ),
                }
            )
    return pd.DataFrame(
        rows, columns=["model", "metric", "mean", "std", "n", "min", "max", "mean_std"]
    )


def summary_table(
    results: Sequence[RunResult],
    metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """`summarize` pivoted to models x metrics of "mean (std)" strings -- the shape to
    paste into a write-up. Use `summarize` to compute with.
    """
    long = summarize(results, metrics)
    if long.empty:
        return long
    wide = long.pivot(index="model", columns="metric", values="mean_std")
    wanted = list(metrics) if metrics is not None else list(DEFAULT_SUMMARY_METRICS)
    return wide[[m for m in wanted if m in wide.columns]]


def align_by_id(
    runs: Sequence[RunResult],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Line the runs' predictions up on a common example order, returning
    `(ids, labels, preds)` with `preds` shaped `(n_runs, n_examples)`.
    """
    # Joined on id, not position: misalignment would mean wrong numbers, silently.
    if not runs:
        raise ValueError("no runs to align")

    for run in runs:
        # First axis, not `.size`: MITweet relevance predicts a 12-vector per row, so one id
        # covers twelve cells and comparing totals would reject every relevance run.
        if run.preds.shape != run.labels.shape or len(run.preds) != len(run.ids):
            raise ValueError(
                f"{run.path}: preds/labels/ids lengths disagree "
                f"({run.preds.shape}/{run.labels.shape}/{len(run.ids)})"
            )
        if len(set(run.ids)) != len(run.ids):
            duplicates = len(run.ids) - len(set(run.ids))
            raise ValueError(
                f"{run.path}: {duplicates} duplicate ids. Predictions cannot be "
                "joined on id; aggregate the duplicates before comparing seeds."
            )

    reference = runs[0]
    ids = list(reference.ids)
    id_set = set(ids)
    for run in runs[1:]:
        if set(run.ids) != id_set:
            only_ref = len(id_set - set(run.ids))
            only_run = len(set(run.ids) - id_set)
            raise ValueError(
                f"{run.path} scored a different test set from {reference.path} "
                f"({only_ref} ids missing, {only_run} extra). These runs are not "
                "comparable per example."
            )

    labels = reference.labels
    # Trailing axes come from the run itself, so a multi-label (n, 12) run stacks correctly.
    preds = np.empty((len(runs), *reference.preds.shape), dtype=np.int64)
    for row, run in enumerate(runs):
        order = {example_id: i for i, example_id in enumerate(run.ids)}
        index = np.fromiter((order[i] for i in ids), dtype=np.int64, count=len(ids))
        preds[row] = run.preds[index]
        if not np.array_equal(run.labels[index], labels):
            raise ValueError(
                f"{run.path} and {reference.path} disagree on the gold label for "
                "at least one id. The runs used different label mappings."
            )

    return ids, labels, preds


def _f1_per_class(confusion: np.ndarray) -> np.ndarray:
    """Per-class F1 from a confusion matrix, rows = true, columns = predicted."""
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    # A class with no true and no predicted rows scores 0, as sklearn's zero_division=0 does.
    return np.divide(
        2 * true_positive, denominator, out=np.zeros_like(true_positive), where=denominator > 0
    )


def _f1_macro(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> float:
    scores = _f1_per_class(_confusion(labels, preds, num_classes))
    # sklearn's macro averages over the classes present in either labels or predictions.
    present = np.union1d(np.unique(labels), np.unique(preds))
    return float(scores[present].mean())


def _f1_positive(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> float:
    """BASIL's reported metric: F1 of the biased class alone."""
    return float(_f1_per_class(_confusion(labels, preds, num_classes))[1])


def _f1_favor_against(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> float:
    """SemEval's own metric: the mean of the FAVOR and AGAINST F1s, ignoring NONE."""
    scores = _f1_per_class(_confusion(labels, preds, num_classes))
    return float(scores[[0, 2]].mean())


def _f1_micro_multilabel(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> float:
    """MITweet relevance: every facet cell flattened into one binary vector.

    Positive-class F1, matching `mitweet_metrics`'s own `f1_micro`, which calls `f1_score`
    with no `average`. sklearn's `average="micro"` on binary input is accuracy, not this.
    """
    true_positive = float(np.sum((labels == 1) & (preds == 1)))
    denominator = float(np.sum(labels == 1) + np.sum(preds == 1))
    return 2 * true_positive / denominator if denominator else 0.0


# Each study is compared on the metric it reports, not on a lowest common denominator.
PAIRED_METRICS = {
    "f1_macro": _f1_macro,
    "f1_positive": _f1_positive,
    "f1_favor_against": _f1_favor_against,
    "f1_micro": _f1_micro_multilabel,
}


def pooled_out_of_fold(runs: Sequence[RunResult], metric: str = "f1_macro") -> Dict[str, Any]:
    """Score one metric over every fold's predictions at once, each row predicted exactly once.

    The right read-out when a fold's own test set is not representative -- an AllSides media
    fold holds out whole outlets, so its test rows are nearly single-class and a per-fold macro
    F1 says little. Requires the folds to be a true partition; raises on overlapping ids, which
    is what a repeated-random-split study like BASIL produces.
    """
    if not runs:
        raise ValueError("no runs to pool")
    scorer = PAIRED_METRICS[metric]

    seen: Dict[str, int] = {}
    for run in runs:
        for example_id in run.ids:
            if example_id in seen:
                raise ValueError(
                    f"id {example_id!r} appears in more than one fold, so these runs are not a "
                    "partition and pooling would score some rows twice. Use the mean over folds."
                )
            seen[example_id] = 1

    preds = np.concatenate([run.preds for run in runs], axis=0)
    labels = np.concatenate([run.labels for run in runs], axis=0)
    num_classes = int(max(labels.max(), preds.max())) + 1
    return {
        "metric": metric,
        "score": round(float(scorer(labels, preds, num_classes)) * 100, 2),
        "n_examples": len(labels),
        "n_folds": len({run.fold for run in runs}),
    }


def paired_bootstrap(
    runs_a: Sequence[RunResult],
    runs_b: Sequence[RunResult],
    metric: str = "f1_macro",
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Bootstrap CI on (model A - model B), joined per example so the pairing is kept.

    Resamples test examples *and* seeds, so the interval covers both the test split and the
    training noise; with five seeds the latter component is coarse. Returns the mean difference
    on the JSONs' x100 scale, its 95% percentile interval, and the share of resamples favouring A.
    """
    if not runs_a or not runs_b:
        raise ValueError("both models need at least one run")
    scorer = PAIRED_METRICS[metric]

    # Aligned together, so `align_by_id` is the one place that checks both models scored the
    # same examples with the same gold labels.
    ids, labels, preds = align_by_id(list(runs_a) + list(runs_b))
    preds_a, preds_b = preds[: len(runs_a)], preds[len(runs_a) :]

    rng = np.random.default_rng(seed)
    n = len(ids)
    num_classes = int(max(labels.max(), preds.max())) + 1
    diffs = np.empty(n_boot)
    for step in range(n_boot):
        rows = rng.integers(0, n, n)
        truth = labels[rows]
        # Every seed is scored once on this resample, then a bootstrap sample *of the seeds* is
        # averaged. That gives the sampling distribution of the multi-seed mean, which is the
        # claim being made -- not of a single training run, which is far more pessimistic.
        scores_a = np.array([scorer(truth, p[rows], num_classes) for p in preds_a])
        scores_b = np.array([scorer(truth, p[rows], num_classes) for p in preds_b])
        score_a = scores_a[rng.integers(0, len(runs_a), len(runs_a))].mean()
        score_b = scores_b[rng.integers(0, len(runs_b), len(runs_b))].mean()
        diffs[step] = (score_a - score_b) * 100

    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "metric": metric,
        "mean_diff": round(float(diffs.mean()), 2),
        "ci_low": round(float(low), 2),
        "ci_high": round(float(high), 2),
        "share_favouring_a": round(float((diffs > 0).mean()), 3),
        # The interval excluding zero is the claim; anything else is a wash.
        "distinguishable": bool(low > 0 or high < 0),
        "n_examples": n,
        "n_boot": n_boot,
    }


def pairwise_table(
    results: Sequence[RunResult],
    reference: str,
    metric: str = "f1_macro",
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Every model against one baseline, as `model - reference` with a bootstrap interval.

    This is the comparison to read, not two independent mean (std) columns: both models scored
    the same examples, and pairing recovers the covariance that a side-by-side table discards.
    """
    grouped = by_model(results)
    if reference not in grouped:
        raise ValueError(f"no runs for reference model {reference!r}; have {sorted(grouped)}")

    rows = []
    for model, runs in sorted(grouped.items()):
        if model == reference:
            continue
        outcome = paired_bootstrap(runs, grouped[reference], metric, n_boot, seed)
        rows.append({"model": model, "vs": reference, **outcome})
    return pd.DataFrame(rows)


def consistency_table(
    roots: Dict[str, str],
    reference: str,
    metrics: Dict[str, str],
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """`pairwise_table` over several study directories, one row per (study, model).

    `roots` maps a label to a directory holding `seed_*`; `metrics` maps the same label to a
    `PAIRED_METRICS` key, since each study is scored on its own metric.
    """
    frames = []
    for study, root in roots.items():
        results = discover_results(root)
        if not results:
            print(f"  SKIP {study}: no runs under {root}")
            continue
        table = pairwise_table(results, reference, metrics[study], n_boot, seed)
        frames.append(table.assign(study=study))
    if not frames:
        raise ValueError("no runs found under any root")
    combined = pd.concat(frames, ignore_index=True)
    return combined[["study", "model", "vs", "metric", "mean_diff", "ci_low", "ci_high",
                     "distinguishable", "n_examples"]]


def sign_test(table: pd.DataFrame) -> pd.DataFrame:
    """Per model, how many studies it beats the reference on, with an exact binomial p.

    The point of running four corpora: a method that wins on 12 of 13 variants is evidence no
    single-corpus confidence interval can reach, however tight.
    """
    rows = []
    for model, group in table.groupby("model"):
        wins = int((group["mean_diff"] > 0).sum())
        losses = int((group["mean_diff"] < 0).sum())
        trials = wins + losses
        # Two-sided exact binomial against p=0.5, ties dropped as the sign test requires.
        if trials == 0:
            p_value = None
        else:
            extreme = max(wins, losses)
            tail = sum(comb(trials, k) for k in range(extreme, trials + 1)) / 2 ** trials
            p_value = round(min(1.0, 2 * tail), 4)
        rows.append({
            "model": model, "wins": wins, "losses": losses, "n_studies": trials,
            "mean_diff": round(float(group["mean_diff"].mean()), 2), "sign_test_p": p_value,
        })
    return pd.DataFrame(rows).sort_values("mean_diff", ascending=False).reset_index(drop=True)


def _vote_counts(preds: np.ndarray, num_classes: int) -> np.ndarray:
    """Per-example votes per class, shape (n_examples, num_classes)."""
    counts = np.zeros((preds.shape[1], num_classes), dtype=np.int64)
    for row in preds:
        counts[np.arange(preds.shape[1]), row] += 1
    return counts


def majority_vote(runs: Sequence[RunResult]) -> Dict[str, Any]:
    """Score the per-example modal prediction across seeds. Comparing it against the
    mean single seed says how much of a model's error is seed noise.
    """
    # Ties (5 seeds, 3 classes) break toward the lowest class; tie_rate counts them.
    ids, labels, preds = align_by_id(runs)
    num_classes = int(max(preds.max(), labels.max())) + 1

    counts = _vote_counts(preds, num_classes)
    top = counts.max(axis=1)
    voted = counts.argmax(axis=1)
    tied = (counts == top[:, None]).sum(axis=1) > 1

    metrics = _compute_classification_metrics(voted, labels)
    per_seed_f1 = np.asarray(
        [r.metrics["f1_macro"] for r in runs if "f1_macro" in r.metrics],
        dtype=np.float64,
    )

    return {
        **metrics,
        "n_seeds": len(runs),
        "n_examples": len(ids),
        "tie_rate": round(float(tied.mean()) * 100, 2),
        "mean_single_seed_f1_macro": (
            round(float(per_seed_f1.mean()), 2) if per_seed_f1.size else None
        ),
        "gain_f1_macro": (
            round(float(metrics["f1_macro"] - per_seed_f1.mean()), 2)
            if per_seed_f1.size
            else None
        ),
        "preds": [int(p) for p in voted],
        "labels": [int(v) for v in labels],
        "ids": ids,
    }


def _fleiss_kappa(counts: np.ndarray) -> Optional[float]:
    """Chance-corrected agreement among the seeds, treating each as a rater. None when
    undefined -- every seed predicting one class makes expected agreement 1.
    """
    n_examples, _ = counts.shape
    if n_examples == 0:
        return None
    n_raters = int(counts[0].sum())
    if n_raters < 2:
        return None

    # Per-example agreement: the fraction of rater pairs that match.
    agreement = (
        (counts * (counts - 1)).sum(axis=1) / (n_raters * (n_raters - 1))
    ).mean()
    # Expected agreement if every rater drew from the marginal class distribution.
    marginals = counts.sum(axis=0) / (n_examples * n_raters)
    expected = float((marginals**2).sum())

    if np.isclose(expected, 1.0):
        return None
    return float((agreement - expected) / (1.0 - expected))


def _pair_disagreement_counts(preds: np.ndarray, num_classes: int) -> np.ndarray:
    """Unordered class-pair counts over every (seed pair, example) comparison.

    Upper-triangular, shape (num_classes, num_classes): seeds are exchangeable,
    so `a` predicting left while `b` predicts right is the same event as the
    reverse.
    """
    n_runs = preds.shape[0]
    pair_counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    for a, b in combinations(range(n_runs), 2):
        differ = preds[a] != preds[b]
        low = np.minimum(preds[a][differ], preds[b][differ])
        high = np.maximum(preds[a][differ], preds[b][differ])
        np.add.at(pair_counts, (low, high), 1)
    return pair_counts


def disagreement(runs: Sequence[RunResult]) -> Dict[str, Any]:
    """How much the seeds of one model disagree per example -- two models can post the
    same mean macro-F1 while only one is stable.
    """
    # Unanimously-wrong is the model or the data; contested is the seed alone.
    ids, labels, preds = align_by_id(runs)
    n_runs, n_examples = preds.shape
    if n_runs < 2:
        raise ValueError(
            f"disagreement needs at least 2 seeds, got {n_runs}. "
            "Run more seeds before asking how much they vary."
        )

    num_classes = int(max(preds.max(), labels.max())) + 1
    counts = _vote_counts(preds, num_classes)

    unanimous = (counts.max(axis=1) == n_runs)
    correct_when_unanimous = unanimous & (counts.argmax(axis=1) == labels)

    pairwise = np.asarray(
        [float((preds[a] != preds[b]).mean()) for a, b in combinations(range(n_runs), 2)]
    )

    # The rate says how often the seeds disagree; the flip share says how badly.
    pair_counts = _pair_disagreement_counts(preds, num_classes)
    pair_total = int(pair_counts.sum())
    flip_total = int(
        sum(
            pair_counts[low, high]
            for low in range(num_classes)
            for high in range(low + 1, num_classes)
            if high - low == POLARITY_FLIP_DISTANCE
        )
    )

    # Normalised so 1.0 is votes spread evenly over every class the model uses.
    probabilities = counts / n_runs
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(probabilities > 0, np.log(probabilities), 0.0)
    entropy = -(probabilities * logs).sum(axis=1)
    used_classes = int((counts.sum(axis=0) > 0).sum())
    if used_classes > 1:
        entropy = entropy / np.log(used_classes)

    return {
        "n_seeds": n_runs,
        "n_examples": n_examples,
        "unanimity_rate": round(float(unanimous.mean()) * 100, 2),
        "unanimous_correct_rate": round(float(correct_when_unanimous.mean()) * 100, 2),
        "unanimous_wrong_rate": round(
            float((unanimous & ~correct_when_unanimous).mean()) * 100, 2
        ),
        "contested_rate": round(float((~unanimous).mean()) * 100, 2),
        "mean_pairwise_disagreement": round(float(pairwise.mean()) * 100, 2),
        "max_pairwise_disagreement": round(float(pairwise.max()) * 100, 2),
        "mean_vote_entropy": round(float(entropy.mean()), 4),
        "fleiss_kappa": _round_or_none(_fleiss_kappa(counts), 4),
        "polarity_flip_share": (
            round(flip_total / pair_total * 100, 2) if pair_total else None
        ),
    }


def _round_or_none(value: Optional[float], digits: int) -> Optional[float]:
    return None if value is None else round(value, digits)


def _confusion(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    """Counts of (true, predicted) pairs, shape (num_classes, num_classes)."""
    # One bincount rather than np.add.at: `paired_bootstrap` calls this tens of thousands of
    # times, where the buffered version is several times slower.
    flat = labels.astype(np.int64) * num_classes + preds.astype(np.int64)
    return np.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def _kind(distance: int, num_classes: int) -> str:
    """Name the severity of a class confusion on the ordinal bias scale."""
    if num_classes != NUM_BIAS_CLASSES:
        return f"distance {distance}"
    return "polarity flip" if distance == POLARITY_FLIP_DISTANCE else "adjacent"


def error_breakdown(runs: Sequence[RunResult]) -> pd.DataFrame:
    """One row per directed `true -> pred` confusion, counted both as a mean over seeds
    and for the majority-vote ensemble.
    """
    # Directed: hedging to center and flipping polarity are different failures.
    _, labels, preds = align_by_id(runs)
    num_classes = int(max(preds.max(), labels.max())) + 1
    n_examples = labels.size

    per_seed = np.stack(
        [_confusion(labels, preds[row], num_classes) for row in range(len(runs))]
    )
    counts = _vote_counts(preds, num_classes)
    vote_confusion = _confusion(labels, counts.argmax(axis=1), num_classes)

    off_diagonal = ~np.eye(num_classes, dtype=bool)
    mean_errors = float(per_seed[:, off_diagonal].sum(axis=1).mean())
    vote_errors = int(vote_confusion[off_diagonal].sum())

    rows = []
    for true in range(num_classes):
        for pred in range(num_classes):
            if true == pred:
                continue
            cell = per_seed[:, true, pred].astype(np.float64)
            distance = abs(true - pred)
            rows.append(
                {
                    "confusion": f"{true}->{pred}",
                    "true": true,
                    "pred": pred,
                    "distance": distance,
                    "kind": _kind(distance, num_classes),
                    "mean_count": round(float(cell.mean()), 1),
                    "std_count": (
                        round(float(cell.std(ddof=1)), 1) if cell.size > 1 else None
                    ),
                    "mean_rate": round(float(cell.mean()) / n_examples * 100, 2),
                    "share_of_errors": (
                        round(float(cell.mean()) / mean_errors * 100, 2)
                        if mean_errors
                        else None
                    ),
                    "vote_count": int(vote_confusion[true, pred]),
                    "vote_share_of_errors": (
                        round(float(vote_confusion[true, pred]) / vote_errors * 100, 2)
                        if vote_errors
                        else None
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    return frame.sort_values("mean_count", ascending=False, ignore_index=True)


def disagreement_breakdown(runs: Sequence[RunResult]) -> pd.DataFrame:
    """One row per unordered class pair the seeds split on -- the detail behind
    `mean_pairwise_disagreement`, which cannot say whether a model wobbles into center
    or flips polarity outright.
    """
    # Unordered because seeds are exchangeable; counted over every (pair, example).
    _, labels, preds = align_by_id(runs)
    n_runs, n_examples = preds.shape
    if n_runs < 2:
        raise ValueError(
            f"disagreement_breakdown needs at least 2 seeds, got {n_runs}."
        )

    num_classes = int(max(preds.max(), labels.max())) + 1
    pair_counts = _pair_disagreement_counts(preds, num_classes)

    comparisons = n_runs * (n_runs - 1) // 2 * n_examples
    total = int(pair_counts.sum())

    rows = []
    for low in range(num_classes):
        for high in range(low + 1, num_classes):
            count = int(pair_counts[low, high])
            distance = high - low
            rows.append(
                {
                    "pair": f"{low}<->{high}",
                    "class_a": low,
                    "class_b": high,
                    "distance": distance,
                    "kind": _kind(distance, num_classes),
                    "count": count,
                    "rate": round(count / comparisons * 100, 2),
                    "share_of_disagreements": (
                        round(count / total * 100, 2) if total else None
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    return frame.sort_values("count", ascending=False, ignore_index=True)


def confusion_frame(runs: Sequence[RunResult], source: str = "mean") -> pd.DataFrame:
    """The confusion matrix as true (rows) x predicted (columns), with `source` either
    "mean" over seeds or "vote" for the ensemble.
    """
    _, labels, preds = align_by_id(runs)
    num_classes = int(max(preds.max(), labels.max())) + 1

    if source == "mean":
        stacked = np.stack(
            [_confusion(labels, preds[row], num_classes) for row in range(len(runs))]
        )
        # Unrounded: rounding each cell stops the matrix summing to the test size.
        matrix = stacked.mean(axis=0)
    elif source == "vote":
        counts = _vote_counts(preds, num_classes)
        matrix = _confusion(labels, counts.argmax(axis=1), num_classes)
    else:
        raise ValueError(f"source must be 'mean' or 'vote', got {source!r}")

    return pd.DataFrame(
        matrix,
        index=pd.Index(range(num_classes), name="true"),
        columns=pd.Index(range(num_classes), name="pred"),
    )


def per_example_frame(runs: Sequence[RunResult]) -> pd.DataFrame:
    """One row per test example: the gold label, each seed's prediction, the vote. Sort
    on `n_distinct` and `n_correct` to find where the model is actually undecided.
    """
    ids, labels, preds = align_by_id(runs)
    num_classes = int(max(preds.max(), labels.max())) + 1
    counts = _vote_counts(preds, num_classes)

    frame = pd.DataFrame({"ID": ids, "label": labels})
    for row, run in enumerate(runs):
        frame[f"seed_{run.seed}"] = preds[row]

    frame["vote"] = counts.argmax(axis=1)
    frame["vote_count"] = counts.max(axis=1)
    frame["n_distinct"] = [len(np.unique(column)) for column in preds.T]
    frame["n_correct"] = (preds == labels[None, :]).sum(axis=0)
    frame["vote_correct"] = frame["vote"] == frame["label"]
    return frame


def model_report(runs: Sequence[RunResult]) -> Dict[str, Any]:
    """Everything this module computes for a single model, in one dict."""
    report: Dict[str, Any] = {
        "model": runs[0].model,
        "seeds": [r.seed for r in runs],
        "summary": summarize(runs),
    }
    report["majority_vote"] = majority_vote(runs)
    report["errors"] = error_breakdown(runs)
    if len(runs) > 1:
        report["disagreement"] = disagreement(runs)
        report["disagreement_pairs"] = disagreement_breakdown(runs)
    return report


def report_lines(report: Dict[str, Any]) -> List[str]:
    """Format a `model_report` as aligned text, for printing."""
    seeds = ", ".join(str(s) for s in report["seeds"])
    lines = [f"  {report['model']} ({len(report['seeds'])} seeds: {seeds})"]

    summary = report["summary"]
    lines.append("    across seeds, mean (std)")
    for _, row in summary.iterrows():
        lines.append(f"      {row['metric']:<28}{row['mean_std']:>12}")

    vote = report["majority_vote"]
    lines.append("    majority vote across seeds")
    for key in ("accuracy", "f1_macro", "f1_weighted", "gain_f1_macro", "tie_rate"):
        value = vote.get(key)
        formatted = "undefined" if value is None else f"{value:.2f}"
        lines.append(f"      {key:<28}{formatted:>12}")

    lines.append("    errors by confusion, mean per seed / majority vote")
    for _, row in report["errors"].iterrows():
        std = "" if row["std_count"] is None else f" (±{row['std_count']:.1f})"
        # None for a model with no errors at all; print rather than raise.
        share = row["share_of_errors"]
        share_text = "  n/a" if share is None else f"{share:>5.1f}%"
        lines.append(
            f"      {row['confusion']} {row['kind']:<14}"
            f"{row['mean_count']:>8.1f}{std:<8} {share_text} "
            f"| vote {row['vote_count']:>6}"
        )

    spread = report.get("disagreement")
    if spread is None:
        lines.append("    disagreement: needs at least 2 seeds")
        return lines

    lines.append("    disagreement across seeds")
    # The rates are percentages; entropy and kappa are not, and need the digits.
    for key, digits in (
        ("unanimity_rate", 2),
        ("unanimous_correct_rate", 2),
        ("unanimous_wrong_rate", 2),
        ("mean_pairwise_disagreement", 2),
        ("polarity_flip_share", 2),
        ("mean_vote_entropy", 4),
        ("fleiss_kappa", 4),
    ):
        value = spread.get(key)
        formatted = "undefined" if value is None else f"{value:.{digits}f}"
        lines.append(f"      {key:<28}{formatted:>12}")

    lines.append("    seed disagreement by class pair")
    for _, row in report["disagreement_pairs"].iterrows():
        # None when the seeds never disagreed anywhere; still worth printing.
        share = row["share_of_disagreements"]
        share_text = "  n/a" if share is None else f"{share:>5.1f}%"
        lines.append(
            f"      {row['pair']} {row['kind']:<14}{row['count']:>8} "
            f"{share_text} of disagreements"
        )
    return lines


def print_reports(results: Sequence[RunResult]) -> None:
    """Print a `report_lines` block for every model found."""
    for _, runs in sorted(by_model(results).items()):
        if not runs:
            continue
        for line in report_lines(model_report(runs)):
            print(line)
        print()
