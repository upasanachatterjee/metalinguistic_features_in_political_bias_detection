"""Training-split statistics for the tone and theme objectives.

Both are computed over exactly the rows a run trains on -- same dataset, same
``require_nonempty_themes_and_tone`` filter, same title dedup -- because this
builds the run's own ``MemoryEfficientDataset`` rather than re-deriving the
selection. Both refuse a non-train split::

    python corpus_stats.py --config run_configs/grad_diagnostics.yaml \\
        --out ./corpus_stats

**Tone.** Describes the regression target: how much of it is spread and how much
is a constant offset, which is what says whether a low tone loss means anything.
Parsing goes through ``RegressionCollator``'s own ``parse_regression_values``, so
the values audited here are byte-for-byte the values the collator feeds the
model. Purely descriptive -- targets are trained in GDELT's own units and nothing
consumes these numbers.

**Themes.** Counts positives per theme over the same rows -- how rare the rare
themes are, and how much of the label space a document actually touches. Parsing
goes through
``MultiLabelCollator``'s ``parse_multilabel`` and the same ``top_themes.txt``
label ordering.


Outputs (into ``--out``)
------------------------
``tone_stats.json``          count/mean/std/min/max/median/p1/p5/p95/p99
``theme_label_stats.json``   summary + one record per theme
``theme_label_stats.csv``    the per-theme records, for spreadsheets/plots
``corpus_stats.txt``         the printed summary, verbatim
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer

from collators.multi_label_collator import MultiLabelCollator, parse_multilabel
from collators.regression_collator import parse_regression_values
from config import RunConfig, load_run_config
from dataset import MemoryEfficientDataset

# Rows pulled out of the Arrow table per chunk. Only bounds peak memory of the
# python-object conversion; the numbers are identical at any chunk size.
CHUNK_ROWS = 100_000

# Prevalence bands the theme audit reports counts for.
PREVALENCE_BANDS = (0.10, 0.01, 0.001)


def _percentiles(values: np.ndarray, points: Dict[str, float]) -> Dict[str, float]:
    return {
        name: float(np.percentile(values, q)) for name, q in points.items()
    }


def _iter_column(dataset, column: str):
    """Yield the column's raw python values in chunks, in row order."""
    table = dataset.dataset.data
    total = len(dataset.dataset)
    for start in range(0, total, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, total)
        yield table.column(column)[start:end].to_pylist()


# ----------------------------------------------------------------------
# tone
# ----------------------------------------------------------------------
def compute_tone_stats(dataset, output_size: int = 1) -> Dict[str, Any]:
    """Fixed tone statistics over the training rows.

    Uses ``parse_regression_values`` -- the collator's own parser -- so a row the
    collator would drop as unparseable is dropped here too, and the mean/std
    describe precisely the population the model sees.
    """
    values: List[float] = []
    unparseable = 0
    for chunk in _iter_column(dataset, "V2Tone"):
        for raw in chunk:
            parsed = parse_regression_values(
                str(raw) if raw is not None else None, output_size
            )
            if parsed is None:
                unparseable += 1
                continue
            values.append(parsed[0])

    if not values:
        raise ValueError("No parseable V2Tone values in the training split.")

    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    # ddof=0: this is the whole training population, not a sample of it.
    std = float(array.std(ddof=0))
    mean_square = float(np.mean(array**2))
    return {
        "count": int(array.size),
        "unparseable_rows": unparseable,
        "mean": mean,
        "std": std,
        "min": float(array.min()),
        "max": float(array.max()),
        "median": float(np.median(array)),
        **_percentiles(array, {"p1": 1, "p5": 5, "p95": 95, "p99": 99}),
        # How much of the MSE a zero-predicting head starts with is just the
        # mean offset. A large share means early tone loss says more about the
        # corpus being systematically negative than about anything learned.
        "mean_offset_share_of_initial_mse": (
            mean**2 / mean_square if mean_square > 0 else 0.0
        ),
    }


# ----------------------------------------------------------------------
# themes
# ----------------------------------------------------------------------
def compute_theme_stats(dataset, themes_path: str) -> Dict[str, Any]:
    """Per-theme positive counts and label-density summary over training rows.

    Uses ``MultiLabelCollator``'s label ordering and ``parse_multilabel``, so
    index i here is index i of the theme head.
    """
    collator = MultiLabelCollator(top_themes_path=themes_path)
    num_labels = collator.num_labels

    positive_counts = np.zeros(num_labels, dtype=np.int64)
    positives_per_example: List[int] = []
    num_examples = 0

    for chunk in _iter_column(dataset, "V2Themes"):
        for raw in chunk:
            num_examples += 1
            themes = parse_multilabel(raw)
            if not themes:
                # None (missing) and "" (present but empty) both collate to an
                # all-zero target vector, exactly as counted here.
                positives_per_example.append(0)
                continue
            indices = {
                collator.theme_to_idx[theme]
                for theme in themes
                if theme in collator.theme_to_idx
            }
            for idx in indices:
                positive_counts[idx] += 1
            positives_per_example.append(len(indices))

    if num_examples == 0:
        raise ValueError("No rows in the training split; nothing to audit.")

    per_example = np.asarray(positives_per_example, dtype=np.int64)
    with_positive = int((per_example > 0).sum())
    negative_counts = num_examples - positive_counts
    prevalence = positive_counts / float(num_examples)

    bands = {}
    for threshold in PREVALENCE_BANDS:
        below = int((prevalence < threshold).sum())
        bands[f"below_{threshold:g}"] = {
            "count": below,
            "fraction": below / float(num_labels),
        }

    summary = {
        "num_theme_classes": num_labels,
        "num_examples": num_examples,
        "num_examples_with_positive": with_positive,
        "fraction_all_zero_targets": 1.0 - with_positive / float(num_examples),
        "mean_positives_per_example": float(per_example.mean()),
        "median_positives_per_example": float(np.median(per_example)),
        "min_positives_per_example": int(per_example.min()),
        "max_positives_per_example": int(per_example.max()),
        "themes_with_zero_positives": int((positive_counts == 0).sum()),
        "prevalence_bands": bands,
    }

    per_label = [
        {
            "index": idx,
            "theme": collator.top_themes[idx],
            "positive_count": int(positive_counts[idx]),
            "negative_count": int(negative_counts[idx]),
            "prevalence": float(prevalence[idx]),
        }
        for idx in range(num_labels)
    ]
    return {"summary": summary, "per_label": per_label}


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def tone_report_lines(stats: Dict[str, Any]) -> List[str]:
    lines = ["", "TONE (training split only)", "-" * 60]
    for key in (
        "count",
        "unparseable_rows",
        "mean",
        "std",
        "min",
        "max",
        "median",
        "p1",
        "p5",
        "p95",
        "p99",
    ):
        value = stats[key]
        formatted = f"{value:,}" if isinstance(value, int) else f"{value:.6f}"
        lines.append(f"  {key:<20}{formatted:>18}")
    lines += [
        "",
        f"  {stats['mean_offset_share_of_initial_mse']:.1%} of the MSE a "
        "zero-predicting head starts with is the mean offset alone, not "
        "article-level signal. Read the tone loss against that, and prefer the "
        "correlations in task_metrics.py over the loss value.",
        "",
        "  Descriptive only: nothing consumes these numbers. Tone is trained in "
        "GDELT's own units -- the field is a defined -100..+100 scale, so "
        "rescaling it would only duplicate loss_weights.tone.",
    ]
    return lines


def theme_report_lines(stats: Dict[str, Any]) -> List[str]:
    summary = stats["summary"]
    lines = ["", "THEMES (training split only)", "-" * 60]
    ordered = [
        ("num_theme_classes", "theme classes"),
        ("num_examples", "examples"),
        ("num_examples_with_positive", "examples with >=1 positive"),
        ("fraction_all_zero_targets", "fraction all-zero targets"),
        ("mean_positives_per_example", "mean positives / example"),
        ("median_positives_per_example", "median positives / example"),
        ("min_positives_per_example", "min positives / example"),
        ("max_positives_per_example", "max positives / example"),
        ("themes_with_zero_positives", "themes with 0 positives"),
    ]
    for key, label in ordered:
        value = summary[key]
        formatted = f"{value:,}" if isinstance(value, int) else f"{value:.6f}"
        lines.append(f"  {label:<30}{formatted:>16}")

    lines += ["", "  Themes below a prevalence threshold:"]
    for name, band in summary["prevalence_bands"].items():
        threshold = name.replace("below_", "")
        lines.append(
            f"    prevalence < {threshold:<8}{band['count']:>8,} themes "
            f"({band['fraction']:.1%})"
        )

    ranked = sorted(
        stats["per_label"], key=lambda row: row["positive_count"], reverse=True
    )
    lines += ["", "  Most and least frequent themes:"]
    for row in ranked[:5] + ranked[-5:]:
        lines.append(
            f"    {row['theme'][:44]:<46}{row['positive_count']:>10,}"
            f"{row['prevalence']:>10.4%}"
        )
    lines += [
        "",
        "  Descriptive only: nothing consumes these numbers. The theme "
        "objective is a plain unweighted BCE. If its gradient is too small, "
        "loss_weights.themes is the knob -- chosen from the gradient "
        "diagnostic, not from the magnitude of the theme loss.",
    ]
    return lines


def write_outputs(
    out_dir: str,
    lines: List[str],
    tone: Optional[Dict[str, Any]],
    themes: Optional[Dict[str, Any]],
    provenance: Dict[str, Any],
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []

    if tone is not None:
        path = os.path.join(out_dir, "tone_stats.json")
        with open(path, "w") as handle:
            json.dump({"provenance": provenance, "tone": tone}, handle, indent=2)
            handle.write("\n")
        written.append(path)

    if themes is not None:
        path = os.path.join(out_dir, "theme_label_stats.json")
        with open(path, "w") as handle:
            json.dump({"provenance": provenance, **themes}, handle, indent=2)
            handle.write("\n")
        written.append(path)

        csv_path = os.path.join(out_dir, "theme_label_stats.csv")
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "index",
                    "theme",
                    "positive_count",
                    "negative_count",
                    "prevalence",
                ],
            )
            writer.writeheader()
            writer.writerows(themes["per_label"])
        written.append(csv_path)

    txt_path = os.path.join(out_dir, "corpus_stats.txt")
    with open(txt_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    written.append(txt_path)
    return written


def build_dataset(cfg: RunConfig):
    """The exact row set the run trains on (same filter, same dedup)."""
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    return MemoryEfficientDataset(
        dataset_name=cfg.task_spec.dataset_name,
        split=cfg.task_spec.split,
        text_col=cfg.task_spec.text_col,
        tokenizer=tokenizer,
        cache_dir=cfg.task_spec.index_cache_dir,
        require_nonempty_themes_and_tone=cfg.task_spec.require_nonempty_themes_and_tone,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a run YAML config")
    parser.add_argument(
        "--out", default="./corpus_stats", help="Directory for the statistics files"
    )
    parser.add_argument("--skip-tone", action="store_true")
    parser.add_argument("--skip-themes", action="store_true")
    parser.add_argument(
        "--allow-non-train-split",
        action="store_true",
        help=(
            "Audit a split whose name is not 'train'. Statistics fitted this way "
            "must NOT be used to normalize training targets."
        ),
    )
    cli = parser.parse_args()

    cfg = load_run_config(cli.config)
    split = cfg.task_spec.split
    if not split.startswith("train") and not cli.allow_non_train_split:
        raise SystemExit(
            f"task_spec.split is '{split}', not a training split. Normalization "
            "and theme weights fitted on validation/test data leak the "
            "evaluation set into training. Pass --allow-non-train-split only if "
            "you are auditing, not fitting."
        )

    dataset = build_dataset(cfg)

    lines = [
        "=" * 78,
        "CORPUS STATISTICS",
        "=" * 78,
        f"  config                        : {cli.config}",
        f"  dataset                       : {cfg.task_spec.dataset_name}",
        f"  split                         : {split}",
        f"  require_nonempty_themes_and_tone: {cfg.task_spec.require_nonempty_themes_and_tone}",
        f"  rows audited                  : {len(dataset):,}",
        f"  themes file                   : {cfg.task_spec.themes_path}",
        f"  computed                      : {datetime.now():%Y-%m-%d %H:%M:%S}",
    ]

    provenance = {
        "config": cli.config,
        "dataset": cfg.task_spec.dataset_name,
        "split": split,
        "require_nonempty_themes_and_tone": cfg.task_spec.require_nonempty_themes_and_tone,
        "rows": len(dataset),
        "themes_path": cfg.task_spec.themes_path,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }

    tone_stats = None
    if not cli.skip_tone:
        tone_stats = compute_tone_stats(dataset)
        lines += tone_report_lines(tone_stats)

    theme_stats = None
    if not cli.skip_themes:
        if cfg.task_spec.themes_path is None:
            lines += ["", "THEMES: skipped (task_spec.themes_path is not set)"]
        else:
            theme_stats = compute_theme_stats(dataset, cfg.task_spec.themes_path)
            lines += theme_report_lines(theme_stats)

    lines.append("=" * 78)
    for line in lines:
        print(line)

    written = write_outputs(cli.out, lines, tone_stats, theme_stats, provenance)
    print("\nWritten:")
    for path in written:
        print(f"   {path}")


if __name__ == "__main__":
    main()
