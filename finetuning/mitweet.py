"""MITweet (EMNLP 2023) relevance recognition and ideology detection, read off the local
checkout at `../MITweet`. Ideology rides the `bias_*` keys because its labels are the AllSides ones.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

MITWEET_DATA_ROOT = "../MITweet/data"

# Facet relevance, in I1..I12 order. The domain-level R1..R5 are unused, as in MITweet's own code.
RELEVANCE_COLUMNS = (
    "R1-1-1", "R2-1-2", "R3-2-1", "R4-2-2", "R5-3-1", "R6-3-2",
    "R7-3-3", "R8-4-1", "R9-4-2", "R10-5-1", "R11-5-2", "R12-5-3",
)
IDEOLOGY_COLUMNS = tuple(f"I{index}" for index in range(1, 13))

# From the paper's schema table; no shipped data file carries the names.
FACET_NAMES = (
    "Political Regime", "State Structure", "Economic Orientation", "Economic Equality",
    "Ethical Pursuit", "Church-State Relations", "Cultural Value", "Diplomatic Strategy",
    "Military Force", "Social Development", "Justice Orientation", "Personal Right",
)

NUM_FACETS = 12
# The ideology sentinel for "this tweet is not about this facet", so there is no label.
UNRELATED = -1
# MITweet's --indicator_num default.
INDICATOR_WORDS = 18
# Four disjoint triples partitioning all 12 facets, each drawing from three different domains
# so no fold holds out a whole domain. Fold 0 is the original hand-picked triple, so it
# reproduces the runs already on disk and checks the rotation against them.
FACET_FOLDS = ((3, 8, 11), (1, 5, 9), (2, 6, 10), (4, 7, 12))
NUM_FACET_FOLDS = len(FACET_FOLDS)
# One facet each from Economy, Diplomacy and Society, all with enough test rows to mean something.
HELD_OUT_FACETS = FACET_FOLDS[0]
# MITweet's --pos_weight times its --weight_scale of 0.4, clamped at 1.
RELEVANCE_POS_WEIGHT = (12.0, 6.0, 2.4, 2.4, 1.0, 16.0, 10.0, 1.2, 1.2, 1.2, 1.0, 1.0)
# An indicator prefix is at most 76 tokens and 99% of tweets are under 86, so nothing truncates.
MAX_LENGTH = 192

TASKS = ("ideology", "relevance")
PREPEND_MODES = ("indicators", "facet_name", "none")
SPLIT_MODES = ("random", "facet")
# Our split names, and the file stems MITweet ships them under.
SPLIT_FILES = (("train", "train"), ("validation", "val"), ("test", "test"))


@dataclass
class MITweetConfig:
    """One MITweet run: which task, how the facet is described, and which split."""

    task: str = "ideology"
    prepend: str = "indicators"
    split: str = "random"
    # Rotates the held-out triple; only the facet split reads it.
    fold: int = 0
    # 1-indexed to match the I1..I12 column names. None takes the fold's triple instead.
    held_out_facets: Optional[Tuple[int, ...]] = None
    data_root: str = MITWEET_DATA_ROOT


def variant_name(config: MITweetConfig) -> str:
    """Directory stem naming one variant, e.g. `ideology_indicators_facet`."""
    if config.task == "relevance":
        return "relevance_random"
    return f"{config.task}_{config.prepend}_{config.split}"


def held_out_for(config: MITweetConfig) -> Tuple[int, ...]:
    """The 1-indexed facets this run holds out: an explicit override, else the fold's triple."""
    if config.held_out_facets is not None:
        return tuple(config.held_out_facets)
    return FACET_FOLDS[config.fold]


def _validate(config: MITweetConfig) -> None:
    """Reject configurations whose results directory would not mean what it says."""
    if config.task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {config.task!r}")
    if config.prepend not in PREPEND_MODES:
        raise ValueError(f"prepend must be one of {PREPEND_MODES}, got {config.prepend!r}")
    if config.split not in SPLIT_MODES:
        raise ValueError(f"split must be one of {SPLIT_MODES}, got {config.split!r}")
    # Relevance sees one row per tweet and predicts all 12 facets at once, so there is
    # nothing for a per-facet prefix to condition on and no facet to hold out.
    if config.task == "relevance" and config.prepend != "none":
        raise ValueError(
            "the relevance task takes the tweet alone; set prepend='none'. A prefix here "
            "would be ignored while still naming the results directory after it."
        )
    if config.task == "relevance" and config.split != "random":
        raise ValueError(
            "the facet split is ideology-only: a held-out facet's relevance output column "
            "never receives a gradient, so testing on it would measure its initialisation."
        )
    if config.split == "facet":
        if config.fold not in range(NUM_FACET_FOLDS):
            raise ValueError(
                f"fold must be in range({NUM_FACET_FOLDS}), got {config.fold!r}"
            )
        # Otherwise a hand-picked triple would be filed under a fold_N directory naming a
        # different one, and the fold mean would pool runs that held out different facets.
        if config.held_out_facets is not None and config.fold != 0:
            raise ValueError(
                "set held_out_facets or fold, not both: held_out_facets="
                f"{config.held_out_facets!r} with fold={config.fold} would be filed under "
                f"fold_{config.fold}, whose triple is {FACET_FOLDS[config.fold]}"
            )
        held_out = set(held_out_for(config))
        if not held_out or not held_out <= set(range(1, NUM_FACETS + 1)):
            raise ValueError(
                f"held_out_facets must be a non-empty subset of 1..{NUM_FACETS}, got "
                f"{config.held_out_facets!r}"
            )
        if len(held_out) == NUM_FACETS:
            raise ValueError("holding out every facet leaves nothing to train on")


def load_indicators(data_root: str, separator: str) -> List[str]:
    """The 12 indicator keyword strings, kept to `INDICATOR_WORDS` words and joined by
    `separator`, exactly as MITweet's `--sep_ind` builds them.
    """
    with open(f"{data_root}/random_split/Indicators.txt", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if len(lines) != NUM_FACETS:
        raise ValueError(f"Indicators.txt has {len(lines)} lines, expected {NUM_FACETS}")
    return [f" {separator} ".join(line.split()[:INDICATOR_WORDS]) for line in lines]


def load_frame(data_root: str, stem: str) -> pd.DataFrame:
    """One shipped random-split CSV, with a `row_id` that is unique within the split."""
    # The header's first field carries a UTF-8 BOM.
    frame = pd.read_csv(f"{data_root}/random_split/{stem}.csv", encoding="utf-8-sig")
    frame = frame.reset_index(drop=True)
    frame["row_id"] = frame.index.astype(str)
    return frame


def _check_sentinel(frame: pd.DataFrame, stem: str) -> None:
    """Fail loudly if `I{k} == -1` is not exactly `R{k} == 0`; the explode rests on it."""
    for relevance_column, ideology_column in zip(RELEVANCE_COLUMNS, IDEOLOGY_COLUMNS):
        mismatched = (frame[ideology_column] == UNRELATED) != (frame[relevance_column] == 0)
        if mismatched.any():
            raise ValueError(
                f"{stem}.csv: {int(mismatched.sum())} rows where {ideology_column} == -1 "
                f"disagrees with {relevance_column} == 0. The relevance and ideology labels "
                "cannot both be trusted; the exploded rows would be silently wrong."
            )


def prefixes_for(config: MITweetConfig, tokenizer: PreTrainedTokenizerBase) -> Optional[List[str]]:
    """The 12 strings prepended before the tweet, or None when the tweet goes in alone."""
    if config.prepend == "none":
        return None
    if config.prepend == "facet_name":
        return list(FACET_NAMES)
    return load_indicators(config.data_root, tokenizer.sep_token)


def _facet_indices(config: MITweetConfig, split_name: str) -> Sequence[int]:
    """The 0-indexed facets one split may draw rows from."""
    if config.split == "random":
        return range(NUM_FACETS)
    # held_out_facets is 1-indexed, matching the I1..I12 names.
    held_out = {facet - 1 for facet in held_out_for(config)}
    # Selecting on a held-out facet would leak the test facets into model selection.
    if split_name == "test":
        return sorted(held_out)
    return [facet for facet in range(NUM_FACETS) if facet not in held_out]


def build_ideology_rows(
    frame: pd.DataFrame, facets: Sequence[int], prefixes: Optional[List[str]]
) -> Dataset:
    """Explode to one row per (tweet, relevant facet), grouped by facet as MITweet does."""
    columns: Dict[str, list] = {"content": [], "bias_labels": [], "facet": [], "ID": []}
    if prefixes is not None:
        columns["prefix"] = []

    for facet in facets:
        related = frame[frame[RELEVANCE_COLUMNS[facet]] == 1]
        columns["content"] += list(related["tweet"].astype(str))
        columns["bias_labels"] += [int(value) for value in related[IDEOLOGY_COLUMNS[facet]]]
        columns["facet"] += [str(facet)] * len(related)
        columns["ID"] += [f"{row_id}-{facet}" for row_id in related["row_id"]]
        if prefixes is not None:
            columns["prefix"] += [prefixes[facet]] * len(related)

    return Dataset.from_dict(columns)


def build_relevance_rows(frame: pd.DataFrame) -> Dataset:
    """One row per tweet, labelled with the 12-facet binary relevance vector."""
    labels = frame[list(RELEVANCE_COLUMNS)].to_numpy(dtype=np.float32)
    return Dataset.from_dict(
        {
            "content": [str(tweet) for tweet in frame["tweet"]],
            "relevance_labels": labels,
            "ID": list(frame["row_id"]),
        }
    )


def tokenize_rows(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase, task: str, use_bias_keys: bool
) -> Dataset:
    """Tokenize to fixed-length rows and rename to the prefixed keys the model dispatches on."""
    has_prefix = "prefix" in dataset.column_names

    def tokenize_batch(examples):
        if has_prefix:
            # Prefix first, tweet second, so only the tweet is ever cut.
            return tokenizer(
                examples["prefix"],
                examples["content"],
                padding="max_length",
                truncation="only_second",
                max_length=MAX_LENGTH,
            )
        return tokenizer(
            examples["content"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["prefix", "content"] if has_prefix else ["content"],
    )

    prefix = "relevance" if task == "relevance" else "bias"
    if use_bias_keys:
        tokenized = tokenized.rename_column("input_ids", f"{prefix}_input_ids")
        tokenized = tokenized.rename_column("attention_mask", f"{prefix}_attention_mask")
    else:
        tokenized = tokenized.rename_column(f"{prefix}_labels", "labels")
    return tokenized


def load_mitweet(
    config: MITweetConfig, tokenizer: PreTrainedTokenizerBase, use_bias_keys: bool
) -> DatasetDict:
    """Build the three tokenized splits for one MITweet variant. Never undersampled: MITweet
    reports on the natural label distribution.
    """
    _validate(config)
    prefixes = prefixes_for(config, tokenizer)

    splits = DatasetDict()
    for split_name, stem in SPLIT_FILES:
        frame = load_frame(config.data_root, stem)
        _check_sentinel(frame, stem)
        if config.task == "relevance":
            rows = build_relevance_rows(frame)
        else:
            rows = build_ideology_rows(frame, _facet_indices(config, split_name), prefixes)
        # An empty split would be reported as a metric over nothing.
        if len(rows) == 0:
            raise ValueError(
                f"{variant_name(config)}: the {split_name} split is empty. Check "
                f"held_out_facets={config.held_out_facets!r}."
            )
        splits[split_name] = tokenize_rows(rows, tokenizer, config.task, use_bias_keys)

    print(f"{variant_name(config)}: " + ", ".join(f"{name}={len(rows)}" for name, rows in splits.items()))
    if config.task == "ideology":
        for split_name, rows in splits.items():
            counts = np.bincount([int(facet) for facet in rows["facet"]], minlength=NUM_FACETS)
            print(f"  {split_name} rows per facet: {counts.tolist()}")
    return splits
