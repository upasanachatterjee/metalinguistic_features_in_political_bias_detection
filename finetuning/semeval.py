"""SemEval-2016 Task 6 stance detection and opinion-target classification, read off the local
`data-all-annotations/` files. Both tasks are 3-class, so both ride the `bias_*` keys.
"""

import csv
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

SEMEVAL_DATA_ROOT = "data-all-annotations"

# The five taskA targets, then taskB's, so a target's index is stable across splits.
ALL_TARGETS = (
    "Atheism",
    "Climate Change is a Real Concern",
    "Feminist Movement",
    "Hillary Clinton",
    "Legalization of Abortion",
    "Donald Trump",
)
TASK_B_TARGET = "Donald Trump"

# NONE sits in the middle so the order matches the bias head's left/center/right, which is
# what makes an AllSides checkpoint's head meaningful here without weight surgery.
STANCE_LABELS = ("AGAINST", "NONE", "FAVOR")
OPINION_LABELS = ("TARGET", "OTHER", "NO ONE")
# The two classes SemEval's official metric averages; NONE is excluded from it by design.
AGAINST, FAVOR = 0, 2

TASKS = ("stance", "opinion")
LABEL_COLUMN = {"stance": "Stance", "opinion": "Opinion towards"}
LABEL_VALUES = {"stance": STANCE_LABELS, "opinion": OPINION_LABELS}

# The longest target+tweet pair in all four files is 74 tokens, so nothing truncates.
MAX_LENGTH = 96

PREPEND_MODES = ("target", "none")
SPLIT_MODES = ("taskA", "taskB")

# The training file carries 12 windows-1252 bytes and is not valid UTF-8; the other three are
# pure ASCII, which windows-1252 decodes identically, so one encoding reads all four.
ENCODING = "windows-1252"

TRAIN_FILE = "trainingdata-all-annotations.txt"
VALIDATION_FILE = "trialdata-all-annotations.txt"
TEST_FILES = {
    "taskA": "testdata-taskA-all-annotations.txt",
    "taskB": "testdata-taskB-all-annotations.txt",
}


@dataclass
class SemEvalConfig:
    """One SemEval run: which label column, how the target is described, and which test set."""

    task: str = "stance"
    prepend: str = "target"
    split: str = "taskA"
    data_root: str = SEMEVAL_DATA_ROOT


def variant_name(config: SemEvalConfig) -> str:
    """Directory stem naming one variant, e.g. `stance_target_taskB`."""
    return f"{config.task}_{config.prepend}_{config.split}"


def _validate(config: SemEvalConfig) -> None:
    """Reject configurations whose results directory would not mean what it says."""
    if config.task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {config.task!r}")
    if config.prepend not in PREPEND_MODES:
        raise ValueError(f"prepend must be one of {PREPEND_MODES}, got {config.prepend!r}")
    if config.split not in SPLIT_MODES:
        raise ValueError(f"split must be one of {SPLIT_MODES}, got {config.split!r}")


def load_frame(data_root: str, filename: str) -> pd.DataFrame:
    """One shipped annotations file, with its own globally unique `ID` column kept as-is."""
    # QUOTE_NONE: tweets carry bare `"` characters that default quoting swallows rows across.
    return pd.read_csv(
        f"{data_root}/{filename}", sep="\t", encoding=ENCODING, quoting=csv.QUOTE_NONE
    )


def _check_no_validation_leak(validation: pd.DataFrame, config: SemEvalConfig) -> None:
    """Fail loudly if the cross-target split's held-out target appears in validation."""
    if config.split != "taskB":
        return
    # Selecting the best epoch on Donald Trump rows would leak the taskB test target.
    leaked = int((validation["Target"] == TASK_B_TARGET).sum())
    if leaked:
        raise ValueError(
            f"{VALIDATION_FILE}: {leaked} rows have Target == {TASK_B_TARGET!r}, which the "
            "taskB split holds out. Model selection would see the test target."
        )


def prefixes_for(config: SemEvalConfig) -> Optional[List[str]]:
    """The strings prepended before the tweet, indexed by target, or None when it goes in alone."""
    if config.prepend == "none":
        return None
    return list(ALL_TARGETS)


def build_rows(frame: pd.DataFrame, task: str, prefixes: Optional[List[str]]) -> Dataset:
    """One row per tweet, labelled from `task`'s annotation column."""
    label_index = {name: index for index, name in enumerate(LABEL_VALUES[task])}
    targets = [ALL_TARGETS.index(name) for name in frame["Target"]]

    columns: Dict[str, list] = {
        "content": [str(tweet) for tweet in frame["Tweet"]],
        "bias_labels": [label_index[value] for value in frame[LABEL_COLUMN[task]]],
        # A string, so the default collator drops it before `forward`; an int would be
        # collated and passed as a kwarg an HF baseline's explicit signature rejects.
        "target": [str(target) for target in targets],
        "ID": [str(row_id) for row_id in frame["ID"]],
    }
    if prefixes is not None:
        columns["prefix"] = [prefixes[target] for target in targets]
    return Dataset.from_dict(columns)


def tokenize_rows(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase, use_bias_keys: bool
) -> Dataset:
    """Tokenize to fixed-length rows and rename to the prefixed keys the model dispatches on."""
    has_prefix = "prefix" in dataset.column_names

    def tokenize_batch(examples):
        if has_prefix:
            # Target first, tweet second, so only the tweet is ever cut.
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

    if use_bias_keys:
        tokenized = tokenized.rename_column("input_ids", "bias_input_ids")
        tokenized = tokenized.rename_column("attention_mask", "bias_attention_mask")
    else:
        tokenized = tokenized.rename_column("bias_labels", "labels")
    return tokenized


def load_semeval(
    config: SemEvalConfig, tokenizer: PreTrainedTokenizerBase, use_bias_keys: bool
) -> DatasetDict:
    """Build the three tokenized splits for one SemEval variant. Never undersampled: SemEval
    reports on the natural label distribution.
    """
    _validate(config)
    prefixes = prefixes_for(config)

    frames = {
        "train": load_frame(config.data_root, TRAIN_FILE),
        "validation": load_frame(config.data_root, VALIDATION_FILE),
        "test": load_frame(config.data_root, TEST_FILES[config.split]),
    }
    _check_no_validation_leak(frames["validation"], config)

    splits = DatasetDict()
    for split_name, frame in frames.items():
        rows = build_rows(frame, config.task, prefixes)
        splits[split_name] = tokenize_rows(rows, tokenizer, use_bias_keys)

    print(f"{variant_name(config)}: " + ", ".join(f"{name}={len(rows)}" for name, rows in splits.items()))
    for split_name, frame in frames.items():
        counts = frame["Target"].value_counts()
        print(f"  {split_name} rows per target: {counts.to_dict()}")
    return splits
