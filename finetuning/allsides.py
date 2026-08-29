"""Fold splitting for the two AllSides studies, which otherwise use the shipped hub splits.

Built for completeness; the shipped split is still the default, because AllSides' 2,565-row test
set makes test-sampling noise about half the size of the seed noise.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import GroupKFold, train_test_split

MEDIA_SPLIT = "upasanachatterjee/AllSides-media-split"
RANDOM_SPLIT = "upasanachatterjee/AllSides-random-split"

SPLIT_MODES = ("media", "random")
NUM_FOLDS = 10

# The shipped geometry: 26,590 / 161 / 2,565 of 29,316 rows. Validation only drives early
# stopping, and is kept tiny so folds stay comparable with the runs already in `allsides/`.
VALIDATION_SIZE = 161
TEST_SIZE = 2565


@dataclass
class AllSidesConfig:
    """One AllSides run: which shipped repo, and which fold of the pooled rows."""

    split: str = "media"
    fold: int = 0


def variant_name(config: AllSidesConfig) -> str:
    """Directory stem naming one variant. The fold is a directory level of its own."""
    return f"{config.split}_split"


def _validate(config: AllSidesConfig) -> None:
    """Reject configurations whose results directory would not mean what it says."""
    if config.split not in SPLIT_MODES:
        raise ValueError(f"split must be one of {SPLIT_MODES}, got {config.split!r}")
    if config.fold not in range(NUM_FOLDS):
        raise ValueError(f"fold must be in range({NUM_FOLDS}), got {config.fold!r}")


def pool(dataset: DatasetDict) -> Dataset:
    """The three shipped splits concatenated back into one 29,316-row set."""
    return concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])


def _media_fold(rows: Dataset, fold: int) -> Dict[str, np.ndarray]:
    """Hold out whole outlets, so a fold's test sources appear in no training row."""
    labels = np.asarray(rows["bias"])
    sources = np.asarray(rows["source"])
    # Grouped, not stratified-by-row: the media split exists to measure cross-outlet
    # generalisation, and a row-level fold would quietly turn it into the random split.
    # GroupKFold and not StratifiedGroupKFold: outlet sizes run from 1 to 2,883 rows, and
    # stratifying on bias instead of on size gave test folds of 96 and 3,212. An outlet has
    # essentially one lean, so holding whole outlets out cannot balance labels anyway.
    splitter = GroupKFold(n_splits=NUM_FOLDS)
    rest, test = list(splitter.split(np.zeros(len(rows)), labels, groups=sources))[fold]
    # Validation is drawn from the training outlets, so model selection never sees a test one.
    rest, validation = train_test_split(
        rest, test_size=VALIDATION_SIZE, stratify=labels[rest], random_state=fold
    )
    return {"train": rest, "validation": validation, "test": test}


def _random_fold(rows: Dataset, fold: int) -> Dict[str, np.ndarray]:
    """A stratified row split at the shipped sizes, seeded by the fold index."""
    labels = np.asarray(rows["bias"])
    index = np.arange(len(rows))
    rest, test = train_test_split(
        index, test_size=TEST_SIZE, stratify=labels, random_state=fold
    )
    rest, validation = train_test_split(
        rest, test_size=VALIDATION_SIZE, stratify=labels[rest], random_state=fold
    )
    return {"train": rest, "validation": validation, "test": test}


def split_pool(rows: Dataset, config: AllSidesConfig) -> DatasetDict:
    """One fold of the pooled rows, as a DatasetDict `get_cleaned_dataset` can consume."""
    _validate(config)
    indices = _media_fold(rows, config.fold) if config.split == "media" else _random_fold(rows, config.fold)

    splits = DatasetDict(
        {name: rows.select(sorted(int(i) for i in index)) for name, index in indices.items()}
    )
    if config.split == "media":
        _check_outlets_disjoint(splits, config.fold)
    print(f"{variant_name(config)} fold {config.fold}: " + ", ".join(
        f"{name}={len(split)}" for name, split in splits.items()
    ))
    return splits


def _check_outlets_disjoint(splits: DatasetDict, fold: int) -> None:
    """Fail loudly if a media fold leaks an outlet, which would inflate every number it reports."""
    shared = set(splits["train"]["source"]) & set(splits["test"]["source"])
    if shared:
        raise ValueError(
            f"media fold {fold}: {len(shared)} outlets appear in both train and test "
            f"({sorted(shared)[:5]}...). This fold measures memorisation, not cross-outlet "
            "generalisation, and its numbers are not comparable with the shipped media split."
        )
