"""BASIL sentence-level bias classification, read off the local `BASIL/` checkout. Two binary
tasks -- does this sentence carry a lexical / an informational bias span -- both on `bias_*` keys.
"""

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerBase

BASIL_DATA_ROOT = "BASIL"

TASKS = ("lexical", "informational")
# The `bias` field's own abbreviations; it is never spelled out in the JSON.
BIAS_CODE = {"lexical": "lex", "informational": "inf"}

# The longest body sentence is 119 roberta tokens, so nothing truncates.
MAX_LENGTH = 128

# The paper's sentence-level split: 6,819 train / 758 validation / 400 test. Validation and
# test are pinned to those sizes and train takes the remainder, rather than dropping rows.
VALIDATION_SIZE = 758
TEST_SIZE = 400

# Negatives over positives, over the whole corpus. Stratification keeps the rate identical in
# every fold's training split, so one constant per task is right for all ten.
BASIL_CLASS_WEIGHTS = {"lexical": (1.0, 16.78), "informational": (1.0, 5.56)}

# "10-fold cross validation" in the paper cannot be a disjoint partition -- that would give a
# 798-sentence test set, not 400 -- so a fold is a fresh stratified split at the published sizes.
NUM_FOLDS = 10


@dataclass
class BasilConfig:
    """One BASIL run: which bias type, and which of the ten stratified splits."""

    task: str = "informational"
    fold: int = 0
    data_root: str = BASIL_DATA_ROOT


def variant_name(config: BasilConfig) -> str:
    """Directory stem naming one variant. The fold is a directory level of its own."""
    return config.task


def _validate(config: BasilConfig) -> None:
    """Reject configurations whose results directory would not mean what it says."""
    if config.task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {config.task!r}")
    if config.fold not in range(NUM_FOLDS):
        raise ValueError(f"fold must be in range({NUM_FOLDS}), got {config.fold!r}")


def _biased_sentences(annotation: dict, sentences: List[str], stem: str) -> Dict[str, set]:
    """The sentence indices carrying a span of each bias type, checked as they go."""
    marked: Dict[str, set] = {task: set() for task in TASKS}
    task_of = {code: task for task, code in BIAS_CODE.items()}
    for span in annotation["phrase-level-annotations"]:
        # Three annotations sit on the title, which is not one of the body sentences.
        if span["id"] == "title":
            continue
        index = int(span["id"][1:])
        # `id` is p{flattened sentence index} and start/end are offsets into that one sentence;
        # a wrong join here would mislabel rows silently, so it is checked on every load.
        if sentences[index][span["start"]:span["end"]] != span["txt"]:
            raise ValueError(
                f"{stem}: annotation {span['id']} [{span['start']}:{span['end']}] reads "
                f"{sentences[index][span['start']:span['end']]!r}, not {span['txt']!r}. The "
                "span-to-sentence join is wrong and every label built from it is suspect."
            )
        marked[task_of[span["bias"]]].add(index)
    return marked


def load_articles(data_root: str = BASIL_DATA_ROOT) -> pd.DataFrame:
    """One row per body sentence, with a binary column per bias type. Titles are excluded."""
    rows = []
    for article_path in sorted(glob.glob(f"{data_root}/articles/*/*.json")):
        stem = os.path.basename(article_path)[: -len(".json")]
        annotation_path = article_path.replace("/articles/", "/annotations/").replace(
            ".json", "_ann.json"
        )
        with open(article_path) as handle:
            article = json.load(handle)
        with open(annotation_path) as handle:
            annotation = json.load(handle)

        sentences = [s for paragraph in article["body-paragraphs"] for s in paragraph]
        marked = _biased_sentences(annotation, sentences, stem)
        for index, sentence in enumerate(sentences):
            rows.append(
                {
                    # Four articles have uuid "empty", so the file stem is the only unique key.
                    "ID": f"{stem}_{index}",
                    "content": sentence,
                    # Cased FOX/HPO/NYT appear in 11 files.
                    "source": article["source"].lower(),
                    **{task: int(index in marked[task]) for task in TASKS},
                }
            )
    return pd.DataFrame(rows)


def split_frame(frame: pd.DataFrame, task: str, fold: int) -> Dict[str, pd.DataFrame]:
    """Stratify on the task's label into the published sizes, seeded by the fold index."""
    labels = frame[task]
    rest, test = train_test_split(
        frame, test_size=TEST_SIZE, stratify=labels, random_state=fold
    )
    train, validation = train_test_split(
        rest, test_size=VALIDATION_SIZE, stratify=rest[task], random_state=fold
    )
    return {"train": train, "validation": validation, "test": test}


def build_rows(frame: pd.DataFrame, task: str) -> Dataset:
    """One row per sentence, labelled 1 when it carries a span of `task`'s bias type."""
    return Dataset.from_dict(
        {
            "content": list(frame["content"]),
            "bias_labels": [int(value) for value in frame[task]],
            "ID": list(frame["ID"]),
            # A string, so the default collator drops it before `forward`; an int would be
            # collated and passed as a kwarg an HF baseline's explicit signature rejects.
            "source": list(frame["source"]),
        }
    )


def tokenize_rows(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase, use_bias_keys: bool
) -> Dataset:
    """Tokenize to fixed-length rows and rename to the prefixed keys the model dispatches on."""

    def tokenize_batch(examples):
        return tokenizer(
            examples["content"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["content"])

    if use_bias_keys:
        tokenized = tokenized.rename_column("input_ids", "bias_input_ids")
        tokenized = tokenized.rename_column("attention_mask", "bias_attention_mask")
    else:
        tokenized = tokenized.rename_column("bias_labels", "labels")
    return tokenized


def load_basil(
    config: BasilConfig, tokenizer: PreTrainedTokenizerBase, use_bias_keys: bool
) -> DatasetDict:
    """Build the three tokenized splits for one BASIL fold. Never undersampled: the paper
    classifies every sentence, and the imbalance is handled by the loss weighting instead.
    """
    _validate(config)
    frame = load_articles(config.data_root)
    frames = split_frame(frame, config.task, config.fold)

    splits = DatasetDict()
    for split_name, split in frames.items():
        splits[split_name] = tokenize_rows(
            build_rows(split, config.task), tokenizer, use_bias_keys
        )

    print(f"{variant_name(config)} fold {config.fold}:")
    for split_name, split in frames.items():
        print(f"  {split_name}: {len(split)} rows, {int(split[config.task].sum())} positive")
    return splits
