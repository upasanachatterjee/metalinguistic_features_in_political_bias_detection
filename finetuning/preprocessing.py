import numpy as np
from collections import Counter
from typing import Any, Dict, List
from transformers import PreTrainedTokenizerBase
from datasets import Dataset

from dotenv import load_dotenv

load_dotenv()


def undersample_per_topic(dataset: Dataset) -> Dataset:
    """Undersample `bias` to its smallest class within each topic, dropping rows
    with a null topic.
    """
    # Per topic rather than globally, or the model learns topic as a proxy for bias.
    topics = np.array(dataset["topic"])
    labels = np.array(dataset["bias"])
    all_indices = []

    # Filter out None topics
    none_mask = np.array([t is not None for t in topics])
    valid_topics = topics[none_mask]
    valid_indices = np.where(none_mask)[0]
    valid_labels = labels[none_mask]
    # Find unique topics
    unique_topics = np.unique(valid_topics)

    for topic in unique_topics:
        # 1) Collect indices belonging to this topic
        topic_mask = valid_topics == topic
        topic_indices = valid_indices[topic_mask]
        topic_labels = valid_labels[topic_mask]

        # 2) Compute label counts within this topic
        label_counts = Counter(topic_labels)
        min_samples = min(label_counts.values())

        # 3) For each label value, randomly choose `min_samples` indices
        for label_value, count in label_counts.items():
            label_mask = topic_labels == label_value
            label_indices = topic_indices[label_mask]

            chosen = np.random.choice(label_indices, size=min_samples, replace=False)
            all_indices.append(chosen)

    # 4) Concatenate all chosen indices from every topic
    balanced_indices = np.concatenate(all_indices)
    # Optional: shuffle them so that the resulting dataset isn't ordered by topic/label
    np.random.shuffle(balanced_indices)

    balanced_dataset = dataset.select(balanced_indices.tolist())
    return balanced_dataset


def get_sign(v, UNK) -> int:
    if isinstance(v, (type(None))):
        return UNK
    if v == 0:
        return 0
    if v < 0:
        return -1
    if v > 0:
        return 1
    else:
        return UNK


def clean_dataset_optimized(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 1024,
    num_proc: int = 4,
    skip_undersampling=False,
    use_bias_keys: bool = True,
) -> Dataset | None:
    """Balance and tokenize a bias split, truncating at `max_length` so one article is
    one row. `use_bias_keys` picks MultiTaskRoberta's `bias_*` names over the plain ones.
    """
    print("Initial label distribution:", Counter(dataset["bias"]))

    if len(dataset) < 1:
        return None

    if skip_undersampling:
        balanced = dataset
    else:
        balanced = undersample_per_topic(dataset)

    prepared = balanced
    prepared = prepared.flatten()

    if use_bias_keys:
        prepared = prepared.rename_column("bias", "bias_labels")
        prepared = prepared.select_columns(["bias_labels", "text", "id"])
    else:
        prepared = prepared.rename_column("bias", "labels")
        prepared = prepared.select_columns(["labels", "text", "id"])

    def tokenize_batch(exs: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tokenizer(
            exs["text"], padding="max_length", truncation=True, max_length=max_length
        )

    tokenized = prepared.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],  # no longer need raw text after tokenization
        num_proc=num_proc,
    )
    if use_bias_keys:
        tokenized = tokenized.rename_column("input_ids", "bias_input_ids")
        tokenized = tokenized.rename_column("attention_mask", "bias_attention_mask")

    return tokenized
