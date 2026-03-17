import ast
import numpy as np
import nltk
from collections import Counter
from nltk.tokenize import sent_tokenize
from typing import Dict, List, Any
from transformers import PreTrainedTokenizerBase
from datasets import Dataset
import re
import unicodedata
import json

from dotenv import load_dotenv
from transformers import set_seed

# Load tag-group mapping once at module level to avoid re-reading per batch
try:
    with open("reversed_tags_kmeans_100.json") as _f:
        _TAG_GROUPS: dict = json.load(_f)
except FileNotFoundError:
    _TAG_GROUPS = {}


# Download required NLTK models once
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


set_seed(42)
load_dotenv()


def sanitize_for_literal_eval(text: str) -> str:
    """
    Clean a string to make it safe for ast.literal_eval:
    - Fixes smart quotes, dashes, ellipses
    - Removes invisible/control characters
    - Ensures quotes are balanced
    """
    # Replace unicode dashes with ASCII hyphen
    text = text.replace("–", "-").replace("—", "-")

    # Replace smart quotes with straight quotes
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # Replace ellipsis with three dots
    text = text.replace("…", "...")

    # Remove invisible Unicode characters (e.g., zero-width space, non-breaking space)
    text = re.sub(r"[\u200B-\u200D\uFEFF\u00A0]", "", text)

    # Normalize Unicode (decompose accents, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Remove other control characters and enforce printable characters
    text = "".join(ch for ch in text if ch.isprintable())

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Handle mismatched quotes: skip malformed string if obviously invalid
    # Count total quotes — if odd, reject or fix
    if text.count("'") % 2 != 0 or text.count('"') % 2 != 0:
        raise ValueError("Unbalanced quotes detected in string.")

    return text


def undersample_per_topic(dataset: Dataset) -> Dataset:
    """
    Undersample so that, within each topic, the `int_bias` classes are balanced.
    Returns a new Dataset containing an equal number of samples per int_bias class
    within each topic.
    """
    # Convert columns to numpy arrays once for speed
    topics = np.array(dataset["topic"])
    labels = np.array(dataset["int_bias"])
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


# -----------------------------------------------------------
# 1) A helper to sanitize and convert a string‐list into plain text
# -----------------------------------------------------------
def _convert_single_stringified_list(text: str) -> str:
    """
    Try to literal‐eval a string (after some sanitization) into a Python list.
    If it really is a list, join its items with spaces. Otherwise, return
    either the sanitized string or the raw fallback.
    """
    try:
        sanitized = sanitize_for_literal_eval(text)
        if sanitized.startswith("[") and sanitized.endswith("]"):
            parsed = ast.literal_eval(sanitized)
            if isinstance(parsed, list):
                return " ".join(map(str, parsed))
            else:
                return str(parsed)
        else:
            return sanitized
    except Exception:
        # If any error occurs, return a best‐effort sanitized string or the raw text
        try:
            return text
        except Exception:
            return text


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


def prepend_sentiments(
    batch: Dict[str, List[Any]], tokenizer: PreTrainedTokenizerBase
) -> Dict[str, List[Any]]:
    out_texts: List[Any] = []
    SEP = tokenizer.sep_token
    UNK = tokenizer.unk_token
    for k1, v1, k2, v2, k3, v3, k4, v4, k5, v5, text in zip(
        batch["text_topic_0"],
        batch["text_sentiment_0"],
        batch["text_topic_1"],
        batch["text_sentiment_1"],
        batch["text_topic_2"],
        batch["text_sentiment_2"],
        batch["text_topic_3"],
        batch["text_sentiment_3"],
        batch["text_topic_4"],
        batch["text_sentiment_4"],
        batch["text"],
    ):
        v1 = get_sign(v1, UNK)
        v2 = get_sign(v2, UNK)
        v3 = get_sign(v3, UNK)
        v4 = get_sign(v4, UNK)
        v5 = get_sign(v5, UNK)

        k1 = _TAG_GROUPS.get(k1, UNK)
        k2 = _TAG_GROUPS.get(k2, UNK)
        k3 = _TAG_GROUPS.get(k3, UNK)
        k4 = _TAG_GROUPS.get(k4, UNK)
        k5 = _TAG_GROUPS.get(k5, UNK)

        for k, v in zip([k2, k3, k4, k5], [v2, v3, v4, v5]):
            if k1 == k and v != UNK:
                if v1 == UNK:
                    v1 = v
                else:
                    v1 += v

        out_texts.append(
            f"{k1} {v1} {SEP} {k2} {v2} {SEP} {k3} {v3} {SEP} {k4} {v4} {SEP} {k5} {v5} {SEP} {text}"
        )

    return {"text": out_texts, "int_bias": batch["int_bias"], "id": batch["id"]}


# -----------------------------------------------------------
# 3) Optimized undersampling using Dataset.filter + random seed
# -----------------------------------------------------------
def _balanced_filter(
    example: Dict[str, Any], counts: Dict[int, int], min_count: int
) -> bool:
    """
    We will call dataset.shuffle(seed=…) and then filter out examples of each class
    beyond the first `min_count`. This requires a tiny bit of bookkeeping at map time:
    we maintain a counter per label.
    """
    label = example["int_bias"]
    if counts[label] < min_count:
        counts[label] += 1
        return True
    else:
        return False


def undersample_optimized(dataset: Dataset, seed: int = 42) -> Dataset:
    """
    Instead of pulling out indices in Python, we shuffle the dataset, then filter
    so that for each label we keep only `min_count` examples.
    """
    # 1) Compute the minimum class count
    label_counts = Counter(dataset["int_bias"])
    min_samples = min(label_counts.values())

    # 2) Shuffle once so that selecting the first N per class is random
    shuffled = dataset.shuffle(seed=seed)

    # 3) Create a mutable counter that map_fn can update
    counts = {lab: 0 for lab in label_counts}

    # 4) Filter: keep only the first `min_samples` of each label
    balanced = shuffled.filter(
        function=lambda ex, counts=counts, minv=min_samples: _balanced_filter(
            ex, counts, minv
        ),
        batched=False,
    )

    return balanced


# -----------------------------------------------------------
# 4) The final “fused” pipeline that puts it all together
# -----------------------------------------------------------
def clean_dataset_optimized(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 1024,
    num_proc: int = 4,
    sentiments=False,
    skip_undersampling=False,
    use_bias_keys: bool = True,
) -> Dataset | None:
    """
    1. Undersample (balance) via `undersample_optimized`.
    2. Convert stringified lists → text (truncation applied at tokenization).
    3. Flatten the result.
    4. Class‐encode labels and remap to "label".
    5. Tokenize all text to input_ids/attention_mask.
    """

    print("Initial label distribution:", Counter(dataset["int_bias"]))

    filtered = dataset

    if len(filtered) < 1:
        return None

    # ---- 1) Undersample (balanced) ----
    if skip_undersampling:
        balanced = filtered
    else:
        balanced = undersample_per_topic(filtered)

    # ---- 2) Optionally prepend sentiments ----
    if sentiments:
        chunked = balanced.map(
            lambda batch: prepend_sentiments(batch, tokenizer),
            batched=True,
            remove_columns=balanced.column_names,
            num_proc=num_proc,
        )
    else:
        chunked = balanced
    # ---- 4) Flatten: each row is now a single chunk with its label and id ----
    chunked = chunked.flatten()

    # ---- 5) Class‐encode labels, rename to "label", select only needed columns ----
    # chunked = chunked.class_encode_column("int_bias")  # now int_bias → class IDs

    if use_bias_keys:
        chunked = chunked.rename_column("int_bias", "bias_labels")
        chunked = chunked.select_columns(["bias_labels", "text", "id"])
    else:
        chunked = chunked.rename_column("int_bias", "labels")
        chunked = chunked.select_columns(["labels", "text", "id"])
    # ---- 6) Tokenize all final chunks (batched) ----
    def tokenize_batch(exs: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tokenizer(
            exs["text"], padding="max_length", truncation=True, max_length=max_length
        )

    tokenized = chunked.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],  # no longer need raw text after tokenization
        num_proc=num_proc,
    )
    if use_bias_keys:
        tokenized = tokenized.rename_column("input_ids", "bias_input_ids")
        tokenized = tokenized.rename_column("attention_mask", "bias_attention_mask")

    return tokenized
