from typing import Any, Dict, List, Sequence, Tuple

import torch

Triplet = Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]


def _pad_or_truncate(seq, max_len, pad_value):
    if len(seq) > max_len:
        return seq[:max_len]
    return seq + [pad_value] * (max_len - len(seq))


def pack_triplets(triplets: Sequence[Triplet], max_length: int) -> Dict[str, torch.Tensor]:
    """Stack (anchor, positive, negative) samples into padded int64 tensors.

    Returns a dict with keys a_ids, a_mask, p_ids, p_mask, n_ids, n_mask, each
    of shape (len(triplets), max_length).
    """

    def column(role: int, field: str) -> torch.Tensor:
        rows: List[List[int]] = [
            _pad_or_truncate(triplet[role][field], max_length, 0)
            for triplet in triplets
        ]
        return torch.tensor(rows, dtype=torch.int64)

    anchor, positive, negative = 0, 1, 2
    return {
        "a_ids": column(anchor, "input_ids"),
        "a_mask": column(anchor, "attention_mask"),
        "p_ids": column(positive, "input_ids"),
        "p_mask": column(positive, "attention_mask"),
        "n_ids": column(negative, "input_ids"),
        "n_mask": column(negative, "attention_mask"),
    }


def empty_triplet_batch() -> Dict[str, Any]:
    """Placeholder batch when no valid triplets can be formed; the trainer skips it."""
    empty_ids = torch.zeros((0, 1), dtype=torch.int64)
    empty_mask = torch.zeros((0, 1), dtype=torch.int64)
    return {
        "a_ids": empty_ids,
        "a_mask": empty_mask,
        "p_ids": empty_ids,
        "p_mask": empty_mask,
        "n_ids": empty_ids,
        "n_mask": empty_mask,
        "_skip": True,
    }
