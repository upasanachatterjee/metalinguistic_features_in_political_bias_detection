import torch


def _pad_or_truncate(seq, max_len, pad_value):
    if len(seq) > max_len:
        return seq[:max_len]
    return seq + [pad_value] * (max_len - len(seq))


def pack_triplets(a_id, a_att, p_id, p_att, n_id, n_att, max_length):
    """Pad/truncate token-id and attention-mask lists to max_length and stack into int64 tensors.

    Returns a dict with keys: a_ids, a_mask, p_ids, p_mask, n_ids, n_mask.
    """
    pad_ids = 0
    pad_mask = 0

    a_id = [_pad_or_truncate(s, max_length, pad_ids) for s in a_id]
    p_id = [_pad_or_truncate(s, max_length, pad_ids) for s in p_id]
    n_id = [_pad_or_truncate(s, max_length, pad_ids) for s in n_id]
    a_att = [_pad_or_truncate(s, max_length, pad_mask) for s in a_att]
    p_att = [_pad_or_truncate(s, max_length, pad_mask) for s in p_att]
    n_att = [_pad_or_truncate(s, max_length, pad_mask) for s in n_att]

    def stack(rows):
        return torch.stack([torch.tensor(r) for r in rows]).to(torch.int64)

    return {
        "a_ids": stack(a_id),
        "a_mask": stack(a_att),
        "p_ids": stack(p_id),
        "p_mask": stack(p_att),
        "n_ids": stack(n_id),
        "n_mask": stack(n_att),
    }


def empty_triplet_batch():
    """Empty batch placeholder when no valid triplets can be formed."""
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
