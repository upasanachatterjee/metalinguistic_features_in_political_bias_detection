import random
from typing import Any, Dict, List, Sequence

from accelerate.state import PartialState

from collators._triplet_utils import Triplet, empty_triplet_batch, pack_triplets

OPPOSITE = {"left": "right", "right": "left"}


def _log(message: str) -> None:
    """Print on rank 0 only.
    """
    if PartialState().is_main_process:
        print(message, flush=True)


class TripletDataCollator:
    """Mines triplets within each batch: anchor and positive share a
    `political_bias`, the negative carries the opposite one.

    Batches with no usable left/right split are returned as a `_skip` batch and
    contribute no triplet loss for that step.
    """

    def __init__(
        self,
        political_bias_field="political_bias",
        max_length=512,
        triplet_downsample_size=16,
    ):
        self.political_bias_field = political_bias_field
        self.max_length = max_length
        self.triplet_downsample_size = triplet_downsample_size

    def __call__(self, batch):
        triplets = sample_triplets(
            batch, self.political_bias_field, self.triplet_downsample_size
        )
        if not triplets:
            return empty_triplet_batch()
        return pack_triplets(triplets, self.max_length)


def group_by_bias(
    batch: Sequence[Dict[str, Any]], political_bias_field: str
) -> Dict[str, List[Dict[str, Any]]]:
    """Split a batch into its left-leaning and right-leaning samples.

    Samples with any other (or malformed) bias value are dropped; the corpus
    also carries "center", which the contrastive objective has no use for.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {"left": [], "right": []}
    for item in batch:
        bias = item.get(political_bias_field)
        if not isinstance(bias, str):
            _log(f"Unknown ideology format: {bias!r}; skipping sample")
            continue
        if bias in groups:
            groups[bias].append(item)
    return groups


def sample_triplets(
    batch: Sequence[Dict[str, Any]],
    political_bias_field: str,
    max_triplets: int,
    rng: random.Random = random,  # type: ignore[assignment]
) -> List[Triplet]:
    """Draw up to `max_triplets` (anchor, positive, negative) samples from a batch.

    Anchor and positive share a bias, the negative has the opposite one.
    """
    groups = group_by_bias(batch, political_bias_field)

    # A triplet needs two distinct samples from one side and one from the other.
    usable = [
        side
        for side in ("left", "right")
        if len(groups[side]) >= 2 and len(groups[OPPOSITE[side]]) >= 1
    ]
    if not usable:
        _log(
            f"Batch has no usable left/right split: {[len(groups[side]) for side in ('left', 'right')]}"
        )
        return []

    distinct_possible = sum(
        len(groups[side]) * (len(groups[side]) - 1) * len(groups[OPPOSITE[side]])
        for side in usable
    )
    wanted = min(max_triplets, distinct_possible)

    triplets: List[Triplet] = []
    seen = set()
    for _ in range(wanted * 20):
        if len(triplets) == wanted:
            break
        side = rng.choice(usable)
        anchor_idx, positive_idx = rng.sample(range(len(groups[side])), 2)
        negative_idx = rng.randrange(len(groups[OPPOSITE[side]]))
        key = (side, anchor_idx, positive_idx, negative_idx)
        if key in seen:
            continue
        seen.add(key)
        triplets.append(
            (
                groups[side][anchor_idx],
                groups[side][positive_idx],
                groups[OPPOSITE[side]][negative_idx],
            )
        )
    return triplets
