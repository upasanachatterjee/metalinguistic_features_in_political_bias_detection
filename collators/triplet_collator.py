import random

from collators._triplet_utils import empty_triplet_batch, pack_triplets


class TripletDataCollator:
    """Ideology triplet mining: anchor/positive share political_bias, negative has opposite bias.

    Ignores group_uid. Mirrors the original behavior used for the POLITICS-style ideology objective.
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
        a_att, a_id, p_att, p_id, n_att, n_id = sample_ideology_triplets(
            batch, self.political_bias_field, self.triplet_downsample_size
        )

        if not a_id:
            return empty_triplet_batch()

        return pack_triplets(a_id, a_att, p_id, p_att, n_id, n_att, self.max_length)


def sample_ideology_triplets(batch, political_bias_field, triplet_downsample_size):
    ideology_groups = {"left": [], "right": []}

    for item in batch:
        ideology = item.get(political_bias_field)

        if ideology in ["left", "right"]:
            ideology_groups[ideology].append(item)

        if not isinstance(ideology, str):
            print(f"Unknown ideology format for item: {item}")
            return [], [], [], [], [], []

    if len(ideology_groups["left"]) < 1 or len(ideology_groups["right"]) < 1:
        return [], [], [], [], [], []

    all_triplets = []

    if len(ideology_groups["left"]) >= 2:
        for i, anchor in enumerate(ideology_groups["left"]):
            pos_candidates = [
                p for j, p in enumerate(ideology_groups["left"]) if i != j
            ]
            neg_candidates = ideology_groups["right"]
            for positive in pos_candidates:
                for negative in neg_candidates:
                    all_triplets.append((anchor, positive, negative))

    if len(ideology_groups["right"]) >= 2:
        for i, anchor in enumerate(ideology_groups["right"]):
            pos_candidates = [
                p for j, p in enumerate(ideology_groups["right"]) if i != j
            ]
            neg_candidates = ideology_groups["left"]
            for positive in pos_candidates:
                for negative in neg_candidates:
                    all_triplets.append((anchor, positive, negative))

    if len(all_triplets) > triplet_downsample_size:
        all_triplets = random.sample(all_triplets, k=triplet_downsample_size)

    anchor_attention, anchor_id = [], []
    positive_attention, positive_id = [], []
    negative_attention, negative_id = [], []

    for anchor, positive, negative in all_triplets:
        anchor_attention.append(anchor["attention_mask"])
        anchor_id.append(anchor["input_ids"])
        positive_attention.append(positive["attention_mask"])
        positive_id.append(positive["input_ids"])
        negative_attention.append(negative["attention_mask"])
        negative_id.append(negative["input_ids"])

    return (
        anchor_attention,
        anchor_id,
        positive_attention,
        positive_id,
        negative_attention,
        negative_id,
    )
