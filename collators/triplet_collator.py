import torch
import random


class TripletDataCollator:
    def __init__(
        self,
        group_field="group_uid",
        political_bias_field="political_bias",
        group_by_uid=False,
        max_length=512,
        triplet_downsample_size=16,
    ):
        print(f"Grouping by uid: {group_by_uid}")
        self.group_field = group_field
        self.political_bias_field = political_bias_field
        self.group_by_uid = group_by_uid
        self.max_length = max_length
        self.triplet_downsample_size = triplet_downsample_size

    def __call__(self, batch):
        a_att, a_id = [], []
        p_att, p_id = [], []
        n_att, n_id = [], []

        if self.group_by_uid:
            # Group items by group_uid and then by political_bias within each group
            valid_groups = create_valid_groups(
                batch, self.group_field, self.political_bias_field
            )

            if not valid_groups:
                # Return empty tensors if no valid triplets can be formed
                return self._create_empty_batch()

            # Create triplets within each valid group
            for group_id, bias_groups in valid_groups.items():
                left_items = bias_groups["left"]
                right_items = bias_groups["right"]
                a_att, a_id, p_att, p_id, n_att, n_id = create_triplets(
                    left_items, right_items
                )
        else:
            # Use the sample_ideology_triplets function when group_field is None
            a_att, a_id, p_att, p_id, n_att, n_id = sample_ideology_triplets(
                batch, self.political_bias_field, self.triplet_downsample_size
            )

        # Handle case where no valid triplets were formed
        if not a_id:
            return self._create_empty_batch()

        # Pad or truncate each sequence to max_length
        pad_value_ids = 0  # Assuming padding token ID is 0
        pad_value_mask = 0

        def pad_or_truncate(seq, max_len, pad_value):
            if len(seq) > max_len:
                return seq[:max_len]
            else:
                return seq + [pad_value] * (max_len - len(seq))

        a_id = [pad_or_truncate(ids, self.max_length, pad_value_ids) for ids in a_id]
        p_id = [pad_or_truncate(ids, self.max_length, pad_value_ids) for ids in p_id]
        n_id = [pad_or_truncate(ids, self.max_length, pad_value_ids) for ids in n_id]
        a_att = [
            pad_or_truncate(mask, self.max_length, pad_value_mask) for mask in a_att
        ]
        p_att = [
            pad_or_truncate(mask, self.max_length, pad_value_mask) for mask in p_att
        ]
        n_att = [
            pad_or_truncate(mask, self.max_length, pad_value_mask) for mask in n_att
        ]

        # Stack tensors
        # Convert lists to tensors before stacking
        anchor_id_tensors = [torch.tensor(ids) for ids in a_id]
        positive_id_tensors = [torch.tensor(ids) for ids in p_id]
        negative_id_tensors = [torch.tensor(ids) for ids in n_id]
        anchor_attention_tensors = [torch.tensor(mask) for mask in a_att]
        positive_attention_tensors = [torch.tensor(mask) for mask in p_att]
        negative_attention_tensors = [torch.tensor(mask) for mask in n_att]

        # Pad sequences to same length before stacking
        anchors_id_tensor = torch.stack(anchor_id_tensors).to(torch.int64)
        positives_tensor = torch.stack(positive_id_tensors).to(torch.int64)
        negatives_tensor = torch.stack(negative_id_tensors).to(torch.int64)
        anchors_attention_tensor = torch.stack(anchor_attention_tensors).to(torch.int64)
        positives_attention_tensor = torch.stack(positive_attention_tensors).to(
            torch.int64
        )
        negatives_attention_tensor = torch.stack(negative_attention_tensors).to(
            torch.int64
        )

        return {
            "a_ids": anchors_id_tensor,
            "a_mask": anchors_attention_tensor,
            "p_ids": positives_tensor,
            "p_mask": positives_attention_tensor,
            "n_ids": negatives_tensor,
            "n_mask": negatives_attention_tensor,
        }

    def _create_empty_batch(self):
        """Create empty batch when no valid triplets can be formed"""
        # Create minimal empty tensors (1x1 for compatibility)
        empty_ids = torch.zeros((0, 1), dtype=torch.int64)
        empty_mask = torch.zeros((0, 1), dtype=torch.int64)

        return {
            "a_ids": empty_ids,
            "a_mask": empty_mask,
            "p_ids": empty_ids,
            "p_mask": empty_mask,
            "n_ids": empty_ids,
            "n_mask": empty_mask,
            "_skip": True,  # Flag to indicate this batch should be skipped
        }


def sample_ideology_triplets(batch, political_bias_field, triplet_downsample_size):
    # Group articles by ideology
    ideology_groups = {"left": [], "right": []}

    # Process batch and group by ideology
    for item in batch:
        # Get the ideology value, could be string or numeric
        ideology = item.get(political_bias_field)

        # Skip center/unknown ideology
        if ideology in ["left", "right"]:
            ideology_groups[ideology].append(item)

        if not isinstance(ideology, (str)):
            print(f"Unknown ideology format for item: {item}")
            return [], [], [], [], [], []

    # Check if we have enough items in each ideology group to form triplets
    if len(ideology_groups["left"]) < 1 or len(ideology_groups["right"]) < 1:
        # Not enough items to form triplets, return empty lists
        return [], [], [], [], [], []

    all_triplets = []

    # Generate triplets with left-wing anchors
    if len(ideology_groups["left"]) >= 2:
        for i, anchor in enumerate(ideology_groups["left"]):
            pos_candidates = [
                p for j, p in enumerate(ideology_groups["left"]) if i != j
            ]
            neg_candidates = ideology_groups["right"]
            for positive in pos_candidates:
                for negative in neg_candidates:
                    all_triplets.append((anchor, positive, negative))

    # Generate triplets with right-wing anchors
    if len(ideology_groups["right"]) >= 2:
        for i, anchor in enumerate(ideology_groups["right"]):
            pos_candidates = [
                p for j, p in enumerate(ideology_groups["right"]) if i != j
            ]
            neg_candidates = ideology_groups["left"]
            for positive in pos_candidates:
                for negative in neg_candidates:
                    all_triplets.append((anchor, positive, negative))

    # Downsample if more than triplet_downsample_size triplets are generated
    if len(all_triplets) > triplet_downsample_size:
        all_triplets = random.sample(all_triplets, k=triplet_downsample_size)

    # Unpack triplets into separate lists
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


def create_valid_groups(batch, group_field, political_bias_field):
    groups = {}
    for item in batch:
        group_id = item.get(group_field, None)
        bias = item.get(political_bias_field, "").lower()

        if group_id is not None and bias:
            if group_id not in groups:
                groups[group_id] = {"left": [], "right": []}

            if bias == "left":
                groups[group_id]["left"].append(item)
            elif bias == "right":
                groups[group_id]["right"].append(item)

    # Filter to groups that have both left and right bias items
    valid_groups = {}
    for group_id, bias_groups in groups.items():
        if len(bias_groups["left"]) >= 1 and len(bias_groups["right"]) >= 1:
            # Also need at least 2 items of the same bias to form anchor-positive pairs
            if len(bias_groups["left"]) >= 2 or len(bias_groups["right"]) >= 2:
                valid_groups[group_id] = bias_groups

    return valid_groups


def create_triplets(left_items, right_items):
    anchor_attention, anchor_id = [], []
    positive_attention, positive_id = [], []
    negative_attention, negative_id = [], []
    # Create triplets: left anchor, left positive, right negative
    if len(left_items) >= 2:
        for i, anchor in enumerate(left_items):
            # Find positive candidates (same group, same bias, different item)
            pos_candidates = [left_items[j] for j in range(len(left_items)) if j != i]
            if pos_candidates:
                positive = random.choice(pos_candidates)
                negative = random.choice(right_items)  # Opposite bias, same group

                anchor_attention.append(anchor["attention_mask"])
                anchor_id.append(anchor["input_ids"])
                positive_attention.append(positive["attention_mask"])
                positive_id.append(positive["input_ids"])
                negative_attention.append(negative["attention_mask"])
                negative_id.append(negative["input_ids"])

    # Create triplets: right anchor, right positive, left negative
    if len(right_items) >= 2:
        for i, anchor in enumerate(right_items):
            # Find positive candidates (same group, same bias, different item)
            pos_candidates = [right_items[j] for j in range(len(right_items)) if j != i]
            if pos_candidates:
                positive = random.choice(pos_candidates)
                negative = random.choice(left_items)  # Opposite bias, same group

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
