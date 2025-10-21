import torch
from torch.nn.utils.rnn import pad_sequence
import random


class TripletDataCollator:
    def __init__(
        self,
        group_field="group_uid",
        political_bias_field="political_bias",
        group_by_uid=False,
    ):
        """
        Initialize the triplet collator.

        Args:
            group_field: The field name containing group information for triplet formation.
                        All samples (anchor, positive, negative) will come from the same group.
                        Within each group:
                        - Anchor and Positive: Same political_bias
                        - Negative: Opposite political_bias
        """
        self.group_field = group_field
        self.political_bias_field = political_bias_field
        self.group_by_uid = group_by_uid

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
                batch, self.political_bias_field
            )

        # Handle case where no valid triplets were formed
        if not a_id:
            return self._create_empty_batch()

        # Stack tensors
        # Convert lists to tensors before stacking
        anchor_id_tensors = [torch.tensor(ids) for ids in a_id]
        positive_id_tensors = [torch.tensor(ids) for ids in p_id]
        negative_id_tensors = [torch.tensor(ids) for ids in n_id]
        anchor_attention_tensors = [torch.tensor(mask) for mask in a_att]
        positive_attention_tensors = [torch.tensor(mask) for mask in p_att]
        negative_attention_tensors = [torch.tensor(mask) for mask in n_att]

        # Pad sequences to same length before stacking
        anchors_id_tensor = pad_sequence(
            anchor_id_tensors, batch_first=True, padding_value=0
        ).to(torch.int64)
        positives_tensor = pad_sequence(
            positive_id_tensors, batch_first=True, padding_value=0
        ).to(torch.int64)
        negatives_tensor = pad_sequence(
            negative_id_tensors, batch_first=True, padding_value=0
        ).to(torch.int64)
        anchors_attention_tensor = pad_sequence(
            anchor_attention_tensors, batch_first=True, padding_value=0
        ).to(torch.int64)
        positives_attention_tensor = pad_sequence(
            positive_attention_tensors, batch_first=True, padding_value=0
        ).to(torch.int64)
        negatives_attention_tensor = pad_sequence(
            negative_attention_tensors, batch_first=True, padding_value=0
        ).to(torch.int64)

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


def sample_ideology_triplets(batch, political_bias_field):
    """
    Sample triplets for contrastive learning based on political ideology.
    
    This function creates triplets where:
    - Anchor: Current article
    - Positive: Article with same political bias/ideology
    - Negative: Article with opposite political bias/ideology
    
    The function ensures that the number of triplets with left anchors
    exactly matches the number of triplets with right anchors.
    
    Arguments:
        batch: A batch of data containing articles
        political_bias_field: Name of the field containing political bias values 
                             (should contain values like "left", "right" or numerical values)
    
    Returns:
        Tuple of lists containing attention masks and input ids for anchors, positives, and negatives
    """
    # Group articles by ideology
    ideology_groups = {"left": [], "right": []}

    # Process batch and group by ideology
    for item in batch:
        # Get the ideology value, could be string or numeric
        ideology = item.get(political_bias_field)

        # Handle different value formats
        if isinstance(ideology, (int, float)):
            if ideology < 0:  # Negative values represent left
                ideology = "left"
            elif ideology > 0:  # Positive values represent right
                ideology = "right"
            else:  # Skip neutral (0)
                continue
                
        # Skip center/unknown ideology
        if ideology in ["left", "right"]:
            ideology_groups[ideology].append(item)
    
    # Check if we have enough items in each ideology group to form triplets
    if len(ideology_groups["left"]) < 2 or len(ideology_groups["right"]) < 2:
        # Not enough items to form triplets, return empty lists
        return [], [], [], [], [], []
    
    # Create separate lists for left and right triplets
    left_triplets = []
    right_triplets = []
    
    # Create triplets from the left ideology group
    for i, anchor in enumerate(ideology_groups["left"]):
        # Find positive candidates (same ideology, different item)
        pos_candidates = [ideology_groups["left"][j] for j in range(len(ideology_groups["left"])) if j != i]
        if pos_candidates:
            positive = random.choice(pos_candidates)
            negative = random.choice(ideology_groups["right"])  # Opposite ideology
            
            # Store triplet as a tuple of (anchor, positive, negative) with their attention masks and input ids
            left_triplets.append((
                anchor["attention_mask"], 
                anchor["input_ids"],
                positive["attention_mask"], 
                positive["input_ids"],
                negative["attention_mask"], 
                negative["input_ids"]
            ))
    
    # Create triplets from the right ideology group
    for i, anchor in enumerate(ideology_groups["right"]):
        # Find positive candidates (same ideology, different item)
        pos_candidates = [ideology_groups["right"][j] for j in range(len(ideology_groups["right"])) if j != i]
        if pos_candidates:
            positive = random.choice(pos_candidates)
            negative = random.choice(ideology_groups["left"])  # Opposite ideology
            
            # Store triplet as a tuple
            right_triplets.append((
                anchor["attention_mask"], 
                anchor["input_ids"],
                positive["attention_mask"], 
                positive["input_ids"],
                negative["attention_mask"], 
                negative["input_ids"]
            ))
    
    # Balance triplets after creation
    min_triplets = min(len(left_triplets), len(right_triplets))
    
    if min_triplets == 0:
        # No valid triplets could be formed
        return [], [], [], [], [], []
    
    # Randomly sample to ensure equal number of triplets from each group
    if len(left_triplets) > min_triplets:
        left_triplets = random.sample(left_triplets, min_triplets)
    if len(right_triplets) > min_triplets:
        right_triplets = random.sample(right_triplets, min_triplets)
    
    # Combine and possibly shuffle triplets
    balanced_triplets = left_triplets + right_triplets
    random.shuffle(balanced_triplets)
    
    # Unpack balanced triplets into separate lists
    anchor_attention, anchor_id = [], []
    positive_attention, positive_id = [], []
    negative_attention, negative_id = [], []
    
    for triplet in balanced_triplets:
        anchor_attention.append(triplet[0])
        anchor_id.append(triplet[1])
        positive_attention.append(triplet[2])
        positive_id.append(triplet[3])
        negative_attention.append(triplet[4])
        negative_id.append(triplet[5])
    
    return anchor_attention, anchor_id, positive_attention, positive_id, negative_attention, negative_id


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
