import random

from collators._triplet_utils import empty_triplet_batch, pack_triplets


class StoryTripletCollator:
    """Story triplet mining: anchor/positive share group_uid; negative has a different group_uid.

    Ideology is ignored. Requires the dataloader to use a group-aware batch sampler so that
    batches contain multi-article stories; otherwise positives will be unfindable.
    """

    def __init__(
        self,
        group_field="group_uid",
        max_length=512,
        triplet_downsample_size=16,
    ):
        self.group_field = group_field
        self.max_length = max_length
        self.triplet_downsample_size = triplet_downsample_size

    def __call__(self, batch):
        groups: dict = {}
        for item in batch:
            gid = item.get(self.group_field)
            if gid is None:
                continue
            groups.setdefault(gid, []).append(item)

        multi_groups = {g: items for g, items in groups.items() if len(items) >= 2}
        if not multi_groups:
            return empty_triplet_batch()

        all_triplets = []

        for gid, items in multi_groups.items():
            other_items = [it for og, lst in groups.items() if og != gid for it in lst]
            if not other_items:
                continue
            for i, anchor in enumerate(items):
                for j, positive in enumerate(items):
                    if i == j:
                        continue
                    negative = random.choice(other_items)
                    all_triplets.append((anchor, positive, negative))

        if not all_triplets:
            return empty_triplet_batch()

        if len(all_triplets) > self.triplet_downsample_size:
            all_triplets = random.sample(all_triplets, k=self.triplet_downsample_size)

        a_att, a_id, p_att, p_id, n_att, n_id = [], [], [], [], [], []
        for anchor, positive, negative in all_triplets:
            a_att.append(anchor["attention_mask"])
            a_id.append(anchor["input_ids"])
            p_att.append(positive["attention_mask"])
            p_id.append(positive["input_ids"])
            n_att.append(negative["attention_mask"])
            n_id.append(negative["input_ids"])

        return pack_triplets(a_id, a_att, p_id, p_att, n_id, n_att, self.max_length)
