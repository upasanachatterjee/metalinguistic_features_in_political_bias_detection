"""One collate_fn that feeds every objective the SAME rows.
"""

from typing import Any, Callable, Dict, List, Mapping

# What DataLoader hands a collate_fn: the list of rows the sampler drew.
CollateFn = Callable[[List[Dict[str, Any]]], Any]


class MultiTaskCollator:
    """Run every active task's collator over one list of rows.

    Returns `{task_name: sub_batch}`; a task whose collator signalled `_skip`
    (no usable left/right split, no parseable tone) is absent.
    """

    def __init__(self, sub_collators: Dict[str, CollateFn]):
        self.sub_collators = sub_collators

    @property
    def tasks(self) -> List[str]:
        return list(self.sub_collators)

    def __call__(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        batches: Dict[str, Any] = {}
        for task_name, collate in self.sub_collators.items():
            sub_batch = collate(rows)
            if sub_batch is None:
                continue
            if isinstance(sub_batch, Mapping) and sub_batch.get("_skip", False):
                continue
            batches[task_name] = sub_batch
        return batches
