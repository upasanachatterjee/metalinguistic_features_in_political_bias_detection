import os
import time
import numpy as np
from accelerate.state import PartialState
from huggingface_hub import login
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from config import TaskSpec, TrainArgs
from row_selection import select_rows
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.tokenization_utils import PreTrainedTokenizer
from collators.multi_task_collator import CollateFn, MultiTaskCollator
from collators.triplet_collator import TripletDataCollator
from collators.multi_label_collator import MultiLabelCollator
from collators.regression_collator import RegressionCollator

TASK_ORDER = ("mlm", "tone", "triplet", "themes")

# Metadata columns no task reads; dropped to keep the mapped table small.
UNUSED_COLUMNS = ["source", "title", "html", "url", "date", "group_uid"]


def _login_once(state: PartialState) -> None:
    """Authenticate to GH.
    """
    if state.is_main_process:
        token = os.getenv("hf_token")
        if token:
            login(token)
            print("   logged in to the Hugging Face Hub", flush=True)
        else:
            print(
                "   no hf_token in the environment; relying on a cached login",
                flush=True,
            )
    state.wait_for_everyone()


class MemoryEfficientDataset(TorchDataset):
    """Memory-mapped view of the corpus, tokenized lazily in `__getitem__`, carrying
    raw `political_bias` / `V2Themes` / `V2Tone` for the collators to turn into labels.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        text_col: str,
        tokenizer: PreTrainedTokenizer,
        cache_dir: Optional[str] = None,
        require_nonempty_themes_and_tone: bool = False,
    ):
        self.tokenizer = tokenizer
        self.text_col = text_col

        state = PartialState()
        self.dataset = self._load(dataset_name, split, cache_dir, state)

        before = len(self.dataset)
        keep_indices = select_rows(
            self.dataset,
            dataset_name=dataset_name,
            split=split,
            cache_dir=cache_dir or "./cache",
            require_nonempty_themes_and_tone=require_nonempty_themes_and_tone,
            state=state,
        )
        if keep_indices is not None:
            self._apply_selection(keep_indices, before, state)

        # Drop columns no task reads, to keep the mapped table small.
        existing_columns = [
            col for col in UNUSED_COLUMNS if col in self.dataset.column_names
        ]
        if existing_columns:
            self.dataset = self.dataset.remove_columns(existing_columns)

        if state.is_main_process:
            print(f"   dataset ready: {len(self.dataset):,} samples", flush=True)

    @staticmethod
    def _load(
        dataset_name: str,
        split: str,
        cache_dir: Optional[str],
        state: PartialState,
    ) -> Dataset:
        """Memory-map the corpus, letting rank 0 warm the HF cache first."""
        # A rank crashing mid-load leaves an orphan .lock that deadlocks the next launch.
        if state.is_main_process:
            print(f"   loading {dataset_name} (memory-mapped)", flush=True)
        t = time.perf_counter()
        with state.main_process_first():
            dataset = load_dataset(
                dataset_name,
                split=split,
                revision="refs/convert/parquet",
                streaming=False,
                cache_dir=cache_dir,
                keep_in_memory=False,
            )
        if state.is_main_process:
            print(f"   loaded in {time.perf_counter() - t:.1f}s", flush=True)

        return dataset

    def _apply_selection(
        self, keep_indices: np.ndarray, before: int, state: PartialState
    ) -> None:
        """Narrow the table to `keep_indices` and materialize the result.

        Both selection rules are resolved into one index array first, so this
        Arrow rewrite -- the expensive part -- is paid exactly once.
        """
        t = time.perf_counter()
        # Staggered so rank 0 writes the indices-mapping file and the rest read it warm.
        with state.main_process_first():
            self.dataset = self.dataset.select(keep_indices)
        # Materialized so `.data.column(...)` sees the selection and __getitem__ skips it.
        with state.main_process_first():
            self.dataset = self.dataset.flatten_indices()
        if state.is_main_process:
            print(
                f"   selected {len(self.dataset):,}/{before:,} rows "
                f"(subsample + title dedup) in {time.perf_counter() - t:.1f}s",
                flush=True,
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Tokenize one row on demand, padded to a fixed 512."""
        # Fixed padding keeps the 8 DDP ranks in lockstep and the collators' padding a no-op.
        sample = self.dataset[idx]
        text = sample.get(self.text_col, "")

        tokenized = self.tokenizer(
            text, truncation=True, padding="max_length", return_attention_mask=True
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "political_bias": sample["political_bias"],
            "V2Themes": sample["V2Themes"],
            "V2Tone": sample["V2Tone"],
        }


def build_dataloader(
    tok: PreTrainedTokenizer,
    task_spec: TaskSpec,
    args: TrainArgs,
    tasks_to_build: Optional[Iterable[str]] = None,
) -> Tuple[DataLoader, List[str]]:
    """One shuffled dataloader feeding every objective the same rows, returning it
    and the tasks it actually collates.
    """
    requested = set(tasks_to_build) if tasks_to_build is not None else set(TASK_ORDER)

    _login_once(PartialState())

    dataset = MemoryEfficientDataset(
        dataset_name=task_spec.dataset_name,
        split=task_spec.split,
        text_col=task_spec.text_col,
        tokenizer=tok,
        cache_dir=task_spec.index_cache_dir,
        require_nonempty_themes_and_tone=task_spec.require_nonempty_themes_and_tone,
    )

    builders: Dict[str, Callable[[], Optional[CollateFn]]] = {
        "mlm": lambda: build_mlm_collator(tok, task_spec),
        "tone": lambda: RegressionCollator(output_size=1),
        "triplet": lambda: TripletDataCollator(
            triplet_downsample_size=task_spec.max_triplet_samples
        ),
        "themes": lambda: (
            None
            if task_spec.themes_path is None
            else MultiLabelCollator(top_themes_path=task_spec.themes_path)
        ),
    }

    sub_collators: Dict[str, CollateFn] = {}
    for task_name in TASK_ORDER:
        if task_name not in requested:
            continue
        collator = builders[task_name]()
        if collator is not None:
            sub_collators[task_name] = collator

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=MultiTaskCollator(sub_collators),
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )
    return dataloader, list(sub_collators)


def build_mlm_collator(tok: PreTrainedTokenizer, spec: TaskSpec) -> CollateFn:
    """Masked-LM batches; drops the label columns the MLM collator doesn't expect."""
    base_collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=True, mlm_probability=spec.mlm_probability
    )

    def filtered_collator(batch: List[Dict[str, Any]]):
        return base_collator(
            [
                {
                    "input_ids": item["input_ids"],
                    "attention_mask": item["attention_mask"],
                }
                for item in batch
            ]
        )

    return filtered_collator
