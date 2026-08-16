import os
import time
import numpy as np
from accelerate.state import PartialState
from huggingface_hub import login
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from config import TaskSpec, TrainArgs
from row_selection import select_rows
from typing import Any, Dict, Iterable, Optional
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.tokenization_utils import PreTrainedTokenizer
from collators.triplet_collator import TripletDataCollator
from collators.multi_label_collator import MultiLabelCollator
from collators.regression_collator import RegressionCollator

# Metadata columns no task reads; dropped to keep the mapped table small.
UNUSED_COLUMNS = ["source", "title", "html", "url", "date", "group_uid"]


def _login_once(state: PartialState) -> None:
    """Authenticate rank 0 before any rank touches the Hub.

    The corpus is a private dataset, so this has to happen before
    `load_dataset`. Logging in from every rank at once races on the same token
    file, hence the rank-0 gate and the barrier.
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
    """Memory-mapped view of the corpus, tokenized lazily in __getitem__.

    Rows are deduplicated by title (first occurrence wins; untitled rows are
    dropped) and, when `require_nonempty_themes_and_tone`, restricted to rows
    that have both GDELT fields populated. Row selection is computed once as
    index arrays cached under `cache_dir` and mmap'd by every rank.

    Each item carries the tokenized text plus the raw `political_bias`,
    `V2Themes` and `V2Tone` values; turning those into task labels is the
    collators' job, so all four tasks share one instance of this dataset.
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
        """Memory-map the corpus, letting rank 0 warm the HF cache first.

        All ranks racing into load_dataset hammers the same cache_dir and its
        filelock; if any rank crashes mid-load it leaves an orphan .lock that
        deadlocks the next launch. main_process_first() lets rank 0 warm (or
        repair) the cache alone, then the other ranks read it warm.
        """
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

        if not isinstance(dataset, Dataset):
            raise ValueError(f"Expected Dataset, got {type(dataset)}")
        return dataset

    def _apply_selection(
        self, keep_indices: np.ndarray, before: int, state: PartialState
    ) -> None:
        """Narrow the table to `keep_indices` and materialize the result.

        Both selection rules are resolved into one index array first, so this
        Arrow rewrite -- the expensive part -- is paid exactly once.
        """
        t = time.perf_counter()
        # `select` writes a per-rank indices-mapping arrow file into the HF
        # cache_dir; stagger it through main_process_first so rank 0 writes
        # first and the other ranks pick up the warm cache instead of all
        # 8 racing to write the same hashed path.
        with state.main_process_first():
            self.dataset = self.dataset.select(keep_indices)
        # Materialize the selection into the underlying arrow table so that
        # `self.dataset.data.column(...)` reflects the selected view, and so
        # __getitem__ doesn't pay the indices-mapping indirection per row.
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
        """Tokenize one row on demand.

        Deliberately padded to the full 512 rather than dynamically per batch:
        every step then costs the same regardless of article length, which keeps
        the 8 DDP ranks in lockstep and makes the step-time ETA meaningful. It
        also means the collators' own padding is a no-op. Switching to
        `padding=False` here (and dropping the fixed `max_length` in
        `_triplet_utils.pack_triplets`) would buy throughput on short articles,
        at the cost of variable step times -- worth doing as its own change,
        not silently.
        """
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


def build_dataloaders(
    tok: PreTrainedTokenizer,
    task_spec: TaskSpec,
    args: TrainArgs,
    tasks_to_build: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """One dataloader per task, all wrapping a single lazily-tokenized dataset.

    Every loader shuffles independently, so each training step shows the four
    objectives four different random subsets of the same corpus. Pass
    `tasks_to_build` ("mlm", "tone", "triplet", "themes") to skip loaders the
    current run doesn't need.
    """
    tasks = set(tasks_to_build) if tasks_to_build is not None else {
        "mlm", "tone", "triplet", "themes",
    }

    _login_once(PartialState())

    dataloaders: Dict[str, Any] = {}

    # Keep per-rank worker fanout low. With 8 ranks, num_workers=8 across 4 loaders
    # can fork hundreds of children whose COW heaps plus pinned buffers overrun
    # host RAM. Tokenization on roberta-base is cheap relative to fwd/bwd, so
    # 2 workers per loader is plenty.
    loader_params = {
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_factor": 2,
    }

    dataset = MemoryEfficientDataset(
        dataset_name=task_spec.dataset_name,
        split=task_spec.split,
        text_col=task_spec.text_col,
        tokenizer=tok,
        cache_dir=task_spec.index_cache_dir,
        require_nonempty_themes_and_tone=task_spec.require_nonempty_themes_and_tone,
    )

    if "mlm" in tasks:
        dataloaders["mlm"] = build_lazy_mlm_dataloader(
            tok, task_spec, args, dataset, **loader_params
        )
    if "tone" in tasks:
        dataloaders["tone"] = build_lazy_regression_dataloader(
            args, dataset, **loader_params
        )
    if "triplet" in tasks:
        dataloaders["triplet"] = build_lazy_triplet_dataloader(
            task_spec, args, dataset, **loader_params
        )
    if "themes" in tasks:
        dataloaders["themes"] = build_lazy_multilabel_dataloader(
            task_spec, args, dataset, **loader_params
        )

    return dataloaders


# ------------------------------
# One dataloader per task, all over the same MemoryEfficientDataset
# ------------------------------


def build_lazy_mlm_dataloader(
    tok: PreTrainedTokenizer,
    spec: TaskSpec,
    args: TrainArgs,
    dataset: TorchDataset,
    **loader_kwargs,
) -> DataLoader:
    """Masked-LM batches; drops the label columns the MLM collator doesn't expect."""
    base_collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=True, mlm_probability=spec.mlm_probability
    )

    def filtered_collator(batch):
        return base_collator(
            [
                {
                    "input_ids": item["input_ids"],
                    "attention_mask": item["attention_mask"],
                }
                for item in batch
            ]
        )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=filtered_collator,
        **loader_kwargs,
    )


def build_lazy_triplet_dataloader(
    spec: TaskSpec, args: TrainArgs, dataset: TorchDataset, **loader_kwargs
) -> DataLoader:
    """Triplets mined within each batch from the political_bias column."""
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=TripletDataCollator(
            triplet_downsample_size=spec.max_triplet_samples
        ),
        **loader_kwargs,
    )


def build_lazy_multilabel_dataloader(
    spec: TaskSpec, args: TrainArgs, dataset: TorchDataset, **loader_kwargs
) -> Optional[DataLoader]:
    """Multi-hot theme labels over the top_themes.txt label space.

    Returns None when the run has no theme label space configured.
    """
    if spec.themes_path is None:
        return None

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=MultiLabelCollator(top_themes_path=spec.themes_path),
        **loader_kwargs,
    )


def build_lazy_regression_dataloader(
    args: TrainArgs, dataset: TorchDataset, **loader_kwargs
) -> DataLoader:
    """Tone regression targets: the first field of GDELT V2Tone (average tone)."""
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=RegressionCollator(output_size=1),
        **loader_kwargs,
    )
