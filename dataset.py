import hashlib
import os
import time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from accelerate.state import PartialState
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from pretraining_utils import TaskSpec, TrainArgs, login_to_huggingface
from typing import Any, Dict, Iterable, Optional
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.tokenization_utils import PreTrainedTokenizer
from collators.triplet_collator import TripletDataCollator
from collators.multi_label_collator import MultiLabelCollator
from collators.regression_collator import RegressionCollator

_hf_login_state = PartialState()
if _hf_login_state.is_main_process:
    login_to_huggingface(os.getenv("hf_token"))
_hf_login_state.wait_for_everyone()


def _filter_index_cache_path(
    cache_dir: str, dataset_name: str, split: str, n_rows: int
) -> str:
    key = f"filter|{dataset_name}|{split}|themes_and_tone|n={n_rows}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"filter_index_{h}.npy")


def _nonempty_mask(col: pa.ChunkedArray) -> pa.ChunkedArray:
    """Null-free boolean mask of rows where `col` is populated.

    Works whether the column is string-typed (treat empty as missing) or numeric
    (just non-null). Operates on Arrow buffers — no Python list copy.

    The result is explicitly null-filled: `utf8_length` propagates nulls, and a
    mask carrying nulls decays to an object-dtype ndarray on the numpy side.
    """
    if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
        # Skip whitespace-trim: it allocates a full string-column copy
        # (multi-GB for V2Themes at 5M rows). GDELT columns are either
        # populated or null in practice, so length-on-original is enough.
        len_ok = pc.greater(pc.utf8_length(col), 0)
        return pc.fill_null(pc.and_(pc.is_valid(col), len_ok), False)
    return pc.is_valid(col)


def _build_or_load_filter_index(
    dataset_raw: Dataset, cache_path: str, state: PartialState
) -> np.ndarray:
    """Compute row indices where V2Themes and V2Tone are both non-empty.

    Built once on rank 0 (scanning the two Arrow columns directly, avoiding the
    per-rank `dataset.filter(...)` that materializes a fresh Arrow table on every
    process), then mmap'd by all ranks.
    """
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    if state.is_main_process and not os.path.exists(cache_path):
        print(f"   [filter-index] Building (one-time) at {cache_path}", flush=True)
        t0 = time.perf_counter()

        t = time.perf_counter()
        themes_ok = _nonempty_mask(dataset_raw.data.column("V2Themes"))
        tone_ok = _nonempty_mask(dataset_raw.data.column("V2Tone"))
        keep_mask = pc.and_(themes_ok, tone_ok)
        print(f"   [filter-index] arrow masks computed in {time.perf_counter() - t:.1f}s", flush=True)

        t = time.perf_counter()
        keep_np = np.concatenate([np.asarray(chunk) for chunk in keep_mask.chunks])
        indices = np.flatnonzero(keep_np).astype(np.int64)
        print(f"   [filter-index] np concat+flatnonzero in {time.perf_counter() - t:.1f}s", flush=True)

        t = time.perf_counter()
        np.save(cache_path, indices)
        print(f"   [filter-index] np.save in {time.perf_counter() - t:.1f}s", flush=True)

        print(
            f"   [filter-index] DONE in {time.perf_counter() - t0:.1f}s — "
            f"kept {len(indices):,}/{len(dataset_raw):,} rows",
            flush=True,
        )

    t_bar = time.perf_counter()
    state.wait_for_everyone()
    print(
        f"   [filter-index] rank={state.process_index} barrier waited "
        f"{time.perf_counter() - t_bar:.1f}s",
        flush=True,
    )
    return np.load(cache_path, mmap_mode="r")


def _title_index_cache_path(
    cache_dir: str, dataset_name: str, split: str, filtered: bool, n_rows: int
) -> str:
    key = f"title|{dataset_name}|{split}|filtered={filtered}|n={n_rows}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"title_index_{h}.npy")


def _build_or_load_first_title_index(
    dataset_raw: Dataset,
    cache_path: str,
    state: PartialState,
    candidate_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Row indices of the first occurrence of each distinct title.

    Rows whose title is null/empty are dropped entirely.

    `candidate_indices` scopes the dedup to rows that survived earlier filtering,
    so "first" means first *among kept rows*; the result is then a subset of it.

    Built once on rank 0 (scanning the Arrow column directly) then mmap'd by all
    ranks, mirroring `_build_or_load_filter_index`.
    """
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    if state.is_main_process and not os.path.exists(cache_path):
        print(f"   [title-index] Building (one-time) at {cache_path}", flush=True)
        t0 = time.perf_counter()

        t = time.perf_counter()
        col = dataset_raw.data.column("title")
        # Dictionary-encode so dedup happens over int32 codes instead of strings.
        # Null titles encode to null indices; fill them with -1 so the array stays
        # integer-typed, and track them in `has_title` so they can be dropped.
        if len(col) == 0:
            # A zero-row column has zero chunks, which np.concatenate rejects.
            codes = np.zeros(0, dtype=np.int32)
            has_title = np.zeros(0, dtype=bool)
        else:
            encoded = pc.dictionary_encode(col)
            codes = np.concatenate(
                [np.asarray(chunk.indices.fill_null(-1)) for chunk in encoded.chunks]
            )
            has_title = np.concatenate(
                [np.asarray(chunk) for chunk in _nonempty_mask(col).chunks]
            )
        print(
            f"   [title-index] dictionary_encode + concat in "
            f"{time.perf_counter() - t:.1f}s ({len(codes):,} rows)",
            flush=True,
        )

        if candidate_indices is not None:
            candidate_indices = np.asarray(candidate_indices)
            codes = codes[candidate_indices]
            has_title = has_title[candidate_indices]

        n_candidates = len(codes)

        t = time.perf_counter()
        # Untitled rows are dropped outright, so they never reach the dedup.
        titled = np.flatnonzero(has_title)
        codes = codes[titled]
        # Stable argsort groups equal codes together while preserving original row
        # order within a group, so the first position of each group is the first
        # occurrence of that title.
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        if len(sorted_codes) == 0:
            keep = np.zeros(0, dtype=np.int64)
        else:
            change = np.empty(len(sorted_codes), dtype=bool)
            change[0] = True
            change[1:] = sorted_codes[1:] != sorted_codes[:-1]
            # `titled` is ascending, so mapping back through it recovers positions
            # in the candidate space.
            keep = titled[order[np.flatnonzero(change)]].astype(np.int64)
        print(f"   [title-index] dedup in {time.perf_counter() - t:.1f}s", flush=True)

        indices = candidate_indices[keep] if candidate_indices is not None else keep
        indices = np.sort(indices).astype(np.int64)

        t = time.perf_counter()
        np.save(cache_path, indices)
        print(f"   [title-index] np.save in {time.perf_counter() - t:.1f}s", flush=True)

        print(
            f"   [title-index] DONE in {time.perf_counter() - t0:.1f}s — "
            f"kept {len(indices):,}/{n_candidates:,} rows "
            f"({n_candidates - len(titled):,} untitled dropped)",
            flush=True,
        )

    t_bar = time.perf_counter()
    state.wait_for_everyone()
    print(
        f"   [title-index] rank={state.process_index} barrier waited "
        f"{time.perf_counter() - t_bar:.1f}s",
        flush=True,
    )
    return np.load(cache_path, mmap_mode="r")


class MemoryEfficientDataset(TorchDataset):
    """
    Alternative approach using HuggingFace datasets with memory mapping.
    Uses datasets' built-in lazy loading with memory mapping.
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

        # All ranks racing into load_dataset hammers the same HF cache_dir and
        # its filelock; if any rank crashes mid-load it leaves an orphan .lock
        # that deadlocks the next launch. main_process_first() lets rank 0 warm
        # (or repair) the cache alone, then the other ranks read it warm.
        print(
            f"Loading {dataset_name} with memory mapping (rank={state.process_index})",
            flush=True,
        )
        t = time.perf_counter()
        with state.main_process_first():
            dataset_raw = load_dataset(
                dataset_name,
                split=split,
                revision="refs/convert/parquet",
                streaming=False,
                cache_dir=cache_dir,
                keep_in_memory=False,
            )
        print(
            f"   [load_dataset] rank={state.process_index} returned in "
            f"{time.perf_counter() - t:.1f}s",
            flush=True,
        )

        if isinstance(dataset_raw, Dataset):
            self.dataset = dataset_raw
        else:
            raise ValueError(f"Expected Dataset, got {type(dataset_raw)}")

        # Row selection is computed as index arrays first and applied as a single
        # select + flatten_indices at the end, so the expensive Arrow rewrite is
        # paid once no matter how many filters are active.
        before = len(self.dataset)
        keep_indices: Optional[np.ndarray] = None

        if require_nonempty_themes_and_tone:
            filter_cache_path = _filter_index_cache_path(
                cache_dir=cache_dir or "./cache",
                dataset_name=dataset_name,
                split=split,
                n_rows=before,
            )
            keep_indices = _build_or_load_filter_index(
                self.dataset, filter_cache_path, state
            )
            if state.is_main_process:
                print(
                    f"🔎 Subsample: {len(keep_indices):,}/{before:,} rows "
                    f"have non-empty V2Themes and V2Tone",
                    flush=True,
                )

        # Deduplicate by title, keeping the first row of each distinct title and
        # dropping rows with no title at all. Scoped to `keep_indices` so a title
        # whose first row was dropped by the subsample falls back to its next
        # surviving row instead of disappearing.
        if "title" in self.dataset.column_names:
            title_cache_path = _title_index_cache_path(
                cache_dir=cache_dir or "./cache",
                dataset_name=dataset_name,
                split=split,
                filtered=require_nonempty_themes_and_tone,
                n_rows=before,
            )
            n_candidates = len(keep_indices) if keep_indices is not None else before
            keep_indices = _build_or_load_first_title_index(
                self.dataset, title_cache_path, state, candidate_indices=keep_indices
            )
            if state.is_main_process:
                print(
                    f"🔎 Dedup: kept {len(keep_indices):,}/{n_candidates:,} rows "
                    f"with distinct, non-empty titles",
                    flush=True,
                )
        else:
            print(
                "   [title-index] No 'title' column found; skipping title dedup",
                flush=True,
            )

        if keep_indices is not None:
            t = time.perf_counter()
            # `select` writes a per-rank indices-mapping arrow file into the HF
            # cache_dir; stagger it through main_process_first so rank 0 writes
            # first and the other ranks pick up the warm cache instead of all
            # 8 racing to write the same hashed path.
            with state.main_process_first():
                self.dataset = self.dataset.select(keep_indices)
            print(
                f"   [select] rank={state.process_index} applied row indices in "
                f"{time.perf_counter() - t:.1f}s",
                flush=True,
            )
            # Materialize the selection into the underlying arrow table so that
            # `self.dataset.data.column(...)` reflects the selected view, and so
            # __getitem__ doesn't pay the indices-mapping indirection per row.
            t = time.perf_counter()
            with state.main_process_first():
                self.dataset = self.dataset.flatten_indices()
            print(
                f"   [flatten_indices] rank={state.process_index} in "
                f"{time.perf_counter() - t:.1f}s",
                flush=True,
            )
            if state.is_main_process:
                print(
                    f"🔎 Kept {len(self.dataset):,}/{before:,} rows after "
                    f"filtering and title dedup",
                    flush=True,
                )

        # Remove columns we don't need to save memory
        t = time.perf_counter()
        columns_to_remove = ["source", "title", "html", "url", "date", "group_uid"]
        existing_columns = [
            col for col in columns_to_remove if col in self.dataset.column_names
        ]
        if existing_columns:
            self.dataset = self.dataset.remove_columns(existing_columns)
        print(f"   [remove_columns] in {time.perf_counter() - t:.1f}s", flush=True)

        print(f" Memory-mapped dataset ready: {len(self.dataset):,} samples", flush=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Load and tokenize a single sample from memory-mapped storage."""
        sample = self.dataset[idx]

        # Extract text - sample is a dict-like object
        text = sample.get(self.text_col, "")

        # Tokenize on-demand
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
    """
    Build dataloaders with lazy loading - no pre-tokenization or RAM loading.

    Pass `tasks_to_build` to skip loaders not needed for the current run. Task names:
      - "mlm", "tone", "themes"
      - "triplet" (random sampler, ideology mining)
    """
    print(" Building memory-efficient dataloaders with lazy loading...")

    tasks = set(tasks_to_build) if tasks_to_build is not None else {
        "mlm", "tone", "triplet", "themes",
    }

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

    print("   Creating lazy datasets...")
    mlm_dataset = MemoryEfficientDataset(
        dataset_name=task_spec.dataset_name,
        split=task_spec.split,
        text_col=task_spec.text_col,
        tokenizer=tok,
        cache_dir=task_spec.index_cache_dir,
        require_nonempty_themes_and_tone=task_spec.require_nonempty_themes_and_tone,
    )

    print("   Building dataloaders...")

    if "mlm" in tasks:
        dataloaders["mlm"] = build_lazy_mlm_dataloader(
            tok, args, mlm_dataset, **loader_params
        )
    if "tone" in tasks:
        dataloaders["tone"] = build_lazy_regression_dataloader(
            tok, task_spec, args, mlm_dataset, **loader_params
        )
    if "triplet" in tasks:
        dataloaders["triplet"] = build_lazy_triplet_dataloader(
            tok, task_spec, args, mlm_dataset, **loader_params
        )
    if "themes" in tasks:
        dataloaders["themes"] = build_lazy_multilabel_dataloader(
            tok, task_spec, args, mlm_dataset, **loader_params
        )

    print("    All lazy dataloaders built")
    return dataloaders



# ------------------------------
# Lazy Dataloaders for each task
# ------------------------------


def build_lazy_mlm_dataloader(
    tok: PreTrainedTokenizer, args: TrainArgs, dataset: TorchDataset, **loader_kwargs
) -> DataLoader:
    """Build MLM dataloader with lazy loading."""
    base_collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=True, mlm_probability=0.15
    )

    # Wrapper to filter batch before passing to MLM collator
    def filtered_collator(batch):
        # Filter to only include MLM-relevant fields
        filtered_batch = []
        for item in batch:
            filtered_item = {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
            }
            filtered_batch.append(filtered_item)
        return base_collator(filtered_batch)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=filtered_collator,
        **loader_kwargs,
    )


def build_lazy_triplet_dataloader(
    tok: PreTrainedTokenizer,
    spec: TaskSpec,
    args: TrainArgs,
    dataset: TorchDataset,
    **loader_kwargs,
) -> DataLoader:
    """Build triplet dataloader with lazy loading."""
    base_collator = TripletDataCollator(
        triplet_downsample_size=spec.max_triplet_samples
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=base_collator,
        **loader_kwargs,
    )


def build_lazy_multilabel_dataloader(
    tok: PreTrainedTokenizer,
    spec: TaskSpec,
    args: TrainArgs,
    dataset: TorchDataset,
    **loader_kwargs,
) -> Optional[DataLoader]:
    """Build multilabel dataloader with lazy loading."""
    if spec.multi_label_col is None or spec.themes_path is None:
        return None

    collator = MultiLabelCollator(top_themes_path=spec.themes_path)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        **loader_kwargs,
    )


def build_lazy_regression_dataloader(
    tok: PreTrainedTokenizer,
    spec: TaskSpec,
    args: TrainArgs,
    dataset: TorchDataset,
    **loader_kwargs,
) -> Optional[DataLoader]:
    """Build regression dataloader with lazy loading."""
    collator = RegressionCollator(output_size=1)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        **loader_kwargs,
    )


# ------------------------------
# Original Dataloaders (for compatibility)
# ------------------------------


def build_mlm_dataloader(
    tok: PreTrainedTokenizer,
    spec: TaskSpec,
    args: TrainArgs,
    ds: Dataset,
    **loader_kwargs,
) -> DataLoader:
    """Build MLM dataloader."""
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=True, mlm_probability=spec.mlm_probability
    )

    # Use optimized parameters or fallback to args
    pin_memory = loader_kwargs.get("pin_memory", args.pin_memory)

    dl = DataLoader(
        ds,  # type: ignore
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=pin_memory,
        persistent_workers=loader_kwargs.get("persistent_workers", False),
        prefetch_factor=loader_kwargs.get("prefetch_factor", 2),
    )
    return dl


def build_triplet_dataloader(
    tok: PreTrainedTokenizer,
    spec: TaskSpec,
    args: TrainArgs,
    ds: Dataset,
    **loader_kwargs,
) -> DataLoader:
    """Build triplet dataloader."""
    collator = TripletDataCollator(triplet_downsample_size=spec.max_triplet_samples)

    # Use optimized parameters or fallback to args
    num_workers = loader_kwargs.get("num_workers", args.dataloader_num_workers)
    pin_memory = loader_kwargs.get("pin_memory", args.pin_memory)

    dl = DataLoader(
        ds,  # type: ignore
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=loader_kwargs.get("persistent_workers", False),
        prefetch_factor=loader_kwargs.get("prefetch_factor", 2),
    )
    return dl


def build_multilabel_dataloader(
    tok: PreTrainedTokenizer,
    spec: TaskSpec,
    args: TrainArgs,
    ds: Dataset,
    **loader_kwargs,
) -> Optional[DataLoader]:
    """Build multilabel dataloader."""
    if spec.multi_label_col is None or spec.themes_path is None:
        print(
            "No multilabel column or themes path specified; skipping multilabel dataloader."
        )
        return None

    collator = MultiLabelCollator(top_themes_path=spec.themes_path)

    # Use optimized parameters or fallback to args
    num_workers = loader_kwargs.get("num_workers", args.dataloader_num_workers)
    pin_memory = loader_kwargs.get("pin_memory", args.pin_memory)

    dl = DataLoader(
        ds,  # type: ignore
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=loader_kwargs.get("persistent_workers", False),
        prefetch_factor=loader_kwargs.get("prefetch_factor", 2),
    )
    return dl


def build_regression_dataloader(
    tok, spec: TaskSpec, args: TrainArgs, ds: Dataset, **loader_kwargs
) -> Optional[DataLoader]:
    """Build regression dataloader with RTX 5090 optimizations."""
    if spec.regression_col is None:
        print("No regression column specified; skipping regression dataloader.")
        return None

    collator = RegressionCollator()

    # Use optimized parameters or fallback to args
    num_workers = loader_kwargs.get("num_workers", args.dataloader_num_workers)
    pin_memory = loader_kwargs.get("pin_memory", args.pin_memory)

    dl = DataLoader(
        ds,  # type: ignore
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=loader_kwargs.get("persistent_workers", False),
        prefetch_factor=loader_kwargs.get("prefetch_factor", 2),
    )
    return dl
