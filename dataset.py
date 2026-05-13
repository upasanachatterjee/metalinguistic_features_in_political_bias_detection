import hashlib
import os
import time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from accelerate.state import PartialState
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset, Sampler
from pretraining_utils import TaskSpec, TrainArgs, login_to_huggingface
from typing import Any, Dict, Iterable, List, Optional, Tuple
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.tokenization_utils import PreTrainedTokenizer
from collators.triplet_collator import TripletDataCollator
from collators.story_collator import StoryTripletCollator
from collators.multi_label_collator import MultiLabelCollator
from collators.regression_collator import RegressionCollator

login_to_huggingface(os.getenv("hf_token"))


def _filter_index_cache_path(
    cache_dir: str, dataset_name: str, split: str, n_rows: int
) -> str:
    key = f"filter|{dataset_name}|{split}|themes_and_tone|n={n_rows}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"filter_index_{h}.npy")


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

        def _nonempty_mask(col: pa.ChunkedArray) -> pa.ChunkedArray:
            # Works whether the column is string-typed (treat empty/whitespace as missing)
            # or numeric (just non-null). Operates on Arrow buffers — no Python list copy.
            if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
                # Skip whitespace-trim: it allocates a full string-column copy
                # (multi-GB for V2Themes at 5M rows). GDELT columns are either
                # populated or null in practice, so length-on-original is enough.
                len_ok = pc.greater(pc.utf8_length(col), 0)
                return pc.and_(pc.is_valid(col), len_ok)
            return pc.is_valid(col)

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
            f"🗂️  Loading {dataset_name} with memory mapping... (rank={state.process_index})",
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

        if require_nonempty_themes_and_tone:
            before = len(self.dataset)
            cache_path = _filter_index_cache_path(
                cache_dir=cache_dir or "./cache",
                dataset_name=dataset_name,
                split=split,
                n_rows=before,
            )
            keep_indices = _build_or_load_filter_index(self.dataset, cache_path, state)
            t = time.perf_counter()
            # `select` writes a per-rank indices-mapping arrow file into the HF
            # cache_dir; stagger it through main_process_first so rank 0 writes
            # first and the other ranks pick up the warm cache instead of all
            # 8 racing to write the same hashed path.
            with state.main_process_first():
                self.dataset = self.dataset.select(keep_indices)
            print(
                f"   [select] rank={state.process_index} applied filter indices in "
                f"{time.perf_counter() - t:.1f}s",
                flush=True,
            )
            if state.is_main_process:
                print(
                    f"🔎 Subsample: kept {len(self.dataset):,}/{before:,} rows "
                    f"with non-empty V2Themes and V2Tone",
                    flush=True,
                )

        # Remove columns we don't need to save memory (keeping group_uid for triplet formation)
        t = time.perf_counter()
        columns_to_remove = ["source", "title", "html", "url", "date"]
        existing_columns = [
            col for col in columns_to_remove if col in self.dataset.column_names
        ]
        if existing_columns:
            self.dataset = self.dataset.remove_columns(existing_columns)
        print(f"   [remove_columns] in {time.perf_counter() - t:.1f}s", flush=True)

        print(f"✅ Memory-mapped dataset ready: {len(self.dataset):,} samples", flush=True)

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
            "group_uid": sample["group_uid"],
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
      - "mlm", "regression", "multilabel"
      - "triplet_ideology" (random sampler, ideology mining)
      - "triplet_story" (group-aware sampler, story mining)
    Legacy "triplet" is treated as "triplet_ideology".
    """
    print("🚀 Building memory-efficient dataloaders with lazy loading...")

    tasks = set(tasks_to_build) if tasks_to_build is not None else {
        "mlm", "regression", "triplet_ideology", "multilabel",
    }
    if "triplet" in tasks:
        tasks.discard("triplet")
        tasks.add("triplet_ideology")

    dataloaders: Dict[str, Any] = {}

    # Keep per-rank worker fanout low. With 8 ranks, num_workers=8 across 4 loaders
    # plus story_triplet's own workers can fork hundreds of children whose COW heaps
    # plus pinned buffers overrun host RAM. Tokenization on roberta-base is cheap
    # relative to fwd/bwd, so 2 workers per loader is plenty.
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
        cache_dir=task_spec.group_index_cache_dir,
        require_nonempty_themes_and_tone=task_spec.require_nonempty_themes_and_tone,
    )

    print("   Building dataloaders...")

    if "mlm" in tasks:
        dataloaders["mlm"] = build_lazy_mlm_dataloader(
            tok, args, mlm_dataset, **loader_params
        )
    if "regression" in tasks:
        dataloaders["regression"] = build_lazy_regression_dataloader(
            tok, task_spec, args, mlm_dataset, **loader_params
        )
    if "triplet_ideology" in tasks:
        dataloaders["triplet_ideology"] = build_lazy_triplet_dataloader(
            tok, args, mlm_dataset, **loader_params
        )
    if "triplet_story" in tasks:
        dataloaders["triplet_story"] = build_lazy_story_triplet_dataloader(
            tok, args, task_spec, mlm_dataset, **loader_params
        )
    if "multilabel" in tasks:
        dataloaders["multilabel"] = build_lazy_multilabel_dataloader(
            tok, task_spec, args, mlm_dataset, **loader_params
        )

    print("   ✅ All lazy dataloaders built")
    return dataloaders


def _group_index_cache_path(
    cache_dir: str, dataset_name: str, split: str, filtered: bool, n_rows: int
) -> str:
    key = f"{dataset_name}|{split}|filtered={filtered}|n={n_rows}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"group_index_{h}.npz")


def build_or_load_group_index(
    dataset: "MemoryEfficientDataset",
    cache_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build or mmap-load a compact group index for `group_uid`.

    Returns:
      sorted_indices : int32[N]   row indices grouped contiguously by group_uid
      group_offsets  : int64[G+1] start positions per unique group
      multi_group_ids: int64[M]   indices into group_offsets for groups with count >= 2
    """
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    state = PartialState()

    if state.is_main_process and not os.path.exists(cache_path):
        print("   [group-index] Building (one-time)...", flush=True)
        t0 = time.perf_counter()

        t = time.perf_counter()
        arrow_col = dataset.dataset.data.column("group_uid")
        encoded = pc.dictionary_encode(arrow_col)
        codes = np.concatenate(
            [np.asarray(chunk.indices) for chunk in encoded.chunks]
        )
        print(
            f"   [group-index] dictionary_encode + concat in "
            f"{time.perf_counter() - t:.1f}s ({len(codes):,} rows)",
            flush=True,
        )

        t = time.perf_counter()
        order = np.argsort(codes, kind="stable")
        sorted_uids = codes[order]
        sorted_indices = order.astype(np.int32, copy=False)
        print(f"   [group-index] argsort in {time.perf_counter() - t:.1f}s", flush=True)

        t = time.perf_counter()
        # Group boundaries: positions where sorted_uids changes.
        if len(sorted_uids) == 0:
            group_offsets = np.zeros(1, dtype=np.int64)
            multi_group_ids = np.zeros(0, dtype=np.int64)
        else:
            change = np.empty(len(sorted_uids), dtype=bool)
            change[0] = True
            change[1:] = sorted_uids[1:] != sorted_uids[:-1]
            starts = np.flatnonzero(change).astype(np.int64)
            group_offsets = np.concatenate([starts, np.array([len(sorted_uids)], dtype=np.int64)])
            counts = np.diff(group_offsets)
            multi_group_ids = np.flatnonzero(counts >= 2).astype(np.int64)
        print(f"   [group-index] boundaries in {time.perf_counter() - t:.1f}s", flush=True)

        t = time.perf_counter()
        np.savez(cache_path, sorted_indices=sorted_indices, group_offsets=group_offsets,
                 multi_group_ids=multi_group_ids)
        print(f"   [group-index] np.savez in {time.perf_counter() - t:.1f}s", flush=True)

        print(
            f"   [group-index] DONE in {time.perf_counter() - t0:.1f}s — "
            f"{len(group_offsets) - 1:,} groups total, "
            f"{len(multi_group_ids):,} multi-article groups, "
            f"{int(np.diff(group_offsets)[multi_group_ids].sum()) if len(multi_group_ids) else 0:,} indices",
            flush=True,
        )

    t_bar = time.perf_counter()
    state.wait_for_everyone()
    print(
        f"   [group-index] rank={state.process_index} barrier waited "
        f"{time.perf_counter() - t_bar:.1f}s",
        flush=True,
    )
    print(f"   Loading cached group index from {cache_path} (mmap)", flush=True)
    data = np.load(cache_path, mmap_mode="r")
    return data["sorted_indices"], data["group_offsets"], data["multi_group_ids"]


class GroupBatchSampler(Sampler[List[int]]):
    """Yields batches of indices that contain multiple articles sharing the same group_uid.

    Backed by a compact NumPy index (mmap'd from disk) rather than a Python dict, so the
    sampler stays small in RAM, cheap to deepcopy (Accelerate's prepare path), and shares
    pages across forked DataLoader workers.

    Each batch is composed of `num_groups` randomly chosen multi-article groups, each
    contributing up to `per_group` indices, truncated to `batch_size`.
    """

    def __init__(
        self,
        sorted_indices: np.ndarray,
        group_offsets: np.ndarray,
        multi_group_ids: np.ndarray,
        batch_size: int,
        num_groups: int,
        per_group: int,
        seed: Optional[int] = None,
    ):
        self.sorted_indices = sorted_indices
        self.group_offsets = group_offsets
        self.multi_group_ids = multi_group_ids
        self.batch_size = batch_size
        self.num_groups = num_groups
        self.per_group = per_group
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if len(multi_group_ids):
            sizes = group_offsets[multi_group_ids + 1] - group_offsets[multi_group_ids]
            self._multi_total = int(sizes.sum())
        else:
            self._multi_total = 0
        print(
            f"   GroupBatchSampler: {len(multi_group_ids):,} multi-article groups, "
            f"{self._multi_total:,} indices, ~{len(self):,} batches/epoch"
        )

    def __iter__(self):
        n_multi = len(self.multi_group_ids)
        if n_multi == 0:
            return
        k = min(self.num_groups, n_multi)
        for _ in range(len(self)):
            chosen = self.rng.choice(self.multi_group_ids, size=k, replace=False)
            batch: List[int] = []
            for g in chosen:
                start = int(self.group_offsets[g])
                end = int(self.group_offsets[g + 1])
                size = end - start
                if size <= self.per_group:
                    batch.extend(int(x) for x in self.sorted_indices[start:end])
                else:
                    picks = self.rng.choice(size, size=self.per_group, replace=False)
                    batch.extend(int(self.sorted_indices[start + p]) for p in picks)
                if len(batch) >= self.batch_size:
                    break
            yield batch[: self.batch_size]

    def __len__(self):
        return max(1, self._multi_total // self.batch_size)

    def __deepcopy__(self, memo):
        # Index arrays are read-only (mmap'd); share by reference so accelerator.prepare's
        # deepcopy doesn't traverse millions of entries or duplicate memory.
        new = GroupBatchSampler.__new__(GroupBatchSampler)
        new.sorted_indices = self.sorted_indices
        new.group_offsets = self.group_offsets
        new.multi_group_ids = self.multi_group_ids
        new.batch_size = self.batch_size
        new.num_groups = self.num_groups
        new.per_group = self.per_group
        new.seed = self.seed
        new._multi_total = self._multi_total
        new.rng = np.random.default_rng(self.seed)
        return new


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
    tok: PreTrainedTokenizer, args: TrainArgs, dataset: TorchDataset, **loader_kwargs
) -> DataLoader:
    """Build triplet dataloader with lazy loading."""
    base_collator = TripletDataCollator()

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=base_collator,
        **loader_kwargs,
    )


def build_lazy_story_triplet_dataloader(
    tok: PreTrainedTokenizer,
    args: TrainArgs,
    spec: TaskSpec,
    dataset: "MemoryEfficientDataset",
    **loader_kwargs,
) -> DataLoader:
    """Build story-triplet dataloader using a group-aware batch sampler."""
    cache_path = _group_index_cache_path(
        cache_dir=spec.group_index_cache_dir,
        dataset_name=spec.dataset_name,
        split=spec.split,
        filtered=spec.require_nonempty_themes_and_tone,
        n_rows=len(dataset),
    )
    sorted_indices, group_offsets, multi_group_ids = build_or_load_group_index(
        dataset, cache_path
    )
    sampler = GroupBatchSampler(
        sorted_indices=sorted_indices,
        group_offsets=group_offsets,
        multi_group_ids=multi_group_ids,
        batch_size=args.batch_size,
        num_groups=spec.group_batch_num_groups,
        per_group=spec.group_batch_per_group,
    )
    collator = StoryTripletCollator(triplet_downsample_size=spec.max_triplet_samples)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
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
    collator = RegressionCollator(num_tones=spec.tones_count)

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
    """Build MLM dataloader with RTX 5090 optimizations."""
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
    """Build triplet dataloader with RTX 5090 optimizations."""
    collator = TripletDataCollator()

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
    """Build multilabel dataloader with RTX 5090 optimizations."""
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
