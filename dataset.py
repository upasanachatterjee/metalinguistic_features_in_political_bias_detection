import random
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset, Sampler
import os
from pretraining_utils import TaskSpec, TrainArgs, login_to_huggingface
from typing import Any, Dict, Iterable, List, Optional
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.tokenization_utils import PreTrainedTokenizer
from collators.triplet_collator import TripletDataCollator
from collators.story_collator import StoryTripletCollator
from collators.multi_label_collator import MultiLabelCollator
from collators.regression_collator import RegressionCollator

login_to_huggingface(os.getenv("hf_token"))


def _has_themes_and_tone(ex: Dict[str, Any]) -> bool:
    t = ex.get("V2Themes")
    n = ex.get("V2Tone")
    themes_ok = t is not None and str(t).strip() != ""
    tone_ok = n is not None and str(n).strip() != ""
    return themes_ok and tone_ok


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

        # Load dataset with memory mapping (doesn't load into RAM)
        print(f"🗂️  Loading {dataset_name} with memory mapping...")
        dataset_raw = load_dataset(
            dataset_name,
            split=split,
            revision="refs/convert/parquet",
            streaming=False,  # Use memory mapping instead of streaming
            cache_dir=cache_dir,
            keep_in_memory=False,  # Don't load into memory
        )

        # Ensure we have a Dataset object (not DatasetDict)
        if isinstance(dataset_raw, Dataset):
            self.dataset = dataset_raw
        else:
            raise ValueError(f"Expected Dataset, got {type(dataset_raw)}")

        if require_nonempty_themes_and_tone:
            before = len(self.dataset)
            self.dataset = self.dataset.filter(_has_themes_and_tone)
            print(
                f"🔎 Subsample: kept {len(self.dataset):,}/{before:,} rows with non-empty V2Themes and V2Tone"
            )

        # Remove columns we don't need to save memory (keeping group_uid for triplet formation)
        columns_to_remove = ["source", "title", "html", "url", "date"]
        existing_columns = [
            col for col in columns_to_remove if col in self.dataset.column_names
        ]
        if existing_columns:
            self.dataset = self.dataset.remove_columns(existing_columns)

        print(f"✅ Memory-mapped dataset ready: {len(self.dataset):,} samples")

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

    loader_params = {
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }

    print("   Creating lazy datasets...")
    mlm_dataset = MemoryEfficientDataset(
        dataset_name=task_spec.dataset_name,
        split=task_spec.split,
        text_col=task_spec.text_col,
        tokenizer=tok,
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


class GroupBatchSampler(Sampler[List[int]]):
    """Yields batches of indices that contain multiple articles sharing the same group_uid.

    Each batch is composed of `num_groups` randomly chosen multi-article groups, each
    contributing up to `per_group` indices, truncated to `batch_size`.
    """

    def __init__(
        self,
        dataset: "MemoryEfficientDataset",
        batch_size: int,
        num_groups: int,
        per_group: int,
        seed: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.num_groups = num_groups
        self.per_group = per_group

        print("   Building group_uid index for GroupBatchSampler...")
        group_uids = dataset.dataset["group_uid"]
        groups: Dict[Any, List[int]] = {}
        for i, g in enumerate(group_uids):
            groups.setdefault(g, []).append(i)
        self.groups = {g: idxs for g, idxs in groups.items() if len(idxs) >= 2}
        self.group_keys = list(self.groups.keys())
        self._multi_total = sum(len(v) for v in self.groups.values())
        self.rng = random.Random(seed)
        print(
            f"   GroupBatchSampler: {len(self.group_keys):,} multi-article groups, "
            f"{self._multi_total:,} indices, ~{len(self):,} batches/epoch"
        )

    def __iter__(self):
        for _ in range(len(self)):
            chosen = self.rng.sample(
                self.group_keys, k=min(self.num_groups, len(self.group_keys))
            )
            batch: List[int] = []
            for g in chosen:
                items = self.groups[g]
                if len(items) <= self.per_group:
                    batch.extend(items)
                else:
                    batch.extend(self.rng.sample(items, k=self.per_group))
                if len(batch) >= self.batch_size:
                    break
            yield batch[: self.batch_size]

    def __len__(self):
        return max(1, self._multi_total // self.batch_size)


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
    sampler = GroupBatchSampler(
        dataset,
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
