from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
import os
from pretraining_utils import TaskSpec, TrainArgs, login_to_huggingface
from typing import Any, Dict, Optional
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.tokenization_utils import PreTrainedTokenizer
from collators.triplet_collator import TripletDataCollator
from collators.multi_label_collator import MultiLabelCollator
from collators.regression_collator import RegressionCollator

login_to_huggingface(os.getenv("hf_token"))


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
    tok: PreTrainedTokenizer, task_spec: TaskSpec, args: TrainArgs
) -> Dict[str, Any]:
    """
    Build dataloaders with lazy loading - no pre-tokenization or RAM loading.
    """
    print("🚀 Building memory-efficient dataloaders with lazy loading...")

    dataloaders = {}

    # Optimized dataloader parameters for RTX 5090
    loader_params = {
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "tones_count": task_spec.tones_count,
    }

    # Build datasets for each task type
    print("   � Creating lazy datasets...")

    # MLM Dataset
    mlm_dataset = MemoryEfficientDataset(
        dataset_name=task_spec.dataset_name,
        split=task_spec.split,
        text_col=task_spec.text_col,
        tokenizer=tok,
    )

    # Create dataloaders
    print("   � Building dataloaders...")

    dataloaders["mlm"] = build_lazy_mlm_dataloader(
        tok, args, mlm_dataset, **loader_params
    )
    dataloaders["regression"] = build_lazy_regression_dataloader(
        tok, task_spec, args, mlm_dataset, **loader_params
    )
    dataloaders["triplet"] = build_lazy_triplet_dataloader(
        tok, args, mlm_dataset, **loader_params
    )
    dataloaders["multilabel"] = build_lazy_multilabel_dataloader(
        tok, task_spec, args, mlm_dataset, **loader_params
    )

    print("   ✅ All lazy dataloaders built")
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
