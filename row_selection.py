"""Which rows of the corpus a run trains on: an optional non-empty themes/tone
filter, then title dedup, both computed at the Arrow level rather than through
`Dataset.filter`.
"""

import hashlib
import os
import time
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from accelerate.state import PartialState
from datasets import Dataset


def _filter_index_cache_path(
    cache_dir: str, dataset_name: str, split: str, n_rows: int
) -> str:
    key = f"filter|{dataset_name}|{split}|themes_and_tone|n={n_rows}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"filter_index_{h}.npy")


def _nonempty_mask(col: pa.ChunkedArray) -> pa.ChunkedArray:
    """Null-free boolean mask of rows where `col` is populated.
    """
    if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
        len_ok = pc.greater(pc.utf8_length(col), 0)
        return pc.fill_null(pc.and_(pc.is_valid(col), len_ok), False)
    return pc.is_valid(col)


def _build_or_load_filter_index(
    dataset_raw: Dataset, cache_path: str, state: PartialState
) -> np.ndarray:
    """Compute row indices where V2Themes and V2Tone are both non-empty.
    """
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    if state.is_main_process and not os.path.exists(cache_path):
        print(f"   [filter-index] building (one-time) at {cache_path}", flush=True)
        t0 = time.perf_counter()

        themes_ok = _nonempty_mask(dataset_raw.data.column("V2Themes"))
        tone_ok = _nonempty_mask(dataset_raw.data.column("V2Tone"))
        keep_mask = pc.and_(themes_ok, tone_ok)
        keep_np = np.concatenate([np.asarray(chunk) for chunk in keep_mask.chunks])
        indices = np.flatnonzero(keep_np).astype(np.int64)
        np.save(cache_path, indices)

        print(
            f"   [filter-index] kept {len(indices):,}/{len(dataset_raw):,} rows "
            f"in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    state.wait_for_everyone()
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
    """
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    if state.is_main_process and not os.path.exists(cache_path):
        print(f"   [title-index] building (one-time) at {cache_path}", flush=True)
        t0 = time.perf_counter()

        col = dataset_raw.data.column("title")
        # Dictionary-encoded so dedup runs over int32 codes; nulls fill to -1.
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

        if candidate_indices is not None:
            candidate_indices = np.asarray(candidate_indices)
            codes = codes[candidate_indices]
            has_title = has_title[candidate_indices]

        n_candidates = len(codes)
        titled = np.flatnonzero(has_title)
        codes = codes[titled]
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        if len(sorted_codes) == 0:
            keep = np.zeros(0, dtype=np.int64)
        else:
            change = np.empty(len(sorted_codes), dtype=bool)
            change[0] = True
            change[1:] = sorted_codes[1:] != sorted_codes[:-1]
            keep = titled[order[np.flatnonzero(change)]].astype(np.int64)

        indices = candidate_indices[keep] if candidate_indices is not None else keep
        indices = np.sort(indices).astype(np.int64)
        np.save(cache_path, indices)

        print(
            f"   [title-index] kept {len(indices):,}/{n_candidates:,} rows "
            f"({n_candidates - len(titled):,} untitled dropped) "
            f"in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    state.wait_for_everyone()
    return np.load(cache_path, mmap_mode="r")


def select_rows(
    dataset: Dataset,
    dataset_name: str,
    split: str,
    cache_dir: str,
    require_nonempty_themes_and_tone: bool,
    state: PartialState,
) -> Optional[np.ndarray]:
    """Row indices this run should train on, or None to keep every row.
    """
    n_rows = len(dataset)
    keep_indices: Optional[np.ndarray] = None

    if require_nonempty_themes_and_tone:
        keep_indices = _build_or_load_filter_index(
            dataset,
            _filter_index_cache_path(cache_dir, dataset_name, split, n_rows),
            state,
        )

    if "title" in dataset.column_names:
        keep_indices = _build_or_load_first_title_index(
            dataset,
            _title_index_cache_path(
                cache_dir,
                dataset_name,
                split,
                require_nonempty_themes_and_tone,
                n_rows,
            ),
            state,
            candidate_indices=keep_indices,
        )
    elif state.is_main_process:
        print("   no 'title' column found; skipping title dedup", flush=True)

    return keep_indices
