#!/usr/bin/env python3
"""Shared split helpers for standard FMT outer 5-fold experiments."""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.data.FMTDataset import grouped_kfold_indices, grouped_train_val_indices


def nested_grouped_cv_indices(
    metadata: pd.DataFrame,
    fold_index: int,
    num_folds: int = 5,
    val_ratio: float = 0.1,
    seed: int = 42,
    group_column: str = "fmt_id",
    label_column: str = "response_label",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split all events into outer train/test, then inner train/val.

    The returned indices always refer to the original metadata table.
    Groups are kept intact across train, validation, and test sets.
    """
    folds = grouped_kfold_indices(
        metadata,
        n_splits=num_folds,
        seed=seed,
        group_column=group_column,
        label_column=label_column,
    )
    if fold_index < 0 or fold_index >= len(folds):
        raise ValueError(f"fold_index must be in [0, {len(folds) - 1}]")

    outer_train_idx, test_idx = folds[fold_index]
    inner_metadata = metadata.iloc[outer_train_idx].reset_index(drop=True)
    inner_train_rel, inner_val_rel = grouped_train_val_indices(
        inner_metadata,
        val_ratio,
        seed + fold_index + 1000,
        group_column=group_column,
    )
    train_idx = outer_train_idx[inner_train_rel]
    val_idx = outer_train_idx[inner_val_rel]
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def split_summary(
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    group_column: str = "fmt_id",
) -> Dict[str, int]:
    """Return sample and group counts for split audit files."""
    summary: Dict[str, int] = {
        "train_events": int(len(train_idx)),
        "val_events": int(len(val_idx)),
        "test_events": int(len(test_idx)),
    }
    if group_column in metadata.columns:
        summary.update(
            {
                "train_groups": int(metadata.iloc[train_idx][group_column].astype(str).nunique()),
                "val_groups": int(metadata.iloc[val_idx][group_column].astype(str).nunique()),
                "test_groups": int(metadata.iloc[test_idx][group_column].astype(str).nunique()),
            }
        )
    return summary


def write_split_audit(
    output_dir: Path,
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    fold_index: int,
    num_folds: int,
    seed: int,
    group_column: str = "fmt_id",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "fold_index": int(fold_index),
        "num_folds": int(num_folds),
        "seed": int(seed),
        "group_column": group_column,
        **split_summary(metadata, train_idx, val_idx, test_idx, group_column=group_column),
    }
    with (output_dir / "fold_split.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
