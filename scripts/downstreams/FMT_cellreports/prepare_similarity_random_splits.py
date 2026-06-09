#!/usr/bin/env python3
"""Prepare Cell Reports FMT donor-similarity labels and random splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import pandas as pd


CLASS_NAMES = ["similarity_increased", "similarity_decreased"]


def parse_args() -> argparse.Namespace:
    data_dir = Path("data/downstreams/FMT_cellreports")
    parser = argparse.ArgumentParser(
        description="Build donor-similarity FMT labels and 72/8/20 random splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, default=data_dir)
    parser.add_argument("--event_table", type=Path, default=None)
    parser.add_argument("--community_h5", type=Path, default=None)
    parser.add_argument("--h5_key", type=str, default="cellreports")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--train_ratio", type=float, default=0.72)
    parser.add_argument("--val_ratio", type=float, default=0.08)
    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--stratify_column", type=str, default="disease_code")
    return parser.parse_args()


def _decode_h5_strings(values) -> List[str]:
    return [
        value.decode("utf-8", errors="replace") if isinstance(value, (bytes, bytearray, np.bytes_)) else str(value)
        for value in values
    ]


def bray_curtis(first: np.ndarray, second: np.ndarray) -> float:
    denom = float(first.sum() + second.sum())
    if denom <= 0:
        return 0.0
    return float(np.abs(first - second).sum() / denom)


def donor_similarity_delta(recipient_pre: np.ndarray, recipient_post: np.ndarray, donor: np.ndarray) -> Tuple[float, float, float, int, str]:
    pre_similarity = 1.0 - bray_curtis(recipient_pre, donor)
    post_similarity = 1.0 - bray_curtis(recipient_post, donor)
    delta = post_similarity - pre_similarity
    label_id = 0 if delta > 0 else 1
    return pre_similarity, post_similarity, delta, label_id, CLASS_NAMES[label_id]


def split_sizes(n_events: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"split ratios must sum to 1.0, got {ratio_sum}")
    n_train = int(round(n_events * train_ratio))
    n_val = int(round(n_events * val_ratio))
    n_test = n_events - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"invalid split sizes for n={n_events}: {(n_train, n_val, n_test)}")
    return n_train, n_val, n_test


def random_split_indices(n_events: int, seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train, n_val, _ = split_sizes(n_events, train_ratio, val_ratio, test_ratio)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_events)
    train_idx = np.sort(order[:n_train])
    val_idx = np.sort(order[n_train : n_train + n_val])
    test_idx = np.sort(order[n_train + n_val :])
    return train_idx, val_idx, test_idx


def _largest_remainder_counts(group_sizes: np.ndarray, target_total: int) -> np.ndarray:
    if target_total <= 0:
        return np.zeros_like(group_sizes, dtype=np.int64)
    total = int(group_sizes.sum())
    if total <= 0:
        raise ValueError("group_sizes must contain at least one event")
    raw = group_sizes.astype(np.float64) * float(target_total) / float(total)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(target_total - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts), kind="mergesort")
        for idx in order[:remainder]:
            counts[idx] += 1
    return np.minimum(counts, group_sizes.astype(np.int64))


def stratified_split_indices(
    metadata: pd.DataFrame,
    stratify_column: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stratify_column not in metadata.columns:
        return random_split_indices(len(metadata), seed, train_ratio, val_ratio, test_ratio)
    n_train, n_val, _ = split_sizes(len(metadata), train_ratio, val_ratio, test_ratio)
    strata = metadata[stratify_column].astype(str).to_numpy()
    unique_strata = np.asarray(pd.unique(strata), dtype=object)
    group_indices = [np.flatnonzero(strata == stratum) for stratum in unique_strata]
    group_sizes = np.asarray([len(indices) for indices in group_indices], dtype=np.int64)
    train_counts = _largest_remainder_counts(group_sizes, n_train)
    remaining_sizes = group_sizes - train_counts
    val_counts = _largest_remainder_counts(remaining_sizes, n_val)

    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    for indices, n_group_train, n_group_val in zip(group_indices, train_counts, val_counts):
        shuffled = np.asarray(indices, dtype=np.int64).copy()
        rng.shuffle(shuffled)
        train_parts.append(shuffled[:n_group_train])
        val_parts.append(shuffled[n_group_train : n_group_train + n_group_val])
        test_parts.append(shuffled[n_group_train + n_group_val :])

    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=np.int64)
    val_idx = np.sort(np.concatenate(val_parts)) if val_parts else np.array([], dtype=np.int64)
    test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=np.int64)
    return train_idx, val_idx, test_idx


def _sample_profile(values, sample_to_idx: Dict[str, int], sample_id: str) -> np.ndarray:
    profile = values[:, sample_to_idx[str(sample_id)]].astype(np.float32)
    total = float(profile.sum())
    if total > 0:
        profile = profile / total
    return profile


def build_similarity_metadata(event_table: Path, community_h5: Path, h5_key: str) -> pd.DataFrame:
    events = pd.read_csv(event_table)
    required = {"donor_sample_id", "recipient_sample_id", "post_sample_id"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"event table missing columns: {missing}")

    with h5py.File(community_h5, "r") as handle:
        group = handle[h5_key]
        sample_ids = _decode_h5_strings(group["axis0"][:])
        sample_to_idx = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
        values = group["block0_values"]
        rows = []
        for _, row in events.iterrows():
            donor = str(row["donor_sample_id"])
            recipient = str(row["recipient_sample_id"])
            post = str(row["post_sample_id"])
            if donor not in sample_to_idx or recipient not in sample_to_idx or post not in sample_to_idx:
                continue
            recipient_pre_profile = _sample_profile(values, sample_to_idx, recipient)
            recipient_post_profile = _sample_profile(values, sample_to_idx, post)
            donor_profile = _sample_profile(values, sample_to_idx, donor)
            pre_sim, post_sim, delta, label_id, label_name = donor_similarity_delta(
                recipient_pre_profile,
                recipient_post_profile,
                donor_profile,
            )
            output = row.to_dict()
            output["clinical_response_label"] = output.get("response_label", "")
            output["response_label"] = label_name
            output["similarity_label_id"] = label_id
            output["similarity_pre"] = pre_sim
            output["similarity_post"] = post_sim
            output["similarity_delta"] = delta
            output["similarity_label_definition"] = "0:post more similar to donor; 1:post less/equal similar to donor"
            rows.append(output)
    if not rows:
        raise ValueError("No events have donor, recipient, and post samples in the community H5 table")
    return pd.DataFrame(rows).reset_index(drop=True)


def write_splits(
    metadata: pd.DataFrame,
    output_dir: Path,
    num_repeats: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    stratify_column: str,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_dir / "similarity_all_meta.csv", index=False)
    with (output_dir / "class_map.json").open("w", encoding="utf-8") as handle:
        json.dump({name: idx for idx, name in enumerate(CLASS_NAMES)}, handle, indent=2)

    summary_rows = []
    for repeat in range(num_repeats):
        split_seed = seed + repeat
        train_idx, val_idx, test_idx = stratified_split_indices(
            metadata,
            stratify_column,
            split_seed,
            train_ratio,
            val_ratio,
            test_ratio,
        )
        split_dir = output_dir / f"split_{repeat:02d}"
        split_dir.mkdir(parents=True, exist_ok=True)
        parts = {
            "train": metadata.iloc[train_idx].copy(),
            "val": metadata.iloc[val_idx].copy(),
            "test": metadata.iloc[test_idx].copy(),
        }
        for name, table in parts.items():
            table.to_csv(split_dir / f"{name}_meta.csv", index=False)
            counts = table["response_label"].value_counts().to_dict()
            summary_rows.append(
                {
                    "split": repeat,
                    "seed": split_seed,
                    "partition": name,
                    "events": int(len(table)),
                    "stratify_column": stratify_column,
                    "strata": int(table[stratify_column].nunique()) if stratify_column in table.columns else 0,
                    "similarity_increased": int(counts.get("similarity_increased", 0)),
                    "similarity_decreased": int(counts.get("similarity_decreased", 0)),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "split_summary.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    event_table = args.event_table or (args.data_dir / "fmt_event_table.csv")
    community_h5 = args.community_h5 or (args.data_dir / "community_all.h5")
    output_dir = args.output_dir or (args.data_dir / "similarity_random_splits")
    metadata = build_similarity_metadata(event_table, community_h5, args.h5_key)
    summary = write_splits(
        metadata,
        output_dir,
        args.num_repeats,
        args.seed,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.stratify_column,
    )
    label_counts = metadata["response_label"].value_counts().to_dict()
    print(
        json.dumps(
            {
                "events": int(len(metadata)),
                "class_counts": {key: int(value) for key, value in label_counts.items()},
                "output_dir": str(output_dir),
                "num_repeats": int(args.num_repeats),
            },
            indent=2,
        )
    )
    print(summary.head(9).to_string(index=False))


if __name__ == "__main__":
    main()
