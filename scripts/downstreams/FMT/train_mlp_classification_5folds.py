#!/usr/bin/env python3
"""Outer 5-fold MLP baseline for FMT response classification."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from fmt_5fold_utils import nested_grouped_cv_indices, split_summary, write_split_audit
from train_mlp_classification_baseline import (
    FMTClassificationArrays,
    FMTMLPClassificationConfig,
    build_arrays,
    predict_probabilities,
    set_seed,
    train_model,
    write_outputs,
)
from src.data.FMTDataset import FMTCommunityClassificationDataset


def subset_arrays(arrays: FMTClassificationArrays, indices: np.ndarray) -> FMTClassificationArrays:
    return FMTClassificationArrays(
        inputs=arrays.inputs[indices],
        labels=arrays.labels[indices],
        donor_raw=arrays.donor_raw[indices],
        recipient_raw=arrays.recipient_raw[indices],
        metadata=arrays.metadata.iloc[indices].reset_index(drop=True),
        feature_ids=arrays.feature_ids,
        class_names=arrays.class_names,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fully connected FMT response baseline with outer grouped 5-fold CV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_dir = Path("data/downstreams/FMT_5folds")
    parser.add_argument("--data_file", type=Path, default=data_dir / "community_all.pkl")
    parser.add_argument("--metadata_file", type=Path, default=data_dir / "response_all_meta.csv")
    parser.add_argument("--ntv3_embedding_path", type=Path, default=data_dir / "community_all.motu_embeddings.h5")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_downstreams/FMT_5folds/mlp_classification/seed_11/fold_0"))
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--group_column", type=str, default="fmt_id")
    parser.add_argument("--label_column", type=str, default="response_label")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--class_weight", choices=["balanced", "none"], default="balanced")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    config = FMTMLPClassificationConfig(
        train_data_file=str(args.data_file),
        train_metadata_file=str(args.metadata_file),
        test_data_file=str(args.data_file),
        test_metadata_file=str(args.metadata_file),
        ntv3_embedding_path=str(args.ntv3_embedding_path),
        test_ntv3_embedding_path=str(args.ntv3_embedding_path),
        output_dir=str(args.output_dir),
        fold_index=args.fold_index,
        num_folds=args.num_folds,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_group_column=args.group_column,
        seed=args.seed,
        device=str(device),
        class_weight=args.class_weight,
    )

    dataset = FMTCommunityClassificationDataset(args.data_file, args.metadata_file, args.ntv3_embedding_path)
    arrays = build_arrays(dataset)
    train_idx, val_idx, test_idx = nested_grouped_cv_indices(
        arrays.metadata,
        fold_index=args.fold_index,
        num_folds=args.num_folds,
        val_ratio=args.val_ratio,
        seed=args.seed,
        group_column=args.group_column,
        label_column=args.label_column,
    )
    counts = split_summary(arrays.metadata, train_idx, val_idx, test_idx, group_column=args.group_column)
    print(
        f"FMT MLP response fold {args.fold_index + 1}/{args.num_folds}: "
        f"train={counts['train_events']} val={counts['val_events']} test={counts['test_events']}"
    )
    print(f"Classes: {arrays.class_names} | input dim: {arrays.inputs.shape[1]} | device: {device}")

    model, scaler, history = train_model(arrays, train_idx, val_idx, config, device)
    train_counts = np.bincount(arrays.labels[train_idx], minlength=len(arrays.class_names))
    majority_class = int(train_counts.argmax())
    test_arrays = subset_arrays(arrays, test_idx)
    history.update(counts)
    history["train_majority_class"] = arrays.class_names[majority_class]
    history["train_majority_test_accuracy"] = float((test_arrays.labels == majority_class).mean())
    probs = predict_probabilities(model, test_arrays, scaler, config, device)
    write_outputs(args.output_dir, test_arrays, probs, config, history)
    write_split_audit(
        args.output_dir,
        arrays.metadata,
        train_idx,
        val_idx,
        test_idx,
        fold_index=args.fold_index,
        num_folds=args.num_folds,
        seed=args.seed,
        group_column=args.group_column,
    )
    print(f"Wrote FMT 5-fold MLP response results to {args.output_dir}")


if __name__ == "__main__":
    main()
