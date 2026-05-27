#!/usr/bin/env python3
"""Outer 5-fold MLP baseline for FMT post-community prediction."""

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
from train_mlp_baseline import (
    FMTMLPArrays,
    FMTMLPConfig,
    build_arrays,
    predict_raw,
    set_seed,
    train_model,
    write_outputs,
)
from src.data.FMTDataset import FMTCommunityPredictionDataset


def subset_arrays(arrays: FMTMLPArrays, indices: np.ndarray) -> FMTMLPArrays:
    return FMTMLPArrays(
        inputs=arrays.inputs[indices],
        targets=arrays.targets[indices],
        recipient_transformed=arrays.recipient_transformed[indices],
        donor_raw=arrays.donor_raw[indices],
        recipient_raw=arrays.recipient_raw[indices],
        post_raw=arrays.post_raw[indices],
        metadata=arrays.metadata.iloc[indices].reset_index(drop=True),
        feature_ids=arrays.feature_ids,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fully connected FMT post-community baseline with outer grouped 5-fold CV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_dir = Path("data/downstreams/FMT_5folds")
    parser.add_argument("--data_file", type=Path, default=data_dir / "community_all.pkl")
    parser.add_argument("--metadata_file", type=Path, default=data_dir / "post_all_meta.csv")
    parser.add_argument("--ntv3_embedding_path", type=Path, default=data_dir / "community_all.motu_embeddings.h5")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_downstreams/FMT_5folds/mlp_community_prediction/seed_11/fold_0"))
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--group_column", type=str, default="fmt_id")
    parser.add_argument("--label_column", type=str, default="response_label")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--raw_zero_threshold", type=float, default=5e-5)
    parser.add_argument("--renormalize_predictions", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    config = FMTMLPConfig(
        train_data_file=str(args.data_file),
        train_metadata_file=str(args.metadata_file),
        test_data_file=str(args.data_file),
        test_metadata_file=str(args.metadata_file),
        ntv3_embedding_path=str(args.ntv3_embedding_path),
        test_ntv3_embedding_path=str(args.ntv3_embedding_path),
        output_dir=str(args.output_dir),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        val_group_column=args.group_column,
        seed=args.seed,
        device=str(device),
        raw_zero_threshold=args.raw_zero_threshold,
        renormalize_predictions=args.renormalize_predictions,
    )

    dataset = FMTCommunityPredictionDataset(args.data_file, args.metadata_file, args.ntv3_embedding_path)
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
        f"FMT MLP community fold {args.fold_index + 1}/{args.num_folds}: "
        f"train={counts['train_events']} val={counts['val_events']} test={counts['test_events']}"
    )
    print(f"Features: {len(arrays.feature_ids)} | input dim: {arrays.inputs.shape[1]} | device: {device}")

    model, input_scaler, delta_scaler, history = train_model(arrays, train_idx, val_idx, config, device)
    history.update(counts)
    test_arrays = subset_arrays(arrays, test_idx)
    pred_raw = predict_raw(model, test_arrays, input_scaler, delta_scaler, config, device)
    write_outputs(args.output_dir, test_arrays, pred_raw, config, history)
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
    with (args.output_dir / "config.json").open("r+", encoding="utf-8") as handle:
        import json

        payload = json.load(handle)
        payload.update({"fold_index": args.fold_index, "num_folds": args.num_folds, "outer_cv": True})
        handle.seek(0)
        json.dump(payload, handle, indent=2)
        handle.truncate()
    print(f"Wrote FMT 5-fold MLP community results to {args.output_dir}")


if __name__ == "__main__":
    main()
