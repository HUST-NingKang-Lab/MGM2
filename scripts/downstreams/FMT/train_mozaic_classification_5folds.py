#!/usr/bin/env python3
"""Outer 5-fold MOZAIC reproduction for FMT response classification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from fmt_5fold_utils import nested_grouped_cv_indices, split_summary, write_split_audit
from train_mozaic_classification_baseline import build_arrays, subset_arrays
from scripts.downstreams.FMT_cellreports.train_mozaic_similarity_random import (
    MOZAICDenseNet1D,
    Standardizer,
    compute_metrics,
    make_loader,
    mozaic_binary_cross_entropy,
    mozaic_l2_regularization,
    predict,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    data_dir = Path("data/downstreams/FMT_5folds")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_file", type=Path, default=data_dir / "community_all.pkl")
    parser.add_argument("--metadata_file", type=Path, default=data_dir / "response_all_meta.csv")
    parser.add_argument("--ntv3_embedding_path", type=Path, default=data_dir / "community_all.motu_embeddings.h5")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_downstreams/FMT_5folds/mozaic_classification/seed_0/fold_0"))
    parser.add_argument("--class_names", nargs="+", default=["non-responder", "responder"])
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--group_column", type=str, default="fmt_id")
    parser.add_argument("--label_column", type=str, default="indication")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--nb_layers", type=int, default=5)
    parser.add_argument("--nb_filter", type=int, default=32)
    parser.add_argument("--growth_rate", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--init_min", type=float, default=-0.05)
    parser.add_argument("--init_max", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def train_one_fold(args: argparse.Namespace) -> Dict[str, float]:
    set_seed(args.seed)
    arrays = build_arrays(args.data_file, args.metadata_file, args.ntv3_embedding_path, args.class_names)
    train_idx, val_idx, test_idx = nested_grouped_cv_indices(
        arrays.metadata,
        fold_index=args.fold_index,
        num_folds=args.num_folds,
        val_ratio=args.val_ratio,
        seed=args.seed,
        group_column=args.group_column,
        label_column=args.label_column,
    )
    train_arrays = subset_arrays(arrays, train_idx)
    val_arrays = subset_arrays(arrays, val_idx)
    test_arrays = subset_arrays(arrays, test_idx)
    counts = split_summary(arrays.metadata, train_idx, val_idx, test_idx, group_column=args.group_column, label_column=args.label_column)

    scaler = Standardizer.fit(train_arrays.donor, train_arrays.recipient)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = MOZAICDenseNet1D(
        num_features=train_arrays.donor.shape[1],
        num_classes=len(args.class_names),
        nb_layers=args.nb_layers,
        nb_filter=args.nb_filter,
        growth_rate=args.growth_rate,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        init_min=args.init_min,
        init_max=args.init_max,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adam_epsilon,
        weight_decay=0.0,
    )
    train_loader = make_loader(train_arrays, scaler, args.batch_size, shuffle=True)

    best_state = None
    best_val_auc = -float("inf")
    best_epoch = -1
    history = []
    for epoch in range(args.max_epochs):
        model.train()
        losses = []
        for donor, recipient, labels in train_loader:
            donor = donor.to(device)
            recipient = recipient.to(device)
            labels = labels.to(device)
            ce_loss = mozaic_binary_cross_entropy(model(donor, recipient), labels, num_classes=len(args.class_names))
            l2_loss = mozaic_l2_regularization(model).to(device) * args.weight_decay
            loss = ce_loss + l2_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_probs = predict(model, val_arrays, scaler, args.batch_size, device)
        val_metrics = compute_metrics(val_arrays.labels, val_probs)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **{f"val_{key}": value for key, value in val_metrics.items()}})
        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"fold={args.fold_index} epoch={epoch + 1:03d} train_loss={train_loss:.4f} val_auc={val_metrics['roc_auc']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_probs = predict(model, test_arrays, scaler, args.batch_size, device)
    test_metrics = compute_metrics(test_arrays.labels, test_probs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = test_arrays.metadata.copy()
    predictions["label_id"] = test_arrays.labels
    predictions["pred_id"] = test_probs.argmax(axis=1)
    predictions["pred_label"] = [args.class_names[idx] for idx in predictions["pred_id"]]
    for class_id, class_name in enumerate(args.class_names):
        predictions[f"prob_{class_name.replace('-', '_')}"] = test_probs[:, class_id]
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)

    metrics = {
        **test_metrics,
        **counts,
        "fold_index": int(args.fold_index),
        "num_folds": int(args.num_folds),
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "best_val_roc_auc": float(best_val_auc),
        "num_features": int(train_arrays.donor.shape[1]),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (args.output_dir / "mozaic_args.json").open("w", encoding="utf-8") as handle:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, handle, indent=2)
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
        label_column=args.label_column,
    )
    return metrics


def main() -> None:
    args = parse_args()
    metrics = train_one_fold(args)
    print(pd.Series(metrics).to_string())
    print(f"Wrote FMT 5-fold MOZAIC response results to {args.output_dir}")


if __name__ == "__main__":
    main()
