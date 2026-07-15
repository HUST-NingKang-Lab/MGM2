#!/usr/bin/env python3
"""MOZAIC reproduction for the original FMT response 5-fold baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.downstreams.FMT_cellreports.train_mozaic_similarity_random import (
    MOZAICArrays,
    MOZAICDenseNet1D,
    Standardizer,
    compute_metrics,
    make_loader,
    mozaic_binary_cross_entropy,
    mozaic_l2_regularization,
    predict,
    set_seed,
)
from src.data.FMTDataset import FMTCommunityClassificationDataset, grouped_kfold_indices


def parse_args() -> argparse.Namespace:
    data_dir = Path("data/downstreams/FMT")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_data_file", type=Path, default=data_dir / "community_train.pkl")
    parser.add_argument("--test_data_file", type=Path, default=data_dir / "community_test.pkl")
    parser.add_argument("--train_metadata_file", type=Path, default=data_dir / "response_train_meta_split.csv")
    parser.add_argument("--test_metadata_file", type=Path, default=data_dir / "response_test_meta_split.csv")
    parser.add_argument("--ntv3_embedding_path", type=Path, default=data_dir / "community_train.resolved_embeddings.h5")
    parser.add_argument("--test_ntv3_embedding_path", type=Path, default=data_dir / "community_test.resolved_embeddings.h5")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_downstreams/FMT/mozaic_classification_baseline"))
    parser.add_argument("--class_names", nargs="+", default=["non-responder", "responder"])
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--val_group_column", type=str, default="fmt_id")
    parser.add_argument("--fold_label_column", type=str, default="response_label")
    parser.add_argument("--seed", type=int, default=11)
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
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def build_arrays(
    community_data_file: Path,
    metadata_file: Path,
    ntv3_embedding_path: Path,
    class_names: List[str],
) -> MOZAICArrays:
    dataset = FMTCommunityClassificationDataset(
        community_data_file=community_data_file,
        metadata_file=metadata_file,
        ntv3_embedding_path=ntv3_embedding_path,
        class_names=class_names,
        sequence_mode="dense",
        deduplicate_group_column=None,
    )
    donor_values = []
    recipient_values = []
    labels = []
    rows = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        donor_values.append(item["donor_abundance_raw"][1:].numpy())
        recipient_values.append(item["recipient_abundance_raw"][1:].numpy())
        labels.append(int(item["label"].item()))
        rows.append(
            {
                "target_id": item["target_id"],
                "fmt_id": item["fmt_id"],
                "study": item["study"],
                "indication": item["indication"],
                "response_label": item["response_label"],
                "donor_sample_id": item["donor_sample_id"],
                "recipient_sample_id": item["recipient_sample_id"],
            }
        )
    return MOZAICArrays(
        donor=np.stack(donor_values).astype(np.float32),
        recipient=np.stack(recipient_values).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        metadata=pd.DataFrame(rows),
        feature_ids=list(dataset.feature_ids),
    )


def subset_arrays(arrays: MOZAICArrays, indices: np.ndarray) -> MOZAICArrays:
    indices = np.asarray(indices, dtype=np.int64)
    return MOZAICArrays(
        donor=arrays.donor[indices],
        recipient=arrays.recipient[indices],
        labels=arrays.labels[indices],
        metadata=arrays.metadata.iloc[indices].reset_index(drop=True),
        feature_ids=arrays.feature_ids,
    )


def train_mozaic(args: argparse.Namespace) -> Dict[str, float]:
    set_seed(args.seed)
    all_train_arrays = build_arrays(args.train_data_file, args.train_metadata_file, args.ntv3_embedding_path, args.class_names)
    test_arrays = build_arrays(args.test_data_file, args.test_metadata_file, args.test_ntv3_embedding_path, args.class_names)
    folds = grouped_kfold_indices(
        all_train_arrays.metadata,
        n_splits=args.num_folds,
        seed=args.seed,
        group_column=args.val_group_column,
        label_column=args.fold_label_column,
    )
    train_idx, val_idx = folds[args.fold_index]
    train_arrays = subset_arrays(all_train_arrays, train_idx)
    val_arrays = subset_arrays(all_train_arrays, val_idx)

    scaler = Standardizer.fit(train_arrays.donor, train_arrays.recipient)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
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
        "fold_index": int(args.fold_index),
        "num_folds": int(args.num_folds),
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "best_val_roc_auc": float(best_val_auc),
        "train_events": int(len(train_arrays.labels)),
        "val_events": int(len(val_arrays.labels)),
        "test_events": int(len(test_arrays.labels)),
        "num_features": int(train_arrays.donor.shape[1]),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (args.output_dir / "mozaic_args.json").open("w", encoding="utf-8") as handle:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, handle, indent=2)
    return metrics


def main() -> None:
    args = parse_args()
    metrics = train_mozaic(args)
    print(pd.Series(metrics).to_string())
    print(f"Wrote original FMT MOZAIC response fold results to {args.output_dir}")


if __name__ == "__main__":
    main()
