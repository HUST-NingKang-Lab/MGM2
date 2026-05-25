#!/usr/bin/env python3
"""Fully connected baseline for FMT response classification."""

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.FMTDataset import FMTCommunityClassificationDataset, grouped_kfold_indices


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = values.mean(axis=0, keepdims=True).astype(np.float32)
        std = values.std(axis=0, keepdims=True).astype(np.float32)
        std[std < 1e-8] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)


@dataclass
class FMTClassificationArrays:
    inputs: np.ndarray
    labels: np.ndarray
    donor_raw: np.ndarray
    recipient_raw: np.ndarray
    metadata: pd.DataFrame
    feature_ids: List[str]
    class_names: List[str]


@dataclass
class FMTMLPClassificationConfig:
    train_data_file: str
    train_metadata_file: str
    test_data_file: str
    test_metadata_file: str
    ntv3_embedding_path: str
    test_ntv3_embedding_path: str
    output_dir: str
    fold_index: int
    num_folds: int
    hidden_size: int
    num_layers: int
    dropout: float
    batch_size: int
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    val_group_column: str
    seed: int
    device: str
    class_weight: str


class FMTResponseMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        layers: List[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_size
        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arrays(dataset: FMTCommunityClassificationDataset) -> FMTClassificationArrays:
    recipient_t, donor_t, labels = [], [], []
    recipient_raw, donor_raw, rows = [], [], []
    for idx in range(len(dataset)):
        item = dataset[idx]
        recipient_values = item["recipient_abundance_values"][1:].numpy()
        donor_values = item["donor_abundance_values"][1:].numpy()
        recipient_t.append(recipient_values)
        donor_t.append(donor_values)
        recipient_raw.append(item["recipient_abundance_raw"][1:].numpy())
        donor_raw.append(item["donor_abundance_raw"][1:].numpy())
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

    recipient_arr = np.stack(recipient_t).astype(np.float32)
    donor_arr = np.stack(donor_t).astype(np.float32)
    inputs = np.concatenate([recipient_arr, donor_arr, donor_arr - recipient_arr], axis=1).astype(np.float32)
    return FMTClassificationArrays(
        inputs=inputs,
        labels=np.asarray(labels, dtype=np.int64),
        donor_raw=np.stack(donor_raw).astype(np.float32),
        recipient_raw=np.stack(recipient_raw).astype(np.float32),
        metadata=pd.DataFrame(rows),
        feature_ids=list(dataset.feature_ids),
        class_names=list(dataset.class_names),
    )


def make_loader(inputs: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_dataset = TensorDataset(
        torch.from_numpy(inputs.astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


def class_weights(labels: np.ndarray, num_classes: int, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = labels.size / np.clip(counts * num_classes, 1.0, None)
    return torch.from_numpy(weights.astype(np.float32))


def train_model(
    arrays: FMTClassificationArrays,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: FMTMLPClassificationConfig,
    device: torch.device,
) -> Tuple[FMTResponseMLP, Standardizer, Dict[str, float]]:
    scaler = Standardizer.fit(arrays.inputs[train_idx])
    train_loader = make_loader(scaler.transform(arrays.inputs[train_idx]), arrays.labels[train_idx], config.batch_size, True)
    val_loader = make_loader(scaler.transform(arrays.inputs[val_idx]), arrays.labels[val_idx], config.batch_size, False)
    model = FMTResponseMLP(
        input_dim=arrays.inputs.shape[1],
        num_classes=len(arrays.class_names),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    weights = class_weights(arrays.labels[train_idx], len(arrays.class_names), config.class_weight)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    best_state = None
    best_val = float("inf")
    stale = 0
    epochs_trained = 0

    for epoch in range(config.max_epochs):
        model.train()
        for input_batch, label_batch in train_loader:
            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device)
            loss = criterion(model(input_batch), label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for input_batch, label_batch in val_loader:
                input_batch = input_batch.to(device)
                label_batch = label_batch.to(device)
                val_losses.append(float(criterion(model(input_batch), label_batch).item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        epochs_trained = epoch + 1
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, {"best_val_loss": best_val, "epochs_trained": epochs_trained}


def predict_probabilities(
    model: FMTResponseMLP,
    arrays: FMTClassificationArrays,
    scaler: Standardizer,
    config: FMTMLPClassificationConfig,
    device: torch.device,
) -> np.ndarray:
    loader = make_loader(scaler.transform(arrays.inputs), arrays.labels, config.batch_size, False)
    chunks = []
    model.eval()
    with torch.no_grad():
        for input_batch, _ in loader:
            chunks.append(torch.softmax(model(input_batch.to(device)), dim=-1).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    unique_scores, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if len(unique_scores) != len(scores):
        rank_sums = np.zeros(len(unique_scores), dtype=np.float64)
        np.add.at(rank_sums, inverse, ranks)
        ranks = rank_sums[inverse] / counts[inverse]
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    precision = tp / (np.arange(len(labels)) + 1)
    return float((precision * (sorted_labels == 1)).sum() / n_pos)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    preds = probs.argmax(axis=1)
    metrics: Dict[str, float] = {"accuracy": float((preds == labels).mean())}
    recalls = []
    f1_values = []
    for class_id in range(len(class_names)):
        pred_mask = preds == class_id
        label_mask = labels == class_id
        tp = float((pred_mask & label_mask).sum())
        fp = float((pred_mask & ~label_mask).sum())
        fn = float((~pred_mask & label_mask).sum())
        if label_mask.any():
            recalls.append(tp / max(tp + fn, 1.0))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        if (tp + fp) > 0 or (tp + fn) > 0:
            f1_values.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    metrics["balanced_accuracy"] = float(np.mean(recalls)) if recalls else float("nan")
    metrics["macro_f1"] = float(np.mean(f1_values)) if f1_values else float("nan")
    if len(class_names) == 2:
        positive_id = class_names.index("responder") if "responder" in class_names else 1
        binary_labels = (labels == positive_id).astype(int)
        scores = probs[:, positive_id]
        metrics["roc_auc"] = binary_auc(binary_labels, scores)
        metrics["average_precision"] = average_precision(binary_labels, scores)
    return metrics


def write_outputs(
    output_dir: Path,
    arrays: FMTClassificationArrays,
    probs: np.ndarray,
    config: FMTMLPClassificationConfig,
    history: Dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preds = probs.argmax(axis=1)
    table = arrays.metadata.copy()
    table["label_id"] = arrays.labels
    table["pred_id"] = preds
    table["pred_label"] = [arrays.class_names[int(idx)] for idx in preds]
    for class_id, class_name in enumerate(arrays.class_names):
        table[f"prob_{class_name.replace('-', '_')}"] = probs[:, class_id]
    table.to_csv(output_dir / "test_predictions.csv", index=False)
    metrics = compute_metrics(arrays.labels, probs, arrays.class_names)
    metrics.update(history)
    metrics["fold_index"] = config.fold_index
    metrics["num_folds"] = config.num_folds
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "class_map.json").open("w", encoding="utf-8") as handle:
        json.dump({name: idx for idx, name in enumerate(arrays.class_names)}, handle, indent=2)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
    print(pd.Series(metrics).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fully connected FMT response classification baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_dir = Path("data/downstreams/FMT")
    parser.add_argument("--train_data_file", type=Path, default=data_dir / "community_train.pkl")
    parser.add_argument("--train_metadata_file", type=Path, default=data_dir / "response_train_meta_split.csv")
    parser.add_argument("--test_data_file", type=Path, default=data_dir / "community_test.pkl")
    parser.add_argument("--test_metadata_file", type=Path, default=data_dir / "response_test_meta_split.csv")
    parser.add_argument("--ntv3_embedding_path", type=Path, default=data_dir / "community_train.resolved_embeddings.h5")
    parser.add_argument("--test_ntv3_embedding_path", type=Path, default=data_dir / "community_test.resolved_embeddings.h5")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_downstreams/FMT/mlp_classification_baseline"))
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_group_column", type=str, default="fmt_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--class_weight", choices=["balanced", "none"], default="balanced")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    config = FMTMLPClassificationConfig(
        train_data_file=str(args.train_data_file),
        train_metadata_file=str(args.train_metadata_file),
        test_data_file=str(args.test_data_file),
        test_metadata_file=str(args.test_metadata_file),
        ntv3_embedding_path=str(args.ntv3_embedding_path),
        test_ntv3_embedding_path=str(args.test_ntv3_embedding_path),
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
        val_group_column=args.val_group_column,
        seed=args.seed,
        device=str(device),
        class_weight=args.class_weight,
    )

    train_dataset = FMTCommunityClassificationDataset(args.train_data_file, args.train_metadata_file, args.ntv3_embedding_path)
    test_dataset = FMTCommunityClassificationDataset(
        args.test_data_file,
        args.test_metadata_file,
        args.test_ntv3_embedding_path,
        class_names=train_dataset.class_names,
    )
    train_arrays = build_arrays(train_dataset)
    test_arrays = build_arrays(test_dataset)
    folds = grouped_kfold_indices(
        train_dataset.metadata,
        n_splits=args.num_folds,
        seed=args.seed,
        group_column=args.val_group_column,
        label_column="response_label",
    )
    train_idx, val_idx = folds[args.fold_index]
    print(
        f"FMT MLP response fold {args.fold_index + 1}/{args.num_folds}: "
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_dataset)}"
    )
    print(f"Classes: {train_arrays.class_names} | input dim: {train_arrays.inputs.shape[1]} | device: {device}")

    model, scaler, history = train_model(train_arrays, train_idx, val_idx, config, device)
    train_counts = np.bincount(train_arrays.labels[train_idx], minlength=len(train_arrays.class_names))
    majority_class = int(train_counts.argmax())
    history["train_majority_class"] = train_arrays.class_names[majority_class]
    history["train_majority_test_accuracy"] = float((test_arrays.labels == majority_class).mean())
    probs = predict_probabilities(model, test_arrays, scaler, config, device)
    write_outputs(args.output_dir, test_arrays, probs, config, history)
    print(f"Wrote FMT MLP response classification results to {args.output_dir}")


if __name__ == "__main__":
    main()
