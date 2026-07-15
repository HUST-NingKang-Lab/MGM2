#!/usr/bin/env python3
"""Train a microbiome-only MOZAIC reproduction on Cell Reports FMT splits.

This is a PyTorch reproduction of the public MOZAIC Keras architecture pattern:
two donor/recipient Conv1D DenseNet branches, feature fusion, and a final
classifier. Only microbiome abundance features are used.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data.FMTDataset import FMTCommunityClassificationDataset

CLASS_NAMES = ["similarity_increased", "similarity_decreased"]


@dataclass
class MOZAICArrays:
    donor: np.ndarray
    recipient: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame
    feature_ids: List[str]


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, *arrays: np.ndarray) -> "Standardizer":
        values = np.concatenate(arrays, axis=0).astype(np.float32)
        mean = values.mean(axis=0, keepdims=True).astype(np.float32)
        std = values.std(axis=0, keepdims=True).astype(np.float32)
        std[std < 1e-8] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)


class DenseLayer1D(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, kernel_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, growth_rate, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.cat([values, self.net(values)], dim=1)


class DenseBlock1D(nn.Module):
    def __init__(self, in_channels: int, nb_layers: int, growth_rate: int, kernel_size: int, dropout: float):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(nb_layers):
            layers.append(DenseLayer1D(channels, growth_rate, kernel_size, dropout))
            channels += growth_rate
        self.layers = nn.ModuleList(layers)
        self.out_channels = channels

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            values = layer(values)
        return values


class MOZAICBranch(nn.Module):
    def __init__(self, nb_layers: int, nb_filter: int, growth_rate: int, kernel_size: int, dropout: float):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(1, nb_filter, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.ReLU(inplace=True),
        )
        self.block1 = DenseBlock1D(nb_filter, nb_layers, growth_rate, kernel_size, dropout)
        self.block2 = DenseBlock1D(self.block1.out_channels, nb_layers, growth_rate, kernel_size, dropout)
        self.out_channels = self.block2.out_channels

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.unsqueeze(1)
        values = self.initial(values)
        values = self.block1(values)
        values = self.block2(values)
        return values


class MOZAICDenseNet1D(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        nb_layers: int = 5,
        nb_filter: int = 32,
        growth_rate: int = 32,
        kernel_size: int = 5,
        dropout: float = 0.2,
        init_min: float = -0.05,
        init_max: float = 0.05,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.donor_branch = MOZAICBranch(nb_layers, nb_filter, growth_rate, kernel_size, dropout)
        self.recipient_branch = MOZAICBranch(nb_layers, nb_filter, growth_rate, kernel_size, dropout)
        fused_channels = self.donor_branch.out_channels + self.recipient_branch.out_channels
        reduced_1 = max(1, int(fused_channels * 0.25))
        reduced_2 = max(1, int(fused_channels * 0.5))
        self.fusion_projection = nn.Sequential(
            nn.Conv1d(fused_channels, reduced_1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_1, reduced_2, kernel_size=1),
        )
        self.fusion_block = DenseBlock1D(reduced_2, nb_layers, growth_rate, kernel_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.fusion_block.out_channels * self.num_features, num_classes)
        self.reset_parameters(init_min, init_max)

    def reset_parameters(self, init_min: float = -0.05, init_max: float = 0.05) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.uniform_(module.weight, a=init_min, b=init_max)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, donor: torch.Tensor, recipient: torch.Tensor) -> torch.Tensor:
        donor_hidden = self.donor_branch(donor)
        recipient_hidden = self.recipient_branch(recipient)
        fused = torch.cat([donor_hidden, recipient_hidden], dim=1)
        fused = self.fusion_projection(fused)
        fused = self.fusion_block(fused)
        fused = torch.flatten(fused, start_dim=1)
        return self.classifier(self.dropout(fused))


def mozaic_l2_regularization(model: nn.Module) -> torch.Tensor:
    """Keras-style L2 regularization over trainable Conv/Dense weights."""
    penalties = []
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            penalties.append(torch.sum(module.weight.pow(2)))
            if module.bias is not None:
                penalties.append(torch.sum(module.bias.pow(2)))
    if not penalties:
        return torch.tensor(0.0)
    return torch.stack(penalties).sum()


def mozaic_binary_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    targets = F.one_hot(labels, num_classes=num_classes).to(dtype=logits.dtype)
    probabilities = torch.softmax(logits, dim=1)
    return F.binary_cross_entropy(probabilities, targets)


def parse_split(path: Path) -> int:
    match = re.search(r"split_(\d+)", str(path))
    return int(match.group(1)) if match else -1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def make_loader(arrays: MOZAICArrays, scaler: Standardizer, batch_size: int, shuffle: bool) -> DataLoader:
    donor = scaler.transform(arrays.donor)
    recipient = scaler.transform(arrays.recipient)
    dataset = TensorDataset(
        torch.from_numpy(donor),
        torch.from_numpy(recipient),
        torch.from_numpy(arrays.labels.astype(np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
    pos_rank_sum = ranks[positives].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


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


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    preds = probs.argmax(axis=1)
    metrics = {"accuracy": float((preds == labels).mean())}
    recalls = []
    f1_values = []
    for class_id in range(2):
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
    metrics["roc_auc"] = binary_auc(labels == 1, probs[:, 1])
    metrics["average_precision"] = average_precision(labels == 1, probs[:, 1])
    return metrics


def predict(model: nn.Module, arrays: MOZAICArrays, scaler: Standardizer, batch_size: int, device: torch.device) -> np.ndarray:
    loader = make_loader(arrays, scaler, batch_size, shuffle=False)
    chunks = []
    model.eval()
    with torch.no_grad():
        for donor, recipient, _ in loader:
            logits = model(donor.to(device), recipient.to(device))
            chunks.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def train_one_split(args: argparse.Namespace, split: int) -> Dict[str, float]:
    split_dir = args.split_root / f"split_{split:02d}"
    train_meta = split_dir / "train_meta.csv"
    val_meta = split_dir / "val_meta.csv"
    test_meta = split_dir / "test_meta.csv"
    for path in [train_meta, val_meta, test_meta]:
        if not path.exists():
            raise FileNotFoundError(path)

    seed = args.seed + split
    set_seed(seed)
    train_arrays = build_arrays(args.data_file, train_meta, args.ntv3_embedding_path, CLASS_NAMES)
    val_arrays = build_arrays(args.data_file, val_meta, args.ntv3_embedding_path, CLASS_NAMES)
    test_arrays = build_arrays(args.data_file, test_meta, args.ntv3_embedding_path, CLASS_NAMES)
    scaler = Standardizer.fit(train_arrays.donor, train_arrays.recipient)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MOZAICDenseNet1D(
        num_features=train_arrays.donor.shape[1],
        num_classes=2,
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
            ce_loss = mozaic_binary_cross_entropy(model(donor, recipient), labels)
            l2_loss = mozaic_l2_regularization(model).to(device) * args.weight_decay
            loss = ce_loss + l2_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_probs = predict(model, val_arrays, scaler, args.batch_size, device)
        val_metrics = compute_metrics(val_arrays.labels, val_probs)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"split={split:02d} epoch={epoch + 1:03d} train_loss={train_loss:.4f} val_auc={val_metrics['roc_auc']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_probs = predict(model, test_arrays, scaler, args.batch_size, device)
    test_metrics = compute_metrics(test_arrays.labels, test_probs)
    output_dir = args.output_dir / f"split_{split:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = test_arrays.metadata.copy()
    predictions["label_id"] = test_arrays.labels
    predictions["pred_id"] = test_probs.argmax(axis=1)
    predictions["pred_label"] = [CLASS_NAMES[idx] for idx in predictions["pred_id"]]
    for class_id, class_name in enumerate(CLASS_NAMES):
        predictions[f"prob_{class_name}"] = test_probs[:, class_id]
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    metrics = {
        **test_metrics,
        "split": split,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_roc_auc": best_val_auc,
        "train_events": int(len(train_arrays.labels)),
        "val_events": int(len(val_arrays.labels)),
        "test_events": int(len(test_arrays.labels)),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    data_dir = Path("data/downstreams/FMT_cellreports")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_file", type=Path, default=data_dir / "community_all.pkl")
    parser.add_argument("--ntv3_embedding_path", type=Path, default=data_dir / "community_all.cellreports_embeddings.h5")
    parser.add_argument("--split_root", type=Path, default=data_dir / "similarity_random_splits")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_downstreams/FMT_cellreports/mozaic_similarity_random"))
    parser.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "mozaic_args.json").open("w", encoding="utf-8") as handle:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, handle, indent=2)
    rows = []
    for split in args.splits:
        rows.append(train_one_split(args, int(split)))
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary[["split", "roc_auc", "accuracy", "balanced_accuracy", "macro_f1", "best_epoch"]].to_string(index=False))


if __name__ == "__main__":
    main()
