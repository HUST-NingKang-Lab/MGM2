#!/usr/bin/env python3
"""Fully connected baseline for FMT post-community prediction."""

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.FMTDataset import FMTCommunityPredictionDataset, grouped_train_val_indices


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

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std + self.mean).astype(np.float32)


@dataclass
class FMTMLPArrays:
    inputs: np.ndarray
    targets: np.ndarray
    recipient_transformed: np.ndarray
    donor_raw: np.ndarray
    recipient_raw: np.ndarray
    post_raw: np.ndarray
    metadata: pd.DataFrame
    feature_ids: List[str]


@dataclass
class FMTMLPConfig:
    train_data_file: str
    train_metadata_file: str
    test_data_file: str
    test_metadata_file: str
    ntv3_embedding_path: str
    test_ntv3_embedding_path: str
    output_dir: str
    hidden_size: int
    num_layers: int
    dropout: float
    batch_size: int
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    val_ratio: float
    val_group_column: str
    seed: int
    device: str
    raw_zero_threshold: float
    renormalize_predictions: bool


class FMTResidualMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 512, num_layers: int = 3, dropout: float = 0.1):
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
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized_timepoint(days: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(days.astype(np.float32), nan=0.0, posinf=365.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    return (np.log1p(values) / np.log(366.0)).reshape(-1, 1).astype(np.float32)


def build_arrays(dataset: FMTCommunityPredictionDataset) -> FMTMLPArrays:
    recipient_t, donor_t, post_t = [], [], []
    recipient_raw, donor_raw, post_raw, timepoints = [], [], [], []
    rows = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        recipient_t.append(item["recipient_abundance_values"][1:].numpy())
        donor_t.append(item["donor_abundance_values"][1:].numpy())
        post_t.append(item["post_abundance_values"][1:].numpy())
        recipient_raw.append(item["recipient_abundance_raw"][1:].numpy())
        donor_raw.append(item["donor_abundance_raw"][1:].numpy())
        post_raw.append(item["post_abundance_raw"][1:].numpy())
        timepoints.append(float(item["timepoint_fmt"].item()))
        rows.append(
            {
                "target_id": item["target_id"],
                "fmt_id": item["fmt_id"],
                "study": item["study"],
                "response_label": item["response_label"],
                "donor_sample_id": item["donor_sample_id"],
                "recipient_sample_id": item["recipient_sample_id"],
                "post_sample_id": item["post_sample_id"],
                "timepoint_fmt": timepoints[-1],
            }
        )

    recipient_t_arr = np.stack(recipient_t).astype(np.float32)
    donor_t_arr = np.stack(donor_t).astype(np.float32)
    post_t_arr = np.stack(post_t).astype(np.float32)
    inputs = np.concatenate(
        [
            recipient_t_arr,
            donor_t_arr,
            normalized_timepoint(np.asarray(timepoints, dtype=np.float32)),
        ],
        axis=1,
    ).astype(np.float32)
    return FMTMLPArrays(
        inputs=inputs,
        targets=post_t_arr,
        recipient_transformed=recipient_t_arr,
        donor_raw=np.stack(donor_raw).astype(np.float32),
        recipient_raw=np.stack(recipient_raw).astype(np.float32),
        post_raw=np.stack(post_raw).astype(np.float32),
        metadata=pd.DataFrame(rows),
        feature_ids=list(dataset.feature_ids),
    )


def make_loader(inputs: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_dataset = TensorDataset(
        torch.from_numpy(inputs.astype(np.float32)),
        torch.from_numpy(targets.astype(np.float32)),
    )
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


def transformed_to_raw(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32), -2.0, 2.0)
    return np.power(10.0, clipped - 2.0).astype(np.float32)


def finalize_raw_predictions(
    values: np.ndarray,
    raw_zero_threshold: float = 5e-5,
    renormalize: bool = True,
) -> np.ndarray:
    matrix = np.clip(values.astype(np.float32), 0.0, None)
    if raw_zero_threshold > 0:
        matrix[matrix <= raw_zero_threshold] = 0.0
    if renormalize:
        row_sum = matrix.sum(axis=1, keepdims=True)
        normalized = np.zeros_like(matrix, dtype=np.float32)
        np.divide(matrix, np.clip(row_sum, 1e-12, None), out=normalized, where=row_sum > 0)
        matrix = normalized
    return matrix.astype(np.float32)


def bray_curtis(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    denom = pred.sum(axis=1) + true.sum(axis=1)
    return np.abs(pred - true).sum(axis=1) / np.clip(denom, 1e-12, None)


def cosine(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
    return (pred * true).sum(axis=1) / np.clip(denom, 1e-12, None)


def train_model(
    train_arrays: FMTMLPArrays,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: FMTMLPConfig,
    device: torch.device,
) -> Tuple[FMTResidualMLP, Standardizer, Standardizer, Dict[str, float]]:
    target_delta = train_arrays.targets - train_arrays.recipient_transformed
    input_scaler = Standardizer.fit(train_arrays.inputs[train_idx])
    delta_scaler = Standardizer.fit(target_delta[train_idx])
    train_loader = make_loader(
        input_scaler.transform(train_arrays.inputs[train_idx]),
        delta_scaler.transform(target_delta[train_idx]),
        config.batch_size,
        True,
    )
    val_loader = make_loader(
        input_scaler.transform(train_arrays.inputs[val_idx]),
        delta_scaler.transform(target_delta[val_idx]),
        config.batch_size,
        False,
    )

    model = FMTResidualMLP(
        input_dim=train_arrays.inputs.shape[1],
        output_dim=train_arrays.targets.shape[1],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    stale = 0
    epochs_trained = 0

    for epoch in range(config.max_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            loss = criterion(model(input_batch), target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                val_losses.append(float(criterion(model(input_batch), target_batch).item()))
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
    return model, input_scaler, delta_scaler, {"best_val_loss": best_val, "epochs_trained": epochs_trained}


def predict_raw(
    model: FMTResidualMLP,
    arrays: FMTMLPArrays,
    input_scaler: Standardizer,
    delta_scaler: Standardizer,
    config: FMTMLPConfig,
    device: torch.device,
) -> np.ndarray:
    loader = make_loader(input_scaler.transform(arrays.inputs), np.zeros_like(arrays.targets), config.batch_size, False)
    chunks = []
    model.eval()
    with torch.no_grad():
        for input_batch, _ in loader:
            chunks.append(model(input_batch.to(device)).cpu().numpy())
    pred_delta = delta_scaler.inverse_transform(np.concatenate(chunks, axis=0))
    pred_transformed = arrays.recipient_transformed + pred_delta
    pred_raw = transformed_to_raw(pred_transformed)
    return finalize_raw_predictions(
        pred_raw,
        raw_zero_threshold=config.raw_zero_threshold,
        renormalize=config.renormalize_predictions,
    )


def _string_array(values: Iterable[Any]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=object)


def write_outputs(output_dir: Path, arrays: FMTMLPArrays, pred_raw: np.ndarray, config: FMTMLPConfig, history: Dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = arrays.metadata.copy()
    metrics["mlp_bray_curtis"] = bray_curtis(pred_raw, arrays.post_raw)
    metrics["recipient_baseline_bray_curtis"] = bray_curtis(arrays.recipient_raw, arrays.post_raw)
    metrics["mlp_cosine"] = cosine(pred_raw, arrays.post_raw)
    metrics["recipient_baseline_cosine"] = cosine(arrays.recipient_raw, arrays.post_raw)
    metrics["best_val_loss"] = history["best_val_loss"]
    metrics["epochs_trained"] = history["epochs_trained"]
    metrics.to_csv(output_dir / "predicted_abundance.metrics.csv", index=False)
    summary = metrics[
        [
            "mlp_bray_curtis",
            "recipient_baseline_bray_curtis",
            "mlp_cosine",
            "recipient_baseline_cosine",
        ]
    ].mean()
    summary.to_csv(output_dir / "summary.csv", header=["mean"])

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_dir / "predicted_abundance.h5", "w") as handle:
        handle.attrs["prediction_space"] = "raw_relative_abundance"
        handle.create_dataset("feature_ids", data=_string_array(arrays.feature_ids), dtype=string_dtype)
        meta_group = handle.create_group("metadata")
        for column in [
            "target_id",
            "fmt_id",
            "study",
            "response_label",
            "donor_sample_id",
            "recipient_sample_id",
            "post_sample_id",
        ]:
            meta_group.create_dataset(column, data=_string_array(arrays.metadata[column]), dtype=string_dtype)
        meta_group.create_dataset("timepoint_fmt", data=arrays.metadata["timepoint_fmt"].astype(np.float32).to_numpy())
        pred_group = handle.create_group("predictions")
        pred_group.create_dataset("mlp_post_raw", data=pred_raw.astype(np.float32), compression="gzip")
        pred_group.create_dataset("true_post_raw", data=arrays.post_raw.astype(np.float32), compression="gzip")
        pred_group.create_dataset("recipient_raw", data=arrays.recipient_raw.astype(np.float32), compression="gzip")
        pred_group.create_dataset("donor_raw", data=arrays.donor_raw.astype(np.float32), compression="gzip")

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
    print(summary.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fully connected FMT baseline using donor and recipient communities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_dir = Path("data/downstreams/FMT")
    parser.add_argument("--train_data_file", type=Path, default=data_dir / "community_train.pkl")
    parser.add_argument("--train_metadata_file", type=Path, default=data_dir / "post_train_meta_split.csv")
    parser.add_argument("--test_data_file", type=Path, default=data_dir / "community_test.pkl")
    parser.add_argument("--test_metadata_file", type=Path, default=data_dir / "post_test_meta_split.csv")
    parser.add_argument("--ntv3_embedding_path", type=Path, default=data_dir / "community_train.resolved_embeddings.h5")
    parser.add_argument("--test_ntv3_embedding_path", type=Path, default=data_dir / "community_test.resolved_embeddings.h5")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_downstreams/FMT/mlp_baseline"))
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--val_group_column", type=str, default="fmt_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--raw_zero_threshold", type=float, default=5e-5)
    parser.add_argument("--renormalize_predictions", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    config = FMTMLPConfig(
        train_data_file=str(args.train_data_file),
        train_metadata_file=str(args.train_metadata_file),
        test_data_file=str(args.test_data_file),
        test_metadata_file=str(args.test_metadata_file),
        ntv3_embedding_path=str(args.ntv3_embedding_path),
        test_ntv3_embedding_path=str(args.test_ntv3_embedding_path),
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
        val_group_column=args.val_group_column,
        seed=args.seed,
        device=str(device),
        raw_zero_threshold=args.raw_zero_threshold,
        renormalize_predictions=args.renormalize_predictions,
    )

    train_dataset = FMTCommunityPredictionDataset(args.train_data_file, args.train_metadata_file, args.ntv3_embedding_path)
    test_dataset = FMTCommunityPredictionDataset(args.test_data_file, args.test_metadata_file, args.test_ntv3_embedding_path)
    train_arrays = build_arrays(train_dataset)
    test_arrays = build_arrays(test_dataset)
    train_idx, val_idx = grouped_train_val_indices(
        train_dataset.metadata,
        args.val_ratio,
        args.seed,
        group_column=args.val_group_column,
    )
    print(f"FMT MLP targets: train={len(train_idx)} val={len(val_idx)} test={len(test_dataset)}")
    print(f"Features: {len(train_dataset.feature_ids)} | input dim: {train_arrays.inputs.shape[1]} | device: {device}")

    model, input_scaler, delta_scaler, history = train_model(train_arrays, train_idx, val_idx, config, device)
    pred_raw = predict_raw(model, test_arrays, input_scaler, delta_scaler, config, device)
    write_outputs(args.output_dir, test_arrays, pred_raw, config, history)
    print(f"Wrote FMT MLP baseline results to {args.output_dir}")


if __name__ == "__main__":
    main()
