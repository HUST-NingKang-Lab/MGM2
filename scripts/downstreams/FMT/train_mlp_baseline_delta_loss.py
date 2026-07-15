#!/usr/bin/env python3
"""Exploratory MLP baseline for FMT post-community prediction with delta loss."""

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch
import torch.nn as nn

from src.data.FMTDataset import FMTCommunityPredictionDataset, grouped_train_val_indices
from train_mlp_baseline import (
    FMTMLPArrays,
    FMTResidualMLP,
    Standardizer,
    build_arrays,
    finalize_raw_predictions,
    make_loader,
    transformed_to_raw,
    write_outputs,
)


@dataclass
class FMTMLPDeltaLossConfig:
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
    delta_loss_weight: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scaled_delta_to_transformed(
    scaled_delta: torch.Tensor,
    delta_mean: torch.Tensor,
    delta_std: torch.Tensor,
) -> torch.Tensor:
    return scaled_delta * delta_std + delta_mean


def _combined_loss(
    pred_scaled_delta: torch.Tensor,
    target_scaled_delta: torch.Tensor,
    recipient_transformed: torch.Tensor,
    post_transformed: torch.Tensor,
    delta_mean: torch.Tensor,
    delta_std: torch.Tensor,
    delta_loss_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_delta = _scaled_delta_to_transformed(pred_scaled_delta, delta_mean, delta_std)
    true_delta = post_transformed - recipient_transformed
    pred_post = recipient_transformed + pred_delta
    post_loss = nn.functional.mse_loss(pred_post, post_transformed)
    delta_loss = nn.functional.mse_loss(pred_delta, true_delta)
    scaled_delta_loss = nn.functional.mse_loss(pred_scaled_delta, target_scaled_delta)
    total = post_loss + float(delta_loss_weight) * delta_loss
    return total, post_loss, delta_loss + 0.0 * scaled_delta_loss


def make_delta_loader(arrays: FMTMLPArrays, indices: np.ndarray, input_scaler: Standardizer, delta_scaler: Standardizer, batch_size: int, shuffle: bool):
    target_delta = arrays.targets - arrays.recipient_transformed
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(input_scaler.transform(arrays.inputs[indices]).astype(np.float32)),
        torch.from_numpy(delta_scaler.transform(target_delta[indices]).astype(np.float32)),
        torch.from_numpy(arrays.recipient_transformed[indices].astype(np.float32)),
        torch.from_numpy(arrays.targets[indices].astype(np.float32)),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    train_arrays: FMTMLPArrays,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: FMTMLPDeltaLossConfig,
    device: torch.device,
) -> Tuple[FMTResidualMLP, Standardizer, Standardizer, Dict[str, float]]:
    target_delta = train_arrays.targets - train_arrays.recipient_transformed
    input_scaler = Standardizer.fit(train_arrays.inputs[train_idx])
    delta_scaler = Standardizer.fit(target_delta[train_idx])
    train_loader = make_delta_loader(train_arrays, train_idx, input_scaler, delta_scaler, config.batch_size, True)
    val_loader = make_delta_loader(train_arrays, val_idx, input_scaler, delta_scaler, config.batch_size, False)

    model = FMTResidualMLP(
        input_dim=train_arrays.inputs.shape[1],
        output_dim=train_arrays.targets.shape[1],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    delta_mean = torch.from_numpy(delta_scaler.mean.astype(np.float32)).to(device)
    delta_std = torch.from_numpy(delta_scaler.std.astype(np.float32)).to(device)
    best_state = None
    best_val = float("inf")
    best_val_post = float("inf")
    best_val_delta = float("inf")
    stale = 0
    epochs_trained = 0

    for epoch in range(config.max_epochs):
        model.train()
        for input_batch, target_batch, recipient_batch, post_batch in train_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            recipient_batch = recipient_batch.to(device)
            post_batch = post_batch.to(device)
            pred = model(input_batch)
            loss, _, _ = _combined_loss(
                pred,
                target_batch,
                recipient_batch,
                post_batch,
                delta_mean,
                delta_std,
                config.delta_loss_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        val_post_losses = []
        val_delta_losses = []
        with torch.no_grad():
            for input_batch, target_batch, recipient_batch, post_batch in val_loader:
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                recipient_batch = recipient_batch.to(device)
                post_batch = post_batch.to(device)
                pred = model(input_batch)
                loss, post_loss, delta_loss = _combined_loss(
                    pred,
                    target_batch,
                    recipient_batch,
                    post_batch,
                    delta_mean,
                    delta_std,
                    config.delta_loss_weight,
                )
                val_losses.append(float(loss.item()))
                val_post_losses.append(float(post_loss.item()))
                val_delta_losses.append(float(delta_loss.item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        val_post_loss = float(np.mean(val_post_losses)) if val_post_losses else float("inf")
        val_delta_loss = float(np.mean(val_delta_losses)) if val_delta_losses else float("inf")
        epochs_trained = epoch + 1
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_val_post = val_post_loss
            best_val_delta = val_delta_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, input_scaler, delta_scaler, {
        "best_val_loss": best_val,
        "best_val_post_loss": best_val_post,
        "best_val_delta_loss": best_val_delta,
        "epochs_trained": epochs_trained,
    }


def predict_raw(
    model: FMTResidualMLP,
    arrays: FMTMLPArrays,
    input_scaler: Standardizer,
    delta_scaler: Standardizer,
    config: FMTMLPDeltaLossConfig,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an exploratory MLP FMT baseline with post + delta loss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_dir = Path("data/downstreams/FMT")
    parser.add_argument("--train_data_file", type=Path, default=data_dir / "community_train.pkl")
    parser.add_argument("--train_metadata_file", type=Path, default=data_dir / "post_train_meta_split.csv")
    parser.add_argument("--test_data_file", type=Path, default=data_dir / "community_test.pkl")
    parser.add_argument("--test_metadata_file", type=Path, default=data_dir / "post_test_meta_split.csv")
    parser.add_argument("--ntv3_embedding_path", type=Path, default=data_dir / "community_train.resolved_embeddings.h5")
    parser.add_argument("--test_ntv3_embedding_path", type=Path, default=data_dir / "community_test.resolved_embeddings.h5")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_downstreams/FMT/mlp_baseline_exploratory/delta_loss"))
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
    parser.add_argument("--delta_loss_weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    config = FMTMLPDeltaLossConfig(
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
        delta_loss_weight=args.delta_loss_weight,
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
    print(f"FMT exploratory MLP targets: train={len(train_idx)} val={len(val_idx)} test={len(test_dataset)}")
    print(f"Features: {len(train_dataset.feature_ids)} | input dim: {train_arrays.inputs.shape[1]} | device: {device}")
    print(f"Loss: post_transformed_mse + {args.delta_loss_weight} * delta_transformed_mse")

    model, input_scaler, delta_scaler, history = train_model(train_arrays, train_idx, val_idx, config, device)
    pred_raw = predict_raw(model, test_arrays, input_scaler, delta_scaler, config, device)
    write_outputs(args.output_dir, test_arrays, pred_raw, config, history)
    with (args.output_dir / "delta_loss_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"Wrote exploratory FMT MLP baseline results to {args.output_dir}")


if __name__ == "__main__":
    main()
