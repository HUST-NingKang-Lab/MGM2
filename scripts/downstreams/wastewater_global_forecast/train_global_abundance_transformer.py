#!/usr/bin/env python3
"""Train a global abundance-only temporal baseline without persistence residuals."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.data.WastewaterDataset import load_wastewater_windows
from src.metrics.wastewater import aggregate_forecast_metrics, compute_forecast_metrics, normalize_abundance
from src.models.WastewaterModel import bray_curtis_loss_torch


DEFAULT_DATASET = Path("data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl")
DEFAULT_OUTPUT_DIR = Path("outputs_downstreams/wastewater_global_forecast/abundance_transformer_top200")


class GlobalAbundanceTransformer(nn.Module):
    """A single global temporal model over abundance vectors, with softmax outputs."""

    def __init__(
        self,
        num_asvs: int,
        horizon: int,
        context_len: int,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_asvs <= 0:
            raise ValueError("num_asvs must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if context_len <= 0:
            raise ValueError("context_len must be positive")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_asvs = int(num_asvs)
        self.horizon = int(horizon)
        self.context_len = int(context_len)
        self.input_projection = nn.Linear(num_asvs, hidden_size)
        self.context_position = nn.Parameter(torch.zeros(1, context_len, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.horizon_embedding = nn.Embedding(horizon, hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_asvs),
        )

    def forward(self, context_abundance: torch.Tensor) -> torch.Tensor:
        if context_abundance.ndim != 3:
            raise ValueError("context_abundance must have shape [batch, context_len, num_asvs]")
        if context_abundance.shape[1] != self.context_len:
            raise ValueError(f"expected context_len={self.context_len}, got {context_abundance.shape[1]}")
        if context_abundance.shape[2] != self.num_asvs:
            raise ValueError(f"expected num_asvs={self.num_asvs}, got {context_abundance.shape[2]}")

        context_abundance = torch.nan_to_num(context_abundance.float(), nan=0.0, posinf=0.0, neginf=0.0)
        context_abundance = torch.clamp(context_abundance, min=0.0)
        totals = context_abundance.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        context_abundance = context_abundance / totals

        tokens = self.input_projection(context_abundance) + self.context_position
        encoded = self.encoder(tokens)
        summary = encoded[:, -1, :]
        horizon_ids = torch.arange(self.horizon, device=context_abundance.device)
        future = summary[:, None, :] + self.horizon_embedding(horizon_ids)[None, :, :]
        logits = self.head(future)
        return F.softmax(logits, dim=-1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plants", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=["test"])
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_tail_val_ratio", type=float, default=0.15)
    parser.add_argument("--loss_mse_weight", type=float, default=0.1)
    parser.add_argument("--top_k_values", nargs="*", type=int, default=[10, 20, 50])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--limit_train_windows", type=int, default=0)
    return parser.parse_args()


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collect_records(dataset: Dict, plants: Iterable[str] | None):
    selected = set(plants) if plants else None
    records = []
    for window in dataset["windows"]:
        plant = window["plant"]
        if selected is not None and plant not in selected:
            continue
        records.append(
            {
                "plant": plant,
                "split": window["split"],
                "context": normalize_abundance(window["context_abundance"]).astype(np.float32),
                "target": normalize_abundance(window["target_abundance"]).astype(np.float32),
                "target_sample_ids": window["target_sample_ids"],
                "target_dates": window["target_dates"],
                "asv_ids": window["asv_ids"],
            }
        )
    return records


def _split_train_tail_by_plant(records: List[Dict], val_ratio: float, limit_train_windows: int = 0):
    train_records = []
    holdout_records = []
    for plant in sorted({record["plant"] for record in records}):
        plant_train = [record for record in records if record["plant"] == plant and record["split"] == "train"]
        if limit_train_windows > 0:
            plant_train = plant_train[:limit_train_windows]
        n_val = max(1, int(round(len(plant_train) * val_ratio))) if len(plant_train) > 1 else 0
        n_val = min(n_val, max(0, len(plant_train) - 1))
        if n_val == 0:
            train_records.extend(plant_train)
        else:
            train_records.extend(plant_train[:-n_val])
            holdout_records.extend(plant_train[-n_val:])
    return train_records, holdout_records


def _tensor_dataset(records: List[Dict]):
    x = torch.from_numpy(np.stack([record["context"] for record in records]).astype(np.float32))
    y = torch.from_numpy(np.stack([record["target"] for record in records]).astype(np.float32))
    return TensorDataset(x, y)


def _combined_loss(pred: torch.Tensor, target: torch.Tensor, mse_weight: float) -> torch.Tensor:
    return bray_curtis_loss_torch(pred, target) + float(mse_weight) * F.mse_loss(pred, target)


def _evaluate_loss(model, records: List[Dict], args, device) -> float:
    if not records:
        return float("inf")
    loader = DataLoader(_tensor_dataset(records), batch_size=args.batch_size, shuffle=False)
    losses = []
    model.eval()
    with torch.no_grad():
        for context, target in loader:
            pred = model(context.to(device))
            losses.append(float(_combined_loss(pred, target.to(device), args.loss_mse_weight).item()))
    return float(np.mean(losses))


def _predict_records(model, records: List[Dict], args, device):
    if not records:
        return None
    loader = DataLoader(_tensor_dataset(records), batch_size=args.batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for context, _ in loader:
            preds.append(model(context.to(device)).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def _prediction_rows(record_group: Dict):
    asv_ids = record_group["asv_ids"]
    for window_idx in range(record_group["y_pred"].shape[0]):
        for horizon_idx in range(record_group["y_pred"].shape[1]):
            for asv_id, y_true, y_pred in zip(
                asv_ids,
                record_group["y_true"][window_idx, horizon_idx],
                record_group["y_pred"][window_idx, horizon_idx],
            ):
                yield {
                    "model": record_group["model"],
                    "plant": record_group["plant"],
                    "split": record_group["split"],
                    "window_idx": window_idx,
                    "horizon": horizon_idx + 1,
                    "sample_id": record_group["target_sample_ids"][window_idx][horizon_idx],
                    "date": record_group["target_dates"][window_idx][horizon_idx],
                    "asv_id": asv_id,
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                }


def main():
    args = parse_args()
    _set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dataset = load_wastewater_windows(args.dataset)
    records = _collect_records(dataset, args.plants)
    if not records:
        raise ValueError("no records found")

    train_records, holdout_records = _split_train_tail_by_plant(records, args.train_tail_val_ratio, args.limit_train_windows)
    if not train_records:
        raise ValueError("no train records found")

    context_len, num_asvs = train_records[0]["context"].shape
    horizon = train_records[0]["target"].shape[0]
    model = GlobalAbundanceTransformer(
        num_asvs=num_asvs,
        horizon=horizon,
        context_len=context_len,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(_tensor_dataset(train_records), batch_size=args.batch_size, shuffle=True)

    best_state = None
    best_holdout = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_losses = []
        for context, target in train_loader:
            pred = model(context.to(device))
            loss = _combined_loss(pred, target.to(device), args.loss_mse_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_records = holdout_records or train_records
        holdout_loss = _evaluate_loss(model, val_records, args, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "train_tail_holdout_loss": holdout_loss})
        print(f"epoch={epoch} train={history[-1]['train_loss']:.4f} holdout={holdout_loss:.4f}")
        if holdout_loss < best_holdout:
            best_holdout = holdout_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "history": history,
            "best_train_tail_holdout_loss": best_holdout,
        },
        output_dir / "checkpoint.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "global_abundance_transformer",
                "train_windows": len(train_records),
                "train_tail_holdout_windows": len(holdout_records),
                "best_train_tail_holdout_loss": best_holdout,
                "epochs": len(history),
                "loss_mse_weight": args.loss_mse_weight,
                "abundance_head": "softmax",
                "residual_persistence": False,
            }
        ]
    ).to_csv(output_dir / "train_summary.csv", index=False)

    eval_records = []
    wanted_splits = set(args.splits)
    for split in args.splits:
        split_records = [record for record in records if record["split"] == split]
        if not split_records:
            continue
        pred = _predict_records(model, split_records, args, device)
        for plant in sorted({record["plant"] for record in split_records}):
            plant_records = [record for record in split_records if record["plant"] == plant]
            indices = [idx for idx, record in enumerate(split_records) if record["plant"] == plant]
            eval_records.append(
                {
                    "model": "global_abundance_transformer",
                    "plant": plant,
                    "split": split,
                    "asv_ids": plant_records[0]["asv_ids"],
                    "target_sample_ids": [record["target_sample_ids"] for record in plant_records],
                    "target_dates": [record["target_dates"] for record in plant_records],
                    "y_true": np.stack([record["target"] for record in plant_records]).astype(np.float32),
                    "y_pred": pred[indices].astype(np.float32),
                }
            )
    if not eval_records:
        raise ValueError(f"no evaluation records found for splits: {sorted(wanted_splits)}")

    summary_rows = []
    for record in eval_records:
        metrics = compute_forecast_metrics(record["y_true"], record["y_pred"], top_k_values=args.top_k_values)
        summary_rows.append(
            {
                "model": record["model"],
                "plant": record["plant"],
                "split": record["split"],
                "n_windows": int(record["y_true"].shape[0]),
                "horizon": int(record["y_true"].shape[1]),
                "n_asvs": int(record["y_true"].shape[2]),
                **metrics,
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)

    horizon_frames = []
    for record in eval_records:
        frame = aggregate_forecast_metrics([record], top_k_values=args.top_k_values)
        frame.insert(0, "model", record["model"])
        horizon_frames.append(frame)
    pd.concat(horizon_frames, ignore_index=True).to_csv(output_dir / "metrics_by_horizon.csv", index=False)

    if args.save_predictions:
        rows = []
        for record in eval_records:
            rows.extend(_prediction_rows(record))
        pd.DataFrame(rows).to_csv(output_dir / "predictions.csv.gz", index=False, compression="gzip")

    print(f"Saved summary: {output_dir / 'summary.csv'}")
    print(f"Saved horizon metrics: {output_dir / 'metrics_by_horizon.csv'}")
    print(f"Saved train summary: {output_dir / 'train_summary.csv'}")


if __name__ == "__main__":
    main()
