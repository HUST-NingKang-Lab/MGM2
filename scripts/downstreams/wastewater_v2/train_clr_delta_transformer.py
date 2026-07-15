#!/usr/bin/env python3
"""Train a clean V2 CLR-delta Transformer for wastewater ASV forecasting.

This script intentionally does not reuse the previous wastewater world-model
classes. It uses frozen per-sample MGM2 CLS embeddings as context states and
predicts future compositional changes in CLR/log-ratio space.
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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
DEFAULT_CACHE = Path("data/downstreams/wastewater_v2/embeddings/mgm2_scale_xlarge_top200_cls.pkl")
DEFAULT_OUTPUT_DIR = Path("outputs_downstreams/wastewater_v2/clr_delta_transformer_scale_xlarge")


def normalize_torch(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    values = torch.clamp(values, min=0.0)
    totals = values.sum(dim=-1, keepdim=True)
    return torch.where(totals > eps, values / totals.clamp_min(eps), torch.zeros_like(values))


def clr_torch(values: torch.Tensor, pseudocount: float = 1e-4) -> torch.Tensor:
    values = normalize_torch(values)
    smoothed = values + float(pseudocount)
    smoothed = smoothed / smoothed.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_values = torch.log(smoothed.clamp_min(1e-12))
    return log_values - log_values.mean(dim=-1, keepdim=True)


def pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred.reshape(-1).float()
    target = target.reshape(-1).float()
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.norm() * target.norm()
    if float(denom.detach().cpu()) <= eps:
        return pred.new_tensor(0.0)
    corr = torch.sum(pred * target) / denom.clamp_min(eps)
    return 1.0 - corr


class PlantSpecificCLRDeltaTransformer(nn.Module):
    """Shared attention encoder with separate plant output heads.

    The separate output heads avoid treating column 17 from different plants as
    the same biological ASV. The shared encoder still learns common temporal
    dynamics from abundance trajectories and MGM2 CLS states.
    """

    def __init__(
        self,
        num_asvs: int,
        context_len: int,
        horizon: int,
        cls_dim: int,
        num_plants: int,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        pseudocount: float = 1e-4,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_asvs = int(num_asvs)
        self.context_len = int(context_len)
        self.horizon = int(horizon)
        self.cls_dim = int(cls_dim)
        self.num_plants = int(num_plants)
        self.pseudocount = float(pseudocount)

        self.clr_projection = nn.Linear(num_asvs, hidden_size)
        self.raw_projection = nn.Linear(num_asvs, hidden_size)
        self.cls_projection = nn.Linear(cls_dim, hidden_size)
        self.plant_embedding = nn.Embedding(num_plants, hidden_size)
        self.context_position = nn.Parameter(torch.zeros(1, context_len, hidden_size))
        self.horizon_embedding = nn.Embedding(horizon, hidden_size)

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
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, num_asvs),
                )
                for _ in range(num_plants)
            ]
        )

    def forward(self, context_abundance: torch.Tensor, context_cls: torch.Tensor, plant_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        if context_abundance.ndim != 3:
            raise ValueError("context_abundance must have shape [batch, context_len, num_asvs]")
        if context_cls.ndim != 3:
            raise ValueError("context_cls must have shape [batch, context_len, cls_dim]")
        context_abundance = normalize_torch(context_abundance)
        context_clr = clr_torch(context_abundance, self.pseudocount)
        plant_state = self.plant_embedding(plant_id)

        tokens = (
            self.clr_projection(context_clr)
            + self.raw_projection(context_abundance)
            + self.cls_projection(context_cls)
            + plant_state[:, None, :]
            + self.context_position
        )
        memory = self.encoder(tokens)
        horizon_ids = torch.arange(self.horizon, device=context_abundance.device)
        query = (
            memory[:, -1:, :]
            + plant_state[:, None, :]
            + self.horizon_embedding(horizon_ids)[None, :, :]
        )
        decoded = self.decoder(query, memory)

        raw_delta = decoded.new_zeros((decoded.shape[0], self.horizon, self.num_asvs))
        for pid in torch.unique(plant_id).tolist():
            mask = plant_id == int(pid)
            raw_delta[mask] = self.output_heads[int(pid)](decoded[mask])

        last_clr = context_clr[:, -1:, :]
        pred_clr = last_clr + raw_delta
        abundance = F.softmax(pred_clr, dim=-1)
        return {"abundance": abundance, "pred_delta_clr": raw_delta, "pred_clr": pred_clr}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--embedding_cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plants", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=["test"])
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pseudocount", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_tail_val_ratio", type=float, default=0.15)
    parser.add_argument("--loss_mse_weight", type=float, default=0.1)
    parser.add_argument("--loss_clr_delta_weight", type=float, default=0.05)
    parser.add_argument("--loss_corr_weight", type=float, default=0.05)
    parser.add_argument("--loss_direction_weight", type=float, default=0.02)
    parser.add_argument("--direction_delta_threshold", type=float, default=1e-4)
    parser.add_argument("--top_k_values", nargs="*", type=int, default=[10, 20, 50])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--limit_train_windows", type=int, default=0)
    parser.add_argument("--limit_eval_windows", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cls_cache(path: Path) -> Dict:
    with Path(path).open("rb") as handle:
        cache = pickle.load(handle)
    required = {"sample_to_idx", "sample_keys", "cls_embeddings", "scale", "hidden_size"}
    missing = required - set(cache)
    if missing:
        raise ValueError(f"embedding cache missing keys: {sorted(missing)}")
    cache["cls_embeddings"] = np.asarray(cache["cls_embeddings"], dtype=np.float32)
    return cache


def collect_records(dataset: Dict, cache: Dict, plants: Iterable[str] | None):
    selected = set(plants) if plants else None
    sample_to_idx = cache["sample_to_idx"]
    cls_embeddings = cache["cls_embeddings"]
    records = []
    plant_names = sorted(dataset["plants"])
    plant_to_id = {plant: idx for idx, plant in enumerate(plant_names)}
    missing_keys = []
    num_asvs = None

    for window in dataset["windows"]:
        plant = window["plant"]
        if selected is not None and plant not in selected:
            continue
        asv_ids = list(window["asv_ids"])
        if num_asvs is None:
            num_asvs = len(asv_ids)
        elif len(asv_ids) != num_asvs:
            raise ValueError("all V2 windows must have the same ASV count; use the top-k dataset")

        context_cls = []
        for sample_id in window["context_sample_ids"]:
            key = (plant, sample_id)
            if key not in sample_to_idx:
                missing_keys.append(key)
                continue
            context_cls.append(cls_embeddings[sample_to_idx[key]])
        if len(context_cls) != len(window["context_sample_ids"]):
            continue

        records.append(
            {
                "plant": plant,
                "plant_id": plant_to_id[plant],
                "split": window["split"],
                "context": normalize_abundance(window["context_abundance"]).astype(np.float32),
                "target": normalize_abundance(window["target_abundance"]).astype(np.float32),
                "context_cls": np.stack(context_cls).astype(np.float32),
                "target_sample_ids": window["target_sample_ids"],
                "target_dates": window["target_dates"],
                "asv_ids": asv_ids,
            }
        )
    if missing_keys:
        preview = ", ".join(map(str, missing_keys[:5]))
        raise KeyError(f"missing {len(missing_keys)} context sample embeddings, e.g. {preview}")
    return records, plant_to_id


def split_train_tail_by_plant(records: List[Dict], val_ratio: float, limit_train_windows: int = 0):
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


def tensor_dataset(records: List[Dict]):
    context = torch.from_numpy(np.stack([record["context"] for record in records]).astype(np.float32))
    target = torch.from_numpy(np.stack([record["target"] for record in records]).astype(np.float32))
    context_cls = torch.from_numpy(np.stack([record["context_cls"] for record in records]).astype(np.float32))
    plant_id = torch.tensor([record["plant_id"] for record in records], dtype=torch.long)
    return TensorDataset(context, target, context_cls, plant_id)


def combined_loss(outputs: Dict[str, torch.Tensor], context: torch.Tensor, target: torch.Tensor, args) -> Dict[str, torch.Tensor]:
    pred = outputs["abundance"]
    target = normalize_torch(target)
    context = normalize_torch(context)
    target_delta_clr = clr_torch(target, args.pseudocount) - clr_torch(context[:, -1:, :], args.pseudocount)

    bray = bray_curtis_loss_torch(pred, target)
    mse = F.mse_loss(pred, target)
    clr_delta = F.mse_loss(outputs["pred_delta_clr"], target_delta_clr)
    corr = pearson_loss(outputs["pred_delta_clr"], target_delta_clr)

    target_abundance_delta = target - context[:, -1:, :]
    direction_mask = target_abundance_delta.abs() > float(args.direction_delta_threshold)
    if bool(direction_mask.any().detach().cpu()):
        direction_target = (target_abundance_delta[direction_mask] > 0).float()
        direction_logits = outputs["pred_delta_clr"][direction_mask]
        direction = F.binary_cross_entropy_with_logits(direction_logits, direction_target)
    else:
        direction = pred.new_tensor(0.0)

    loss = (
        bray
        + float(args.loss_mse_weight) * mse
        + float(args.loss_clr_delta_weight) * clr_delta
        + float(args.loss_corr_weight) * corr
        + float(args.loss_direction_weight) * direction
    )
    return {
        "loss": loss,
        "bray_loss": bray.detach(),
        "mse_loss": mse.detach(),
        "clr_delta_loss": clr_delta.detach(),
        "corr_loss": corr.detach(),
        "direction_loss": direction.detach(),
    }


def evaluate_loss(model, records: List[Dict], args, device) -> float:
    if not records:
        return float("inf")
    loader = DataLoader(tensor_dataset(records), batch_size=args.batch_size, shuffle=False)
    losses = []
    model.eval()
    with torch.no_grad():
        for context, target, context_cls, plant_id in loader:
            context = context.to(device)
            target = target.to(device)
            outputs = model(context, context_cls.to(device), plant_id.to(device))
            losses.append(float(combined_loss(outputs, context, target, args)["loss"].item()))
    return float(np.mean(losses))


def predict_records(model, records: List[Dict], args, device):
    if not records:
        return None
    loader = DataLoader(tensor_dataset(records), batch_size=args.batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for context, _, context_cls, plant_id in loader:
            outputs = model(context.to(device), context_cls.to(device), plant_id.to(device))
            preds.append(outputs["abundance"].detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def safe_corr(x: np.ndarray, y: np.ndarray, min_std: float = 1e-8) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    x = x[valid]
    y = y[valid]
    if np.std(x) < min_std or np.std(y) < min_std:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def change_metrics(context: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, direction_threshold: float = 1e-4) -> Dict[str, float]:
    context = normalize_abundance(context)
    true = normalize_abundance(y_true)
    pred = normalize_abundance(y_pred)
    last = context[:, -1:, :]
    true_delta = true - last
    pred_delta = pred - last
    flat_true = true_delta.reshape(-1)
    flat_pred = pred_delta.reshape(-1)
    direction_mask = np.abs(flat_true) > direction_threshold
    rows = {
        "abundance_delta_pearson": safe_corr(flat_true, flat_pred),
        "abundance_delta_mae": float(np.mean(np.abs(flat_pred - flat_true))),
        "direction_accuracy": float(np.mean((flat_pred[direction_mask] > 0) == (flat_true[direction_mask] > 0)))
        if np.any(direction_mask)
        else float("nan"),
        "direction_n": int(np.sum(direction_mask)),
    }

    per_asv = []
    asv_std = np.std(true_delta.reshape(-1, true_delta.shape[-1]), axis=0)
    for asv_idx in range(true_delta.shape[-1]):
        corr = safe_corr(true_delta[..., asv_idx], pred_delta[..., asv_idx])
        if not math.isnan(corr):
            per_asv.append((asv_idx, float(asv_std[asv_idx]), corr))
    if per_asv:
        rows["mean_asv_delta_pearson"] = float(np.mean([item[2] for item in per_asv]))
        rows["median_asv_delta_pearson"] = float(np.median([item[2] for item in per_asv]))
        top = sorted(per_asv, key=lambda item: item[1], reverse=True)[: min(50, len(per_asv))]
        rows["top_variable_asv_delta_pearson"] = float(np.mean([item[2] for item in top]))
        rows["n_asv_delta_pearson"] = int(len(per_asv))
    else:
        rows["mean_asv_delta_pearson"] = float("nan")
        rows["median_asv_delta_pearson"] = float("nan")
        rows["top_variable_asv_delta_pearson"] = float("nan")
        rows["n_asv_delta_pearson"] = 0
    return rows


def prediction_rows(record_group: Dict):
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
                    "scale": record_group["scale"],
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
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dataset = load_wastewater_windows(args.dataset)
    cache = load_cls_cache(args.embedding_cache)
    records, plant_to_id = collect_records(dataset, cache, args.plants)
    if not records:
        raise ValueError("no records found")

    train_records, holdout_records = split_train_tail_by_plant(records, args.train_tail_val_ratio, args.limit_train_windows)
    if not train_records:
        raise ValueError("no train records found")

    context_len, num_asvs = train_records[0]["context"].shape
    horizon = train_records[0]["target"].shape[0]
    cls_dim = train_records[0]["context_cls"].shape[-1]
    model = PlantSpecificCLRDeltaTransformer(
        num_asvs=num_asvs,
        context_len=context_len,
        horizon=horizon,
        cls_dim=cls_dim,
        num_plants=len(plant_to_id),
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pseudocount=args.pseudocount,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(tensor_dataset(train_records), batch_size=args.batch_size, shuffle=True)

    best_state = None
    best_holdout = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_logs = defaultdict(list)
        for context, target, context_cls, plant_id in train_loader:
            context = context.to(device)
            target = target.to(device)
            outputs = model(context, context_cls.to(device), plant_id.to(device))
            losses = combined_loss(outputs, context, target, args)
            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in losses.items():
                epoch_logs[key].append(float(value.detach().cpu()))

        val_records = holdout_records or train_records
        holdout_loss = evaluate_loss(model, val_records, args, device)
        row = {"epoch": epoch, "train_tail_holdout_loss": holdout_loss}
        row.update({f"train_{key}": float(np.mean(values)) for key, values in epoch_logs.items()})
        history.append(row)
        print(
            f"epoch={epoch} train={row['train_loss']:.4f} holdout={holdout_loss:.4f} "
            f"bc={row['train_bray_loss']:.4f} clr={row['train_clr_delta_loss']:.4f}"
        )
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
    pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "plant_to_id": plant_to_id,
            "best_train_tail_holdout_loss": best_holdout,
            "embedding_cache": str(args.embedding_cache),
            "scale": cache.get("scale", "unknown"),
        },
        output_dir / "checkpoint.pt",
    )
    pd.DataFrame(
        [
            {
                "model": "v2_clr_delta_transformer",
                "scale": cache.get("scale", "unknown"),
                "train_windows": len(train_records),
                "train_tail_holdout_windows": len(holdout_records),
                "best_train_tail_holdout_loss": best_holdout,
                "epochs": len(history),
                "num_plants": len(plant_to_id),
                "num_asvs_per_plant": num_asvs,
                "cls_dim": cls_dim,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
            }
        ]
    ).to_csv(output_dir / "train_summary.csv", index=False)

    eval_records = []
    for split in args.splits:
        split_records = [record for record in records if record["split"] == split]
        if args.limit_eval_windows > 0:
            limited = []
            for plant in sorted({record["plant"] for record in split_records}):
                limited.extend([record for record in split_records if record["plant"] == plant][: args.limit_eval_windows])
            split_records = limited
        if not split_records:
            continue
        pred = predict_records(model, split_records, args, device)
        for plant in sorted({record["plant"] for record in split_records}):
            plant_records = [record for record in split_records if record["plant"] == plant]
            indices = [idx for idx, record in enumerate(split_records) if record["plant"] == plant]
            eval_records.append(
                {
                    "model": "v2_clr_delta_transformer",
                    "scale": cache.get("scale", "unknown"),
                    "plant": plant,
                    "split": split,
                    "asv_ids": plant_records[0]["asv_ids"],
                    "target_sample_ids": [record["target_sample_ids"] for record in plant_records],
                    "target_dates": [record["target_dates"] for record in plant_records],
                    "context": np.stack([record["context"] for record in plant_records]).astype(np.float32),
                    "y_true": np.stack([record["target"] for record in plant_records]).astype(np.float32),
                    "y_pred": pred[indices].astype(np.float32),
                }
            )
    if not eval_records:
        raise ValueError(f"no evaluation records found for splits: {args.splits}")

    summary_rows = []
    for record in eval_records:
        metrics = compute_forecast_metrics(record["y_true"], record["y_pred"], top_k_values=args.top_k_values)
        metrics.update(change_metrics(record["context"], record["y_true"], record["y_pred"], args.direction_delta_threshold))
        summary_rows.append(
            {
                "model": record["model"],
                "scale": record["scale"],
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
        frame.insert(1, "scale", record["scale"])
        horizon_frames.append(frame)
    pd.concat(horizon_frames, ignore_index=True).to_csv(output_dir / "metrics_by_horizon.csv", index=False)

    if args.save_predictions:
        rows = []
        for record in eval_records:
            rows.extend(prediction_rows(record))
        pd.DataFrame(rows).to_csv(output_dir / "predictions.csv.gz", index=False, compression="gzip")

    print(f"Saved summary: {output_dir / 'summary.csv'}")
    print(f"Saved horizon metrics: {output_dir / 'metrics_by_horizon.csv'}")
    print(f"Saved train summary: {output_dir / 'train_summary.csv'}")


if __name__ == "__main__":
    main()
