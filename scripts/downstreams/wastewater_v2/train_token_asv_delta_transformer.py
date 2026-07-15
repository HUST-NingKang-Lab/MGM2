#!/usr/bin/env python3
"""Train an ASV-conditioned token model for wastewater forecasting.

This V2 model uses frozen MGM2 ASV-token states instead of only sample-level CLS
states. It predicts future CLR deltas per ASV and closes predictions back to
relative abundance with softmax.
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.data.WastewaterDataset import load_wastewater_windows
from src.metrics.wastewater import aggregate_forecast_metrics, compute_forecast_metrics, normalize_abundance
from src.models.WastewaterModel import bray_curtis_loss_torch


DEFAULT_DATASET = Path("data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl")
DEFAULT_TOKEN_CACHE = Path("data/downstreams/wastewater_v2/embeddings/mgm2_scale_xlarge_top200_tokens.pkl")
DEFAULT_OUTPUT_DIR = Path("outputs_downstreams/wastewater_v2/token_asv_delta_transformer/scale_xlarge")


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
    return 1.0 - torch.sum(pred * target) / denom.clamp_min(eps)


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sum(weights * (pred - target) ** 2) / torch.sum(weights).clamp_min(eps)


def weighted_bray_curtis_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = normalize_torch(pred, eps=eps)
    target = normalize_torch(target, eps=eps)
    weights = torch.nan_to_num(weights.float(), nan=1.0, posinf=1.0, neginf=1.0).clamp_min(0.0)
    shared = torch.minimum(pred, target) * weights
    denom = (pred + target) * weights
    loss = 1.0 - (2.0 * shared.sum(dim=-1) / denom.sum(dim=-1).clamp_min(eps))
    return loss.mean()


def per_asv_delta_corr_loss(
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    weights: torch.Tensor | None = None,
    min_std: float = 1e-5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean 1-r over ASVs, computed across batch and horizon."""
    pred = pred_delta.float().permute(2, 0, 1).reshape(pred_delta.shape[-1], -1)
    target = target_delta.float().permute(2, 0, 1).reshape(target_delta.shape[-1], -1)
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    pred_std = torch.sqrt(torch.mean(pred**2, dim=1) + eps)
    target_std = torch.sqrt(torch.mean(target**2, dim=1) + eps)
    valid = target_std > float(min_std)
    if not bool(valid.any().detach().cpu()):
        return pred_delta.new_tensor(0.0)
    corr = torch.mean(pred * target, dim=1) / (pred_std * target_std).clamp_min(eps)
    corr = corr.clamp(-1.0, 1.0)
    if weights is None:
        asv_weights = target_std.detach()
    else:
        asv_weights = weights.detach().mean(dim=(0, 1))
    asv_weights = torch.where(valid, asv_weights, torch.zeros_like(asv_weights))
    return torch.sum((1.0 - corr) * asv_weights) / torch.sum(asv_weights).clamp_min(eps)


def make_change_weights(context: torch.Tensor, target: torch.Tensor, args) -> torch.Tensor:
    """Weights ASVs with real abundance or future movement more strongly."""
    context = normalize_torch(context)
    target = normalize_torch(target)
    last = context[:, -1:, :]
    abundance_ref = torch.maximum(target, last)
    target_delta = torch.abs(target - last)
    abundance_scale = abundance_ref.mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    delta_scale = target_delta.mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    power = float(args.loss_weight_power)
    weights = torch.ones_like(target)
    if float(args.loss_weight_abundance_factor) > 0:
        abundance_term = (abundance_ref / abundance_scale).clamp_min(0.0).pow(power)
        weights = weights + float(args.loss_weight_abundance_factor) * abundance_term
    if float(args.loss_weight_delta_factor) > 0:
        delta_term = (target_delta / delta_scale).clamp_min(0.0).pow(power)
        weights = weights + float(args.loss_weight_delta_factor) * delta_term
    return weights.clamp(max=float(args.loss_weight_max)).detach()


class TokenASVDeltaTransformer(nn.Module):
    """ASV-conditioned CLR-delta forecaster.

    It first encodes each ASV's 10-step token/abundance trajectory, then applies
    ASV interaction attention separately for each future horizon.
    """

    def __init__(
        self,
        num_asvs: int,
        context_len: int,
        horizon: int,
        token_dim: int,
        num_plants: int,
        hidden_size: int = 256,
        num_heads: int = 4,
        temporal_layers: int = 2,
        asv_layers: int = 1,
        dropout: float = 0.1,
        pseudocount: float = 1e-4,
        token_bottleneck_dim: int = 0,
        pred_delta_scale: float = 1.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_asvs = int(num_asvs)
        self.context_len = int(context_len)
        self.horizon = int(horizon)
        self.token_dim = int(token_dim)
        self.num_plants = int(num_plants)
        self.pseudocount = float(pseudocount)
        self.token_bottleneck_dim = int(token_bottleneck_dim)
        self.pred_delta_scale = float(pred_delta_scale)

        if self.token_bottleneck_dim > 0:
            self.token_projection = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, self.token_bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.token_bottleneck_dim, hidden_size),
            )
        else:
            self.token_projection = nn.Linear(token_dim, hidden_size)
        self.abundance_feature_projection = nn.Linear(4, hidden_size)
        self.plant_embedding = nn.Embedding(num_plants, hidden_size)
        self.context_position = nn.Parameter(torch.zeros(1, context_len, 1, hidden_size))
        self.horizon_embedding = nn.Embedding(horizon, hidden_size)
        self.community_projection = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size))

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)

        asv_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.asv_interaction = nn.TransformerEncoder(asv_layer, num_layers=asv_layers)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, context_abundance: torch.Tensor, context_tokens: torch.Tensor, plant_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        if context_abundance.ndim != 3:
            raise ValueError("context_abundance must have shape [batch, context_len, num_asvs]")
        if context_tokens.ndim != 4:
            raise ValueError("context_tokens must have shape [batch, context_len, num_asvs, token_dim]")
        batch_size, context_len, num_asvs = context_abundance.shape
        if context_len != self.context_len or num_asvs != self.num_asvs:
            raise ValueError(f"expected context shape [*, {self.context_len}, {self.num_asvs}], got {tuple(context_abundance.shape)}")
        if context_tokens.shape[:3] != context_abundance.shape:
            raise ValueError("context_tokens and context_abundance first three dimensions must match")

        context_abundance = normalize_torch(context_abundance)
        context_clr = clr_torch(context_abundance, self.pseudocount)
        raw_delta = torch.zeros_like(context_abundance)
        clr_delta = torch.zeros_like(context_clr)
        raw_delta[:, 1:, :] = context_abundance[:, 1:, :] - context_abundance[:, :-1, :]
        clr_delta[:, 1:, :] = context_clr[:, 1:, :] - context_clr[:, :-1, :]
        abundance_features = torch.stack([context_abundance, context_clr, raw_delta, clr_delta], dim=-1)

        plant_state = self.plant_embedding(plant_id)
        tokens = (
            self.token_projection(context_tokens.float())
            + self.abundance_feature_projection(abundance_features)
            + plant_state[:, None, None, :]
            + self.context_position
        )
        temporal_input = tokens.permute(0, 2, 1, 3).reshape(batch_size * num_asvs, context_len, -1)
        temporal_encoded = self.temporal_encoder(temporal_input)
        asv_state = temporal_encoded[:, -1, :].reshape(batch_size, num_asvs, -1)

        last_abundance = context_abundance[:, -1, :]
        community_state = torch.sum(asv_state * last_abundance[:, :, None], dim=1)
        community_state = self.community_projection(community_state)

        horizon_ids = torch.arange(self.horizon, device=context_abundance.device)
        future = (
            asv_state[:, None, :, :]
            + community_state[:, None, None, :]
            + plant_state[:, None, None, :]
            + self.horizon_embedding(horizon_ids)[None, :, None, :]
        )
        future = future.reshape(batch_size * self.horizon, num_asvs, -1)
        future = self.asv_interaction(future)
        pred_delta_clr = self.delta_head(future).squeeze(-1).reshape(batch_size, self.horizon, num_asvs)
        pred_delta_clr = pred_delta_clr * self.pred_delta_scale
        pred_clr = context_clr[:, -1:, :] + pred_delta_clr
        abundance = F.softmax(pred_clr, dim=-1)
        return {"abundance": abundance, "pred_delta_clr": pred_delta_clr, "pred_clr": pred_clr}


class WastewaterTokenWindowDataset(Dataset):
    def __init__(self, records: List[Dict], token_states: np.ndarray):
        self.records = records
        self.token_states = token_states

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        token = self.token_states[np.asarray(record["context_token_indices"], dtype=np.int64)]
        return {
            "context": torch.from_numpy(record["context"]),
            "target": torch.from_numpy(record["target"]),
            "tokens": torch.from_numpy(np.asarray(token, dtype=np.float32)),
            "plant_id": torch.tensor(record["plant_id"], dtype=torch.long),
        }


def collate_batch(items: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "context": torch.stack([item["context"] for item in items], dim=0),
        "target": torch.stack([item["target"] for item in items], dim=0),
        "tokens": torch.stack([item["tokens"] for item in items], dim=0),
        "plant_id": torch.stack([item["plant_id"] for item in items], dim=0),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--token_cache", type=Path, default=DEFAULT_TOKEN_CACHE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plants", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=["test"])
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--token_bottleneck_dim", type=int, default=0)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--temporal_layers", type=int, default=2)
    parser.add_argument("--asv_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pseudocount", type=float, default=1e-4)
    parser.add_argument("--pred_delta_scale", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_tail_val_ratio", type=float, default=0.15)
    parser.add_argument("--loss_mse_weight", type=float, default=0.1)
    parser.add_argument("--loss_clr_delta_weight", type=float, default=0.05)
    parser.add_argument("--loss_corr_weight", type=float, default=0.05)
    parser.add_argument("--loss_direction_weight", type=float, default=0.02)
    parser.add_argument("--loss_weighted_bray_weight", type=float, default=0.0)
    parser.add_argument("--loss_weighted_mse_weight", type=float, default=0.0)
    parser.add_argument("--loss_asv_corr_weight", type=float, default=0.0)
    parser.add_argument("--loss_delta_l2_weight", type=float, default=0.0)
    parser.add_argument("--loss_weight_abundance_factor", type=float, default=1.0)
    parser.add_argument("--loss_weight_delta_factor", type=float, default=2.0)
    parser.add_argument("--loss_weight_power", type=float, default=0.5)
    parser.add_argument("--loss_weight_max", type=float, default=8.0)
    parser.add_argument("--asv_corr_min_std", type=float, default=1e-5)
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


def load_token_cache(path: Path) -> Dict:
    with Path(path).open("rb") as handle:
        cache = pickle.load(handle)
    required = {"sample_to_idx", "sample_keys", "token_states", "scale", "hidden_size"}
    missing = required - set(cache)
    if missing:
        raise ValueError(f"token cache missing keys: {sorted(missing)}")
    cache["token_states"] = np.asarray(cache["token_states"], dtype=np.float32)
    return cache


def collect_records(dataset: Dict, cache: Dict, plants: Iterable[str] | None):
    selected = set(plants) if plants else None
    sample_to_idx = cache["sample_to_idx"]
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

        context_indices = []
        for sample_id in window["context_sample_ids"]:
            key = (plant, sample_id)
            if key not in sample_to_idx:
                missing_keys.append(key)
                continue
            context_indices.append(int(sample_to_idx[key]))
        if len(context_indices) != len(window["context_sample_ids"]):
            continue

        records.append(
            {
                "plant": plant,
                "plant_id": plant_to_id[plant],
                "split": window["split"],
                "context": normalize_abundance(window["context_abundance"]).astype(np.float32),
                "target": normalize_abundance(window["target_abundance"]).astype(np.float32),
                "context_token_indices": np.asarray(context_indices, dtype=np.int64),
                "target_sample_ids": window["target_sample_ids"],
                "target_dates": window["target_dates"],
                "asv_ids": asv_ids,
            }
        )
    if missing_keys:
        preview = ", ".join(map(str, missing_keys[:5]))
        raise KeyError(f"missing {len(missing_keys)} context token states, e.g. {preview}")
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


def make_loader(records: List[Dict], token_states: np.ndarray, batch_size: int, shuffle: bool):
    return DataLoader(
        WastewaterTokenWindowDataset(records, token_states),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_batch,
    )


def combined_loss(outputs: Dict[str, torch.Tensor], context: torch.Tensor, target: torch.Tensor, args) -> Dict[str, torch.Tensor]:
    pred = outputs["abundance"]
    target = normalize_torch(target)
    context = normalize_torch(context)
    target_delta_clr = clr_torch(target, args.pseudocount) - clr_torch(context[:, -1:, :], args.pseudocount)
    change_weights = make_change_weights(context, target, args)

    bray = bray_curtis_loss_torch(pred, target)
    mse = F.mse_loss(pred, target)
    clr_delta = F.mse_loss(outputs["pred_delta_clr"], target_delta_clr)
    corr = pearson_loss(outputs["pred_delta_clr"], target_delta_clr)
    weighted_bray = weighted_bray_curtis_loss(pred, target, change_weights)
    weighted_mse = weighted_mse_loss(pred, target, change_weights)
    asv_corr = per_asv_delta_corr_loss(
        outputs["pred_delta_clr"],
        target_delta_clr,
        weights=change_weights,
        min_std=float(args.asv_corr_min_std),
    )
    delta_l2 = torch.mean(outputs["pred_delta_clr"] ** 2)

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
        + float(args.loss_weighted_bray_weight) * weighted_bray
        + float(args.loss_weighted_mse_weight) * weighted_mse
        + float(args.loss_asv_corr_weight) * asv_corr
        + float(args.loss_delta_l2_weight) * delta_l2
    )
    return {
        "loss": loss,
        "bray_loss": bray.detach(),
        "mse_loss": mse.detach(),
        "clr_delta_loss": clr_delta.detach(),
        "corr_loss": corr.detach(),
        "direction_loss": direction.detach(),
        "weighted_bray_loss": weighted_bray.detach(),
        "weighted_mse_loss": weighted_mse.detach(),
        "asv_corr_loss": asv_corr.detach(),
        "delta_l2_loss": delta_l2.detach(),
    }


def evaluate_loss(model, records: List[Dict], token_states: np.ndarray, args, device) -> float:
    if not records:
        return float("inf")
    loader = make_loader(records, token_states, args.eval_batch_size, shuffle=False)
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            context = batch["context"].to(device)
            target = batch["target"].to(device)
            outputs = model(context, batch["tokens"].to(device), batch["plant_id"].to(device))
            losses.append(float(combined_loss(outputs, context, target, args)["loss"].item()))
    return float(np.mean(losses))


def predict_records(model, records: List[Dict], token_states: np.ndarray, args, device):
    if not records:
        return None
    loader = make_loader(records, token_states, args.eval_batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["context"].to(device), batch["tokens"].to(device), batch["plant_id"].to(device))
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
    cache = load_token_cache(args.token_cache)
    token_states = cache["token_states"]
    records, plant_to_id = collect_records(dataset, cache, args.plants)
    if not records:
        raise ValueError("no records found")

    train_records, holdout_records = split_train_tail_by_plant(records, args.train_tail_val_ratio, args.limit_train_windows)
    if not train_records:
        raise ValueError("no train records found")

    context_len, num_asvs = train_records[0]["context"].shape
    horizon = train_records[0]["target"].shape[0]
    token_dim = int(token_states.shape[-1])
    model = TokenASVDeltaTransformer(
        num_asvs=num_asvs,
        context_len=context_len,
        horizon=horizon,
        token_dim=token_dim,
        num_plants=len(plant_to_id),
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        temporal_layers=args.temporal_layers,
        asv_layers=args.asv_layers,
        dropout=args.dropout,
        pseudocount=args.pseudocount,
        token_bottleneck_dim=args.token_bottleneck_dim,
        pred_delta_scale=args.pred_delta_scale,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = make_loader(train_records, token_states, args.batch_size, shuffle=True)

    best_state = None
    best_holdout = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_logs = defaultdict(list)
        for batch in train_loader:
            context = batch["context"].to(device)
            target = batch["target"].to(device)
            outputs = model(context, batch["tokens"].to(device), batch["plant_id"].to(device))
            losses = combined_loss(outputs, context, target, args)
            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in losses.items():
                epoch_logs[key].append(float(value.detach().cpu()))

        val_records = holdout_records or train_records
        holdout_loss = evaluate_loss(model, val_records, token_states, args, device)
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
            "token_cache": str(args.token_cache),
            "scale": cache.get("scale", "unknown"),
        },
        output_dir / "checkpoint.pt",
    )
    pd.DataFrame(
        [
            {
                "model": "v2_token_asv_delta_transformer",
                "scale": cache.get("scale", "unknown"),
                "train_windows": len(train_records),
                "train_tail_holdout_windows": len(holdout_records),
                "best_train_tail_holdout_loss": best_holdout,
                "epochs": len(history),
                "num_plants": len(plant_to_id),
                "num_asvs_per_plant": num_asvs,
                "token_dim": token_dim,
                "hidden_size": args.hidden_size,
                "token_bottleneck_dim": args.token_bottleneck_dim,
                "pred_delta_scale": args.pred_delta_scale,
                "temporal_layers": args.temporal_layers,
                "asv_layers": args.asv_layers,
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
        pred = predict_records(model, split_records, token_states, args, device)
        for plant in sorted({record["plant"] for record in split_records}):
            plant_records = [record for record in split_records if record["plant"] == plant]
            indices = [idx for idx, record in enumerate(split_records) if record["plant"] == plant]
            eval_records.append(
                {
                    "model": "v2_token_asv_delta_transformer",
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
