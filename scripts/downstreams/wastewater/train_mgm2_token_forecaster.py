#!/usr/bin/env python3
"""Train token-level MGM2 cross-attention forecasters on cached token states."""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.data.WastewaterDataset import load_wastewater_windows
from src.metrics.wastewater import aggregate_forecast_metrics, compute_forecast_metrics, normalize_abundance
from src.models.WastewaterModel import MGM2TokenCrossAttentionForecaster, bray_curtis_loss_torch


DEFAULT_DATASET = Path("data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl")
DEFAULT_CACHE = Path("data/downstreams/wastewater/cache/mgm2_scale_small_top200_tokens.pkl")
DEFAULT_OUTPUT_DIR = Path("outputs_downstreams/wastewater/mgm2_token_cross_attention/scale_small_top200")


class TokenWindowDataset(Dataset):
    def __init__(self, records: List[Dict], cache: Dict):
        self.records = records
        self.sample_to_idx = cache["sample_to_idx"]
        self.token_states = cache["token_states"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        indices = [(record["plant"], sample_id) for sample_id in record["context_sample_ids"]]
        token_indices = [self.sample_to_idx[key] for key in indices]
        context_states = self.token_states[token_indices].astype(np.float32)
        context_abundance = record["context_abundance"].astype(np.float32)
        target = record["target_abundance"].astype(np.float32)
        return (
            torch.from_numpy(context_states),
            torch.from_numpy(context_abundance),
            torch.from_numpy(target),
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Wastewater windows pickle")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="MGM2 token cache pickle")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--plants", nargs="*", default=None, help="Optional plant subset")
    parser.add_argument("--splits", nargs="*", default=["val", "test"], help="Official splits to evaluate")
    parser.add_argument("--num_heads", type=int, default=4, help="Cross-attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--max_epochs", type=int, default=80, help="Maximum epochs per plant")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--internal_val_ratio", type=float, default=0.15, help="Tail fraction of train windows for internal validation")
    parser.add_argument("--loss_mse_weight", type=float, default=0.1, help="Weight for abundance MSE auxiliary loss")
    parser.add_argument("--top_k_values", nargs="*", type=int, default=[10, 20, 50], help="Top-k overlap values")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_predictions", action="store_true", help="Save compressed long-format predictions")
    parser.add_argument("--min_train_windows", type=int, default=20, help="Skip plants with fewer train windows")
    parser.add_argument("--no_residual_persistence", action="store_true", help="Disable residual correction around persistence")
    parser.add_argument("--residual_scale", type=float, default=0.1, help="Residual correction scale")
    parser.add_argument(
        "--abundance_head",
        choices=["additive_residual", "logratio_residual", "gated_softmax", "softmax", "paper_relu"],
        default="additive_residual",
        help="Abundance decoding head",
    )
    parser.add_argument(
        "--query_mode",
        choices=["last", "temporal_attention"],
        default="last",
        help="Future token query source",
    )
    parser.add_argument("--max_context_len", type=int, default=64, help="Maximum context length for temporal_attention queries")
    return parser.parse_args()


def _load_cache(path: Path) -> Dict:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collect_records(dataset: Dict, plants: Iterable[str] | None):
    selected = set(plants) if plants else None
    by_plant: Dict[str, List[Dict]] = {}
    for window in dataset["windows"]:
        plant = window["plant"]
        if selected is not None and plant not in selected:
            continue
        by_plant.setdefault(plant, []).append(
            {
                "plant": plant,
                "split": window["split"],
                "context_sample_ids": window["context_sample_ids"],
                "context_abundance": normalize_abundance(window["context_abundance"]).astype(np.float32),
                "target_abundance": normalize_abundance(window["target_abundance"]).astype(np.float32),
                "target_sample_ids": window["target_sample_ids"],
                "target_dates": window["target_dates"],
                "asv_ids": window["asv_ids"],
            }
        )
    return by_plant


def _split_train_internal(records: List[Dict], val_ratio: float):
    train = [record for record in records if record["split"] == "train"]
    if not train:
        return [], []
    n_val = max(1, int(round(len(train) * val_ratio))) if len(train) > 1 else 0
    n_val = min(n_val, max(0, len(train) - 1))
    if n_val == 0:
        return train, []
    return train[:-n_val], train[-n_val:]


def _loss(pred, target, mse_weight: float):
    return bray_curtis_loss_torch(pred, target) + mse_weight * F.mse_loss(pred, target)


def _evaluate_loss(model, records: List[Dict], cache: Dict, device, batch_size: int, mse_weight: float):
    if not records:
        return float("inf")
    loader = DataLoader(TokenWindowDataset(records, cache), batch_size=batch_size, shuffle=False, num_workers=0)
    losses = []
    model.eval()
    with torch.no_grad():
        for states, abundance, target in loader:
            states = states.to(device)
            abundance = abundance.to(device)
            target = target.to(device)
            losses.append(float(_loss(model(states, abundance), target, mse_weight).item()))
    return float(np.mean(losses))


def _train_one_plant(plant: str, records: List[Dict], cache: Dict, args, device):
    train_records, internal_val_records = _split_train_internal(records, args.internal_val_ratio)
    if len(train_records) < args.min_train_windows:
        return None, {"plant": plant, "status": "skipped", "reason": f"train_windows={len(train_records)}"}

    sample = train_records[0]
    num_asvs = sample["target_abundance"].shape[-1]
    horizon = sample["target_abundance"].shape[0]
    hidden_size = int(cache["hidden_size"])
    model = MGM2TokenCrossAttentionForecaster(
        hidden_size=hidden_size,
        horizon=horizon,
        num_asvs=num_asvs,
        num_heads=args.num_heads,
        dropout=args.dropout,
        residual_persistence=not args.no_residual_persistence,
        residual_scale=args.residual_scale,
        abundance_head=args.abundance_head,
        query_mode=args.query_mode,
        max_context_len=args.max_context_len,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(TokenWindowDataset(train_records, cache), batch_size=args.batch_size, shuffle=True, num_workers=0)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_losses = []
        for states, abundance, target in train_loader:
            states = states.to(device)
            abundance = abundance.to(device)
            target = target.to(device)
            pred = model(states, abundance)
            loss = _loss(pred, target, args.loss_mse_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_records = internal_val_records or train_records
        val_loss = _evaluate_loss(model, val_records, cache, device, args.batch_size, args.loss_mse_weight)
        history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "internal_val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    info = {
        "plant": plant,
        "status": "trained",
        "train_windows": len(train_records),
        "internal_val_windows": len(internal_val_records),
        "best_internal_val_loss": best_val,
        "epochs": len(history),
        "abundance_head": args.abundance_head,
        "residual_scale": args.residual_scale,
        "query_mode": args.query_mode,
        "history": history,
    }
    return model, info


def _predict_records(model, records: List[Dict], cache: Dict, device, batch_size: int):
    if not records:
        return None
    loader = DataLoader(TokenWindowDataset(records, cache), batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    model.eval()
    with torch.no_grad():
        for states, abundance, _ in loader:
            preds.append(model(states.to(device), abundance.to(device)).detach().cpu().numpy())
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
    cache = _load_cache(args.cache)
    by_plant = _collect_records(dataset, args.plants)
    if not by_plant:
        raise ValueError("no plant records found")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_name = "mgm2_token_cross_attention"

    eval_records = []
    train_infos = []
    for plant, records in sorted(by_plant.items()):
        model, info = _train_one_plant(plant, records, cache, args, device)
        train_infos.append({key: value for key, value in info.items() if key != "history"})
        if model is None:
            print(f"Skipping {plant}: {info['reason']}")
            continue
        torch.save({"model_state_dict": model.state_dict(), "train_info": info}, checkpoint_dir / f"{plant.replace('/', '_')}.pt")
        for split in args.splits:
            split_records = [record for record in records if record["split"] == split]
            if not split_records:
                continue
            pred = _predict_records(model, split_records, cache, device, args.batch_size)
            eval_records.append(
                {
                    "model": model_name,
                    "plant": plant,
                    "split": split,
                    "asv_ids": split_records[0]["asv_ids"],
                    "target_sample_ids": [record["target_sample_ids"] for record in split_records],
                    "target_dates": [record["target_dates"] for record in split_records],
                    "y_true": np.stack([record["target_abundance"] for record in split_records]).astype(np.float32),
                    "y_pred": pred.astype(np.float32),
                }
            )
        print(f"{plant}: epochs={info['epochs']} train={info['train_windows']} internal_val={info['internal_val_windows']} best={info['best_internal_val_loss']:.4f}")

    if not eval_records:
        raise ValueError("no evaluation records were produced")

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
    pd.DataFrame(train_infos).to_csv(output_dir / "train_summary.csv", index=False)

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
