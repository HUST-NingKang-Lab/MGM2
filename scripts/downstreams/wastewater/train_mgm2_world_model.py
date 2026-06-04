#!/usr/bin/env python3
"""Train all-plant MAE or JEPA wastewater world models from MGM2 token caches."""

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
from src.models.WastewaterWorldModel import MGM2DeltaMAEWorldModel, MGM2JEPAWorldModel, MGM2MAEWorldModel


DEFAULT_DATASET = Path("data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl")
DEFAULT_CACHE = Path("data/downstreams/wastewater/cache/mgm2_scale_small_top200_tokens.pkl")
DEFAULT_OUTPUT_DIR = Path("outputs_downstreams/wastewater/mgm2_world_model/mae_scale_small_top200")


class WorldModelWindowDataset(Dataset):
    def __init__(self, records: List[Dict], cache: Dict):
        self.records = records
        self.sample_to_idx = cache["sample_to_idx"]
        self.token_states = cache["token_states"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        context_indices = [(record["plant"], sample_id) for sample_id in record["context_sample_ids"]]
        target_indices = [(record["plant"], sample_id) for sample_id in record["target_sample_ids"]]
        context_token_idx = [self.sample_to_idx[key] for key in context_indices]
        target_token_idx = [self.sample_to_idx[key] for key in target_indices]
        return (
            torch.from_numpy(self.token_states[context_token_idx].astype(np.float32)),
            torch.from_numpy(record["context_abundance"].astype(np.float32)),
            torch.from_numpy(record["target_abundance"].astype(np.float32)),
            torch.from_numpy(self.token_states[target_token_idx].astype(np.float32)),
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["mae", "delta_mae", "jepa"], default="mae")
    parser.add_argument("--plants", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=["val", "test"])
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_tail_val_ratio", type=float, default=0.15)
    parser.add_argument("--internal_val_ratio", type=float, dest="train_tail_val_ratio", help=argparse.SUPPRESS)
    parser.add_argument("--loss_mse_weight", type=float, default=0.0, help="Abundance MSE auxiliary weight; 0 uses pure Bray-Curtis for MAE/probe training")
    parser.add_argument(
        "--bc_loss_mode",
        choices=["normalized", "raw"],
        default="normalized",
        help="normalized applies closure before Bray-Curtis; raw matches the paper loss directly on model outputs",
    )
    parser.add_argument("--delta_loss_weight", type=float, default=0.5, help="Only used for Delta-MAE change supervision")
    parser.add_argument("--ema_momentum", type=float, default=0.996, help="Only used for JEPA target encoder updates")
    parser.add_argument("--probe_epochs", type=int, default=30, help="Only used for JEPA abundance probe training")
    parser.add_argument("--probe_patience", type=int, default=6, help="Only used for JEPA abundance probe early stopping")
    parser.add_argument("--probe_learning_rate", type=float, default=1e-3, help="Only used for JEPA abundance probe training")
    parser.add_argument("--top_k_values", nargs="*", type=int, default=[10, 20, 50])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--no_residual_persistence", action="store_true")
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument(
        "--abundance_head",
        choices=["additive_residual", "logratio_residual", "gated_softmax", "softmax", "paper_relu"],
        default="logratio_residual",
        help="Abundance decoding head. logratio_residual predicts log-fold changes around persistence.",
    )
    parser.add_argument(
        "--query_mode",
        choices=["last", "temporal_attention"],
        default="temporal_attention",
        help="Future token query source. temporal_attention summarizes all context steps per ASV.",
    )
    parser.add_argument("--max_context_len", type=int, default=64, help="Maximum context length for temporal_attention queries")
    parser.add_argument("--limit_train_windows", type=int, default=0, help="Debug: keep only first N train windows")
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
    records = []
    for window in dataset["windows"]:
        plant = window["plant"]
        if selected is not None and plant not in selected:
            continue
        records.append(
            {
                "plant": plant,
                "split": window["split"],
                "context_sample_ids": window["context_sample_ids"],
                "target_sample_ids": window["target_sample_ids"],
                "target_dates": window["target_dates"],
                "asv_ids": window["asv_ids"],
                "context_abundance": normalize_abundance(window["context_abundance"]).astype(np.float32),
                "target_abundance": normalize_abundance(window["target_abundance"]).astype(np.float32),
            }
        )
    return records


def _selected_plants(records: List[Dict]) -> str:
    return "|".join(sorted({record["plant"] for record in records}))


def _split_train_tail_holdout(records: List[Dict], val_ratio: float, limit_train_windows: int = 0):
    train = [record for record in records if record["split"] == "train"]
    if limit_train_windows > 0:
        train = train[:limit_train_windows]
    n_val = max(1, int(round(len(train) * val_ratio))) if len(train) > 1 else 0
    n_val = min(n_val, max(0, len(train) - 1))
    if n_val == 0:
        return train, []
    return train[:-n_val], train[-n_val:]


def _build_model(args, cache: Dict, sample_record: Dict, device):
    hidden_size = int(cache["hidden_size"])
    horizon, num_asvs = sample_record["target_abundance"].shape
    kwargs = {
        "hidden_size": hidden_size,
        "horizon": horizon,
        "num_asvs": num_asvs,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "residual_persistence": not args.no_residual_persistence,
        "residual_scale": args.residual_scale,
        "abundance_head": args.abundance_head,
        "query_mode": args.query_mode,
        "max_context_len": args.max_context_len,
    }
    if args.mode == "mae":
        model = MGM2MAEWorldModel(**kwargs)
    elif args.mode == "delta_mae":
        model = MGM2DeltaMAEWorldModel(**kwargs)
    else:
        model = MGM2JEPAWorldModel(**kwargs)
    return model.to(device)


def _world_parameters(model, mode: str):
    if mode in {"mae", "delta_mae"}:
        return model.parameters()
    return list(model.online_encoder.parameters()) + list(model.predictor.parameters())


def _world_loss(model, context_states, context_abundance, target_abundance, target_states, args) -> Dict[str, torch.Tensor]:
    if args.mode == "jepa":
        return model.training_loss(context_states, target_states)
    if args.mode == "delta_mae":
        return model.training_loss(
            context_states,
            context_abundance,
            target_abundance,
            loss_mse_weight=args.loss_mse_weight,
            delta_loss_weight=args.delta_loss_weight,
            bc_loss_mode=args.bc_loss_mode,
        )
    return model.training_loss(context_states, context_abundance, target_abundance, args.loss_mse_weight, args.bc_loss_mode)


def _evaluate_loss(model, records: List[Dict], cache: Dict, args, device) -> float:
    if not records:
        return float("inf")
    loader = DataLoader(WorldModelWindowDataset(records, cache), batch_size=args.batch_size, shuffle=False, num_workers=0)
    losses = []
    model.eval()
    with torch.no_grad():
        for context_states, context_abundance, target_abundance, target_states in loader:
            context_states = context_states.to(device)
            context_abundance = context_abundance.to(device)
            target_abundance = target_abundance.to(device)
            target_states = target_states.to(device)
            loss_dict = _world_loss(model, context_states, context_abundance, target_abundance, target_states, args)
            losses.append(float(loss_dict["loss"].item()))
    return float(np.mean(losses))


def _evaluate_probe_loss(model, records: List[Dict], cache: Dict, args, device) -> float:
    if not records:
        return float("inf")
    loader = DataLoader(WorldModelWindowDataset(records, cache), batch_size=args.batch_size, shuffle=False, num_workers=0)
    losses = []
    model.eval()
    with torch.no_grad():
        for context_states, context_abundance, target_abundance, _ in loader:
            loss_dict = model.probe_loss(
                context_states.to(device),
                context_abundance.to(device),
                target_abundance.to(device),
                args.loss_mse_weight,
                args.bc_loss_mode,
            )
            losses.append(float(loss_dict["loss"].item()))
    return float(np.mean(losses))


def _jepa_latent_diagnostics(model, records: List[Dict], cache: Dict, args, device) -> Dict[str, float]:
    if args.mode != "jepa" or not records:
        return {"latent_pred_std": np.nan, "latent_target_std": np.nan, "latent_pred_target_cosine": np.nan}
    loader = DataLoader(WorldModelWindowDataset(records, cache), batch_size=args.batch_size, shuffle=False, num_workers=0)
    pred_stds = []
    target_stds = []
    cosines = []
    model.eval()
    with torch.no_grad():
        for context_states, _, _, target_states in loader:
            pred = model(context_states.to(device))["predicted_states"]
            target = model.encode_target(target_states.to(device))
            pred_norm = F.normalize(pred, dim=-1)
            target_norm = F.normalize(target, dim=-1)
            pred_stds.append(float(pred.std(dim=-1).mean().item()))
            target_stds.append(float(target.std(dim=-1).mean().item()))
            cosines.append(float((pred_norm * target_norm).sum(dim=-1).mean().item()))
    return {
        "latent_pred_std": float(np.mean(pred_stds)),
        "latent_target_std": float(np.mean(target_stds)),
        "latent_pred_target_cosine": float(np.mean(cosines)),
    }


def _set_probe_train_mode(model):
    model.online_encoder.eval()
    model.predictor.eval()
    model.target_encoder.eval()
    model.abundance_head.train()


def _predict_records(model, records: List[Dict], cache: Dict, args, device):
    if not records:
        return None
    loader = DataLoader(WorldModelWindowDataset(records, cache), batch_size=args.batch_size, shuffle=False, num_workers=0)
    preds = []
    model.eval()
    with torch.no_grad():
        for context_states, context_abundance, _, _ in loader:
            if args.mode == "jepa":
                outputs = model(context_states.to(device), context_abundance.to(device), return_abundance=True)
            else:
                outputs = model(context_states.to(device), context_abundance.to(device))
            preds.append(outputs["abundance"].detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def _train_jepa_probe(model, train_records, holdout_records, cache, args, device):
    model.freeze_world_model()
    for parameter in model.abundance_head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.abundance_head.parameters(), lr=args.probe_learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(WorldModelWindowDataset(train_records, cache), batch_size=args.batch_size, shuffle=True, num_workers=0)
    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, args.probe_epochs + 1):
        _set_probe_train_mode(model)
        epoch_losses = []
        for context_states, context_abundance, target_abundance, _ in train_loader:
            loss_dict = model.probe_loss(
                context_states.to(device),
                context_abundance.to(device),
                target_abundance.to(device),
                args.loss_mse_weight,
                args.bc_loss_mode,
            )
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.abundance_head.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss_dict["loss"].item()))
        val_records = holdout_records or train_records
        val_loss = _evaluate_probe_loss(model, val_records, cache, args, device)
        history.append({"epoch": epoch, "probe_train_loss": float(np.mean(epoch_losses)), "probe_holdout_loss": val_loss})
        print(f"probe_epoch={epoch} train={history[-1]['probe_train_loss']:.4f} holdout={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.abundance_head.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.probe_patience:
                break
    if best_state is not None:
        model.abundance_head.load_state_dict(best_state)
    model.eval()
    return history, best_val


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
    records = _collect_records(dataset, args.plants)
    if not records:
        raise ValueError("no records found")
    selected_plants = _selected_plants(records)

    train_records, holdout_records = _split_train_tail_holdout(records, args.train_tail_val_ratio, args.limit_train_windows)
    if not train_records:
        raise ValueError("no train records found")
    model = _build_model(args, cache, train_records[0], device)
    optimizer = torch.optim.AdamW(_world_parameters(model, args.mode), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(WorldModelWindowDataset(train_records, cache), batch_size=args.batch_size, shuffle=True, num_workers=0)

    best_state = None
    best_holdout = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_losses = []
        for context_states, context_abundance, target_abundance, target_states in train_loader:
            context_states = context_states.to(device)
            context_abundance = context_abundance.to(device)
            target_abundance = target_abundance.to(device)
            target_states = target_states.to(device)
            loss_dict = _world_loss(model, context_states, context_abundance, target_abundance, target_states, args)
            loss = loss_dict["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_world_parameters(model, args.mode), 1.0)
            optimizer.step()
            if args.mode == "jepa":
                model.update_target_encoder(args.ema_momentum)
            epoch_losses.append(float(loss.item()))

        val_records = holdout_records or train_records
        holdout_loss = _evaluate_loss(model, val_records, cache, args, device)
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

    latent_diagnostics = _jepa_latent_diagnostics(model, holdout_records or train_records, cache, args, device)
    probe_history = []
    best_probe_holdout = np.nan
    if args.mode == "jepa":
        probe_history, best_probe_holdout = _train_jepa_probe(model, train_records, holdout_records, cache, args, device)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "history": history,
            "probe_history": probe_history,
            "best_train_tail_holdout_loss": best_holdout,
            "best_probe_holdout_loss": best_probe_holdout,
            "latent_diagnostics": latent_diagnostics,
        },
        output_dir / "checkpoint.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False)
    if probe_history:
        pd.DataFrame(probe_history).to_csv(output_dir / "probe_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "mode": args.mode,
                "selected_plants": selected_plants,
                "train_windows": len(train_records),
                "train_tail_holdout_windows": len(holdout_records),
                "best_train_tail_holdout_loss": best_holdout,
                "best_probe_holdout_loss": best_probe_holdout,
                "epochs": len(history),
                "probe_epochs": len(probe_history),
                "abundance_head": args.abundance_head,
                "residual_scale": args.residual_scale,
                "query_mode": args.query_mode,
                "bc_loss_mode": args.bc_loss_mode,
                **latent_diagnostics,
            }
        ]
    ).to_csv(output_dir / "train_summary.csv", index=False)

    eval_records = []
    model_name = f"mgm2_world_{args.mode}"
    for split in args.splits:
        split_records = [record for record in records if record["split"] == split]
        if not split_records:
            continue
        pred = _predict_records(model, split_records, cache, args, device)
        for plant in sorted(set(record["plant"] for record in split_records)):
            plant_records = [record for record in split_records if record["plant"] == plant]
            indices = [idx for idx, record in enumerate(split_records) if record["plant"] == plant]
            eval_records.append(
                {
                    "model": model_name,
                    "plant": plant,
                    "split": split,
                    "asv_ids": plant_records[0]["asv_ids"],
                    "target_sample_ids": [record["target_sample_ids"] for record in plant_records],
                    "target_dates": [record["target_dates"] for record in plant_records],
                    "y_true": np.stack([record["target_abundance"] for record in plant_records]).astype(np.float32),
                    "y_pred": pred[indices].astype(np.float32),
                }
            )

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
