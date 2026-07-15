#!/usr/bin/env python3
"""Export predictions from a trained V2 token ASV-delta checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.downstreams.wastewater_v2.train_token_asv_delta_transformer import (
    TokenASVDeltaTransformer,
    collect_records,
    load_token_cache,
    predict_records,
)
from src.data.WastewaterDataset import load_wastewater_windows


DEFAULT_DATASET = Path("data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl")
DEFAULT_OUTPUT = Path("outputs_downstreams/wastewater_v2/token_asv_delta_transformer/scale_large/predictions.csv.gz")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--token_cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", nargs="*", default=["test"])
    parser.add_argument("--plants", nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    return parser.parse_args()


def _prediction_rows(record_group: dict):
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
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    dataset = load_wastewater_windows(args.dataset)
    cache = load_token_cache(args.token_cache)
    records, plant_to_id = collect_records(dataset, cache, args.plants)
    if not records:
        raise ValueError("no records found")

    first = records[0]
    context_len, num_asvs = first["context"].shape
    horizon = first["target"].shape[0]
    token_dim = int(cache["token_states"].shape[-1])
    model = TokenASVDeltaTransformer(
        num_asvs=num_asvs,
        context_len=context_len,
        horizon=horizon,
        token_dim=token_dim,
        num_plants=len(plant_to_id),
        hidden_size=int(ckpt_args.get("hidden_size", 256)),
        num_heads=int(ckpt_args.get("num_heads", 4)),
        temporal_layers=int(ckpt_args.get("temporal_layers", 2)),
        asv_layers=int(ckpt_args.get("asv_layers", 1)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
        pseudocount=float(ckpt_args.get("pseudocount", 1e-4)),
        token_bottleneck_dim=int(ckpt_args.get("token_bottleneck_dim", 0)),
        pred_delta_scale=float(ckpt_args.get("pred_delta_scale", 1.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    class EvalArgs:
        eval_batch_size = int(args.eval_batch_size)

    rows = []
    for split in args.splits:
        split_records = [record for record in records if record["split"] == split]
        if not split_records:
            continue
        pred = predict_records(model, split_records, cache["token_states"], EvalArgs, device)
        for plant in sorted({record["plant"] for record in split_records}):
            plant_records = [record for record in split_records if record["plant"] == plant]
            indices = [idx for idx, record in enumerate(split_records) if record["plant"] == plant]
            group = {
                "model": "v2_token_asv_delta_transformer",
                "scale": cache.get("scale", "unknown"),
                "plant": plant,
                "split": split,
                "asv_ids": plant_records[0]["asv_ids"],
                "target_sample_ids": [record["target_sample_ids"] for record in plant_records],
                "target_dates": [record["target_dates"] for record in plant_records],
                "y_true": np.stack([record["target"] for record in plant_records]).astype(np.float32),
                "y_pred": pred[indices].astype(np.float32),
            }
            rows.extend(_prediction_rows(group))

    if not rows:
        raise ValueError(f"no prediction rows exported for splits={args.splits}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, compression="gzip")
    print(f"Saved predictions: {args.output}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
