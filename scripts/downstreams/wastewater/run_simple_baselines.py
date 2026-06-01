#!/usr/bin/env python3
"""Run simple wastewater forecasting baselines on cached windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.data.WastewaterDataset import load_wastewater_windows
from src.metrics.wastewater import aggregate_forecast_metrics, compute_forecast_metrics, normalize_abundance


DEFAULT_DATASET = Path("data/downstreams/wastewater/wastewater_windows_ctx10_h10.pkl")
DEFAULT_OUTPUT_DIR = Path("outputs_downstreams/wastewater/baselines")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Cached wastewater window pickle")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--splits", nargs="*", default=["val", "test"], help="Splits to evaluate")
    parser.add_argument("--top_k_values", nargs="*", type=int, default=[10, 20, 50], help="Top-k overlap values")
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save compressed long-format prediction tables. This can be large.",
    )
    return parser.parse_args()


def _predict(window: Dict, baseline: str) -> np.ndarray:
    context = normalize_abundance(window["context_abundance"])
    horizon = window["target_abundance"].shape[0]
    if baseline == "persistence":
        pred = np.repeat(context[-1:, :], repeats=horizon, axis=0)
    elif baseline == "context_mean":
        pred = np.repeat(context.mean(axis=0, keepdims=True), repeats=horizon, axis=0)
    else:
        raise ValueError(f"unknown baseline: {baseline}")
    return normalize_abundance(pred)


def _group_predictions(dataset: Dict, baseline: str, splits: Iterable[str]) -> List[Dict]:
    grouped: Dict[tuple, Dict] = {}
    wanted_splits = set(splits)
    for window in dataset["windows"]:
        split = window["split"]
        if split not in wanted_splits:
            continue
        key = (window["plant"], split)
        group = grouped.setdefault(
            key,
            {
                "plant": window["plant"],
                "split": split,
                "asv_ids": window["asv_ids"],
                "target_sample_ids": [],
                "target_dates": [],
                "y_true": [],
                "y_pred": [],
            },
        )
        group["target_sample_ids"].append(window["target_sample_ids"])
        group["target_dates"].append(window["target_dates"])
        group["y_true"].append(normalize_abundance(window["target_abundance"]))
        group["y_pred"].append(_predict(window, baseline))

    records = []
    for group in grouped.values():
        records.append(
            {
                **group,
                "model": baseline,
                "y_true": np.stack(group["y_true"]).astype(np.float32),
                "y_pred": np.stack(group["y_pred"]).astype(np.float32),
            }
        )
    return records


def _summary_rows(records: List[Dict], top_k_values: List[int]) -> List[Dict]:
    rows = []
    for record in records:
        metrics = compute_forecast_metrics(record["y_true"], record["y_pred"], top_k_values=top_k_values)
        rows.append(
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
    return rows


def _prediction_rows(record: Dict):
    asv_ids = record["asv_ids"]
    for window_idx in range(record["y_pred"].shape[0]):
        for horizon_idx in range(record["y_pred"].shape[1]):
            sample_id = record["target_sample_ids"][window_idx][horizon_idx]
            date = record["target_dates"][window_idx][horizon_idx]
            true = record["y_true"][window_idx, horizon_idx]
            pred = record["y_pred"][window_idx, horizon_idx]
            for asv_id, y_true, y_pred in zip(asv_ids, true, pred):
                yield {
                    "model": record["model"],
                    "plant": record["plant"],
                    "split": record["split"],
                    "window_idx": window_idx,
                    "horizon": horizon_idx + 1,
                    "sample_id": sample_id,
                    "date": date,
                    "asv_id": asv_id,
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                }


def main():
    args = parse_args()
    dataset = load_wastewater_windows(args.dataset)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for baseline in ("persistence", "context_mean"):
        all_records.extend(_group_predictions(dataset, baseline, args.splits))

    if not all_records:
        raise ValueError(f"no windows found for splits: {args.splits}")

    summary = pd.DataFrame(_summary_rows(all_records, args.top_k_values))
    summary.to_csv(output_dir / "summary.csv", index=False)

    horizon_frames = []
    for record in all_records:
        frame = aggregate_forecast_metrics([record], top_k_values=args.top_k_values)
        frame.insert(0, "model", record["model"])
        horizon_frames.append(frame)
    pd.concat(horizon_frames, ignore_index=True).to_csv(output_dir / "metrics_by_horizon.csv", index=False)

    if args.save_predictions:
        rows = []
        for record in all_records:
            rows.extend(_prediction_rows(record))
        pd.DataFrame(rows).to_csv(output_dir / "predictions.csv.gz", index=False, compression="gzip")

    print(f"Saved summary: {output_dir / 'summary.csv'}")
    print(f"Saved horizon metrics: {output_dir / 'metrics_by_horizon.csv'}")
    if args.save_predictions:
        print(f"Saved predictions: {output_dir / 'predictions.csv.gz'}")
    print(summary.sort_values(["split", "plant", "model"]).to_string(index=False))


if __name__ == "__main__":
    main()
