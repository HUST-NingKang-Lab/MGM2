#!/usr/bin/env python3
"""Evaluate horizon hybrids between persistence and learned wastewater forecasts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.metrics.wastewater import aggregate_forecast_metrics, compute_forecast_metrics


DEFAULT_BASELINE_PREDICTIONS = Path("outputs_downstreams/wastewater/baseline/baseline_top200/predictions.csv.gz")
DEFAULT_LEARNED_PREDICTIONS = Path("outputs_downstreams/wastewater/vector_temporal/tcn_top200/predictions.csv.gz")
DEFAULT_OUTPUT_DIR = Path("outputs_downstreams/wastewater/hybrid/tcn_persistence_top200")

KEY_COLUMNS = ["plant", "split", "window_idx", "horizon", "sample_id", "date", "asv_id"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_predictions", type=Path, default=DEFAULT_BASELINE_PREDICTIONS)
    parser.add_argument("--learned_predictions", type=Path, default=DEFAULT_LEARNED_PREDICTIONS)
    parser.add_argument("--learned_model", type=str, default=None, help="Model name in learned predictions. Defaults to the first model.")
    parser.add_argument("--baseline_model", type=str, default="persistence", help="Baseline model name")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_horizon", type=int, default=None, help="Maximum horizon. Defaults to max horizon in predictions.")
    parser.add_argument("--cutoff", type=int, default=None, help="Fixed cutoff. If omitted, choose by validation Bray-Curtis.")
    parser.add_argument("--top_k_values", nargs="*", type=int, default=[10, 20, 50])
    parser.add_argument("--save_predictions", action="store_true", help="Save hybrid predictions.csv.gz")
    return parser.parse_args()


def _filter_model(df: pd.DataFrame, model: str | None) -> pd.DataFrame:
    if model is None:
        models = sorted(df["model"].unique().tolist())
        if len(models) != 1:
            raise ValueError(f"learned_model must be specified; found models: {models}")
        model = models[0]
    filtered = df[df["model"] == model].copy()
    if filtered.empty:
        raise ValueError(f"model {model!r} not found in predictions")
    return filtered


def build_hybrid_predictions(
    persistence_predictions: pd.DataFrame,
    learned_predictions: pd.DataFrame,
    learned_model: str,
    cutoff: int,
    hybrid_model_name: str,
) -> pd.DataFrame:
    """Use persistence for horizons <= cutoff and learned predictions afterward."""
    persistence = persistence_predictions.copy()
    learned = learned_predictions.copy()
    if learned_model is not None:
        learned = learned[learned["model"] == learned_model].copy()
    persistence_part = persistence[persistence["horizon"] <= cutoff].copy()
    learned_part = learned[learned["horizon"] > cutoff].copy()
    hybrid = pd.concat([persistence_part, learned_part], ignore_index=True)
    hybrid["model"] = hybrid_model_name
    return hybrid.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _records_from_prediction_table(df: pd.DataFrame) -> List[Dict]:
    records = []
    for (model, plant, split), group in df.groupby(["model", "plant", "split"], sort=True):
        asv_ids = sorted(group["asv_id"].astype(str).unique().tolist())
        sample_keys = group[["window_idx", "horizon", "sample_id", "date"]].drop_duplicates()
        num_windows = int(sample_keys["window_idx"].nunique())
        horizon = int(sample_keys["horizon"].max())
        asv_to_idx = {asv_id: idx for idx, asv_id in enumerate(asv_ids)}
        true = np.zeros((num_windows, horizon, len(asv_ids)), dtype=np.float32)
        pred = np.zeros_like(true)
        target_sample_ids = [[None for _ in range(horizon)] for _ in range(num_windows)]
        target_dates = [[None for _ in range(horizon)] for _ in range(num_windows)]
        for row in group.itertuples(index=False):
            window_idx = int(row.window_idx)
            horizon_idx = int(row.horizon) - 1
            asv_idx = asv_to_idx[str(row.asv_id)]
            true[window_idx, horizon_idx, asv_idx] = float(row.y_true)
            pred[window_idx, horizon_idx, asv_idx] = float(row.y_pred)
            target_sample_ids[window_idx][horizon_idx] = row.sample_id
            target_dates[window_idx][horizon_idx] = row.date
        records.append(
            {
                "model": model,
                "plant": plant,
                "split": split,
                "asv_ids": asv_ids,
                "target_sample_ids": target_sample_ids,
                "target_dates": target_dates,
                "y_true": true,
                "y_pred": pred,
            }
        )
    return records


def _summary_from_predictions(df: pd.DataFrame, top_k_values: Sequence[int]) -> pd.DataFrame:
    rows = []
    for record in _records_from_prediction_table(df):
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
    return pd.DataFrame(rows)


def _horizon_from_predictions(df: pd.DataFrame, top_k_values: Sequence[int]) -> pd.DataFrame:
    frames = []
    for record in _records_from_prediction_table(df):
        frame = aggregate_forecast_metrics([record], top_k_values=top_k_values)
        frame.insert(0, "model", record["model"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _weighted_split_metric(summary: pd.DataFrame, split: str) -> Dict[str, float]:
    rows = []
    for cutoff, group in summary[summary["split"] == split].groupby("cutoff"):
        weights = group["n_windows"]
        rows.append(
            {
                "cutoff": int(cutoff),
                "split": split,
                "bray_curtis": float((group["bray_curtis"] * weights).sum() / weights.sum()),
                "top10_overlap": float((group["top10_overlap"] * weights).sum() / weights.sum()) if "top10_overlap" in group else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def select_cutoff_by_val(cutoff_metrics: pd.DataFrame) -> int:
    """Select lowest validation Bray-Curtis; break ties by higher top10 then larger cutoff."""
    val = cutoff_metrics[cutoff_metrics["split"] == "val"].copy()
    if val.empty:
        raise ValueError("cannot select cutoff without validation metrics")
    val = val.sort_values(["bray_curtis", "top10_overlap", "cutoff"], ascending=[True, False, False])
    return int(val.iloc[0]["cutoff"])


def main():
    args = parse_args()
    baseline = pd.read_csv(args.baseline_predictions)
    learned = pd.read_csv(args.learned_predictions)
    persistence = _filter_model(baseline, args.baseline_model)
    learned_model = args.learned_model or sorted(learned["model"].unique().tolist())[0]
    learned = _filter_model(learned, learned_model)

    max_horizon = args.max_horizon or int(min(persistence["horizon"].max(), learned["horizon"].max()))
    cutoff_summaries = []
    hybrid_tables = {}
    for cutoff in range(0, max_horizon + 1):
        model_name = f"hybrid_{learned_model}_persistence_cutoff{cutoff}"
        hybrid = build_hybrid_predictions(persistence, learned, learned_model, cutoff, model_name)
        summary = _summary_from_predictions(hybrid, top_k_values=args.top_k_values)
        summary.insert(0, "cutoff", cutoff)
        cutoff_summaries.append(summary)
        hybrid_tables[cutoff] = hybrid

    all_cutoff_summary = pd.concat(cutoff_summaries, ignore_index=True)
    cutoff_metrics = pd.concat(
        [
            _weighted_split_metric(all_cutoff_summary, split)
            for split in sorted(all_cutoff_summary["split"].unique())
        ],
        ignore_index=True,
    )
    selected_cutoff = args.cutoff if args.cutoff is not None else select_cutoff_by_val(cutoff_metrics)
    selected = hybrid_tables[selected_cutoff].copy()
    selected["model"] = f"hybrid_{learned_model}_persistence_cutoff{selected_cutoff}"
    selected_summary = _summary_from_predictions(selected, top_k_values=args.top_k_values)
    selected_horizon = _horizon_from_predictions(selected, top_k_values=args.top_k_values)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_cutoff_summary.to_csv(args.output_dir / "all_cutoff_summary.csv", index=False)
    cutoff_metrics.to_csv(args.output_dir / "cutoff_metrics.csv", index=False)
    selected_summary.to_csv(args.output_dir / "summary.csv", index=False)
    selected_horizon.to_csv(args.output_dir / "metrics_by_horizon.csv", index=False)
    if args.save_predictions:
        selected.to_csv(args.output_dir / "predictions.csv.gz", index=False, compression="gzip")

    print(f"Selected cutoff: {selected_cutoff}")
    print(f"Saved summary: {args.output_dir / 'summary.csv'}")
    print(f"Saved cutoff metrics: {args.output_dir / 'cutoff_metrics.csv'}")
    print(cutoff_metrics.sort_values(["split", "bray_curtis"]).to_string(index=False))


if __name__ == "__main__":
    main()
