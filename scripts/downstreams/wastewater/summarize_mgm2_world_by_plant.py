#!/usr/bin/env python3
"""Collect per-plant wastewater world-model runs into analysis-ready CSV files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


METRICS = ["bray_curtis", "mae", "rmse", "spearman", "top50_overlap"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs_downstreams/wastewater/mgm2_world_model_ntv3_asv_by_plant"),
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def _parse_run_dir(path: Path):
    match = re.match(r"^(?P<mode>.+)_scale_(?P<scale>small|medium|large|xlarge)_top200$", path.name)
    if not match:
        return None
    return match.group("mode"), match.group("scale")


def _weighted_by(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        weights = pd.to_numeric(group["n_windows"], errors="coerce").fillna(0.0)
        row = {column: value for column, value in zip(keys, key)}
        if "plant" not in keys and "plant" in group.columns:
            row["n_plants"] = int(group["plant"].nunique())
        row["n_windows"] = int(weights.sum())
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            mask = values.notna() & (weights > 0)
            row[metric] = float((values[mask] * weights[mask]).sum() / weights[mask].sum()) if mask.any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys)


def _weighted_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return _weighted_by(frame, ["mode", "scale"])


def _add_run_columns(frame: pd.DataFrame, run_dir: Path, mode: str, scale: str, plant_slug: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["run_dir"] = str(run_dir)
    frame["mode"] = mode
    frame["scale"] = scale
    frame["plant_slug"] = plant_slug
    ordered = ["run_dir", "mode", "scale", "plant_slug"]
    return frame[ordered + [column for column in frame.columns if column not in ordered]]


def main():
    args = parse_args()
    output_dir = args.output_dir or args.root
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_frames = []
    train_frames = []
    horizon_frames = []
    for run_dir in sorted(args.root.glob("*_scale_*_top200")):
        parsed = _parse_run_dir(run_dir)
        if parsed is None:
            continue
        mode, scale = parsed
        for plant_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
            summary_path = plant_dir / "summary.csv"
            if summary_path.exists():
                frame = pd.read_csv(summary_path)
                summary_frames.append(_add_run_columns(frame, plant_dir, mode, scale, plant_dir.name))

            train_path = plant_dir / "train_summary.csv"
            if train_path.exists():
                frame = pd.read_csv(train_path)
                train_frames.append(_add_run_columns(frame, plant_dir, mode, scale, plant_dir.name))

            horizon_path = plant_dir / "metrics_by_horizon.csv"
            if horizon_path.exists():
                frame = pd.read_csv(horizon_path)
                horizon_frames.append(_add_run_columns(frame, plant_dir, mode, scale, plant_dir.name))

    if not summary_frames:
        raise FileNotFoundError(f"no per-plant summary.csv files found under {args.root}")

    summary = pd.concat(summary_frames, ignore_index=True)
    summary.to_csv(output_dir / "per_plant_summary.csv", index=False)
    _weighted_by(summary, ["mode", "scale", "plant"]).to_csv(output_dir / "per_plant_model_summary.csv", index=False)
    _weighted_summary(summary).to_csv(output_dir / "weighted_summary.csv", index=False)

    if train_frames:
        pd.concat(train_frames, ignore_index=True).to_csv(output_dir / "train_summary.csv", index=False)
    if horizon_frames:
        pd.concat(horizon_frames, ignore_index=True).to_csv(output_dir / "metrics_by_horizon.csv", index=False)

    print(f"Saved per-plant summary: {output_dir / 'per_plant_summary.csv'}")
    print(f"Saved weighted summary: {output_dir / 'weighted_summary.csv'}")


if __name__ == "__main__":
    main()
