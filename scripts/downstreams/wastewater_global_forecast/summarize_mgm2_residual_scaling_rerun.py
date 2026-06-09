#!/usr/bin/env python3
"""Summarize residual MGM2 scaling reruns."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["bray_curtis", "mae", "rmse", "spearman", "top50_overlap"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs_downstreams/wastewater_global_forecast/mgm2_residual_ntv3_asv_rerun"),
    )
    return parser.parse_args()


def _weighted(frame: pd.DataFrame) -> dict:
    weights = pd.to_numeric(frame["n_windows"], errors="coerce").fillna(0.0)
    row = {"n_windows": int(weights.sum()), "n_plants": int(frame["plant"].nunique())}
    for metric in METRICS:
        if metric not in frame.columns:
            continue
        values = pd.to_numeric(frame[metric], errors="coerce")
        mask = values.notna() & (weights > 0)
        row[metric] = float((values[mask] * weights[mask]).sum() / weights[mask].sum()) if mask.any() else np.nan
    return row


def _parse_run(path: Path):
    seed_match = re.match(r"seed_(\d+)$", path.parent.parent.name)
    scale_match = re.match(r"mae_scale_(small|medium|large|xlarge)_top200$", path.parent.name)
    if seed_match is None or scale_match is None:
        return None
    return int(seed_match.group(1)), scale_match.group(1)


def main():
    args = parse_args()
    rows = []
    plant_frames = []
    for summary_path in sorted(args.root.glob("seed_*/mae_scale_*_top200/summary.csv")):
        parsed = _parse_run(summary_path)
        if parsed is None:
            continue
        seed, scale = parsed
        frame = pd.read_csv(summary_path)
        frame = frame[frame["split"] == "test"].copy()
        if frame.empty:
            continue
        frame["seed"] = seed
        frame["scale"] = scale
        frame["run_dir"] = str(summary_path.parent)
        plant_frames.append(frame)
        row = _weighted(frame)
        row.update({"seed": seed, "scale": scale, "run_dir": str(summary_path.parent)})
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"no summary.csv files found under {args.root}")

    by_seed = pd.DataFrame(rows).sort_values(["seed", "scale"])
    by_seed.to_csv(args.root / "weighted_by_seed_scale.csv", index=False)
    pd.concat(plant_frames, ignore_index=True).to_csv(args.root / "plant_summary.csv", index=False)

    agg_rows = []
    for scale, group in by_seed.groupby("scale", dropna=False):
        row = {"scale": scale, "n_seeds": int(group["seed"].nunique())}
        for metric in ["n_windows", "n_plants", *METRICS]:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            if metric in METRICS:
                row[f"{metric}_std"] = float(values.std(ddof=0))
        agg_rows.append(row)
    scale_summary = pd.DataFrame(agg_rows).sort_values("bray_curtis_mean")
    scale_summary.to_csv(args.root / "scale_summary.csv", index=False)

    print(f"Saved weighted by seed/scale: {args.root / 'weighted_by_seed_scale.csv'}")
    print(f"Saved scale summary: {args.root / 'scale_summary.csv'}")
    print(scale_summary.to_string(index=False))


if __name__ == "__main__":
    main()
