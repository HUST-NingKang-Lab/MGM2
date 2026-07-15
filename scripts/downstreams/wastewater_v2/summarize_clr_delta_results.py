#!/usr/bin/env python3
"""Summarize V2 CLR-delta wastewater forecasting results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_RESULT_ROOT = Path("outputs_downstreams/wastewater_v2/clr_delta_transformer")
DEFAULT_OLD_MGM2 = Path("outputs_downstreams/wastewater_global_forecast/mgm2_no_residual_ntv3_asv/mae_scale_xlarge_top200/summary.csv")
DEFAULT_BASELINES = Path("outputs_downstreams/wastewater_global_forecast/simple_baselines_top200/summary.csv")

METRICS = [
    "bray_curtis",
    "mae",
    "rmse",
    "spearman",
    "abundance_delta_pearson",
    "abundance_delta_mae",
    "direction_accuracy",
    "mean_asv_delta_pearson",
    "median_asv_delta_pearson",
    "top_variable_asv_delta_pearson",
]

LOWER_IS_BETTER = {"bray_curtis", "mae", "rmse", "abundance_delta_mae"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--old_mgm2_summary", type=Path, default=DEFAULT_OLD_MGM2)
    parser.add_argument("--baseline_summary", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def weighted_mean(frame: pd.DataFrame, metric: str, weight_col: str = "n_windows") -> float:
    valid = frame[metric].notna()
    if not valid.any():
        return float("nan")
    weights = frame.loc[valid, weight_col].astype(float)
    values = frame.loc[valid, metric].astype(float)
    return float((values * weights).sum() / weights.sum())


def summarize_scales(result_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(result_root.glob("scale_*/summary.csv")):
        frame = pd.read_csv(path)
        frame = frame[frame["split"].eq("test")].copy()
        if frame.empty:
            continue
        row = {
            "model_group": f"v2_clr_delta_{path.parent.name}",
            "scale": path.parent.name,
            "n_windows": int(frame["n_windows"].sum()),
            "n_plants": int(frame["plant"].nunique()),
        }
        for metric in METRICS:
            if metric in frame.columns:
                row[metric] = weighted_mean(frame, metric)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no scale summary.csv files found under {result_root}")
    return pd.DataFrame(rows).sort_values("bray_curtis")


def _pairwise_rows(v2: pd.DataFrame, reference: pd.DataFrame, reference_name: str, metrics: Iterable[str]):
    rows = []
    for metric in metrics:
        if metric not in v2.columns or metric not in reference.columns:
            continue
        joined = v2.set_index("plant")[[metric, "n_windows"]].join(
            reference.set_index("plant")[[metric]].rename(columns={metric: "reference"}),
            how="inner",
        )
        if joined.empty:
            continue
        if metric in LOWER_IS_BETTER:
            improvement = joined["reference"] - joined[metric]
            wins = joined[metric] < joined["reference"]
        else:
            improvement = joined[metric] - joined["reference"]
            wins = joined[metric] > joined["reference"]
        rows.append(
            {
                "comparison": reference_name,
                "metric": metric,
                "wins": int(wins.sum()),
                "n": int(len(joined)),
                "weighted_delta": float((improvement * joined["n_windows"]).sum() / joined["n_windows"].sum()),
                "median_delta": float(improvement.median()),
            }
        )
    return rows


def pairwise_wins(result_root: Path, old_mgm2_summary: Path, baseline_summary: Path) -> pd.DataFrame:
    v2_path = result_root / "scale_xlarge" / "summary.csv"
    if not v2_path.exists():
        return pd.DataFrame()
    v2 = pd.read_csv(v2_path)
    v2 = v2[v2["split"].eq("test")].copy()
    rows = []
    metrics = ["bray_curtis", "mae", "rmse", "spearman"]
    if old_mgm2_summary.exists():
        old = pd.read_csv(old_mgm2_summary)
        old = old[old["split"].eq("test")].copy()
        rows.extend(_pairwise_rows(v2, old, "v2_xlarge_vs_old_mgm2_mae_xlarge", metrics))
    if baseline_summary.exists():
        base = pd.read_csv(baseline_summary)
        base = base[base["split"].eq("test")].copy()
        if "model" in base.columns:
            for model in sorted(base["model"].dropna().unique()):
                rows.extend(_pairwise_rows(v2, base[base["model"].eq(model)].copy(), f"v2_xlarge_vs_{model}", metrics))
    return pd.DataFrame(rows)


def plant_extremes(result_root: Path) -> pd.DataFrame:
    path = result_root / "scale_xlarge" / "summary.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame = frame[frame["split"].eq("test")].copy()
    keep = [
        "plant",
        "n_windows",
        "bray_curtis",
        "spearman",
        "abundance_delta_pearson",
        "direction_accuracy",
        "top_variable_asv_delta_pearson",
        "mean_asv_delta_pearson",
    ]
    return frame[[col for col in keep if col in frame.columns]].sort_values("abundance_delta_pearson", ascending=False)


def main():
    args = parse_args()
    output_dir = args.output_dir or args.result_root
    output_dir.mkdir(parents=True, exist_ok=True)

    weighted = summarize_scales(args.result_root)
    weighted.to_csv(output_dir / "weighted_summary.csv", index=False)
    print("Weighted scale summary:")
    print(
        weighted[
            [
                "model_group",
                "n_windows",
                "n_plants",
                "bray_curtis",
                "mae",
                "rmse",
                "spearman",
                "abundance_delta_pearson",
                "direction_accuracy",
                "top_variable_asv_delta_pearson",
            ]
        ].to_string(index=False)
    )

    wins = pairwise_wins(args.result_root, args.old_mgm2_summary, args.baseline_summary)
    if not wins.empty:
        wins.to_csv(output_dir / "v2_xlarge_pairwise_wins.csv", index=False)
        print("\nV2 xlarge pairwise wins:")
        print(wins.to_string(index=False))

    extremes = plant_extremes(args.result_root)
    if not extremes.empty:
        extremes.to_csv(output_dir / "scale_xlarge_plant_delta_rank.csv", index=False)
        print("\nBest plants by abundance_delta_pearson:")
        print(extremes.head(8).to_string(index=False))
        print("\nWorst plants by abundance_delta_pearson:")
        print(extremes.tail(8).sort_values("abundance_delta_pearson").to_string(index=False))


if __name__ == "__main__":
    main()
