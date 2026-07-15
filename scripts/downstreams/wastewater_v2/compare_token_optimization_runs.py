#!/usr/bin/env python3
"""Compare base and optimized token ASV-delta runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUNS = {
    "base": Path("outputs_downstreams/wastewater_v2/token_asv_delta_transformer/weighted_summary.csv"),
    "change_focus": Path("outputs_downstreams/wastewater_v2/token_asv_delta_transformer_optimized/change_focus/weighted_summary.csv"),
    "conservative": Path("outputs_downstreams/wastewater_v2/token_asv_delta_transformer_optimized/conservative/weighted_summary.csv"),
}

METRICS = [
    "bray_curtis",
    "mae",
    "rmse",
    "spearman",
    "abundance_delta_pearson",
    "direction_accuracy",
    "top_variable_asv_delta_pearson",
    "mean_asv_delta_pearson",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs_downstreams/wastewater_v2/token_optimization_comparison.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    frames = []
    for run_name, path in RUNS.items():
        if not path.exists():
            print(f"[skip] missing {run_name}: {path}")
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "run", run_name)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("no weighted_summary.csv files found")
    combined = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)

    cols = ["run", "scale", "n_windows", "n_plants", *[metric for metric in METRICS if metric in combined.columns]]
    print(combined[cols].sort_values(["bray_curtis", "run"]).to_string(index=False))

    for metric, ascending in [
        ("bray_curtis", True),
        ("abundance_delta_pearson", False),
        ("top_variable_asv_delta_pearson", False),
        ("direction_accuracy", False),
    ]:
        if metric not in combined.columns:
            continue
        best = combined.sort_values(metric, ascending=ascending).iloc[0]
        print(f"best {metric}: {best['run']} {best['scale']} = {best[metric]:.6f}")
    print(f"Saved comparison: {args.output}")


if __name__ == "__main__":
    main()
