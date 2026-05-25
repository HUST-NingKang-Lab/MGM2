#!/usr/bin/env python3
"""Summarize small MGM2 drug-perturbation tuning runs against MLP/baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


METRICS = [
    "bray_curtis_mean",
    "clr_mse_mean",
    "delta_cosine_mean",
    "delta_l1_mae_mean",
    "magnitude_mae_mean",
]
HIGHER_IS_BETTER = {"delta_cosine_mean"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MGM2 quick-tune variants against top1024 MLP and mean baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--quick_root", type=Path, required=True, help="Root containing <variant>/seed_<seed>/fold_metrics.csv")
    parser.add_argument("--mlp_dir", type=Path, default=Path("outputs_downstreams/drug_perturbation/mlp_pair_top1024_full"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_fold_metrics(path: Path, method_prefix: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if method_prefix is not None:
        frame = frame.copy()
        frame["method"] = method_prefix
    required = {"group", "method", *METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame[["group", "method", *METRICS]].copy()


def discover_mgm2_runs(quick_root: Path) -> List[pd.DataFrame]:
    frames = []
    for path in sorted(quick_root.glob("*/seed_*/fold_metrics.csv")):
        variant = path.parents[1].name
        frames.append(load_fold_metrics(path, method_prefix=f"mgm2_{variant}"))
    if not frames:
        raise FileNotFoundError(f"No fold_metrics.csv found under {quick_root}")
    return frames


def metric_improvement(candidate: pd.Series, reference: pd.Series, metric: str) -> pd.Series:
    if metric in HIGHER_IS_BETTER:
        return candidate - reference
    return reference - candidate


def summarize_method(wide_by_metric: Dict[str, pd.DataFrame], method: str, references: List[str]) -> Dict[str, float | str | int]:
    row: Dict[str, float | str | int] = {"method": method}
    first_metric = next(iter(wide_by_metric.values()))
    row["n_folds"] = int(first_metric[method].notna().sum())
    for metric, wide in wide_by_metric.items():
        values = wide[method].dropna()
        row[f"{metric}"] = float(values.mean()) if len(values) else np.nan
        for reference in references:
            if reference not in wide.columns:
                continue
            if method == reference:
                row[f"{metric}_win_vs_{reference}"] = np.nan
                row[f"{metric}_improve_vs_{reference}"] = np.nan
                continue
            common = wide[[method, reference]].dropna()
            if common.empty:
                row[f"{metric}_win_vs_{reference}"] = np.nan
                row[f"{metric}_improve_vs_{reference}"] = np.nan
                continue
            improve = metric_improvement(common[method], common[reference], metric)
            row[f"{metric}_win_vs_{reference}"] = float((improve > 0).mean())
            row[f"{metric}_improve_vs_{reference}"] = float(improve.mean())
    return row


def main() -> None:
    args = parse_args()
    mlp_path = args.mlp_dir / "fold_metrics.csv"
    if not mlp_path.exists():
        raise FileNotFoundError(f"MLP fold metrics not found: {mlp_path}")

    frames = []
    mlp = load_fold_metrics(mlp_path)
    keep_mlp_methods = [
        "no_effect",
        "drug_mean_delta",
        "additive_mean_delta",
        "global_mean_delta",
        "donor_mean_delta",
        "mlp_delta",
    ]
    mgm2_frames = discover_mgm2_runs(args.quick_root)
    quick_groups = sorted(set().union(*(set(frame["group"]) for frame in mgm2_frames)))
    frames.append(mlp[(mlp["method"].isin(keep_mlp_methods)) & (mlp["group"].isin(quick_groups))])
    frames.extend(mgm2_frames)
    all_metrics = pd.concat(frames, ignore_index=True)

    wide_by_metric = {
        metric: all_metrics.pivot_table(index="group", columns="method", values=metric, aggfunc="mean")
        for metric in METRICS
    }
    methods = sorted(all_metrics["method"].unique())
    references = [reference for reference in ["no_effect", "mlp_delta"] if reference in methods]
    summary = pd.DataFrame([summarize_method(wide_by_metric, method, references) for method in methods])

    sort_cols = ["clr_mse_mean", "delta_cosine_mean", "bray_curtis_mean"]
    summary = summary.sort_values(sort_cols, ascending=[True, False, True]).reset_index(drop=True)

    display_cols = [
        "method",
        "n_folds",
        "bray_curtis_mean",
        "clr_mse_mean",
        "delta_cosine_mean",
        "delta_l1_mae_mean",
        "magnitude_mae_mean",
    ]
    for reference in references:
        display_cols.extend(
            [
                f"clr_mse_mean_win_vs_{reference}",
                f"delta_cosine_mean_win_vs_{reference}",
                f"bray_curtis_mean_win_vs_{reference}",
                f"magnitude_mae_mean_win_vs_{reference}",
            ]
        )
    display_cols = [col for col in display_cols if col in summary.columns]

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 240)
    print(summary[display_cols].to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    print("\nBest method counts by fold:")
    for metric, wide in wide_by_metric.items():
        cols = [col for col in summary["method"].tolist() if col in wide.columns]
        if metric in HIGHER_IS_BETTER:
            best = wide[cols].idxmax(axis=1)
        else:
            best = wide[cols].idxmin(axis=1)
        print(f"\n{metric}")
        print(best.value_counts().to_string())

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
