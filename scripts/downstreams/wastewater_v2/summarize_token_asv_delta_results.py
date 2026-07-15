#!/usr/bin/env python3
"""Summarize V2 token ASV-delta wastewater forecasting results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from summarize_clr_delta_results import pairwise_wins, plant_extremes, summarize_scales


DEFAULT_RESULT_ROOT = Path("outputs_downstreams/wastewater_v2/token_asv_delta_transformer")
DEFAULT_OLD_MGM2 = Path("outputs_downstreams/wastewater_global_forecast/mgm2_no_residual_ntv3_asv/mae_scale_xlarge_top200/summary.csv")
DEFAULT_BASELINES = Path("outputs_downstreams/wastewater_global_forecast/simple_baselines_top200/summary.csv")


LOWER_IS_BETTER = {"bray_curtis", "mae", "rmse"}
PAIRWISE_METRICS = ["bray_curtis", "mae", "rmse", "spearman"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--old_mgm2_summary", type=Path, default=DEFAULT_OLD_MGM2)
    parser.add_argument("--baseline_summary", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def _read_test_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[frame["split"].eq("test")].copy()


def _pairwise_rows(candidate: pd.DataFrame, reference: pd.DataFrame, comparison: str) -> list[dict]:
    rows = []
    for metric in PAIRWISE_METRICS:
        if metric not in candidate.columns or metric not in reference.columns:
            continue
        joined = candidate.set_index("plant")[[metric, "n_windows"]].join(
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
                "comparison": comparison,
                "metric": metric,
                "wins": int(wins.sum()),
                "n": int(len(joined)),
                "weighted_delta": float((improvement * joined["n_windows"]).sum() / joined["n_windows"].sum()),
                "median_delta": float(improvement.median()),
            }
        )
    return rows


def best_scale_pairwise_wins(
    result_root: Path,
    weighted: pd.DataFrame,
    old_mgm2_summary: Path,
    baseline_summary: Path,
) -> pd.DataFrame:
    best_scale = str(weighted.sort_values("bray_curtis").iloc[0]["scale"])
    candidate_path = result_root / best_scale / "summary.csv"
    if not candidate_path.exists():
        return pd.DataFrame()
    candidate = _read_test_summary(candidate_path)
    rows = []
    if old_mgm2_summary.exists():
        rows.extend(
            _pairwise_rows(
                candidate,
                _read_test_summary(old_mgm2_summary),
                f"token_{best_scale}_vs_old_mgm2_mae_xlarge",
            )
        )
    if baseline_summary.exists():
        baseline = _read_test_summary(baseline_summary)
        if "model" in baseline.columns:
            for model in sorted(baseline["model"].dropna().unique()):
                rows.extend(
                    _pairwise_rows(
                        candidate,
                        baseline[baseline["model"].eq(model)].copy(),
                        f"token_{best_scale}_vs_{model}",
                    )
                )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    output_dir = args.output_dir or args.result_root
    output_dir.mkdir(parents=True, exist_ok=True)

    weighted = summarize_scales(args.result_root)
    weighted["model_group"] = weighted["model_group"].str.replace("v2_clr_delta_", "v2_token_asv_delta_", regex=False)
    weighted.to_csv(output_dir / "weighted_summary.csv", index=False)
    display_cols = [
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
    print("Weighted scale summary:")
    print(weighted[[col for col in display_cols if col in weighted.columns]].to_string(index=False))

    wins = pairwise_wins(args.result_root, args.old_mgm2_summary, args.baseline_summary)
    if not wins.empty:
        wins["comparison"] = wins["comparison"].str.replace("v2_xlarge", "v2_token_asv_xlarge", regex=False)
        wins.to_csv(output_dir / "v2_token_asv_xlarge_pairwise_wins.csv", index=False)
        print("\nV2 token ASV xlarge pairwise wins:")
        print(wins.to_string(index=False))

    best_wins = best_scale_pairwise_wins(args.result_root, weighted, args.old_mgm2_summary, args.baseline_summary)
    if not best_wins.empty:
        best_wins.to_csv(output_dir / "best_scale_pairwise_wins.csv", index=False)
        print("\nV2 token ASV best-scale pairwise wins:")
        print(best_wins.to_string(index=False))

    extremes = plant_extremes(args.result_root)
    if not extremes.empty:
        extremes.to_csv(output_dir / "scale_xlarge_plant_delta_rank.csv", index=False)
        print("\nBest plants by abundance_delta_pearson:")
        print(extremes.head(8).to_string(index=False))
        print("\nWorst plants by abundance_delta_pearson:")
        print(extremes.tail(8).sort_values("abundance_delta_pearson").to_string(index=False))


if __name__ == "__main__":
    main()
