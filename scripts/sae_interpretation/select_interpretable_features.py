#!/usr/bin/env python3
"""Select SAE features for interpretation from global deterministic scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.sae_interpretation.config import cfg, load_config


def _parse_ids(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _add_reason(selected: dict[int, set[str]], feature_ids, reason: str) -> None:
    for feature_id in feature_ids:
        selected.setdefault(int(feature_id), set()).add(reason)


def _top(df: pd.DataFrame, column: str, n: int) -> pd.Series:
    if n <= 0 or column not in df.columns:
        return pd.Series([], dtype=np.int64)
    return df.sort_values([column, "feature_id"], ascending=[False, True])["feature_id"].head(n)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sae_interpretation/otu_xlarge_k64_aux0p03.json"))
    parser.add_argument("--feature-scores", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--total-n", type=int, default=500)
    parser.add_argument("--activation-sum-n", type=int, default=160)
    parser.add_argument("--mean-positive-n", type=int, default=120)
    parser.add_argument("--activation-max-n", type=int, default=120)
    parser.add_argument("--rare-strong-n", type=int, default=100)
    parser.add_argument("--rare-positive-fraction-max", type=float, default=0.02)
    parser.add_argument("--curated-feature-ids", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    root = Path(cfg(config, "output_dir"))
    scores_path = args.feature_scores or Path(cfg(config, "feature_scores", root / "feature_scores.csv"))
    output_json = args.output_json or root / "selected_interpretable_feature_ids.json"
    output_csv = args.output_csv or root / "selected_interpretable_features.csv"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(scores_path)
    if "feature_id" not in scores.columns:
        raise SystemExit(f"feature scores missing feature_id column: {scores_path}")

    selected: dict[int, set[str]] = {}
    _add_reason(selected, _parse_ids(args.curated_feature_ids), "curated")
    _add_reason(selected, _top(scores, "activation_sum", args.activation_sum_n), "activation_sum")
    _add_reason(selected, _top(scores, "mean_positive_activation", args.mean_positive_n), "mean_positive_activation")
    _add_reason(selected, _top(scores, "activation_max", args.activation_max_n), "activation_max")

    rare = scores.copy()
    if "positive_fraction" in rare.columns:
        rare = rare[rare["positive_fraction"] <= float(args.rare_positive_fraction_max)].copy()
    if not rare.empty:
        for column in ["mean_positive_activation", "activation_max"]:
            values = rare[column].to_numpy(dtype=np.float64)
            span = float(np.nanmax(values) - np.nanmin(values))
            if span > 0:
                rare[f"{column}_scaled"] = (values - float(np.nanmin(values))) / span
            else:
                rare[f"{column}_scaled"] = 0.0
        rare["rare_strong_score"] = rare["mean_positive_activation_scaled"] + rare["activation_max_scaled"]
        _add_reason(selected, _top(rare, "rare_strong_score", args.rare_strong_n), "rare_strong")

    if len(selected) < int(args.total_n):
        ranked = scores.sort_values(["activation_sum", "feature_id"], ascending=[False, True])["feature_id"]
        needed = int(args.total_n) - len(selected)
        fill = [int(fid) for fid in ranked if int(fid) not in selected][:needed]
        _add_reason(selected, fill, "fill_activation_sum")

    selected_ids = sorted(selected)
    if len(selected_ids) > int(args.total_n):
        reason_priority = {
            "curated": 0,
            "activation_sum": 1,
            "mean_positive_activation": 2,
            "activation_max": 3,
            "rare_strong": 4,
            "fill_activation_sum": 5,
        }

        def key(feature_id: int):
            reasons = selected[feature_id]
            best_reason = min(reason_priority.get(reason, 99) for reason in reasons)
            score_row = scores.loc[scores["feature_id"] == feature_id]
            activation_sum = float(score_row["activation_sum"].iloc[0]) if not score_row.empty and "activation_sum" in score_row else 0.0
            return (best_reason, -activation_sum, feature_id)

        selected_ids = sorted(selected_ids, key=key)[: int(args.total_n)]

    selected_df = scores[scores["feature_id"].isin(selected_ids)].copy()
    selected_df["selection_reasons"] = selected_df["feature_id"].map(
        lambda fid: ";".join(sorted(selected[int(fid)]))
    )
    selected_df = selected_df.sort_values(["selection_reasons", "activation_sum", "feature_id"], ascending=[True, False, True])
    selected_df.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps({"feature_ids": [int(fid) for fid in selected_ids]}, indent=2))
    summary = {
        "feature_scores": str(scores_path),
        "output_json": str(output_json),
        "output_csv": str(output_csv),
        "selected_features": int(len(selected_ids)),
        "requested_total_n": int(args.total_n),
        "reason_counts": selected_df["selection_reasons"].value_counts().to_dict(),
    }
    (output_json.parent / "selected_interpretable_features.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
