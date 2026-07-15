#!/usr/bin/env python3
"""Verify and summarize MGM2 xlarge OTU SAE feature-analysis outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_TOP_K = {8, 16, 32, 64}
EXPECTED_FIGURE_STEMS = [
    "training_and_analysis_composition",
    "feature_firing_frequency_histograms",
    "feature_density_and_activation_by_primary_env",
    "preferred_primary_env_feature_counts",
    "feature_weight_embedding_by_firing_frequency",
    "feature_weight_embedding_by_primary_env_specificity",
    "sae_feature_analysis_overview",
]
EXPECTED_REPORTS = [
    "README.md",
    "sae_feature_analysis_report.md",
]
EXPECTED_TABLES = [
    "feature_embedding_coordinates.csv",
    "top_environment_specific_features.csv",
    "top_firing_features.csv",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=Path("outputs_sae_analysis/otu_sample500k_feature_analysis"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/sae/otu_sample500k_feature_analysis"))
    parser.add_argument(
        "--expected-index-cache",
        type=Path,
        default=Path("outputs_sae_analysis/otu_sample500k_feature_analysis/selected_indices_val.npy"),
        help="Selected OTU token index cache used to verify full val coverage when not using --allow-preview.",
    )
    parser.add_argument("--allow-preview", action="store_true", help="Do not fail when analysis used --max-tokens.")
    return parser.parse_args()


def _require(path: Path, errors: list[str]):
    if not path.exists():
        errors.append(f"missing: {path}")
        return False
    if path.is_file() and path.stat().st_size == 0:
        errors.append(f"empty file: {path}")
        return False
    return True


def main():
    args = parse_args()
    errors: list[str] = []

    run_summary_path = args.analysis_dir / "run_summary.csv"
    all_feature_path = args.analysis_dir / "all_feature_summary.csv"
    all_group_path = args.analysis_dir / "all_group_summary.csv"
    config_path = args.analysis_dir / "analysis_config.json"
    coordinate_path = args.analysis_dir / "feature_embedding_coordinates.csv"

    for path in [run_summary_path, all_feature_path, all_group_path, config_path]:
        _require(path, errors)

    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2, sort_keys=True))
        raise SystemExit(1)

    run_summary = pd.read_csv(run_summary_path)
    all_features = pd.read_csv(all_feature_path, usecols=["top_k", "feature_id", "firing_frequency", "mean_nonzero_activation"])
    all_groups = pd.read_csv(all_group_path)
    config = json.loads(config_path.read_text())

    observed_top_k = set(int(value) for value in run_summary["top_k"])
    if observed_top_k != EXPECTED_TOP_K:
        errors.append(f"run_summary top_k mismatch: expected {sorted(EXPECTED_TOP_K)}, got {sorted(observed_top_k)}")
    feature_top_k = set(int(value) for value in all_features["top_k"])
    if feature_top_k != EXPECTED_TOP_K:
        errors.append(f"all_feature_summary top_k mismatch: expected {sorted(EXPECTED_TOP_K)}, got {sorted(feature_top_k)}")
    group_top_k = set(int(value) for value in all_groups["top_k"])
    if group_top_k != EXPECTED_TOP_K:
        errors.append(f"all_group_summary top_k mismatch: expected {sorted(EXPECTED_TOP_K)}, got {sorted(group_top_k)}")

    feature_counts = all_features.groupby("top_k")["feature_id"].nunique().to_dict()
    for top_k in EXPECTED_TOP_K:
        count = int(feature_counts.get(top_k, 0))
        if count != 4096:
            errors.append(f"k={top_k} feature count is {count}, expected 4096")

    for top_k in EXPECTED_TOP_K:
        run_matches = run_summary[run_summary["top_k"] == top_k]["run"].tolist()
        if len(run_matches) != 1:
            errors.append(f"k={top_k} has {len(run_matches)} runs in run_summary")
            continue
        run = run_matches[0]
        _require(args.analysis_dir / run / "feature_summary.csv", errors)
        _require(args.analysis_dir / run / "group_summary.csv", errors)

    for stem in EXPECTED_FIGURE_STEMS:
        _require(args.figure_dir / f"{stem}.png", errors)
        _require(args.figure_dir / f"{stem}.pdf", errors)
    for report_name in EXPECTED_REPORTS:
        _require(args.analysis_dir / report_name, errors)
    for table_name in EXPECTED_TABLES:
        _require(args.analysis_dir / table_name, errors)

    if coordinate_path.exists():
        coords = pd.read_csv(coordinate_path, usecols=["top_k", "feature_id", "embedding_method", "embedding_x", "embedding_y"])
        coord_top_k = set(int(value) for value in coords["top_k"])
        if coord_top_k != EXPECTED_TOP_K:
            errors.append(f"feature_embedding_coordinates top_k mismatch: expected {sorted(EXPECTED_TOP_K)}, got {sorted(coord_top_k)}")
        coord_counts = coords.groupby("top_k")["feature_id"].nunique().to_dict()
        for top_k in EXPECTED_TOP_K:
            count = int(coord_counts.get(top_k, 0))
            if count != 4096:
                errors.append(f"k={top_k} coordinate feature count is {count}, expected 4096")

    max_tokens = int(config.get("max_tokens") or 0)
    if max_tokens != 0 and not args.allow_preview:
        errors.append(f"analysis_config max_tokens={max_tokens}; this is a preview/subsample, not full analysis")
    expected_tokens = None
    if args.expected_index_cache.exists():
        expected_tokens = int(len(np.load(args.expected_index_cache, mmap_mode="r")))
        if not args.allow_preview:
            observed_tokens = set(int(value) for value in run_summary["tokens_analyzed"])
            if observed_tokens != {expected_tokens}:
                errors.append(
                    "tokens_analyzed mismatch: "
                    f"expected every top_k to analyze {expected_tokens} tokens from {args.expected_index_cache}, "
                    f"got {sorted(observed_tokens)}"
                )

    summary = run_summary.sort_values("top_k").copy()
    summary_out = args.analysis_dir / "verified_feature_analysis_summary.csv"
    summary.to_csv(summary_out, index=False)

    result = {
        "ok": not errors,
        "analysis_dir": str(args.analysis_dir),
        "figure_dir": str(args.figure_dir),
        "top_k": sorted(observed_top_k),
        "tokens_analyzed_by_top_k": dict(zip(summary["top_k"].astype(int), summary["tokens_analyzed"].astype(int))),
        "feature_counts": {int(k): int(v) for k, v in feature_counts.items()},
        "summary_csv": str(summary_out),
        "max_tokens": max_tokens,
        "expected_tokens": expected_tokens,
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
