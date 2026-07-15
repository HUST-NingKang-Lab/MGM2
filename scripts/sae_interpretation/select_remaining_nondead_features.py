#!/usr/bin/env python3
"""Select SAE features that still need interpretation, excluding dead features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.sae_interpretation.config import cfg, load_config


def _read_feature_ids(path: Path | None) -> set[int]:
    if path is None or not path.exists():
        return set()
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("feature_ids", data.get("selected_feature_ids", []))
    return {int(feature_id) for feature_id in data}


def _read_feature_dirs(path: Path | None) -> set[int]:
    if path is None or not path.exists():
        return set()
    feature_ids = set()
    for child in path.glob("feature_*"):
        if not child.is_dir():
            continue
        try:
            feature_ids.add(int(child.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return feature_ids


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sae_interpretation/otu_xlarge_sample500k_k64_aux0p03125_sparse_topk_global.json"),
    )
    parser.add_argument("--feature-scores", type=Path, default=None)
    parser.add_argument("--existing-outputs-dir", type=Path, default=None)
    parser.add_argument("--exclude-feature-ids-file", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--min-positive-count", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=None, help="Optional pilot limit after filtering.")
    parser.add_argument(
        "--sort-by",
        choices=["feature_id", "activation_sum", "positive_count", "activation_max", "mean_positive_activation"],
        default="feature_id",
    )
    parser.add_argument("--descending", action="store_true", help="Sort descending for score-like columns.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    root = Path(cfg(config, "output_dir"))
    scores_path = args.feature_scores or Path(cfg(config, "feature_scores", root / "feature_scores.csv"))
    existing_outputs_dir = args.existing_outputs_dir or root / "codex_outputs"
    exclude_ids_file = args.exclude_feature_ids_file or root / "selected_interpretable_feature_ids.json"

    scores = pd.read_csv(scores_path)
    required = {"feature_id", "positive_count"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise SystemExit(f"feature scores missing required columns {missing}: {scores_path}")

    scores["feature_id"] = scores["feature_id"].astype(int)
    scores["positive_count"] = pd.to_numeric(scores["positive_count"], errors="coerce").fillna(0).astype("int64")
    dead_mask = scores["positive_count"] < int(args.min_positive_count)
    dead_ids = set(scores.loc[dead_mask, "feature_id"].astype(int))

    excluded_ids = _read_feature_ids(exclude_ids_file) | _read_feature_dirs(existing_outputs_dir)
    remaining = scores[~scores["feature_id"].isin(dead_ids | excluded_ids)].copy()
    ascending = not bool(args.descending)
    if args.sort_by == "feature_id":
        ascending = True
    remaining = remaining.sort_values([args.sort_by, "feature_id"], ascending=[ascending, True])
    if args.max_features is not None:
        remaining = remaining.head(int(args.max_features)).copy()

    feature_ids = remaining["feature_id"].astype(int).tolist()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "feature_ids": feature_ids,
                "selection": "remaining_nondead",
                "source_feature_scores": str(scores_path),
                "excluded_outputs_dir": str(existing_outputs_dir),
                "excluded_feature_ids_file": str(exclude_ids_file),
                "min_positive_count": int(args.min_positive_count),
            },
            indent=2,
            sort_keys=True,
        )
    )
    remaining.to_csv(args.output_csv, index=False)

    summary = {
        "feature_scores": str(scores_path),
        "total_features": int(len(scores)),
        "dead_features": int(len(dead_ids)),
        "excluded_already_interpreted_or_selected": int(len(excluded_ids)),
        "remaining_nondead_features": int(len(feature_ids)),
        "output_json": str(args.output_json),
        "output_csv": str(args.output_csv),
        "existing_outputs_dir": str(existing_outputs_dir),
        "exclude_feature_ids_file": str(exclude_ids_file),
        "min_positive_count": int(args.min_positive_count),
        "max_features": args.max_features,
    }
    summary_path = args.output_json.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
