#!/usr/bin/env python3
"""Build cached wastewater forecasting windows from organized MC-prediction data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.data.WastewaterDataset import build_wastewater_windows, save_wastewater_windows


DEFAULT_DATA_ROOT = Path("/home/zhanghaohong/data6/databases/wastewater/MC-prediction/data_organized")
DEFAULT_OUTPUT = Path("data/downstreams/wastewater/wastewater_windows_ctx10_h10.pkl")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT, help="Path to data_organized")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output pickle path")
    parser.add_argument("--context_len", type=int, default=10, help="Number of context samples")
    parser.add_argument("--horizon", type=int, default=10, help="Number of future samples to predict")
    parser.add_argument(
        "--plants",
        nargs="*",
        default=None,
        help="Optional plant names. If omitted, all plant directories are used.",
    )
    parser.add_argument("--top_k_asvs", type=int, default=0, help="Keep top-K ASVs per plant; 0 keeps all")
    parser.add_argument(
        "--add_other_bucket",
        action="store_true",
        help="When top-K is used, add an __OTHER__ bucket for remaining ASV mass",
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=None,
        help="Optional path for a JSON summary. Defaults to output path with .summary.json suffix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = build_wastewater_windows(
        data_root=args.data_root,
        context_len=args.context_len,
        horizon=args.horizon,
        plants=args.plants,
        top_k_asvs=args.top_k_asvs,
        add_other_bucket=args.add_other_bucket,
    )
    save_wastewater_windows(dataset, args.output)

    summary_path = args.summary_json or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as handle:
        json.dump(dataset["summary"], handle, indent=2, ensure_ascii=False)

    print(f"Saved wastewater windows: {args.output}")
    print(f"Saved summary: {summary_path}")
    print(f"Plants: {dataset['summary']['num_plants']}")
    print(f"Windows: {dataset['summary']['num_windows']}")
    print(f"Split windows: {dataset['summary']['split_window_counts']}")
    warnings = dataset["summary"].get("warnings", [])
    if warnings:
        print("Warnings:")
        for warning in warnings[:20]:
            print(f"  - {warning}")
        if len(warnings) > 20:
            print(f"  ... {len(warnings) - 20} more")


if __name__ == "__main__":
    main()
