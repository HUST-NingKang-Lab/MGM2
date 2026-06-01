#!/usr/bin/env python3
"""Write leave-one-cohort-out metadata splits for Cell Reports FMT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cellreports_utils import safe_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default=Path("data/downstreams/FMT_cellreports"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/downstreams/FMT_cellreports/loco_splits"))
    parser.add_argument("--cohort_column", type=str, default="cohort")
    parser.add_argument("--min_test_events", type=int, default=3)
    return parser.parse_args()


def write_task_splits(table: pd.DataFrame, task: str, args: argparse.Namespace) -> list[dict]:
    rows = []
    for cohort, test in table.groupby(args.cohort_column, sort=True):
        if len(test) < args.min_test_events:
            continue
        train = table[table[args.cohort_column] != cohort].copy()
        if task == "response" and train["response_label"].nunique() < 2:
            continue
        split_dir = args.output_dir / safe_name(str(cohort))
        split_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(split_dir / f"{task}_train_meta.csv", index=False)
        test.to_csv(split_dir / f"{task}_test_meta.csv", index=False)
        rows.append(
            {
                "task": task,
                "cohort": cohort,
                "split_dir": str(split_dir),
                "train_events": int(len(train)),
                "test_events": int(len(test)),
                "test_response_counts": test["response_label"].value_counts().to_dict() if "response_label" in test.columns else {},
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    post = pd.read_csv(args.data_dir / "post_all_meta.csv")
    response = pd.read_csv(args.data_dir / "response_all_meta.csv")
    rows = write_task_splits(post, "post", args) + write_task_splits(response, "response", args)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "loco_split_summary.csv", index=False)
    print(json.dumps({"n_splits": int(summary["cohort"].nunique()) if not summary.empty else 0, "rows": int(len(summary))}, indent=2))


if __name__ == "__main__":
    main()
