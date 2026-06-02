#!/usr/bin/env python3
"""Inspect Cell Reports FMT metadata and feature-table coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cellreports_utils import KINGDOM_FILES, load_metadata, normalize_sample_id, xlsx_first_column_values


def table_sample_ids(path: Path) -> set[str]:
    return {normalize_sample_id(value) for value in xlsx_first_column_values(path) if value != ""}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default=Path("/home/zhanghaohong/data6/databases/fmt_strain/cell_reports_data"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/downstreams/FMT_cellreports"))
    parser.add_argument("--kingdoms", nargs="+", default=["bacteria", "archaea", "fungi"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.data_dir / "FMT_metadata_summary.xlsx")
    sample_sets = {kingdom: table_sample_ids(args.data_dir / KINGDOM_FILES[kingdom]) for kingdom in args.kingdoms}
    available = set().union(*sample_sets.values())
    metadata["covered"] = [
        row.recipient_sample_id in available and row.post_sample_id in available and row.donor_sample_id in available
        for row in metadata.itertuples(index=False)
    ]
    metadata["response_available"] = metadata["response_label"].isin(["responder", "non-responder"])

    cohort = (
        metadata.groupby("cohort")
        .agg(
            events=("fmt_id", "size"),
            covered_events=("covered", "sum"),
            response_events=("response_available", "sum"),
            covered_response_events=("response_available", lambda s: int((s & metadata.loc[s.index, "covered"]).sum())),
            responders=("response_code", lambda s: int((s == 1).sum())),
            non_responders=("response_code", lambda s: int((s == 0).sum())),
            disease_code=("disease_code", "first"),
        )
        .reset_index()
        .sort_values(["disease_code", "cohort"])
    )
    coverage_rows = []
    all_metadata_samples = set(metadata["recipient_sample_id"]) | set(metadata["post_sample_id"]) | set(metadata["donor_sample_id"])
    for kingdom, samples in sample_sets.items():
        coverage_rows.append(
            {
                "kingdom": kingdom,
                "samples": len(samples),
                "metadata_sample_matches": len(samples & all_metadata_samples),
                "metadata_pre_matches": len(samples & set(metadata["recipient_sample_id"])),
                "metadata_post_matches": len(samples & set(metadata["post_sample_id"])),
                "metadata_donor_matches": len(samples & set(metadata["donor_sample_id"])),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    metadata.to_csv(args.output_dir / "cellreports_metadata_normalized.csv", index=False)
    cohort.to_csv(args.output_dir / "cohort_summary.csv", index=False)
    coverage.to_csv(args.output_dir / "coverage_summary.csv", index=False)
    summary = {
        "events": int(len(metadata)),
        "covered_events": int(metadata["covered"].sum()),
        "response_events": int(metadata["response_available"].sum()),
        "covered_response_events": int((metadata["covered"] & metadata["response_available"]).sum()),
        "cohorts": int(metadata["cohort"].nunique()),
    }
    with (args.output_dir / "inspect_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
