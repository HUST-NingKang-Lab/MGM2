#!/usr/bin/env python3
"""Prepare all-event FMT tables for standard internal 5-fold experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_fmt_mgm2_tables import (  # noqa: E402
    DEFAULT_DATA_DIR,
    choose_motus,
    build_event_tables,
    write_community_table,
)


DEFAULT_OUTPUT_DIR = Path("data/downstreams/FMT_5folds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare all-event FMT community tables for internal 5-fold downstream runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_motus", type=int, default=1024)
    parser.add_argument("--min_post_timepoint", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fmt = pd.read_csv(args.data_dir / "fmt_metadata.csv")
    samples = pd.read_csv(args.data_dir / "sample_metadata.csv")
    taxonomy = pd.read_csv(args.data_dir / "mOTU_taxonomy.csv")
    rel = pd.read_csv(args.data_dir / "mOTU_rel_abundance.csv", index_col=0)
    rel.index = rel.index.astype(str)

    selected = choose_motus(rel, taxonomy, args.max_motus)
    # For FMT_5folds we use precomputed mOTU embeddings directly. Keep the
    # table feature ids as mOTU ids instead of taxonomy strings so
    # src/data/build_dataset.py can perform direct embedding lookup.
    selected["feature_id"] = selected["mOTU.ID"].astype(str)
    rel_selected = rel.loc[selected["mOTU.ID"].astype(str).tolist()]
    event_table, post_targets = build_event_tables(fmt, samples, rel_selected, args.min_post_timepoint)

    response_events = event_table[
        event_table["has_donor_recipient"] & event_table["response_available"]
    ].copy()
    if response_events.empty:
        raise RuntimeError("No response-labeled donor-recipient FMT events found")

    response_events["TargetID"] = response_events["fmt_id"]
    post_targets = post_targets.copy()

    response_meta_cols = [
        "TargetID",
        "fmt_id",
        "study",
        "indication",
        "donor_sample_id",
        "recipient_sample_id",
        "response_label",
    ]
    post_meta_cols = [
        "TargetID",
        "fmt_id",
        "study",
        "indication",
        "fmt_type",
        "donor_sample_id",
        "recipient_sample_id",
        "post_sample_id",
        "timepoint_fmt",
        "timepoint_bin",
        "response_label",
    ]
    response_events[response_meta_cols].to_csv(args.output_dir / "response_all_meta.csv", index=False)
    post_targets[post_meta_cols].to_csv(args.output_dir / "post_all_meta.csv", index=False)
    event_table.to_csv(args.output_dir / "fmt_event_table.csv", index=False)
    selected.to_csv(args.output_dir / "selected_motus.tsv", sep="\t", index=False)

    response_samples = response_events[["donor_sample_id", "recipient_sample_id"]].to_numpy().ravel().tolist()
    post_samples = post_targets[["donor_sample_id", "recipient_sample_id", "post_sample_id"]].to_numpy().ravel().tolist()
    all_samples = response_samples + post_samples
    write_community_table(rel_selected, selected, all_samples, args.output_dir / "community_all_otu_table.csv")

    summary = {
        "n_selected_motus": int(len(selected)),
        "n_response_events": int(len(response_events)),
        "response_counts": response_events["response_label"].value_counts().to_dict(),
        "n_post_targets": int(len(post_targets)),
        "n_post_events": int(post_targets["fmt_id"].nunique()),
        "post_response_counts": post_targets["response_label"].value_counts().to_dict(),
        "n_community_all_samples": int(len(set(all_samples))),
    }
    with (args.output_dir / "prepare_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
