#!/usr/bin/env python3
"""Prepare Cell Reports FMT all-event tables and all-feature community HDF5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cellreports_utils import KINGDOM_FILES, load_metadata, read_summary_workbook, write_h5_community


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default=Path("/data9/zhanghaohong/databases/fmt_strain/cell_reports_data"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/downstreams/FMT_cellreports"))
    parser.add_argument("--kingdoms", nargs="+", default=["bacteria", "archaea", "fungi"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.data_dir / "FMT_metadata_summary.xlsx")

    sample_order: list[str] = []
    sample_seen: set[str] = set()
    kingdom_payloads = []
    for kingdom in args.kingdoms:
        sample_ids, feature_ids, matrix = read_summary_workbook(args.data_dir / KINGDOM_FILES[kingdom], kingdom)
        for sample_id in sample_ids:
            if sample_id not in sample_seen:
                sample_seen.add(sample_id)
                sample_order.append(sample_id)
        kingdom_payloads.append((kingdom, sample_ids, feature_ids, matrix))

    sample_to_idx = {sample_id: idx for idx, sample_id in enumerate(sample_order)}
    feature_ids_all: list[str] = []
    blocks = []
    feature_rows = []
    for kingdom, sample_ids, feature_ids, matrix in kingdom_payloads:
        block = np.zeros((len(sample_order), len(feature_ids)), dtype=np.float32)
        row_indices = [sample_to_idx[sample_id] for sample_id in sample_ids]
        block[row_indices, :] = matrix
        blocks.append(block)
        feature_ids_all.extend(feature_ids)
        feature_rows.extend({"feature_id": feature_id, "kingdom": kingdom, "feature_name": feature_id.split("|", 2)[2]} for feature_id in feature_ids)
    combined = np.concatenate(blocks, axis=1).astype(np.float32)
    write_h5_community(args.output_dir / "community_all.h5", sample_order, feature_ids_all, combined)
    pd.DataFrame(feature_rows).to_csv(args.output_dir / "cellreports_features.tsv", sep="\t", index=False)

    available = set(sample_order)
    metadata["has_donor_recipient"] = [
        row.recipient_sample_id in available and row.donor_sample_id in available for row in metadata.itertuples(index=False)
    ]
    metadata["has_post"] = [
        row.recipient_sample_id in available and row.post_sample_id in available and row.donor_sample_id in available
        for row in metadata.itertuples(index=False)
    ]
    metadata["response_available"] = metadata["response_label"].isin(["responder", "non-responder"])
    event_table = metadata.copy()
    event_table["TargetID"] = event_table["fmt_id"]
    event_table.to_csv(args.output_dir / "fmt_event_table.csv", index=False)

    response = event_table[event_table["has_donor_recipient"] & event_table["response_available"]].copy()
    post = event_table[event_table["has_post"]].copy()
    post["TargetID"] = post["fmt_id"] + "::" + post["post_sample_id"]

    response_cols = ["TargetID", "fmt_id", "study", "indication", "cohort", "donor_sample_id", "recipient_sample_id", "response_label"]
    post_cols = [
        "TargetID",
        "fmt_id",
        "study",
        "indication",
        "cohort",
        "donor_sample_id",
        "recipient_sample_id",
        "post_sample_id",
        "response_label",
    ]
    response[response_cols].to_csv(args.output_dir / "response_all_meta.csv", index=False)
    post[post_cols].to_csv(args.output_dir / "post_all_meta.csv", index=False)

    summary = {
        "kingdoms": args.kingdoms,
        "n_samples": len(sample_order),
        "n_features": len(feature_ids_all),
        "n_response_events": int(len(response)),
        "response_counts": response["response_label"].value_counts().to_dict(),
        "n_post_targets": int(len(post)),
        "n_cohorts": int(event_table["cohort"].nunique()),
    }
    with (args.output_dir / "prepare_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
