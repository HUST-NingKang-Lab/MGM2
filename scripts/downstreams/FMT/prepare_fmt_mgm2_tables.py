#!/usr/bin/env python3
"""Prepare MGnify-style FMT downstream tables and metadata.

This script does not build hidden-state caches. It creates standard community
tables that are then converted by src/data/build_dataset.py, matching the
MGnify_mst downstream organization:

  data/downstreams/FMT/response_train.pkl
  data/downstreams/FMT/response_test.pkl
  data/downstreams/FMT/response_train_meta_split.csv
  data/downstreams/FMT/response_test_meta_split.csv
  data/downstreams/FMT/response_train.resolved_embeddings.h5
  data/downstreams/FMT/response_test.resolved_embeddings.h5

For response prediction, each model sample is a synthetic donor-recipient pair
community. Recipient and donor features are kept as separate tokens by appending
an extra role level after the seven taxonomy ranks; build_dataset.py ignores the
extra level for fallback matching, but the feature IDs remain unique.

The script also writes post-community target metadata for future custom FMT
regression models.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path("/home/zhanghaohong/data6/databases/fmt_strain/data")
DEFAULT_OUTPUT_DIR = Path("data/downstreams/FMT")
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MGnify-style FMT tables for MGM2 downstream training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_motus", type=int, default=1024)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_post_timepoint", type=float, default=0.0)
    return parser.parse_args()


def clean_sample_id(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_taxon(value) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"^\d+\s+", "", str(value).strip())
    if text.lower() in {"", "nan", "none", "unknown", "unclassified"}:
        return ""
    return text


def feature_label(row: pd.Series, role: str | None = None) -> str:
    levels = [clean_taxon(row.get(rank, "")) for rank in RANKS]
    levels = [level for level in levels if level]
    if not levels:
        levels = ["unknown"]
    levels.append(str(row["mOTU.ID"]))
    if role:
        levels.append(role)
    return ";".join(levels)


def choose_motus(rel: pd.DataFrame, taxonomy: pd.DataFrame, max_motus: int) -> pd.DataFrame:
    stats = pd.DataFrame(
        {
            "mOTU.ID": rel.index.astype(str),
            "prevalence": (rel > 0).mean(axis=1).to_numpy(),
            "mean_abundance": rel.mean(axis=1).to_numpy(),
        }
    )
    selected = stats.merge(taxonomy, on="mOTU.ID", how="left")
    selected = selected.sort_values(["prevalence", "mean_abundance", "mOTU.ID"], ascending=[False, False, True])
    if max_motus > 0:
        selected = selected.head(max_motus)
    selected = selected.reset_index(drop=True)
    selected["feature_id"] = selected.apply(lambda row: feature_label(row), axis=1)
    selected["recipient_feature_id"] = selected.apply(lambda row: feature_label(row, "recipient"), axis=1)
    selected["donor_feature_id"] = selected.apply(lambda row: feature_label(row, "donor"), axis=1)
    return selected


def build_post_lookup(samples: pd.DataFrame, abundance_samples: set[str], min_post_timepoint: float) -> Dict[str, pd.DataFrame]:
    samples = samples.copy()
    samples["sample.name"] = samples["sample.name"].map(clean_sample_id)
    samples["fmt.id"] = samples["fmt.id"].map(clean_sample_id)
    samples["timepoint.fmt.numeric"] = pd.to_numeric(samples["timepoint.fmt"], errors="coerce")
    post = samples[
        (samples["subject_type"].astype(str) == "recipient")
        & (samples["timepoint.fmt.numeric"] > min_post_timepoint)
        & (samples["sample.name"].isin(abundance_samples))
    ].copy()
    return {
        str(fmt_id): group.sort_values(["timepoint.fmt.numeric", "sample.name"])
        for fmt_id, group in post.groupby("fmt.id")
    }


def stratified_event_split(events: pd.DataFrame, key_col: str, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    train_ids = []
    test_ids = []
    for _, group in events.groupby(key_col, dropna=False):
        ids = group["fmt_id"].drop_duplicates().to_numpy(dtype=object)
        rng.shuffle(ids)
        if len(ids) <= 1:
            train_ids.extend(ids.tolist())
            continue
        n_test = int(round(len(ids) * test_ratio))
        n_test = min(max(1, n_test), len(ids) - 1)
        test_ids.extend(ids[:n_test].tolist())
        train_ids.extend(ids[n_test:].tolist())
    return set(train_ids), set(test_ids)


def build_event_tables(fmt: pd.DataFrame, samples: pd.DataFrame, rel: pd.DataFrame, min_post_timepoint: float):
    abundance_samples = set(rel.columns)
    post_lookup = build_post_lookup(samples, abundance_samples, min_post_timepoint)
    event_rows = []
    post_rows = []
    for _, row in fmt.iterrows():
        if str(row.get("fmt.type", "")) != "FMT":
            continue
        fmt_id = clean_sample_id(row.get("fmt.id", ""))
        donor_sample = clean_sample_id(row.get("sample.donor", ""))
        recipient_sample = clean_sample_id(row.get("sample.pre_fmt", ""))
        has_donor_recipient = donor_sample in abundance_samples and recipient_sample in abundance_samples
        response = row.get("clinical.response.summary")
        response_label = str(response) if pd.notna(response) else "missing"
        post_frame = post_lookup.get(fmt_id, pd.DataFrame())
        event_rows.append(
            {
                "fmt_id": fmt_id,
                "study": row.get("study"),
                "indication": row.get("indication"),
                "fmt_type": row.get("fmt.type"),
                "donor_sample_id": donor_sample,
                "recipient_sample_id": recipient_sample,
                "donor_subject": row.get("subject.donor"),
                "recipient_subject": row.get("subject.recipient"),
                "response_label": response_label,
                "response_available": response_label in {"responder", "non-responder"},
                "has_donor_recipient": has_donor_recipient,
                "n_post_samples": int(len(post_frame)),
                "community_prediction_ready": bool(has_donor_recipient and len(post_frame) > 0),
            }
        )
        if not has_donor_recipient or post_frame.empty:
            continue
        for _, post_row in post_frame.iterrows():
            post_rows.append(
                {
                    "TargetID": f"{fmt_id}::{clean_sample_id(post_row['sample.name'])}",
                    "fmt_id": fmt_id,
                    "study": row.get("study"),
                    "indication": row.get("indication"),
                    "fmt_type": row.get("fmt.type"),
                    "donor_sample_id": donor_sample,
                    "recipient_sample_id": recipient_sample,
                    "post_sample_id": clean_sample_id(post_row["sample.name"]),
                    "timepoint_fmt": float(post_row["timepoint.fmt.numeric"]),
                    "timepoint_bin": post_row.get("timepoint.bin"),
                    "response_label": response_label,
                }
            )
    return pd.DataFrame(event_rows), pd.DataFrame(post_rows)


def write_pair_response_table(rel_selected: pd.DataFrame, selected: pd.DataFrame, events: pd.DataFrame, output_path: Path) -> None:
    recipient_values = []
    donor_values = []
    columns = []
    for _, row in events.iterrows():
        columns.append(row["TargetID"])
        recipient_values.append(rel_selected[row["recipient_sample_id"]].to_numpy(dtype=np.float32))
        donor_values.append(rel_selected[row["donor_sample_id"]].to_numpy(dtype=np.float32))
    if not columns:
        raise ValueError(f"No response examples for {output_path}")
    matrix = np.vstack([np.stack(recipient_values, axis=1), np.stack(donor_values, axis=1)])
    index = selected["recipient_feature_id"].tolist() + selected["donor_feature_id"].tolist()
    frame = pd.DataFrame(matrix, index=index, columns=columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path)


def write_community_table(rel_selected: pd.DataFrame, selected: pd.DataFrame, sample_ids: List[str], output_path: Path) -> None:
    sample_ids = sorted(set(sample_ids))
    if not sample_ids:
        raise ValueError(f"No community samples for {output_path}")
    frame = rel_selected[sample_ids].copy()
    frame.index = selected["feature_id"].tolist()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fmt = pd.read_csv(args.data_dir / "fmt_metadata.csv")
    samples = pd.read_csv(args.data_dir / "sample_metadata.csv")
    taxonomy = pd.read_csv(args.data_dir / "mOTU_taxonomy.csv")
    rel = pd.read_csv(args.data_dir / "mOTU_rel_abundance.csv", index_col=0)
    rel.index = rel.index.astype(str)

    selected = choose_motus(rel, taxonomy, args.max_motus)
    rel_selected = rel.loc[selected["mOTU.ID"].astype(str).tolist()]
    event_table, post_targets = build_event_tables(fmt, samples, rel_selected, args.min_post_timepoint)

    response_events = event_table[
        event_table["has_donor_recipient"] & event_table["response_available"]
    ].copy()
    if response_events.empty:
        raise RuntimeError("No response-labeled donor-recipient FMT events found")
    train_ids, test_ids = stratified_event_split(response_events, "response_label", args.test_ratio, args.seed)

    response_events["split"] = np.where(response_events["fmt_id"].isin(test_ids), "test", "train")
    response_events["TargetID"] = response_events["fmt_id"]
    post_targets["split"] = np.where(post_targets["fmt_id"].isin(test_ids), "test", "train")

    train_response = response_events[response_events["split"] == "train"].copy()
    test_response = response_events[response_events["split"] == "test"].copy()
    train_post = post_targets[post_targets["split"] == "train"].copy()
    test_post = post_targets[post_targets["split"] == "test"].copy()

    response_meta_cols = [
        "TargetID",
        "fmt_id",
        "study",
        "indication",
        "donor_sample_id",
        "recipient_sample_id",
        "response_label",
    ]
    train_response[response_meta_cols].to_csv(args.output_dir / "response_train_meta_split.csv", index=False)
    test_response[response_meta_cols].to_csv(args.output_dir / "response_test_meta_split.csv", index=False)
    train_post.to_csv(args.output_dir / "post_train_meta_split.csv", index=False)
    test_post.to_csv(args.output_dir / "post_test_meta_split.csv", index=False)
    event_table.to_csv(args.output_dir / "fmt_event_table.csv", index=False)
    selected.to_csv(args.output_dir / "selected_motus.tsv", sep="\t", index=False)

    write_pair_response_table(rel_selected, selected, train_response, args.output_dir / "response_train_otu_table.csv")
    write_pair_response_table(rel_selected, selected, test_response, args.output_dir / "response_test_otu_table.csv")

    train_samples = train_post[["donor_sample_id", "recipient_sample_id", "post_sample_id"]].to_numpy().ravel().tolist()
    test_samples = test_post[["donor_sample_id", "recipient_sample_id", "post_sample_id"]].to_numpy().ravel().tolist()
    write_community_table(rel_selected, selected, train_samples, args.output_dir / "community_train_otu_table.csv")
    write_community_table(rel_selected, selected, test_samples, args.output_dir / "community_test_otu_table.csv")

    summary = {
        "n_selected_motus": int(len(selected)),
        "n_response_events": int(len(response_events)),
        "n_response_train": int(len(train_response)),
        "n_response_test": int(len(test_response)),
        "response_train_counts": train_response["response_label"].value_counts().to_dict(),
        "response_test_counts": test_response["response_label"].value_counts().to_dict(),
        "n_post_targets": int(len(post_targets)),
        "n_post_train_targets": int(len(train_post)),
        "n_post_test_targets": int(len(test_post)),
        "n_community_train_samples": int(len(set(train_samples))),
        "n_community_test_samples": int(len(set(test_samples))),
    }
    with (args.output_dir / "prepare_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
