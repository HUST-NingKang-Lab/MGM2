#!/usr/bin/env python3
"""Dataset evaluation for MGM2 FMT prediction experiments.

The script summarizes whether the FMT strain dataset supports two downstream
tasks:
  1. donor + recipient pre-FMT community -> post-FMT community
  2. donor + recipient pre-FMT community -> clinical response

It writes compact tables and figures intended for paper-ready dataset
description and model feasibility checks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path("/home/zhanghaohong/data6/databases/fmt_strain/data")
DEFAULT_OUTPUT_DIR = Path("outputs_downstreams/FMT/dataset_eval")
ABUNDANCE_THRESHOLD = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and evaluate the fmt_strain dataset for MGM2 FMT prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--abundance_threshold", type=float, default=ABUNDANCE_THRESHOLD)
    return parser.parse_args()


def clean_sample_id(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def bray_curtis(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.clip(a.sum() + b.sum(), 1e-12, None))
    return float(np.abs(a - b).sum() / denom)


def shannon(profile: np.ndarray) -> float:
    total = float(profile.sum())
    if total <= 0:
        return 0.0
    p = profile[profile > 0] / total
    return float(-(p * np.log(p)).sum())


def richness(profile: np.ndarray, threshold: float) -> int:
    return int((profile > threshold).sum())


def safe_value_counts(frame: pd.DataFrame, column: str, name: str) -> pd.DataFrame:
    out = (
        frame[column]
        .fillna("missing")
        .astype(str)
        .value_counts()
        .rename_axis(name)
        .reset_index(name="n")
    )
    out["fraction"] = out["n"] / max(out["n"].sum(), 1)
    return out


def save_barh(frame: pd.DataFrame, label_col: str, value_col: str, title: str, output: Path, max_rows: int = 25) -> None:
    plot_frame = frame.head(max_rows).iloc[::-1]
    fig_height = max(3.5, 0.28 * len(plot_frame) + 1.0)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(plot_frame[label_col].astype(str), plot_frame[value_col], color="#4E79A7")
    ax.set_xlabel("Count")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_boxplot(frame: pd.DataFrame, columns: List[str], labels: List[str], title: str, ylabel: str, output: Path) -> None:
    values = [frame[col].dropna().to_numpy() for col in columns]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch in ax.patches:
        patch.set_facecolor("#A0CBE8")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_grouped_boxplot(frame: pd.DataFrame, group_col: str, value_col: str, title: str, ylabel: str, output: Path, max_groups: int = 10) -> None:
    counts = frame[group_col].fillna("missing").astype(str).value_counts()
    groups = counts.head(max_groups).index.tolist()
    values = [frame.loc[frame[group_col].fillna("missing").astype(str) == group, value_col].dropna().to_numpy() for group in groups]
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(groups)), 4.5))
    ax.boxplot(values, tick_labels=groups, showfliers=False, patch_artist=True)
    for patch in ax.patches:
        patch.set_facecolor("#59A14F")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_inputs(data_dir: Path):
    fmt = pd.read_csv(data_dir / "fmt_metadata.csv")
    samples = pd.read_csv(data_dir / "sample_metadata.csv")
    rel = pd.read_csv(data_dir / "mOTU_rel_abundance.csv", index_col=0)
    counts = pd.read_csv(data_dir / "mOTU_count_abundance.csv", index_col=0)
    taxonomy = pd.read_csv(data_dir / "mOTU_taxonomy.csv")
    phenotype = pd.read_csv(data_dir / "phenotype.csv")
    return fmt, samples, rel, counts, taxonomy, phenotype


def abundance_vector(rel: pd.DataFrame, sample_id: str) -> np.ndarray:
    return rel[sample_id].to_numpy(dtype=np.float32)


def build_event_and_target_tables(fmt: pd.DataFrame, samples: pd.DataFrame, rel: pd.DataFrame, threshold: float):
    abundance_samples = set(rel.columns)
    samples = samples.copy()
    samples["sample.name"] = samples["sample.name"].map(clean_sample_id)
    samples["fmt.id"] = samples["fmt.id"].map(clean_sample_id)
    samples["timepoint.fmt.numeric"] = pd.to_numeric(samples["timepoint.fmt"], errors="coerce")

    post_samples_by_fmt: Dict[str, pd.DataFrame] = {}
    recipient_post = samples[
        (samples["subject_type"].astype(str) == "recipient")
        & (samples["timepoint.fmt.numeric"] > 0)
        & (samples["sample.name"].isin(abundance_samples))
    ].copy()
    for fmt_id, group in recipient_post.groupby("fmt.id"):
        post_samples_by_fmt[str(fmt_id)] = group.sort_values(["timepoint.fmt.numeric", "sample.name"])

    event_rows = []
    target_rows = []
    for _, row in fmt.iterrows():
        fmt_id = clean_sample_id(row.get("fmt.id", ""))
        donor_sample = clean_sample_id(row.get("sample.donor", ""))
        pre_sample = clean_sample_id(row.get("sample.pre_fmt", ""))
        donor_available = donor_sample in abundance_samples
        pre_available = pre_sample in abundance_samples
        post_frame = post_samples_by_fmt.get(fmt_id, pd.DataFrame())
        n_post = int(len(post_frame))
        n_post_timepoints = int(post_frame["timepoint.fmt.numeric"].nunique()) if n_post else 0
        has_response = pd.notna(row.get("clinical.response.summary"))

        event_rows.append(
            {
                "fmt_id": fmt_id,
                "study": row.get("study"),
                "fmt_type": row.get("fmt.type"),
                "indication": row.get("indication"),
                "donor_sample": donor_sample,
                "pre_sample": pre_sample,
                "donor_available": donor_available,
                "pre_available": pre_available,
                "n_post_samples": n_post,
                "n_post_timepoints": n_post_timepoints,
                "has_post_sample": n_post > 0,
                "has_donor_pre": donor_available and pre_available,
                "community_prediction_ready": donor_available and pre_available and n_post > 0,
                "has_clinical_response": bool(has_response),
                "response_summary": row.get("clinical.response.summary"),
                "response_raw": row.get("clinical.response"),
                "donor_relation": row.get("donor.relation"),
                "fmt_route": row.get("fmt.route"),
                "fmt_sample_prep": row.get("fmt.sample_prep"),
                "antibiotics_summary": row.get("antibiotics.summary"),
                "bowel_prep_summary": row.get("bowel.prep.summary"),
            }
        )

        if not (donor_available and pre_available and n_post > 0):
            continue

        donor = abundance_vector(rel, donor_sample)
        pre = abundance_vector(rel, pre_sample)
        bray_pre_donor = bray_curtis(pre, donor)
        donor_unique = (donor > threshold) & (pre <= threshold)
        recipient_unique = (pre > threshold) & (donor <= threshold)
        donor_richness = richness(donor, threshold)
        pre_richness = richness(pre, threshold)

        for _, post_row in post_frame.iterrows():
            post_sample = clean_sample_id(post_row["sample.name"])
            post = abundance_vector(rel, post_sample)
            post_sum = float(np.clip(post.sum(), 1e-12, None))
            donor_unique_fraction = float(post[donor_unique].sum() / post_sum)
            recipient_unique_fraction = float(post[recipient_unique].sum() / post_sum)
            bray_pre_post = bray_curtis(pre, post)
            bray_donor_post = bray_curtis(donor, post)
            target_rows.append(
                {
                    "fmt_id": fmt_id,
                    "study": row.get("study"),
                    "fmt_type": row.get("fmt.type"),
                    "indication": row.get("indication"),
                    "response_summary": row.get("clinical.response.summary"),
                    "response_raw": row.get("clinical.response"),
                    "donor_sample": donor_sample,
                    "pre_sample": pre_sample,
                    "post_sample": post_sample,
                    "timepoint_fmt": float(post_row.get("timepoint.fmt.numeric")),
                    "timepoint_bin": post_row.get("timepoint.bin"),
                    "bray_pre_donor": bray_pre_donor,
                    "bray_pre_post": bray_pre_post,
                    "bray_donor_post": bray_donor_post,
                    "post_closer_to_donor_than_pre": bray_donor_post < bray_pre_post,
                    "donor_unique_taxa": int(donor_unique.sum()),
                    "recipient_unique_taxa": int(recipient_unique.sum()),
                    "donor_unique_post_fraction": donor_unique_fraction,
                    "recipient_unique_post_fraction": recipient_unique_fraction,
                    "donor_richness": donor_richness,
                    "pre_richness": pre_richness,
                    "post_richness": richness(post, threshold),
                    "donor_shannon": shannon(donor),
                    "pre_shannon": shannon(pre),
                    "post_shannon": shannon(post),
                }
            )

    return pd.DataFrame(event_rows), pd.DataFrame(target_rows)


def summarize_task_feasibility(event_table: pd.DataFrame, target_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fmt_type in ["all", "FMT", "placebo"]:
        events = event_table if fmt_type == "all" else event_table[event_table["fmt_type"] == fmt_type]
        targets = target_table if fmt_type == "all" else target_table[target_table["fmt_type"] == fmt_type]
        ready_events = events[events["community_prediction_ready"]]
        response_ready = events[
            (events["fmt_type"] == "FMT")
            & events["has_donor_pre"]
            & events["has_clinical_response"]
        ]
        rows.append(
            {
                "fmt_type": fmt_type,
                "n_events": int(len(events)),
                "n_events_with_donor_pre": int(events["has_donor_pre"].sum()),
                "n_events_with_post": int(events["has_post_sample"].sum()),
                "n_community_prediction_events": int(len(ready_events)),
                "n_community_prediction_targets": int(len(targets)),
                "n_response_prediction_events": int(len(response_ready)),
                "n_studies": int(events["study"].nunique()),
                "n_indications": int(events["indication"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(args, fmt, samples, rel, counts, taxonomy, phenotype, event_table, target_table):
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_nonzero = int((counts.to_numpy() > 0).sum())
    matrix_total = int(np.prod(counts.shape))
    overview = pd.DataFrame(
        [
            {"item": "clinical_studies", "value": int(fmt["study"].nunique())},
            {"item": "fmt_events_total", "value": int(len(fmt))},
            {"item": "fmt_events_actual_FMT", "value": int((fmt["fmt.type"] == "FMT").sum())},
            {"item": "fmt_events_placebo", "value": int((fmt["fmt.type"] == "placebo").sum())},
            {"item": "sample_metadata_rows", "value": int(len(samples))},
            {"item": "abundance_samples", "value": int(rel.shape[1])},
            {"item": "mOTU_features", "value": int(rel.shape[0])},
            {"item": "taxonomy_rows_mOTU", "value": int(len(taxonomy))},
            {"item": "phenotype_species_rows", "value": int(len(phenotype))},
            {"item": "count_matrix_sparsity", "value": float(1.0 - matrix_nonzero / matrix_total)},
        ]
    )
    overview.to_csv(output_dir / "table_dataset_overview.csv", index=False)

    event_table.to_csv(output_dir / "fmt_event_coverage.csv", index=False)
    target_table.to_csv(output_dir / "fmt_post_targets_with_similarity.csv", index=False)

    feasibility = summarize_task_feasibility(event_table, target_table)
    feasibility.to_csv(output_dir / "table_task_feasibility.csv", index=False)

    ready_fmt_events = event_table[
        (event_table["fmt_type"] == "FMT") & event_table["community_prediction_ready"]
    ].copy()
    if not ready_fmt_events.empty:
        ready_by_indication = ready_fmt_events.groupby("indication", dropna=False).agg(
            n_ready_events=("fmt_id", "size"),
            n_response_labeled=("has_clinical_response", "sum"),
            n_responders=("response_summary", lambda s: int((s == "responder").sum())),
            n_non_responders=("response_summary", lambda s: int((s == "non-responder").sum())),
            n_studies=("study", "nunique"),
        ).reset_index().sort_values(["n_ready_events", "indication"], ascending=[False, True])
        ready_by_indication.to_csv(output_dir / "table_ready_events_by_indication.csv", index=False)

    if not target_table.empty:
        targets_by_study_indication = target_table.groupby(["study", "indication"], dropna=False).agg(
            n_targets=("post_sample", "size"),
            n_events=("fmt_id", "nunique"),
            n_timepoint_bins=("timepoint_bin", "nunique"),
            median_timepoint_fmt=("timepoint_fmt", "median"),
        ).reset_index().sort_values(["n_targets", "study"], ascending=[False, True])
        targets_by_study_indication.to_csv(output_dir / "table_targets_by_study_indication.csv", index=False)

        timepoint_by_response = target_table[target_table["fmt_type"] == "FMT"].copy()
        timepoint_by_response["response_summary"] = timepoint_by_response["response_summary"].fillna("missing")
        pd.crosstab(
            timepoint_by_response["timepoint_bin"],
            timepoint_by_response["response_summary"],
        ).to_csv(output_dir / "table_timepoint_by_response.csv")

    study_counts = safe_value_counts(fmt, "study", "study")
    indication_counts = safe_value_counts(fmt, "indication", "indication")
    response_counts = safe_value_counts(fmt, "clinical.response.summary", "response_summary")
    type_counts = safe_value_counts(fmt, "fmt.type", "fmt_type")
    timepoint_counts = safe_value_counts(target_table, "timepoint_bin", "timepoint_bin") if not target_table.empty else pd.DataFrame()
    for name, frame in [
        ("table_fmt_events_by_study.csv", study_counts),
        ("table_fmt_events_by_indication.csv", indication_counts),
        ("table_clinical_response_distribution.csv", response_counts),
        ("table_fmt_type_distribution.csv", type_counts),
        ("table_post_target_timepoints.csv", timepoint_counts),
    ]:
        frame.to_csv(output_dir / name, index=False)

    if not target_table.empty:
        similarity_summary = target_table.groupby(["fmt_type"], dropna=False).agg(
            n_targets=("post_sample", "size"),
            bray_pre_post_median=("bray_pre_post", "median"),
            bray_donor_post_median=("bray_donor_post", "median"),
            bray_pre_donor_median=("bray_pre_donor", "median"),
            donor_closer_fraction=("post_closer_to_donor_than_pre", "mean"),
            donor_unique_post_fraction_median=("donor_unique_post_fraction", "median"),
            post_richness_median=("post_richness", "median"),
            post_shannon_median=("post_shannon", "median"),
        ).reset_index()
        similarity_summary.to_csv(output_dir / "table_similarity_and_engraftment_summary.csv", index=False)

        response_similarity = target_table[target_table["fmt_type"] == "FMT"].groupby(["response_summary"], dropna=False).agg(
            n_targets=("post_sample", "size"),
            bray_pre_post_median=("bray_pre_post", "median"),
            bray_donor_post_median=("bray_donor_post", "median"),
            donor_closer_fraction=("post_closer_to_donor_than_pre", "mean"),
            donor_unique_post_fraction_median=("donor_unique_post_fraction", "median"),
        ).reset_index()
        response_similarity.to_csv(output_dir / "table_similarity_by_response.csv", index=False)

    save_barh(study_counts, "study", "n", "FMT events by study", output_dir / "fig_fmt_events_by_study.png")
    save_barh(indication_counts, "indication", "n", "FMT events by indication", output_dir / "fig_fmt_events_by_indication.png")
    save_barh(response_counts, "response_summary", "n", "Clinical response labels", output_dir / "fig_clinical_response_distribution.png")
    if not timepoint_counts.empty:
        save_barh(timepoint_counts, "timepoint_bin", "n", "Post-FMT target samples by timepoint bin", output_dir / "fig_post_target_timepoints.png")
    if not target_table.empty:
        ready_fmt = target_table[target_table["fmt_type"] == "FMT"].copy()
        save_boxplot(
            ready_fmt,
            ["bray_pre_donor", "bray_pre_post", "bray_donor_post"],
            ["pre vs donor", "pre vs post", "donor vs post"],
            "Community dissimilarity in FMT-ready targets",
            "Bray-Curtis dissimilarity",
            output_dir / "fig_bray_relationships_fmt.png",
        )
        save_grouped_boxplot(
            ready_fmt,
            "timepoint_bin",
            "donor_unique_post_fraction",
            "Donor-unique post-FMT abundance fraction by timepoint",
            "Fraction of post-FMT abundance",
            output_dir / "fig_donor_unique_fraction_by_timepoint.png",
        )
        response_ready = ready_fmt[ready_fmt["response_summary"].notna()].copy()
        if not response_ready.empty:
            save_grouped_boxplot(
                response_ready,
                "response_summary",
                "donor_unique_post_fraction",
                "Donor-unique post-FMT abundance fraction by response",
                "Fraction of post-FMT abundance",
                output_dir / "fig_donor_unique_fraction_by_response.png",
                max_groups=6,
            )
            save_grouped_boxplot(
                response_ready,
                "response_summary",
                "bray_donor_post",
                "Donor-post dissimilarity by clinical response",
                "Bray-Curtis dissimilarity",
                output_dir / "fig_donor_post_bray_by_response.png",
                max_groups=6,
            )

    summary = {
        "data_dir": str(args.data_dir),
        "output_dir": str(output_dir),
        "abundance_threshold": float(args.abundance_threshold),
        "overview": {row["item"]: row["value"] for _, row in overview.iterrows()},
        "task_feasibility": feasibility.to_dict(orient="records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Dataset overview")
    print(overview.to_string(index=False))
    print("\nTask feasibility")
    print(feasibility.to_string(index=False))
    if not target_table.empty:
        print("\nSimilarity and engraftment summary")
        print(pd.read_csv(output_dir / "table_similarity_and_engraftment_summary.csv").to_string(index=False))
    print(f"\nWrote outputs to {output_dir}")


def main() -> None:
    args = parse_args()
    fmt, samples, rel, counts, taxonomy, phenotype = load_inputs(args.data_dir)
    event_table, target_table = build_event_and_target_tables(fmt, samples, rel, args.abundance_threshold)
    write_outputs(args, fmt, samples, rel, counts, taxonomy, phenotype, event_table, target_table)


if __name__ == "__main__":
    main()
