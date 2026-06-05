#!/usr/bin/env python3
"""Plot selected OTU SAE feature interpretation evidence panels."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mgm2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mgm2-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.sae_interpretation.config import cfg, load_config


DEFAULT_FEATURES = "2650,2670,2659,2664,2658,2661"
ENV_COLORS = {
    "animal": "#4C78A8",
    "aquatic": "#2A9D8F",
    "plant": "#59A14F",
    "soil": "#9C755F",
    "<missing>": "#8A8A8A",
}
TECH_COLORS = {
    "AMPLICON": "#E15759",
    "WGS": "#4E79A7",
    "RNAseq": "#B07AA1",
    "<missing>": "#8A8A8A",
}
RANK_COLORS = {
    "kingdom": "#4C78A8",
    "phylum": "#F28E2B",
    "class": "#59A14F",
    "order": "#E15759",
    "family": "#B07AA1",
    "genus": "#9C755F",
}

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sae_interpretation/otu_xlarge_k64_aux0p03.json"))
    parser.add_argument("--feature-ids", type=str, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--scatter-sample", type=int, default=1500)
    return parser.parse_args()


def _feature_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _records(mapping: dict, key: str, top_n: int) -> list[dict]:
    rows = mapping.get(key, [])
    return rows[:top_n] if isinstance(rows, list) else []


def _rank_records(evidence: dict, top_n: int) -> tuple[str, list[dict]]:
    taxonomy = evidence.get("enrichment", {}).get("taxonomy", {})
    for rank in ["genus", "family", "order", "class", "phylum"]:
        rows = [row for row in _records(taxonomy, rank, top_n) if row.get("value") != "unknown"]
        if rows and rows[0].get("fraction", 0.0) >= 0.2:
            return rank, _records(taxonomy, rank, top_n)
    for rank in ["phylum", "class", "order", "family", "genus"]:
        rows = _records(taxonomy, rank, top_n)
        if rows:
            return rank, rows
    return "taxonomy", []


def _barh(ax, rows: list[dict], title: str, color: str, xlabel: str = "Fraction") -> None:
    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10, color="#666666")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_axis_off()
        return
    labels = [str(row.get("value", "")) for row in rows][::-1]
    values = [float(row.get("fraction", 0.0)) for row in rows][::-1]
    counts = [int(row.get("count", 0)) for row in rows][::-1]
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, alpha=0.86)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, max(1.0, max(values) * 1.12 if values else 1.0))
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.8, alpha=0.7)
    for yi, value, count in zip(y, values, counts):
        ax.text(value + 0.01, yi, f"{value:.2f} ({count})", va="center", fontsize=7)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)


def _top_otu_records(events: pd.DataFrame, top_n: int) -> list[dict]:
    grouped = (
        events.groupby(["otu_id", "taxonomy"], dropna=False)
        .agg(count=("activation", "size"), mean_activation=("activation", "mean"), max_activation=("activation", "max"))
        .reset_index()
        .sort_values(["mean_activation", "count"], ascending=False)
        .head(top_n)
    )
    rows = []
    for _, row in grouped.iterrows():
        tax = str(row["taxonomy"])
        compact_tax = tax.split(";")[-2:] if ";" in tax else [tax]
        label = f"{row['otu_id']} | {';'.join(compact_tax)}"
        rows.append({"value": label, "fraction": float(row["mean_activation"]), "count": int(row["count"])})
    return rows


def _plot_scatter(ax, events: pd.DataFrame, feature_id: int, sample_n: int) -> None:
    data = events.dropna(subset=["activation", "abundance", "abundance_rank"]).copy()
    if len(data) > sample_n:
        data = data.sample(sample_n, random_state=feature_id)
    data["abundance"] = pd.to_numeric(data["abundance"], errors="coerce")
    data["abundance_rank"] = pd.to_numeric(data["abundance_rank"], errors="coerce")
    data["activation"] = pd.to_numeric(data["activation"], errors="coerce")
    data = data.dropna(subset=["activation", "abundance", "abundance_rank"])
    if data.empty:
        ax.text(0.5, 0.5, "No event data", ha="center", va="center", fontsize=10, color="#666666")
        ax.set_axis_off()
        return
    envs = data["primary_env"].fillna("<missing>").astype(str)
    colors = [ENV_COLORS.get(env, "#BAB0AC") for env in envs]
    x = np.log10(data["abundance"].clip(lower=1e-8))
    ax.scatter(x, data["activation"], c=colors, s=14, alpha=0.5, linewidths=0)
    ax.set_xlabel("log10(relative abundance)", fontsize=8)
    ax.set_ylabel("Feature activation", fontsize=8)
    ax.set_title("Top-event activation vs abundance", loc="left", fontsize=11, fontweight="bold")
    ax.grid(color="#DDDDDD", linewidth=0.8, alpha=0.7)
    present = [env for env in ENV_COLORS if env in set(envs)]
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=ENV_COLORS[env], label=env, markersize=5)
        for env in present[:5]
    ]
    if handles:
        ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper right")


def _plot_feature(root: Path, feature_id: int, output_dir: Path, top_n: int, scatter_sample: int) -> list[Path]:
    evidence = _load_json(root / "evidence" / f"feature_{feature_id:06d}.evidence.json")
    unified = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "unified_description.json")
    classification = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "classification.json")
    final_validation = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "final_validation.json")
    events = pd.read_csv(root / "feature_lookup" / f"feature_{feature_id:06d}.events.csv.gz")

    fig, axes = plt.subplots(3, 2, figsize=(16, 13), constrained_layout=True)
    fig.suptitle(f"Feature {feature_id}: {unified.get('short_name', '')}", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.set_axis_off()
    primary = classification.get("primary_class", "")
    secondary = classification.get("secondary_classes", [])
    classes = "; ".join([primary] + list(secondary)) if primary else "; ".join(secondary)
    stats = evidence.get("global_stats", {})
    artifact = evidence.get("diagnostics", {}).get("study_batch_artifact", {})
    text = "\n".join(
        [
            f"Classes: {classes}",
            f"Acceptance: {final_validation.get('acceptance', '')}",
            f"Confidence: {unified.get('confidence', '')}",
            f"Events / OTUs / samples: {stats.get('lookup_top_events', '')} / "
            f"{stats.get('lookup_unique_otus', '')} / {stats.get('lookup_unique_samples', '')}",
            f"Artifact suspected: {artifact.get('suspected', '')}; "
            f"max project={artifact.get('max_project_fraction', math.nan):.2f}; "
            f"max technology={artifact.get('max_technology_fraction', math.nan):.2f}",
            "",
            textwrap.fill(unified.get("summary", ""), width=82),
        ]
    )
    ax.text(0.0, 1.0, text, ha="left", va="top", fontsize=9, linespacing=1.4)

    rank, rows = _rank_records(evidence, top_n)
    _barh(axes[0, 1], rows, f"Taxonomy enrichment: {rank}", "#4C78A8")

    metadata = evidence.get("enrichment", {}).get("metadata", {})
    _barh(axes[1, 0], _records(metadata, "primary_env", top_n), "Primary environment", "#59A14F")

    project_rows = _records(metadata, "project", min(top_n, 6))
    technology_rows = _records(metadata, "technology", min(top_n, 4))
    combined = project_rows + [{"value": f"technology:{r['value']}", "count": r["count"], "fraction": r["fraction"]} for r in technology_rows]
    _barh(axes[1, 1], combined, "Project and technology concentration", "#E15759")

    _plot_scatter(axes[2, 0], events, feature_id, scatter_sample)
    _barh(axes[2, 1], _top_otu_records(events, top_n), "Top OTUs by mean activation", "#F28E2B", xlabel="Mean activation")

    path = output_dir / f"feature_{feature_id:06d}_evidence_panel.png"
    pdf_path = output_dir / f"feature_{feature_id:06d}_evidence_panel.pdf"
    fig.savefig(path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [path, pdf_path]


def _plot_overview(root: Path, feature_ids: list[int], output_dir: Path) -> list[Path]:
    rows = []
    for feature_id in feature_ids:
        evidence = _load_json(root / "evidence" / f"feature_{feature_id:06d}.evidence.json")
        unified = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "unified_description.json")
        classification = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "classification.json")
        artifact = evidence.get("diagnostics", {}).get("study_batch_artifact", {})
        stats = evidence.get("global_stats", {})
        rows.append(
            {
                "feature_id": feature_id,
                "short_name": unified.get("short_name", str(feature_id)),
                "primary_class": classification.get("primary_class", ""),
                "max_project_fraction": float(artifact.get("max_project_fraction", 0.0)),
                "max_technology_fraction": float(artifact.get("max_technology_fraction", 0.0)),
                "unique_otus": int(stats.get("lookup_unique_otus", 0)),
                "unique_samples": int(stats.get("lookup_unique_samples", 0)),
                "artifact_suspected": bool(artifact.get("suspected", False)),
            }
        )
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    labels = [f"{row.feature_id}\n{row.short_name}" for row in df.itertuples()]
    x = np.arange(len(df))
    width = 0.38
    axes[0].bar(x - width / 2, df["max_project_fraction"], width=width, color="#E15759", label="max project")
    axes[0].bar(x + width / 2, df["max_technology_fraction"], width=width, color="#4E79A7", label="max technology")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Artifact diagnostics", loc="left", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Fraction among top events")
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.7)
    axes[0].legend(frameon=False)

    colors = ["#E15759" if row.artifact_suspected else "#59A14F" for row in df.itertuples()]
    axes[1].scatter(df["unique_samples"], df["unique_otus"], s=130, c=colors, alpha=0.85, edgecolor="white")
    for row in df.itertuples():
        axes[1].text(row.unique_samples, row.unique_otus, str(row.feature_id), fontsize=8, ha="center", va="center")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Unique samples in lookup top events")
    axes[1].set_ylabel("Unique OTUs in lookup top events")
    axes[1].set_title("Coverage profile", loc="left", fontsize=12, fontweight="bold")
    axes[1].grid(color="#DDDDDD", linewidth=0.8, alpha=0.7)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="#E15759", label="artifact suspected", markersize=8),
        plt.Line2D([0], [0], marker="o", linestyle="", color="#59A14F", label="artifact not suspected", markersize=8),
    ]
    axes[1].legend(handles=handles, frameon=False)

    path = output_dir / "selected_features_overview.png"
    pdf_path = output_dir / "selected_features_overview.pdf"
    fig.savefig(path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [path, pdf_path]


def _format_taxon_label(rows: list[dict], max_items: int = 3) -> str:
    if not rows:
        return "no data"
    parts = []
    for row in rows[:max_items]:
        value = str(row.get("value", ""))
        fraction = float(row.get("fraction", 0.0))
        parts.append(f"{value} {fraction:.0%}")
    return "\n".join(parts)


def _plot_taxonomy_ladder(root: Path, feature_ids: list[int], output_dir: Path) -> list[Path]:
    ranks = ["kingdom", "phylum", "class", "order", "family", "genus"]
    fig, axes = plt.subplots(len(feature_ids), len(ranks), figsize=(20, 2.25 * len(feature_ids)), constrained_layout=True)
    if len(feature_ids) == 1:
        axes = np.array([axes])

    for row_idx, feature_id in enumerate(feature_ids):
        evidence = _load_json(root / "evidence" / f"feature_{feature_id:06d}.evidence.json")
        unified = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "unified_description.json")
        taxonomy = evidence.get("enrichment", {}).get("taxonomy", {})
        for col_idx, rank in enumerate(ranks):
            ax = axes[row_idx, col_idx]
            records = _records(taxonomy, rank, 5)
            top = records[0] if records else {}
            top_fraction = float(top.get("fraction", 0.0))
            ax.barh([0], [top_fraction], color=RANK_COLORS[rank], alpha=0.86)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.7, 0.7)
            ax.set_yticks([])
            ax.set_xticks([0, 0.5, 1.0] if row_idx == len(feature_ids) - 1 else [])
            if row_idx == 0:
                ax.set_title(rank, fontsize=11, fontweight="bold")
            label = _format_taxon_label(records, max_items=3)
            ax.text(0.02, 0, label, ha="left", va="center", fontsize=8)
            ax.grid(axis="x", color="#DDDDDD", linewidth=0.8, alpha=0.7)
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)
        label_ax = axes[row_idx, 0]
        label_ax.set_ylabel(
            f"{feature_id}\n{unified.get('short_name', '')}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=9,
            labelpad=68,
        )

    fig.suptitle("Selected OTU SAE Features: Taxonomy Ladder", fontsize=16, fontweight="bold")
    path = output_dir / "selected_features_taxonomy_ladder.png"
    pdf_path = output_dir / "selected_features_taxonomy_ladder.pdf"
    fig.savefig(path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [path, pdf_path]


def main():
    args = parse_args()
    config = load_config(args.config)
    root = Path(cfg(config, "output_dir"))
    output_dir = args.output_dir or root / "reports" / "selected_feature_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_ids = _feature_ids(args.feature_ids)

    paths = []
    paths.extend(_plot_overview(root, feature_ids, output_dir))
    paths.extend(_plot_taxonomy_ladder(root, feature_ids, output_dir))
    for feature_id in feature_ids:
        paths.extend(_plot_feature(root, feature_id, output_dir, args.top_n, args.scatter_sample))
    print(json.dumps({"figures": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
