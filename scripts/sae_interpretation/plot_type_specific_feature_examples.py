#!/usr/bin/env python3
"""Plot type-specific figures for selected OTU SAE feature interpretations."""

from __future__ import annotations

import argparse
import json
import os
import re
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


DEFAULT_FEATURES = (
    "2131,1821,3915,3435,"
    "686,758,2711,3602,"
    "9,25,2032,2647,"
    "973,1757,2400,"
    "3358,523,"
    "1974,4081"
)

ENV_COLORS = {
    "animal": "#4E79A7",
    "aquatic": "#2A9D8F",
    "soil": "#9C755F",
    "plant": "#59A14F",
    "<missing>": "#8A8A8A",
}
CLASS_COLORS = {
    "Taxonomic identity": "#4E79A7",
    "Ecological niche": "#2A9D8F",
    "Host-associated": "#59A14F",
    "Study/batch/artifact": "#E15759",
    "Abundance/rank": "#F28E2B",
    "Unclear": "#8A8A8A",
}
RANK_COLORS = {
    "kingdom": "#4E79A7",
    "phylum": "#F28E2B",
    "class": "#59A14F",
    "order": "#E15759",
    "family": "#B07AA1",
    "genus": "#9C755F",
    "species": "#76B7B2",
}

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sae_interpretation/otu_xlarge_sample500k_k64_aux0p03125_sparse_topk_global.json"),
    )
    parser.add_argument("--feature-ids", type=str, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=7)
    parser.add_argument("--scatter-sample", type=int, default=1800)
    return parser.parse_args()


def _feature_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _records(mapping: dict, key: str, top_n: int) -> list[dict]:
    value = mapping.get(key, [])
    return value[:top_n] if isinstance(value, list) else []


def _clean_label(value: object, max_len: int = 34) -> str:
    text = "<missing>" if pd.isna(value) else str(value)
    text = text.replace("d__", "").replace("k__", "").replace("p__", "").replace("c__", "")
    text = text.replace("o__", "").replace("f__", "").replace("g__", "").replace("s__", "")
    if len(text) > max_len:
        text = text[: max_len - 1] + "..."
    return text


def _save(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def _barh(ax, rows: list[dict], title: str, color: str, xlabel: str = "Fraction", top_n: int = 7) -> None:
    rows = rows[:top_n]
    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color="#666666")
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontweight="bold")
        return
    labels = [_clean_label(row.get("value", "")) for row in rows][::-1]
    values = [float(row.get("fraction", 0.0)) for row in rows][::-1]
    counts = [int(row.get("count", 0)) for row in rows][::-1]
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, max(1.0, max(values) * 1.15))
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.8, alpha=0.7)
    for yi, value, count in zip(y, values, counts):
        ax.text(value + 0.01, yi, f"{value:.2f} ({count})", va="center", fontsize=7)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)


def _background_map(background: dict, key: str) -> dict[str, float]:
    rows = background.get("distributions", {}).get(key, [])
    return {str(row.get("value", "")): float(row.get("fraction", 0.0)) for row in rows}


def _taxonomy_ladder(evidence: dict) -> list[dict]:
    taxonomy = evidence.get("enrichment", {}).get("taxonomy", {})
    rows = []
    for rank in ["kingdom", "phylum", "class", "order", "family", "genus", "species"]:
        candidates = [row for row in taxonomy.get(rank, []) if row.get("value") not in {"unknown", "<missing>", ""}]
        if not candidates:
            candidates = taxonomy.get(rank, [])[:1]
        if candidates:
            best = candidates[0]
            rows.append({"rank": rank, "value": str(best.get("value", "unknown")), "fraction": float(best.get("fraction", 0.0))})
    return rows


def _top_otu_rows(events: pd.DataFrame, top_n: int) -> list[dict]:
    if events.empty:
        return []
    grouped = (
        events.groupby(["otu_id", "taxonomy"], dropna=False)
        .agg(count=("activation", "size"), mean_activation=("activation", "mean"), max_activation=("activation", "max"))
        .reset_index()
        .sort_values(["mean_activation", "count"], ascending=False)
        .head(top_n)
    )
    rows = []
    for _, row in grouped.iterrows():
        tax = _clean_label(";".join(str(row["taxonomy"]).split(";")[-2:]), 45)
        rows.append({"value": f"{row['otu_id']} | {tax}", "fraction": float(row["mean_activation"]), "count": int(row["count"])})
    return rows


def _plot_title(fig, feature_id: int, short_name: str, primary: str, confidence: str) -> None:
    fig.suptitle(
        f"Feature {feature_id}: {short_name}\n{primary} | confidence={confidence}",
        fontsize=14,
        fontweight="bold",
    )


def _scatter_abundance(ax, events: pd.DataFrame, feature_id: int, sample_n: int, x_col: str = "abundance") -> None:
    data = events.copy()
    for col in ["activation", "abundance", "abundance_rank"]:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["activation", x_col])
    if len(data) > sample_n:
        data = data.sample(sample_n, random_state=feature_id)
    if data.empty:
        ax.text(0.5, 0.5, "No event data", ha="center", va="center")
        ax.set_axis_off()
        return
    env = data.get("primary_env", pd.Series(["<missing>"] * len(data))).fillna("<missing>").astype(str)
    colors = [ENV_COLORS.get(x, "#BAB0AC") for x in env]
    if x_col == "abundance":
        x = np.log10(data[x_col].clip(lower=1e-8))
        xlabel = "log10(relative abundance)"
    else:
        x = data[x_col]
        xlabel = x_col
    ax.scatter(x, data["activation"], c=colors, s=12, alpha=0.45, linewidths=0)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Activation", fontsize=8)
    ax.grid(color="#DDDDDD", linewidth=0.8, alpha=0.7)


def _plot_taxonomic(root: Path, feature_id: int, output_dir: Path, top_n: int, scatter_sample: int, background: dict) -> list[Path]:
    evidence, unified, cls, final, events = _load_feature(root, feature_id)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    _plot_title(fig, feature_id, unified.get("short_name", ""), cls.get("primary_class", ""), final.get("confidence", ""))

    ladder = _taxonomy_ladder(evidence)
    y = np.arange(len(ladder))[::-1]
    axes[0].barh(y, [r["fraction"] for r in ladder][::-1], color=[RANK_COLORS.get(r["rank"], "#999999") for r in ladder][::-1])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{r['rank']}: {_clean_label(r['value'], 24)}" for r in ladder][::-1], fontsize=8)
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("Top taxon fraction")
    axes[0].set_title("Taxonomic ladder", loc="left", fontweight="bold")
    axes[0].grid(axis="x", color="#DDDDDD", alpha=0.7)

    best_rank = evidence.get("diagnostics", {}).get("taxonomy_signal", {}).get("best_rank", "phylum")
    tax_rows = evidence.get("enrichment", {}).get("taxonomy", {}).get(best_rank, [])
    _barh(axes[1], tax_rows, f"{best_rank} composition", "#4E79A7", top_n=top_n)
    _barh(axes[2], _top_otu_rows(events, top_n), "Top OTUs by mean activation", "#F28E2B", xlabel="Mean activation", top_n=top_n)
    return _save(fig, output_dir, f"feature_{feature_id:06d}_taxonomic_identity")


def _plot_ecological(root: Path, feature_id: int, output_dir: Path, top_n: int, scatter_sample: int, background: dict) -> list[Path]:
    evidence, unified, cls, final, events = _load_feature(root, feature_id)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10), constrained_layout=True)
    _plot_title(fig, feature_id, unified.get("short_name", ""), cls.get("primary_class", ""), final.get("confidence", ""))

    env_rows = _records(evidence.get("enrichment", {}).get("metadata", {}), "primary_env", top_n)
    bg = _background_map(background, "primary_env")
    labels = [_clean_label(r.get("value", "")) for r in env_rows]
    top_vals = [float(r.get("fraction", 0.0)) for r in env_rows]
    bg_vals = [bg.get(str(r.get("value", "")), 0.0) for r in env_rows]
    x = np.arange(len(labels))
    axes[0, 0].bar(x - 0.18, top_vals, width=0.36, color="#2A9D8F", label="feature top events")
    axes[0, 0].bar(x + 0.18, bg_vals, width=0.36, color="#BBBBBB", label="background")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axes[0, 0].set_ylim(0, max(1.0, max(top_vals + bg_vals + [0]) * 1.18))
    axes[0, 0].set_title("Environment vs background", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].grid(axis="y", color="#DDDDDD", alpha=0.7)

    _barh(axes[0, 1], _records(evidence.get("enrichment", {}).get("metadata", {}), "env_first2", top_n), "Environment subtypes", "#59A14F", top_n=top_n)
    _scatter_abundance(axes[1, 0], events, feature_id, scatter_sample, "abundance")
    axes[1, 0].set_title("Activation distribution by environment color", loc="left", fontweight="bold")
    _barh(axes[1, 1], evidence.get("enrichment", {}).get("taxonomy", {}).get("phylum", []), "Phylum context", "#4E79A7", top_n=top_n)
    return _save(fig, output_dir, f"feature_{feature_id:06d}_ecological_niche")


def _plot_host(root: Path, feature_id: int, output_dir: Path, top_n: int, scatter_sample: int, background: dict) -> list[Path]:
    evidence, unified, cls, final, events = _load_feature(root, feature_id)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10), constrained_layout=True)
    _plot_title(fig, feature_id, unified.get("short_name", ""), cls.get("primary_class", ""), final.get("confidence", ""))
    _barh(axes[0, 0], _records(evidence.get("enrichment", {}).get("metadata", {}), "env_first2", top_n), "Host/body-site subcontext", "#4E79A7", top_n=top_n)
    _barh(axes[0, 1], evidence.get("enrichment", {}).get("taxonomy", {}).get("class", []), "Class-level host taxa", "#59A14F", top_n=top_n)
    _scatter_abundance(axes[1, 0], events, feature_id, scatter_sample, "abundance_rank")
    axes[1, 0].invert_xaxis()
    axes[1, 0].set_title("Activation vs abundance rank", loc="left", fontweight="bold")
    sample = (
        events.groupby(["sample_id", "env_first2"], dropna=False)["activation"]
        .sum()
        .reset_index()
        .sort_values("activation", ascending=False)
        .head(top_n)
    )
    rows = [{"value": f"{r.sample_id} | {_clean_label(r.env_first2, 18)}", "fraction": float(r.activation), "count": 1} for r in sample.itertuples()]
    _barh(axes[1, 1], rows, "Top host-context samples", "#B07AA1", xlabel="Activation sum", top_n=top_n)
    return _save(fig, output_dir, f"feature_{feature_id:06d}_host_associated")


def _plot_artifact(root: Path, feature_id: int, output_dir: Path, top_n: int, scatter_sample: int, background: dict) -> list[Path]:
    evidence, unified, cls, final, events = _load_feature(root, feature_id)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10), constrained_layout=True)
    _plot_title(fig, feature_id, unified.get("short_name", ""), cls.get("primary_class", ""), final.get("confidence", ""))
    diag = evidence.get("diagnostics", {})
    project = diag.get("project_batch_artifact", {})
    tech = diag.get("technology_bias", {})
    sample = diag.get("sample_concentration", {})
    labels = ["project", "technology", "sample concentration"]
    values = [
        float(project.get("max_project_fraction", 0.0)),
        float(tech.get("top_fraction", tech.get("max_technology_fraction", 0.0))),
        min(1.0, float(sample.get("events_per_sample", 0.0)) / 50.0),
    ]
    colors = ["#E15759" if project.get("suspected") else "#CFCFCF", "#E15759" if tech.get("suspected") else "#CFCFCF", "#E15759" if sample.get("suspected") else "#CFCFCF"]
    axes[0, 0].bar(labels, values, color=colors)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("Artifact diagnostic flags", loc="left", fontweight="bold")
    axes[0, 0].grid(axis="y", color="#DDDDDD", alpha=0.7)
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 0].text(0.02, 0.95, textwrap.fill(final.get("artifact_judgment", ""), 62), transform=axes[0, 0].transAxes, va="top", fontsize=8)

    _barh(axes[0, 1], _records(evidence.get("enrichment", {}).get("metadata", {}), "project", top_n), "Project concentration", "#E15759", top_n=top_n)
    _barh(axes[1, 0], _records(evidence.get("enrichment", {}).get("metadata", {}), "technology", top_n), "Technology composition", "#4E79A7", top_n=top_n)
    _barh(axes[1, 1], evidence.get("enrichment", {}).get("taxonomy", {}).get("phylum", []), "Biological signal caveat", "#9C755F", top_n=top_n)
    return _save(fig, output_dir, f"feature_{feature_id:06d}_artifact_diagnostics")


def _plot_abundance(root: Path, feature_id: int, output_dir: Path, top_n: int, scatter_sample: int, background: dict) -> list[Path]:
    evidence, unified, cls, final, events = _load_feature(root, feature_id)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10), constrained_layout=True)
    _plot_title(fig, feature_id, unified.get("short_name", ""), cls.get("primary_class", ""), final.get("confidence", ""))
    _scatter_abundance(axes[0, 0], events, feature_id, scatter_sample, "abundance")
    axes[0, 0].set_title("Activation vs relative abundance", loc="left", fontweight="bold")
    _scatter_abundance(axes[0, 1], events, feature_id, scatter_sample, "abundance_rank")
    axes[0, 1].invert_xaxis()
    axes[0, 1].set_title("Activation vs abundance rank", loc="left", fontweight="bold")
    data = events.copy()
    data["abundance_rank"] = pd.to_numeric(data["abundance_rank"], errors="coerce")
    data["activation"] = pd.to_numeric(data["activation"], errors="coerce")
    data = data.dropna(subset=["abundance_rank", "activation"])
    data["rank_bin"] = pd.qcut(data["abundance_rank"], q=min(8, max(2, data["abundance_rank"].nunique())), duplicates="drop")
    prof = data.groupby("rank_bin", observed=False)["activation"].mean().reset_index()
    axes[1, 0].plot(range(len(prof)), prof["activation"], marker="o", color="#F28E2B")
    axes[1, 0].set_xticks(range(len(prof)))
    axes[1, 0].set_xticklabels([str(x) for x in prof["rank_bin"]], rotation=35, ha="right", fontsize=7)
    axes[1, 0].set_ylabel("Mean activation")
    axes[1, 0].set_title("Mean activation by rank bin", loc="left", fontweight="bold")
    axes[1, 0].grid(color="#DDDDDD", alpha=0.7)
    corr = evidence.get("enrichment", {}).get("abundance_correlation", {})
    text = "\n".join([f"{k}: {v:.3f}" for k, v in corr.items() if isinstance(v, (int, float))])
    axes[1, 1].axis("off")
    axes[1, 1].text(0, 1, "Abundance/rank diagnostics\n\n" + text + "\n\n" + textwrap.fill(unified.get("abundance_context", ""), 65), va="top", fontsize=10)
    return _save(fig, output_dir, f"feature_{feature_id:06d}_abundance_rank")


def _plot_unclear(root: Path, feature_id: int, output_dir: Path, top_n: int, scatter_sample: int, background: dict) -> list[Path]:
    evidence, unified, cls, final, events = _load_feature(root, feature_id)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10), constrained_layout=True)
    _plot_title(fig, feature_id, unified.get("short_name", ""), cls.get("primary_class", ""), final.get("confidence", ""))
    catalog = pd.read_csv(root / "reports" / "feature_interpretation_merged_catalog.csv")
    row = catalog[catalog["feature_id"] == feature_id].iloc[0]
    labels = ["taxonomy", "environment", "project", "technology", "positive_fraction"]
    values = [
        float(row.get("taxonomy_best_fraction", 0.0)),
        float(row.get("top_primary_env_fraction", 0.0)),
        float(row.get("max_project_fraction", 0.0)),
        float(row.get("max_technology_fraction", 0.0)),
        min(1.0, float(row.get("positive_fraction", 0.0))),
    ]
    axes[0, 0].bar(labels, values, color=["#4E79A7", "#2A9D8F", "#E15759", "#B07AA1", "#F28E2B"])
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("Mixed evidence balance", loc="left", fontweight="bold")
    axes[0, 0].tick_params(axis="x", rotation=30)
    axes[0, 0].grid(axis="y", color="#DDDDDD", alpha=0.7)
    _barh(axes[0, 1], _records(evidence.get("enrichment", {}).get("metadata", {}), "primary_env", top_n), "Environment mixture", "#8A8A8A", top_n=top_n)
    _barh(axes[1, 0], evidence.get("enrichment", {}).get("taxonomy", {}).get("phylum", []), "Taxonomy mixture", "#8A8A8A", top_n=top_n)
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0,
        1,
        "Why this remains unclear\n\n"
        + textwrap.fill(final.get("specificity", ""), 72)
        + "\n\nOvergeneralization risk:\n"
        + textwrap.fill(final.get("overgeneralization_risk", ""), 72),
        va="top",
        fontsize=10,
    )
    return _save(fig, output_dir, f"feature_{feature_id:06d}_unclear_evidence_balance")


def _load_feature(root: Path, feature_id: int):
    evidence = _load_json(root / "evidence" / f"feature_{feature_id:06d}.evidence.json")
    unified = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "unified_description.json")
    cls = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "classification.json")
    final = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "final_validation.json")
    events = pd.read_csv(root / "feature_lookup" / f"feature_{feature_id:06d}.events.csv.gz")
    events["activation"] = pd.to_numeric(events.get("activation"), errors="coerce")
    return evidence, unified, cls, final, events


def main():
    args = parse_args()
    config = load_config(args.config)
    root = Path(cfg(config, "output_dir"))
    output_dir = args.output_dir or root / "reports" / "type_specific_feature_figures"
    background = _load_json(Path(cfg(config, "background_metadata", root / "background_metadata.json")))
    plotters = {
        "Taxonomic identity": _plot_taxonomic,
        "Ecological niche": _plot_ecological,
        "Host-associated": _plot_host,
        "Study/batch/artifact": _plot_artifact,
        "Abundance/rank": _plot_abundance,
        "Unclear": _plot_unclear,
    }
    manifest = []
    for feature_id in _feature_ids(args.feature_ids):
        _, unified, cls, final, _ = _load_feature(root, feature_id)
        primary = cls.get("primary_class", "Unclear")
        plotter = plotters.get(primary, _plot_unclear)
        paths = plotter(root, feature_id, output_dir, args.top_n, args.scatter_sample, background)
        manifest.append(
            {
                "feature_id": feature_id,
                "short_name": unified.get("short_name", ""),
                "primary_class": primary,
                "figure_type": paths[0].stem.replace(f"feature_{feature_id:06d}_", ""),
                "png": str(paths[0]),
                "pdf": str(paths[1]),
                "confidence": final.get("confidence", ""),
                "acceptance": final.get("acceptance", ""),
            }
        )
    manifest_path = output_dir / "type_specific_feature_figures_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(json.dumps({"figures": len(manifest), "manifest": str(manifest_path), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
