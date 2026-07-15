#!/usr/bin/env python3
"""Plot macro-level statistics for OTU SAE feature interpretations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mgm2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mgm2-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.sae_interpretation.config import cfg, load_config


plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})

CLASS_COLORS = {
    "Taxonomic identity": "#4E79A7",
    "Ecological niche": "#2A9D8F",
    "Host-associated": "#59A14F",
    "Study/batch/artifact": "#E15759",
    "Abundance/rank": "#F28E2B",
    "Community interaction/co-occurrence": "#B07AA1",
    "Disease-associated context": "#EDC948",
    "Perturbation/transition": "#9C755F",
    "Unclear": "#8A8A8A",
}
CONFIDENCE_ORDER = ["high", "medium", "low", "unknown"]
CONFIDENCE_COLORS = {"high": "#2A9D8F", "medium": "#F28E2B", "low": "#E15759", "unknown": "#BDBDBD"}
ENV_COLORS = {"animal": "#4E79A7", "aquatic": "#2A9D8F", "soil": "#9C755F", "plant": "#59A14F", "<missing>": "#8A8A8A"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sae_interpretation/otu_xlarge_sample500k_k64_aux0p03125_sparse_topk_global.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _safe_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean(value: object, max_len: int = 38) -> str:
    text = "<missing>" if pd.isna(value) else str(value)
    for prefix in ["d__", "k__", "p__", "c__", "o__", "f__", "g__", "s__"]:
        text = text.replace(prefix, "")
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _list_first(rows: list[dict], key: str, default="unknown"):
    if not rows:
        return default
    return rows[0].get(key, default)


def _collect_summary(root: Path) -> pd.DataFrame:
    records = []
    evidence_paths = sorted((root / "evidence").glob("feature_*.evidence.json"))
    for evidence_path in evidence_paths:
        feature_id = int(evidence_path.stem.split(".")[0].split("_")[1])
        evidence = _load_json(evidence_path)
        classification = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "classification.json")
        final_validation = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "final_validation.json")
        unified = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "unified_description.json")
        diagnostics = evidence.get("diagnostics", {})
        enrichment = evidence.get("enrichment", {})
        metadata = enrichment.get("metadata", {})
        taxonomy = enrichment.get("taxonomy", {})
        global_stats = evidence.get("global_stats", {})
        tax_diag = diagnostics.get("taxonomy_signal", {})
        env_diag = diagnostics.get("environment_enrichment", {})
        project_diag = diagnostics.get("project_batch_artifact", {})
        tech_diag = diagnostics.get("technology_bias", {})
        sample_diag = diagnostics.get("sample_concentration", {})
        bio_diag = diagnostics.get("biological_signal_strength", {})
        abundance_corr = enrichment.get("abundance_correlation", {})
        top_tax_rank = tax_diag.get("best_rank", "unknown")
        top_tax_rows = taxonomy.get(str(top_tax_rank), [])
        top_env_rows = metadata.get("primary_env", [])
        record = {
            "feature_id": feature_id,
            "short_name": unified.get("short_name", ""),
            "primary_class": classification.get("primary_class", "Unclear"),
            "secondary_classes": ";".join(classification.get("secondary_classes", []) or []),
            "classification_confidence": classification.get("confidence", "unknown"),
            "final_confidence": final_validation.get("confidence", classification.get("confidence", "unknown")),
            "acceptance": final_validation.get("acceptance", "unknown"),
            "taxonomy_best_fraction": _safe_float(tax_diag.get("best_fraction")),
            "taxonomy_best_rank": top_tax_rank,
            "taxonomy_best_value": tax_diag.get("best_value", _list_first(top_tax_rows, "value")),
            "taxonomy_strong": _as_bool(tax_diag.get("strong", False)),
            "taxonomy_moderate": _as_bool(tax_diag.get("moderate", False)),
            "environment_top": env_diag.get("top_environment", _list_first(top_env_rows, "value")),
            "environment_max_fraction": _safe_float(env_diag.get("max_environment_fraction")),
            "environment_background_fraction": _safe_float(env_diag.get("background_fraction")),
            "environment_enrichment_ratio": _safe_float(env_diag.get("enrichment_ratio")),
            "environment_strong": _as_bool(env_diag.get("strong", False)),
            "project_top": project_diag.get("top_project", "unknown"),
            "project_max_fraction": _safe_float(project_diag.get("max_project_fraction")),
            "project_background_fraction": _safe_float(project_diag.get("background_fraction")),
            "project_enrichment_ratio": _safe_float(project_diag.get("enrichment_ratio")),
            "project_artifact_suspected": _as_bool(project_diag.get("suspected", False)),
            "project_artifact_strong": _as_bool(project_diag.get("strong", False)),
            "technology_top": tech_diag.get("top_technology", "unknown"),
            "technology_max_fraction": _safe_float(tech_diag.get("max_technology_fraction")),
            "technology_background_fraction": _safe_float(tech_diag.get("background_fraction")),
            "technology_enrichment_ratio": _safe_float(tech_diag.get("enrichment_ratio")),
            "technology_artifact_suspected": _as_bool(tech_diag.get("suspected", False)),
            "technology_caveat": _as_bool(tech_diag.get("caveat", False)),
            "sample_concentration_suspected": _as_bool(sample_diag.get("suspected", False)),
            "events_per_sample": _safe_float(sample_diag.get("events_per_sample")),
            "lookup_unique_otus": _safe_float(sample_diag.get("unique_otus", global_stats.get("lookup_unique_otus"))),
            "lookup_unique_samples": _safe_float(sample_diag.get("unique_samples", global_stats.get("lookup_unique_samples"))),
            "interpret_as_artifact_first": _as_bool(bio_diag.get("interpret_as_artifact_first", False)),
            "positive_fraction": _safe_float(global_stats.get("positive_fraction")),
            "num_positive_events": _safe_float(global_stats.get("num_positive_events")),
            "activation_max": _safe_float(global_stats.get("activation_max")),
            "mean_positive_activation": _safe_float(global_stats.get("mean_positive_activation")),
            "activation_sum": _safe_float(global_stats.get("activation_sum")),
            "activation_vs_abundance": _safe_float(abundance_corr.get("activation_vs_abundance")),
            "activation_vs_abundance_rank": _safe_float(abundance_corr.get("activation_vs_abundance_rank")),
        }
        records.append(record)
    df = pd.DataFrame.from_records(records).sort_values("feature_id").reset_index(drop=True)
    for col in ["primary_class", "final_confidence", "taxonomy_best_rank", "environment_top", "technology_top"]:
        if col in df:
            df[col] = df[col].fillna("unknown").astype(str)
    return df


def _class_order(df: pd.DataFrame) -> list[str]:
    counts = df["primary_class"].value_counts()
    return counts.index.tolist()


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return {"figure": stem, "png": str(png), "pdf": str(pdf)}


def _barh(ax, labels, values, color, title, xlabel="Feature count"):
    labels = list(labels)[::-1]
    values = list(values)[::-1]
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([_clean(x, 42) for x in labels], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", color="#DDDDDD", alpha=0.7)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)


def _class_color(values: pd.Series) -> list[str]:
    return [CLASS_COLORS.get(str(value), "#BAB0AC") for value in values]


def _plot_class_overview(df: pd.DataFrame, output_dir: Path) -> dict:
    order = _class_order(df)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    counts = df["primary_class"].value_counts().reindex(order)
    _barh(axes[0, 0], counts.index, counts.values, [CLASS_COLORS.get(c, "#BAB0AC") for c in counts.index], "Primary interpretation classes")

    conf_table = pd.crosstab(df["primary_class"], df["final_confidence"]).reindex(order).fillna(0)
    for confidence in CONFIDENCE_ORDER:
        if confidence not in conf_table.columns:
            conf_table[confidence] = 0
    left = np.zeros(len(conf_table))
    x = np.arange(len(conf_table))
    for confidence in CONFIDENCE_ORDER:
        values = conf_table[confidence].to_numpy(dtype=float)
        axes[0, 1].bar(x, values, bottom=left, color=CONFIDENCE_COLORS[confidence], label=confidence)
        left += values
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([_clean(c, 18) for c in conf_table.index], rotation=35, ha="right", fontsize=8)
    axes[0, 1].set_ylabel("Feature count")
    axes[0, 1].set_title("Confidence by primary class", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].grid(axis="y", color="#DDDDDD", alpha=0.7)

    secondary = (
        df["secondary_classes"]
        .fillna("")
        .str.split(";")
        .explode()
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .head(12)
    )
    _barh(axes[1, 0], secondary.index, secondary.values, "#B07AA1", "Most common secondary classes")

    acceptance = df["acceptance"].fillna("unknown").astype(str).replace("", "unknown").value_counts()
    _barh(axes[1, 1], acceptance.index, acceptance.values, "#8A8A8A", "Final validation acceptance")
    fig.suptitle(f"Macro overview of {len(df)} interpreted OTU SAE features", fontsize=15, fontweight="bold")
    return _save(fig, output_dir, "01_interpretation_class_overview")


def _axis_heatmap(ax, matrix: pd.DataFrame, title: str, cmap="viridis"):
    if matrix.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([_clean(c, 18) for c in matrix.columns], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([_clean(i, 28) for i in matrix.index], fontsize=8)
    ax.set_title(title, loc="left", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_diagnostic_landscape(df: pd.DataFrame, output_dir: Path) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    sizes = np.clip(df["positive_fraction"].fillna(0).to_numpy() * 6000, 12, 120)
    axes[0, 0].scatter(
        df["taxonomy_best_fraction"],
        df["environment_max_fraction"],
        s=sizes,
        c=_class_color(df["primary_class"]),
        alpha=0.75,
        linewidths=0.3,
        edgecolors="#333333",
    )
    axes[0, 0].axvline(0.5, color="#999999", ls="--", lw=1)
    axes[0, 0].axhline(0.5, color="#999999", ls="--", lw=1)
    axes[0, 0].set_xlabel("Best taxon fraction")
    axes[0, 0].set_ylabel("Top environment fraction")
    axes[0, 0].set_title("Taxonomy vs environment specificity", loc="left", fontweight="bold")
    axes[0, 0].grid(color="#DDDDDD", alpha=0.5)

    artifact_color = np.where(df["interpret_as_artifact_first"], "#E15759", "#9A9A9A")
    axes[0, 1].scatter(
        df["project_max_fraction"],
        df["technology_max_fraction"],
        c=artifact_color,
        s=42,
        alpha=0.75,
        linewidths=0.3,
        edgecolors="#333333",
    )
    axes[0, 1].axvline(0.5, color="#999999", ls="--", lw=1)
    axes[0, 1].axhline(0.95, color="#999999", ls="--", lw=1)
    axes[0, 1].set_xlabel("Top project fraction")
    axes[0, 1].set_ylabel("Top technology fraction")
    axes[0, 1].set_title("Artifact-risk axes", loc="left", fontweight="bold")
    axes[0, 1].grid(color="#DDDDDD", alpha=0.5)

    axis_cols = {
        "taxonomy": "taxonomy_best_fraction",
        "environment": "environment_max_fraction",
        "project": "project_max_fraction",
        "technology": "technology_max_fraction",
    }
    bins = np.linspace(0, 1, 21)
    for label, col in axis_cols.items():
        axes[1, 0].hist(df[col].dropna(), bins=bins, histtype="step", lw=2, label=label)
    axes[1, 0].set_xlabel("Axis top fraction")
    axes[1, 0].set_ylabel("Feature count")
    axes[1, 0].set_title("Specificity distributions", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].grid(axis="y", color="#DDDDDD", alpha=0.7)

    heat = df.groupby("primary_class")[list(axis_cols.values())].mean().reindex(_class_order(df))
    heat.columns = list(axis_cols.keys())
    _axis_heatmap(axes[1, 1], heat, "Mean diagnostic strength by class", cmap="magma")
    fig.suptitle("Global diagnostic-axis landscape", fontsize=15, fontweight="bold")
    return _save(fig, output_dir, "02_diagnostic_axis_landscape")


def _plot_taxonomy_landscape(df: pd.DataFrame, output_dir: Path, top_n: int) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), constrained_layout=True)
    rank_counts = df["taxonomy_best_rank"].replace("unknown", np.nan).dropna().value_counts()
    rank_order = [r for r in ["kingdom", "phylum", "class", "order", "family", "genus", "species"] if r in rank_counts.index]
    rank_counts = rank_counts.reindex(rank_order)
    _barh(axes[0, 0], rank_counts.index, rank_counts.values, "#4E79A7", "Best taxonomic rank distribution")

    top_taxa = df["taxonomy_best_value"].replace(["unknown", "<missing>", ""], np.nan).dropna().value_counts().head(top_n)
    _barh(axes[0, 1], top_taxa.index, top_taxa.values, "#76B7B2", f"Top {len(top_taxa)} best taxa")

    order = _class_order(df)
    class_tax = df.groupby("primary_class")["taxonomy_strong"].mean().reindex(order)
    _barh(axes[1, 0], class_tax.index, class_tax.values, [CLASS_COLORS.get(c, "#BAB0AC") for c in class_tax.index], "Fraction taxonomy_strong by class", xlabel="Fraction")
    axes[1, 0].set_xlim(0, 1)

    data = [df.loc[df["primary_class"].eq(cls), "taxonomy_best_fraction"].dropna().to_numpy() for cls in order]
    axes[1, 1].boxplot(data, tick_labels=[_clean(c, 16) for c in order], showfliers=False)
    axes[1, 1].tick_params(axis="x", rotation=35)
    axes[1, 1].set_ylabel("Best taxon fraction")
    axes[1, 1].set_title("Taxonomy specificity by primary class", loc="left", fontweight="bold")
    axes[1, 1].grid(axis="y", color="#DDDDDD", alpha=0.7)
    fig.suptitle("Taxonomic structure across interpreted features", fontsize=15, fontweight="bold")
    return _save(fig, output_dir, "03_taxonomy_landscape")


def _plot_environment_landscape(df: pd.DataFrame, output_dir: Path, top_n: int) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), constrained_layout=True)
    env_counts = df["environment_top"].replace(["unknown", ""], np.nan).dropna().value_counts().head(top_n)
    colors = [ENV_COLORS.get(env, "#BAB0AC") for env in env_counts.index]
    _barh(axes[0, 0], env_counts.index, env_counts.values, colors, f"Top {len(env_counts)} environments")

    order = _class_order(df)
    class_env = df.groupby("primary_class")["environment_strong"].mean().reindex(order)
    _barh(axes[0, 1], class_env.index, class_env.values, [CLASS_COLORS.get(c, "#BAB0AC") for c in class_env.index], "Fraction environment_strong by class", xlabel="Fraction")
    axes[0, 1].set_xlim(0, 1)

    table = pd.crosstab(df["primary_class"], df["environment_top"]).reindex(order).fillna(0)
    top_envs = df["environment_top"].value_counts().head(8).index
    table = table.loc[:, table.columns.intersection(top_envs)]
    table = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    _axis_heatmap(axes[1, 0], table, "Environment composition within class", cmap="YlGnBu")

    env_ratio = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["environment_enrichment_ratio", "environment_top"])
    top_envs = env_ratio["environment_top"].value_counts().head(6).index
    data = [env_ratio.loc[env_ratio["environment_top"].eq(env), "environment_enrichment_ratio"].clip(upper=10).to_numpy() for env in top_envs]
    axes[1, 1].boxplot(data, tick_labels=[_clean(e, 14) for e in top_envs], showfliers=False)
    axes[1, 1].tick_params(axis="x", rotation=30)
    axes[1, 1].set_ylabel("Environment enrichment ratio (clipped at 10)")
    axes[1, 1].set_title("Background-corrected environment enrichment", loc="left", fontweight="bold")
    axes[1, 1].grid(axis="y", color="#DDDDDD", alpha=0.7)
    fig.suptitle("Environmental and host-context structure", fontsize=15, fontweight="bold")
    return _save(fig, output_dir, "04_environment_landscape")


def _plot_artifact_landscape(df: pd.DataFrame, output_dir: Path, top_n: int) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), constrained_layout=True)
    artifact = df["interpret_as_artifact_first"]
    axes[0, 0].scatter(
        df["project_max_fraction"],
        np.log10(df["project_enrichment_ratio"].replace([np.inf, -np.inf], np.nan).clip(lower=1e-6)),
        c=np.where(artifact, "#E15759", "#9A9A9A"),
        s=44,
        alpha=0.75,
        linewidths=0.3,
        edgecolors="#333333",
    )
    axes[0, 0].set_xlabel("Top project fraction")
    axes[0, 0].set_ylabel("log10(project enrichment ratio)")
    axes[0, 0].set_title("Project concentration vs background enrichment", loc="left", fontweight="bold")
    axes[0, 0].grid(color="#DDDDDD", alpha=0.6)

    axes[0, 1].scatter(
        df["technology_max_fraction"],
        np.log10(df["technology_enrichment_ratio"].replace([np.inf, -np.inf], np.nan).clip(lower=1e-6)),
        c=np.where(df["technology_artifact_suspected"], "#E15759", "#9A9A9A"),
        s=44,
        alpha=0.75,
        linewidths=0.3,
        edgecolors="#333333",
    )
    axes[0, 1].set_xlabel("Top technology fraction")
    axes[0, 1].set_ylabel("log10(technology enrichment ratio)")
    axes[0, 1].set_title("Technology concentration vs background enrichment", loc="left", fontweight="bold")
    axes[0, 1].grid(color="#DDDDDD", alpha=0.6)

    flags = pd.DataFrame(
        {
            "project_suspected": df["project_artifact_suspected"],
            "technology_suspected": df["technology_artifact_suspected"],
            "artifact_first": df["interpret_as_artifact_first"],
            "sample_concentration": df["sample_concentration_suspected"],
        }
    )
    flag_counts = flags.sum().sort_values(ascending=False)
    _barh(axes[1, 0], flag_counts.index, flag_counts.values, "#E15759", "Artifact-related diagnostic flags")

    top_projects = df.loc[df["interpret_as_artifact_first"], "project_top"].replace(["unknown", ""], np.nan).dropna().value_counts().head(top_n)
    if top_projects.empty:
        top_projects = df["project_top"].replace(["unknown", ""], np.nan).dropna().value_counts().head(top_n)
    _barh(axes[1, 1], top_projects.index, top_projects.values, "#FF9DA7", f"Top projects among artifact-first features")
    fig.suptitle("Study, project, and technology artifact diagnostics", fontsize=15, fontweight="bold")
    return _save(fig, output_dir, "05_artifact_landscape")


def _plot_activation_landscape(df: pd.DataFrame, output_dir: Path) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), constrained_layout=True)
    axes[0, 0].hist(df["positive_fraction"].dropna(), bins=30, color="#4E79A7", alpha=0.85)
    axes[0, 0].set_xlabel("Positive fraction")
    axes[0, 0].set_ylabel("Feature count")
    axes[0, 0].set_title("Activation sparsity across features", loc="left", fontweight="bold")
    axes[0, 0].grid(axis="y", color="#DDDDDD", alpha=0.7)

    axes[0, 1].scatter(df["mean_positive_activation"], df["activation_max"], c=_class_color(df["primary_class"]), s=38, alpha=0.75)
    axes[0, 1].set_xlabel("Mean positive activation")
    axes[0, 1].set_ylabel("Max activation")
    axes[0, 1].set_title("Activation magnitude range", loc="left", fontweight="bold")
    axes[0, 1].grid(color="#DDDDDD", alpha=0.6)

    axes[1, 0].scatter(df["lookup_unique_otus"], df["lookup_unique_samples"], c=_class_color(df["primary_class"]), s=38, alpha=0.75)
    axes[1, 0].set_xlabel("Unique OTUs in top events")
    axes[1, 0].set_ylabel("Unique samples in top events")
    axes[1, 0].set_title("Lookup breadth: OTUs vs samples", loc="left", fontweight="bold")
    axes[1, 0].grid(color="#DDDDDD", alpha=0.6)

    order = _class_order(df)
    data = [df.loc[df["primary_class"].eq(cls), "positive_fraction"].dropna().to_numpy() for cls in order]
    axes[1, 1].boxplot(data, tick_labels=[_clean(c, 16) for c in order], showfliers=False)
    axes[1, 1].tick_params(axis="x", rotation=35)
    axes[1, 1].set_ylabel("Positive fraction")
    axes[1, 1].set_title("Sparsity by primary class", loc="left", fontweight="bold")
    axes[1, 1].grid(axis="y", color="#DDDDDD", alpha=0.7)
    fig.suptitle("Activation and lookup-level feature statistics", fontsize=15, fontweight="bold")
    return _save(fig, output_dir, "06_activation_lookup_landscape")


def _plot_rank_abundance_landscape(df: pd.DataFrame, output_dir: Path) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].scatter(
        df["activation_vs_abundance"],
        df["activation_vs_abundance_rank"],
        c=_class_color(df["primary_class"]),
        s=42,
        alpha=0.75,
        linewidths=0.3,
        edgecolors="#333333",
    )
    axes[0].axvline(0, color="#888888", lw=1)
    axes[0].axhline(0, color="#888888", lw=1)
    axes[0].set_xlabel("Activation vs abundance")
    axes[0].set_ylabel("Activation vs abundance rank")
    axes[0].set_title("Abundance correlation axes", loc="left", fontweight="bold")
    axes[0].grid(color="#DDDDDD", alpha=0.6)

    rank_abs = df["activation_vs_abundance_rank"].abs().dropna()
    axes[1].hist(rank_abs, bins=25, color="#F28E2B", alpha=0.85)
    axes[1].set_xlabel("|activation vs abundance rank|")
    axes[1].set_ylabel("Feature count")
    axes[1].set_title("Rank-correlation effect-size distribution", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#DDDDDD", alpha=0.7)

    order = _class_order(df)
    class_rank = df.groupby("primary_class")["activation_vs_abundance_rank"].apply(lambda s: s.abs().mean()).reindex(order)
    _barh(axes[2], class_rank.index, class_rank.values, [CLASS_COLORS.get(c, "#BAB0AC") for c in class_rank.index], "Mean absolute rank correlation by class", xlabel="Mean |rho|")
    fig.suptitle("Abundance and rank behavior across features", fontsize=15, fontweight="bold")
    return _save(fig, output_dir, "07_abundance_rank_landscape")


def _write_readme(output_dir: Path, df: pd.DataFrame, figures: list[dict]) -> Path:
    class_counts = df["primary_class"].value_counts()
    high_conf = int(df["final_confidence"].eq("high").sum())
    medium_conf = int(df["final_confidence"].eq("medium").sum())
    low_conf = int(df["final_confidence"].eq("low").sum())
    tax_strong = int(df["taxonomy_strong"].sum())
    env_strong = int(df["environment_strong"].sum())
    artifact_first = int(df["interpret_as_artifact_first"].sum())
    rank_candidates = int((df["activation_vs_abundance_rank"].abs() >= 0.1).sum())
    top_env = df["environment_top"].value_counts().head(5)
    top_tax = df["taxonomy_best_value"].replace(["unknown", "<missing>", ""], np.nan).dropna().value_counts().head(5)
    lines = [
        "# 宏观 feature 统计图",
        "",
        f"本目录基于 {len(df)} 个已完成解释的 OTU SAE feature 生成宏观统计图。统计只使用 `evidence/*.json` 和 `codex_outputs/*/*.json`，不重新读取每个 feature 的大规模 activation event 表。",
        "",
        "## 总体计数",
        "",
        f"- high / medium / low confidence: {high_conf} / {medium_conf} / {low_conf}",
        f"- taxonomy_strong features: {tax_strong}",
        f"- environment_strong features: {env_strong}",
        f"- artifact-first features: {artifact_first}",
        f"- |activation_vs_abundance_rank| >= 0.1 的 rank-related candidates: {rank_candidates}",
        "",
        "## Primary class 分布",
        "",
    ]
    for label, count in class_counts.items():
        lines.append(f"- {label}: {int(count)}")
    lines.extend(["", "## Top environment", ""])
    for label, count in top_env.items():
        lines.append(f"- {label}: {int(count)}")
    lines.extend(["", "## Top best taxon", ""])
    for label, count in top_tax.items():
        lines.append(f"- {label}: {int(count)}")
    lines.extend(["", "## 图目录", ""])
    descriptions = {
        "01_interpretation_class_overview": "解释类别、置信度、secondary class 和 final acceptance 总览。",
        "02_diagnostic_axis_landscape": "taxonomy、environment、project、technology 四个诊断轴之间的关系。",
        "03_taxonomy_landscape": "best rank、best taxon、taxonomy_strong 和 taxonomy specificity 分布。",
        "04_environment_landscape": "top environment、environment_strong 和背景校正 enrichment 分布。",
        "05_artifact_landscape": "project/technology artifact 风险、artifact flags 和高风险项目。",
        "06_activation_lookup_landscape": "activation 稀疏性、activation magnitude、top event 覆盖的 OTU/sample breadth。",
        "07_abundance_rank_landscape": "activation 与 abundance / abundance rank 的相关结构。",
    }
    for fig in figures:
        stem = fig["figure"]
        lines.append(f"- `{stem}.png` / `{stem}.pdf`: {descriptions.get(stem, '')}")
    lines.extend(
        [
            "",
            "## 注意",
            "",
            "- 这些图是宏观 QC 和结果整理，不替代单个 feature 的 evidence package。",
            "- project enrichment ratio 可能很高但 top project fraction 很低，因此 artifact-first 判断仍应同时看 project fraction、technology、sample concentration 和 biological signal。",
            "- AMPLICON fraction 高本身不等价于 artifact；这里保留 technology caveat 与 suspected 两个不同字段。",
        ]
    )
    readme = output_dir / "README.zh.md"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = Path(cfg(config, "output_dir"))
    output_dir = args.output_dir or root / "reports" / "feature_macro_statistics"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _collect_summary(root)
    summary_csv = output_dir / "feature_macro_summary.csv"
    df.to_csv(summary_csv, index=False)
    figures = [
        _plot_class_overview(df, output_dir),
        _plot_diagnostic_landscape(df, output_dir),
        _plot_taxonomy_landscape(df, output_dir, args.top_n),
        _plot_environment_landscape(df, output_dir, args.top_n),
        _plot_artifact_landscape(df, output_dir, args.top_n),
        _plot_activation_landscape(df, output_dir),
        _plot_rank_abundance_landscape(df, output_dir),
    ]
    manifest = output_dir / "feature_macro_figures_manifest.csv"
    pd.DataFrame(figures).to_csv(manifest, index=False)
    readme = _write_readme(output_dir, df, figures)
    print(
        json.dumps(
            {
                "features": int(len(df)),
                "figures": len(figures),
                "summary_csv": str(summary_csv),
                "manifest": str(manifest),
                "readme": str(readme),
                "output_dir": str(output_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
