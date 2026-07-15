#!/usr/bin/env python3
"""Plot feature-specific story figures for selected OTU SAE interpretations.

Unlike the generic evidence panels, this script chooses a small set of plots for
each feature based on its interpreted class and deterministic evidence. The goal
is to make each figure set test the feature's specific description.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mgm2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mgm2-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.sae_interpretation.config import cfg, load_config
from src.sae_interpretation.taxonomy import parse_taxonomy


DEFAULT_FEATURES = (
    "2131,1821,3915,3435,409,412,"
    "686,758,2711,3602,"
    "9,25,70,2032,2647,"
    "973,1757,2400,"
    "3358,523,"
    "1659,899,"
    "1974,4081"
)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})

COLORS = {
    "target": "#2A9D8F",
    "control": "#BDBDBD",
    "taxonomy": "#4E79A7",
    "host": "#59A14F",
    "artifact": "#E15759",
    "abundance": "#F28E2B",
    "unclear": "#8A8A8A",
    "purple": "#B07AA1",
}
ENV_COLORS = {
    "animal": "#4E79A7",
    "aquatic": "#2A9D8F",
    "soil": "#9C755F",
    "plant": "#59A14F",
    "<missing>": "#8A8A8A",
}
DISEASE_KEYWORD_GROUPS = {
    "disease": ["pneumonia", "tuberculosis", "copd", "covid", "ill"],
    "clinical_infection": ["infection", "hct", "bloodstream"],
    "respiratory_site": ["sputum", "lung", "pharynx", "throat", "nose", "respiration"],
    "oral_upper_airway": ["oral", "saliva"],
    "host_surface": ["human", "skin", "sebum"],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sae_interpretation/otu_xlarge_sample500k_k64_aux0p03125_sparse_topk_global.json"),
    )
    parser.add_argument("--feature-ids", type=str, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--scatter-sample", type=int, default=2500)
    return parser.parse_args()


def _feature_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _clean(value: object, max_len: int = 36) -> str:
    text = "<missing>" if pd.isna(value) else str(value)
    for prefix in ["d__", "k__", "p__", "c__", "o__", "f__", "g__", "s__"]:
        text = text.replace(prefix, "")
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


def _load_feature(root: Path, feature_id: int):
    evidence = _load_json(root / "evidence" / f"feature_{feature_id:06d}.evidence.json")
    unified = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "unified_description.json")
    classification = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "classification.json")
    final_validation = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "final_validation.json")
    events = pd.read_csv(root / "feature_lookup" / f"feature_{feature_id:06d}.events.csv.gz")
    for col in ["activation", "abundance", "abundance_rank"]:
        if col in events:
            events[col] = pd.to_numeric(events[col], errors="coerce")
    for col in ["primary_env", "env_first2", "project", "technology", "taxonomy", "otu_id", "sample_id"]:
        if col not in events:
            events[col] = "<missing>"
        events[col] = events[col].fillna("<missing>").astype(str)
    return evidence, unified, classification, final_validation, events


def _save(fig, feature_dir: Path, stem: str) -> tuple[Path, Path]:
    feature_dir.mkdir(parents=True, exist_ok=True)
    png = feature_dir / f"{stem}.png"
    pdf = feature_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _title(fig, feature_id: int, unified: dict, classification: dict, final_validation: dict, subtitle: str) -> None:
    fig.suptitle(
        f"Feature {feature_id}: {unified.get('short_name', '')}\n"
        f"{classification.get('primary_class', '')} | confidence={final_validation.get('confidence', '')} | {subtitle}",
        fontsize=13,
        fontweight="bold",
    )


def _background_fraction(background: dict, key: str, value: str) -> float | None:
    for row in background.get("distributions", {}).get(key, []):
        if str(row.get("value", "")) == str(value):
            return float(row.get("fraction", 0.0))
    return None


def _background_fraction_many(background: dict, key: str, values: list[str]) -> float | None:
    found = False
    total = 0.0
    for value in values:
        fraction = _background_fraction(background, key, value)
        if fraction is not None:
            found = True
            total += float(fraction)
    return min(total, 1.0) if found else None


def _format_pvalue(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    if value < 1e-300:
        return "<1e-300"
    if value < 1e-3:
        return f"{value:.1e}"
    return f"{value:.3f}"


def _spearman_safe(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan, np.nan
    rho, pvalue = spearmanr(data["x"], data["y"])
    return float(rho), float(pvalue)


def _taxonomy_mask(events: pd.DataFrame, rank: str, value: str) -> pd.Series:
    def matches(taxonomy: str) -> bool:
        parsed = parse_taxonomy(taxonomy)
        return str(parsed.get(rank, "")) == str(value)

    return events["taxonomy"].map(matches)


def _group_activation_summary(events: pd.DataFrame, group: pd.Series | np.ndarray) -> dict:
    group = pd.Series(group, index=events.index).fillna(False).astype(bool)
    target = events[group]
    control = events[~group]
    total_mass = float(events["activation"].sum())
    return {
        "target_event_fraction": len(target) / max(1, len(events)),
        "control_event_fraction": len(control) / max(1, len(events)),
        "target_activation_mass_fraction": float(target["activation"].sum()) / max(total_mass, 1e-12),
        "control_activation_mass_fraction": float(control["activation"].sum()) / max(total_mass, 1e-12),
        "target_mean_activation": float(target["activation"].mean()) if len(target) else 0.0,
        "control_mean_activation": float(control["activation"].mean()) if len(control) else 0.0,
        "target_events": int(len(target)),
        "control_events": int(len(control)),
    }


def _plot_group_proof(ax, events: pd.DataFrame, group: pd.Series, target_label: str, control_label: str, background_fraction: float | None, color: str) -> dict:
    summary = _group_activation_summary(events, group)
    if background_fraction is not None:
        background_fraction = min(max(float(background_fraction), 1e-12), 1.0 - 1e-12)
        summary["background_fraction"] = background_fraction
        summary["event_enrichment_over_background"] = summary["target_event_fraction"] / background_fraction
        summary["binomial_pvalue"] = float(
            binomtest(summary["target_events"], summary["target_events"] + summary["control_events"], background_fraction, alternative="greater").pvalue
        )
    else:
        summary["background_fraction"] = np.nan
        summary["event_enrichment_over_background"] = np.nan
        summary["binomial_pvalue"] = np.nan
    labels = [target_label, control_label]
    event_share = [summary["target_event_fraction"], summary["control_event_fraction"]]
    mass_share = [summary["target_activation_mass_fraction"], summary["control_activation_mass_fraction"]]
    x = np.arange(2)
    ax.bar(x - 0.18, event_share, width=0.36, color=[color, COLORS["control"]], label="positive event share")
    ax.bar(x + 0.18, mass_share, width=0.36, color=[color, COLORS["control"]], alpha=0.55, label="activation mass share")
    if background_fraction is not None:
        ax.scatter([0, 1], [background_fraction, 1.0 - background_fraction], color="#222222", s=38, zorder=3, label="background token share")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", color="#DDDDDD", alpha=0.7)
    if background_fraction is not None:
        ax.text(
            0.02,
            0.95,
            f"enrichment={summary['event_enrichment_over_background']:.2f}x\nbinom p={_format_pvalue(summary['binomial_pvalue'])}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.85},
        )
    return summary


def _plot_group_mean_box(ax, events: pd.DataFrame, group: pd.Series, target_label: str, control_label: str, color: str) -> None:
    group = pd.Series(group, index=events.index).fillna(False).astype(bool)
    target = events.loc[group, "activation"].dropna()
    control = events.loc[~group, "activation"].dropna()
    if len(target) == 0 and len(control) == 0:
        ax.text(0.5, 0.5, "No activation data", ha="center", va="center")
        return
    data = [target.to_numpy(), control.to_numpy()]
    parts = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=False)
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(color if idx == 0 else COLORS["control"])
        body.set_edgecolor("none")
        body.set_alpha(0.65)
    for key in ["cmeans", "cmedians"]:
        if key in parts:
            parts[key].set_color("#222222")
            parts[key].set_linewidth(1.0)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([target_label, control_label], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Activation among sparse positive events")
    ax.grid(axis="y", color="#DDDDDD", alpha=0.7)


def _taxonomy_top_examples(evidence: dict) -> pd.DataFrame:
    rows = evidence.get("top_examples", [])
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=["taxonomy", "kingdom", "phylum", "class", "order", "family", "genus", "species"])
    data = pd.DataFrame(rows)
    if "taxonomy" not in data:
        data["taxonomy"] = ""
    parsed = data["taxonomy"].map(parse_taxonomy)
    for rank in ["kingdom", "phylum", "class", "order", "family", "genus", "species"]:
        if rank not in data:
            data[rank] = parsed.map(lambda item, r=rank: item.get(r, "unknown"))
        data[rank] = data[rank].fillna("unknown").astype(str)
    return data


def _valid_taxon(value: object) -> bool:
    text = str(value).strip().lower()
    return text not in {"", "unknown", "<missing>", "nan", "none", "null"}


def _join_taxa(values: list[str], max_values: int = 5) -> str:
    values = [str(value) for value in values if _valid_taxon(value)]
    if not values:
        return "unknown"
    if len(values) <= max_values:
        return " + ".join(values)
    return " + ".join(values[:max_values]) + f" + {len(values) - max_values} more"


def _best_taxon_ladder(evidence: dict) -> list[tuple[str, str, float]]:
    """Return a lineage-aware ladder for the diagnostic best taxon.

    Taxonomy strings can mix naming systems. A lower-rank taxon may therefore
    appear below several parent labels, such as Lachnospiraceae under
    Lachnospirales, Clostridiales, and Eubacteriales. For ranks above the
    diagnostic best taxon, show the union of parent labels that lead to that
    taxon and compute coverage with that union.
    """
    ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    top = _taxonomy_top_examples(evidence)
    signal = evidence.get("diagnostics", {}).get("taxonomy_signal", {})
    anchor_rank = str(signal.get("best_rank", ""))
    anchor_value = str(signal.get("best_value", ""))
    if not top.empty and anchor_rank in ranks and _valid_taxon(anchor_value):
        total = max(len(top), 1)
        anchor_mask = top[anchor_rank].eq(anchor_value)
        if anchor_mask.any():
            anchor_depth = ranks.index(anchor_rank)
            ladder = []
            for depth, rank in enumerate(ranks):
                if depth < anchor_depth:
                    parent_values = (
                        top.loc[anchor_mask, rank]
                        .loc[lambda values: values.map(_valid_taxon)]
                        .drop_duplicates()
                        .tolist()
                    )
                    if parent_values:
                        union_mask = top[rank].isin(parent_values)
                        ladder.append((rank, _join_taxa(parent_values), float(union_mask.sum() / total)))
                elif depth == anchor_depth:
                    ladder.append((rank, anchor_value, float(anchor_mask.sum() / total)))
                else:
                    child_values = top.loc[anchor_mask, rank].loc[lambda values: values.map(_valid_taxon)]
                    if len(child_values):
                        counts = child_values.value_counts()
                        ladder.append((rank, str(counts.index[0]), float(counts.iloc[0] / total)))
            return ladder

    taxonomy = evidence.get("enrichment", {}).get("taxonomy", {})
    ladder = []
    for rank in ranks:
        rows = [row for row in taxonomy.get(rank, []) if _valid_taxon(row.get("value"))]
        if rows:
            ladder.append((rank, str(rows[0].get("value", "")), float(rows[0].get("fraction", 0.0))))
    return ladder


def _is_taxonomy_related(classification: dict, evidence: dict) -> bool:
    classes = [classification.get("primary_class", "")] + list(classification.get("secondary_classes", []) or [])
    signal = evidence.get("diagnostics", {}).get("taxonomy_signal", {})
    return (
        "Taxonomic identity" in classes
        or bool(signal.get("strong"))
        or float(signal.get("best_fraction", 0.0) or 0.0) >= 0.5
    )


def _plot_best_taxon_rank_figure(root: Path, feature_id: int, out_dir: Path, top_n: int) -> dict:
    evidence, unified, classification, final_validation, events = _load_feature(root, feature_id)
    feature_dir = out_dir / f"feature_{feature_id:06d}"
    ladder = _best_taxon_ladder(evidence)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    _title(fig, feature_id, unified, classification, final_validation, "lineage-aware taxon ladder")
    _barh(
        axes[0],
        [f"{rank}: {value}" for rank, value, _ in ladder],
        [fraction for _, _, fraction in ladder],
        COLORS["taxonomy"],
        "Lineage-aware taxon ladder",
        "Fraction among positive events",
        label_max_len=68,
    )
    signal = evidence.get("diagnostics", {}).get("taxonomy_signal", {})
    rank = signal.get("best_rank", "phylum")
    data = events.copy()
    data["taxon"] = data["taxonomy"].map(lambda x: parse_taxonomy(str(x)).get(rank, "unknown"))
    labels, values = _activation_mass_by(data, "taxon", top_n)
    _barh(axes[1], labels, values, COLORS["abundance"], f"{rank} activation mass", "Activation mass")
    png, pdf = _save(fig, feature_dir, "00_best_taxon_at_each_rank")
    return {
        "figure": "00_best_taxon_at_each_rank",
        "png": str(png),
        "pdf": str(pdf),
        "purpose": "Shows a lineage-aware taxon ladder; parent ranks are merged when the diagnostic lower-rank taxon appears under multiple parent labels.",
    }


def _select_environment_targets(evidence: dict, unified: dict, host_mode: bool = False) -> list[str]:
    if host_mode:
        return ["animal"]
    rows = evidence.get("enrichment", {}).get("metadata", {}).get("primary_env", [])
    if not rows:
        return []
    text = " ".join([unified.get("short_name", ""), unified.get("summary", ""), unified.get("ecological_context", "")]).lower()
    targets = []
    for row in rows:
        value = str(row.get("value", ""))
        fraction = float(row.get("fraction", 0.0))
        if value in {"<missing>", "unknown", ""}:
            continue
        if not targets:
            targets.append(value)
            continue
        if fraction >= 0.20:
            targets.append(value)
        elif value in text and fraction >= 0.10:
            targets.append(value)
    return targets[:3]


def _barh(
    ax,
    labels: list[str],
    values: list[float],
    color: str,
    title: str,
    xlabel: str = "Activation mass",
    label_max_len: int = 42,
) -> None:
    labels = labels[::-1]
    values = values[::-1]
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels([_clean(label, label_max_len) for label in labels], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", color="#DDDDDD", alpha=0.7)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)


def _activation_mass_by(events: pd.DataFrame, col: str, top_n: int) -> tuple[list[str], list[float]]:
    grouped = (
        events.groupby(col, dropna=False)["activation"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    return grouped.index.astype(str).tolist(), grouped.astype(float).tolist()


def _keyword_activation_table(events: pd.DataFrame, terms: set[str] | None = None, top_n: int = 12) -> pd.DataFrame:
    if "keywords" not in events:
        return pd.DataFrame(columns=["keyword", "events", "activation_mass"])
    rows = []
    for _, row in events[["keywords", "activation"]].dropna(subset=["keywords"]).iterrows():
        for token in str(row["keywords"]).replace(",", ";").split(";"):
            keyword = token.strip().lower()
            if not keyword or keyword in {"<missing>", "unknown", "none", "nan"}:
                continue
            if terms is not None and keyword not in terms:
                continue
            rows.append((keyword, float(row["activation"])))
    if not rows:
        return pd.DataFrame(columns=["keyword", "events", "activation_mass"])
    table = pd.DataFrame(rows, columns=["keyword", "activation"])
    return (
        table.groupby("keyword")
        .agg(events=("activation", "size"), activation_mass=("activation", "sum"))
        .sort_values("activation_mass", ascending=False)
        .head(top_n)
        .reset_index()
    )


def _keyword_statistics(events: pd.DataFrame, grouped_terms: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    if "keywords" in events:
        for _, row in events[["keywords", "activation"]].dropna(subset=["keywords"]).iterrows():
            keywords = {
                token.strip().lower()
                for token in str(row["keywords"]).replace(",", ";").split(";")
                if token.strip().lower() not in {"", "<missing>", "unknown", "none", "nan"}
            }
            for group, terms in grouped_terms.items():
                for term in terms:
                    if term in keywords:
                        rows.append({"group": group, "keyword": term, "activation": float(row["activation"])})
    if rows:
        table = (
            pd.DataFrame(rows)
            .groupby(["group", "keyword"], as_index=False)
            .agg(events=("activation", "size"), activation_mass=("activation", "sum"))
        )
    else:
        table = pd.DataFrame(columns=["group", "keyword", "events", "activation_mass"])
    all_terms = pd.DataFrame(
        [{"group": group, "keyword": term} for group, terms in grouped_terms.items() for term in terms]
    )
    table = all_terms.merge(table, on=["group", "keyword"], how="left").fillna({"events": 0, "activation_mass": 0.0})
    table["events"] = table["events"].astype(int)
    table["activation_mass"] = table["activation_mass"].astype(float)
    total_mass = float(events["activation"].sum()) if "activation" in events else 0.0
    table["activation_mass_fraction"] = table["activation_mass"] / max(total_mass, 1e-12)
    table["event_fraction"] = table["events"] / max(len(events), 1)
    return table


def _plot_grouped_keyword_statistics(ax, table: pd.DataFrame, title: str) -> None:
    if table.empty:
        ax.text(0.5, 0.5, "No keyword data", ha="center", va="center")
        ax.set_axis_off()
        return
    plot_table = table.sort_values(["group", "activation_mass"], ascending=[True, False]).copy()
    labels = [f"{row.keyword} ({row.group})" for row in plot_table.itertuples()]
    colors = {
        "disease": COLORS["artifact"],
        "clinical_infection": COLORS["purple"],
        "respiratory_site": COLORS["host"],
        "oral_upper_airway": COLORS["abundance"],
        "host_surface": COLORS["taxonomy"],
    }
    y = np.arange(len(plot_table))
    ax.barh(y, plot_table["activation_mass_fraction"], color=[colors.get(group, COLORS["unclear"]) for group in plot_table["group"]], alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Activation mass fraction among positive events")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", color="#DDDDDD", alpha=0.7)
    for yi, row in enumerate(plot_table.itertuples()):
        ax.text(
            float(row.activation_mass_fraction) + 0.002,
            yi,
            f"n={int(row.events)}",
            va="center",
            fontsize=7,
        )
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)


def _barh_table(ax, table: pd.DataFrame, label_col: str, value_col: str, color: str, title: str, xlabel: str = "Activation mass") -> None:
    if table.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return
    _barh(ax, table[label_col].astype(str).tolist(), table[value_col].astype(float).tolist(), color, title, xlabel)


def _cooccurring_table(evidence: dict, top_n: int) -> pd.DataFrame:
    rows = evidence.get("enrichment", {}).get("co_occurring_taxa", [])
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=["value", "fraction"])
    table = pd.DataFrame(rows)
    if "fraction" not in table:
        table["fraction"] = table.get("count", 0) / max(float(table.get("count", pd.Series([0])).sum()), 1.0)
    return table.head(top_n)


def _heatmap_env_by_taxon(ax, events: pd.DataFrame, tax_rank: str = "phylum", top_n: int = 6, title: str = "Environment x taxon activation mass") -> None:
    data = events.copy()
    data["taxon"] = data["taxonomy"].map(lambda x: parse_taxonomy(str(x)).get(tax_rank, "unknown"))
    top_taxa = data.groupby("taxon")["activation"].sum().sort_values(ascending=False).head(top_n).index
    top_envs = data.groupby("primary_env")["activation"].sum().sort_values(ascending=False).head(5).index
    table = (
        data[data["taxon"].isin(top_taxa) & data["primary_env"].isin(top_envs)]
        .pivot_table(index="primary_env", columns="taxon", values="activation", aggfunc="sum", fill_value=0.0)
    )
    if table.empty:
        ax.text(0.5, 0.5, "No heatmap data", ha="center", va="center")
        ax.set_axis_off()
        return
    matrix = table.to_numpy(dtype=float)
    matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-12)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels([_clean(x, 18) for x in table.columns], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels([_clean(x, 18) for x in table.index], fontsize=8)
    ax.set_title(title, loc="left", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized activation")


def _story_card(ax, unified: dict, final_validation: dict) -> None:
    ax.axis("off")
    text = "\n\n".join(
        [
            "Feature description",
            textwrap.fill(unified.get("summary", ""), 88),
            "Specificity / risk",
            textwrap.fill(final_validation.get("specificity", ""), 88),
        ]
    )
    ax.text(0, 1, text, va="top", fontsize=9, linespacing=1.35)


def _plot_taxonomy_story(root: Path, feature_id: int, out_dir: Path, top_n: int, background: dict, scatter_sample: int) -> list[dict]:
    evidence, unified, classification, final_validation, events = _load_feature(root, feature_id)
    signal = evidence.get("diagnostics", {}).get("taxonomy_signal", {})
    rank = signal.get("best_rank", "phylum")
    value = signal.get("best_value", "")
    mask = _taxonomy_mask(events, rank, value)
    feature_dir = out_dir / f"feature_{feature_id:06d}"
    manifest = []

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    _title(fig, feature_id, unified, classification, final_validation, f"{rank}={value} activation specificity")
    summary = _plot_group_proof(axes[0], events, mask, _clean(value, 24), "other taxa", None, COLORS["taxonomy"])
    axes[0].set_title("Target taxon vs other taxa", loc="left", fontweight="bold")
    labels, values = _activation_mass_by(events[mask] if mask.any() else events, "primary_env", top_n)
    _barh(axes[1], labels, values, COLORS["target"], f"Where {rank}={_clean(value, 24)} activates")
    _heatmap_env_by_taxon(axes[2], events, tax_rank=rank if rank in {"phylum", "class", "order", "family", "genus"} else "phylum", top_n=top_n, title="Environment x target rank context")
    png, pdf = _save(fig, feature_dir, "01_target_taxon_vs_other_activation")
    manifest.append({"figure": "01_target_taxon_vs_other_activation", "png": str(png), "pdf": str(pdf), "purpose": f"Tests whether {rank}={value} carries most positive activation mass and whether the taxon signal is context-specific.", "target_rank": rank, "target_taxon": value, **summary})

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    _title(fig, feature_id, unified, classification, final_validation, "taxonomic hierarchy and top OTUs")
    ladder = _best_taxon_ladder(evidence)
    _barh(
        axes[0],
        [f"{rank}: {value}" for rank, value, _ in ladder],
        [fraction for _, _, fraction in ladder],
        COLORS["taxonomy"],
        "Lineage-aware taxon ladder",
        "Fraction",
        label_max_len=68,
    )
    labels, values = _activation_mass_by(events.assign(otu_label=events["otu_id"] + " | " + events["taxonomy"].map(lambda x: _clean(";".join(str(x).split(";")[-2:]), 24))), "otu_label", top_n)
    _barh(axes[1], labels, values, COLORS["abundance"], "Top OTUs by activation mass")
    png, pdf = _save(fig, feature_dir, "02_taxonomic_hierarchy_and_top_otus")
    manifest.append({"figure": "02_taxonomic_hierarchy_and_top_otus", "png": str(png), "pdf": str(pdf), "purpose": "Shows whether the interpretation is broad-rank or specific-OTU driven."})
    return manifest


def _plot_environment_story(root: Path, feature_id: int, out_dir: Path, top_n: int, background: dict, scatter_sample: int, host_mode: bool = False) -> list[dict]:
    evidence, unified, classification, final_validation, events = _load_feature(root, feature_id)
    target_envs = _select_environment_targets(evidence, unified, host_mode=host_mode)
    target_env = "+".join(target_envs)
    mask = events["primary_env"].isin(target_envs)
    bg_fraction = _background_fraction_many(background, "primary_env", target_envs)
    color = COLORS["host"] if host_mode else COLORS["target"]
    feature_dir = out_dir / f"feature_{feature_id:06d}"
    manifest = []

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    subtitle = f"{target_env} vs other environments" if target_env else "environment claim"
    _title(fig, feature_id, unified, classification, final_validation, subtitle)
    summary = _plot_group_proof(axes[0], events, mask, target_env, "other environments", bg_fraction, color)
    axes[0].set_title("Target environment vs control/background", loc="left", fontweight="bold")
    env_rows = evidence.get("enrichment", {}).get("metadata", {}).get("primary_env", [])
    env_labels = [str(row.get("value", "")) for row in env_rows[: min(top_n, 6)]]
    env_values = [float(events.loc[events["primary_env"].eq(label), "activation"].sum()) for label in env_labels]
    _barh(axes[1], env_labels, env_values, color, "Activation mass by primary environment")
    labels, values = _activation_mass_by(events[mask] if mask.any() else events, "env_first2", top_n)
    _barh(axes[2], labels, values, color, f"Subtype activation mass within {target_env or 'target'}")
    stem = "01_host_vs_nonhost_activation" if host_mode else "01_environment_vs_control_activation"
    png, pdf = _save(fig, feature_dir, stem)
    manifest.append({"figure": stem, "png": str(png), "pdf": str(pdf), "purpose": f"Tests whether positive activation is concentrated in {target_env} relative to controls/background.", "target_environments": ";".join(target_envs), **summary})

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    _title(fig, feature_id, unified, classification, final_validation, "community context of the target environment")
    _heatmap_env_by_taxon(axes[0], events, tax_rank="phylum", top_n=top_n)
    tax_rank = "class" if host_mode else "phylum"
    data = events.copy()
    data["taxon"] = data["taxonomy"].map(lambda x: parse_taxonomy(str(x)).get(tax_rank, "unknown"))
    target_data = data[data["primary_env"].isin(target_envs)]
    labels, values = _activation_mass_by(target_data if len(target_data) else data, "taxon", top_n)
    _barh(axes[1], labels, values, COLORS["taxonomy"], f"Taxa active within {target_env}")
    stem = "02_host_taxon_context" if host_mode else "02_environment_taxon_context"
    png, pdf = _save(fig, feature_dir, stem)
    manifest.append({"figure": stem, "png": str(png), "pdf": str(pdf), "purpose": f"Shows which taxa carry activation inside the interpreted {target_env} context."})
    return manifest


def _plot_artifact_story(root: Path, feature_id: int, out_dir: Path, top_n: int, background: dict, scatter_sample: int) -> list[dict]:
    evidence, unified, classification, final_validation, events = _load_feature(root, feature_id)
    project_diag = evidence.get("diagnostics", {}).get("project_batch_artifact", {})
    tech_diag = evidence.get("diagnostics", {}).get("technology_bias", {})
    target_project = str(project_diag.get("top_project", ""))
    target_tech = str(tech_diag.get("top_technology", ""))
    feature_dir = out_dir / f"feature_{feature_id:06d}"
    manifest = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.3), constrained_layout=True)
    _title(fig, feature_id, unified, classification, final_validation, "artifact claim: project and technology concentration")
    project_mask = events["project"].eq(target_project)
    tech_mask = events["technology"].eq(target_tech)
    project_summary = _plot_group_proof(axes[0, 0], events, project_mask, _clean(target_project, 18), "other projects", _background_fraction(background, "project", target_project), COLORS["artifact"])
    axes[0, 0].set_title("Project vs all other projects", loc="left", fontweight="bold")
    labels, values = _activation_mass_by(events, "project", min(top_n, 8))
    _barh(axes[0, 1], labels, values, COLORS["artifact"], "Top projects by activation mass")
    tech_summary = _plot_group_proof(axes[1, 0], events, tech_mask, _clean(target_tech, 18), "other technologies", _background_fraction(background, "technology", target_tech), COLORS["purple"])
    axes[1, 0].set_title("Technology vs all other technologies", loc="left", fontweight="bold")
    table = pd.pivot_table(events, index="technology", columns="project", values="activation", aggfunc="sum", fill_value=0.0)
    top_projects = events.groupby("project")["activation"].sum().sort_values(ascending=False).head(min(top_n, 6)).index
    table = table.loc[:, table.columns.intersection(top_projects)]
    if not table.empty:
        matrix = table.to_numpy(dtype=float)
        im = axes[1, 1].imshow(matrix / np.maximum(matrix.sum(), 1e-12), aspect="auto", cmap="Reds")
        axes[1, 1].set_xticks(np.arange(table.shape[1]))
        axes[1, 1].set_xticklabels([_clean(x, 16) for x in table.columns], rotation=35, ha="right", fontsize=8)
        axes[1, 1].set_yticks(np.arange(table.shape[0]))
        axes[1, 1].set_yticklabels([_clean(x, 16) for x in table.index], fontsize=8)
        axes[1, 1].set_title("Project x technology activation mass", loc="left", fontweight="bold")
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    png, pdf = _save(fig, feature_dir, "01_project_technology_artifact_tests")
    manifest.append({"figure": "01_project_technology_artifact_tests", "png": str(png), "pdf": str(pdf), "purpose": "Tests whether artifact interpretation is supported by project and technology concentration.", **{f"project_{k}": v for k, v in project_summary.items()}, **{f"technology_{k}": v for k, v in tech_summary.items()}})

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    _title(fig, feature_id, unified, classification, final_validation, "biological signal caveat for artifact feature")
    labels, values = _activation_mass_by(events, "primary_env", top_n)
    _barh(axes[0], labels, values, COLORS["target"], "Environment activation mass")
    data = events.copy()
    data["phylum"] = data["taxonomy"].map(lambda x: parse_taxonomy(str(x)).get("phylum", "unknown"))
    labels, values = _activation_mass_by(data, "phylum", top_n)
    _barh(axes[1], labels, values, COLORS["taxonomy"], "Phylum activation mass")
    png, pdf = _save(fig, feature_dir, "02_biological_signal_caveat")
    manifest.append({"figure": "02_biological_signal_caveat", "png": str(png), "pdf": str(pdf), "purpose": "Shows the biological pattern that should be treated as caveat rather than primary interpretation."})
    return manifest


def _plot_abundance_story(root: Path, feature_id: int, out_dir: Path, top_n: int, background: dict, scatter_sample: int) -> list[dict]:
    evidence, unified, classification, final_validation, events = _load_feature(root, feature_id)
    data = events.dropna(subset=["activation", "abundance_rank", "abundance"]).copy()
    feature_dir = out_dir / f"feature_{feature_id:06d}"
    manifest = []
    if len(data) > scatter_sample:
        sample = data.sample(scatter_sample, random_state=feature_id)
    else:
        sample = data
    bins = np.unique(np.quantile(data["abundance_rank"], np.linspace(0, 1, 21))).astype(float)
    if len(bins) < 3:
        bins = np.linspace(float(data["abundance_rank"].min()), float(data["abundance_rank"].max()), 12)
    data["rank_bin"] = pd.cut(data["abundance_rank"], bins=bins, include_lowest=True, duplicates="drop")
    rank_profile = (
        data.groupby("rank_bin", observed=False)
        .agg(events=("activation", "size"), activation_mass=("activation", "sum"), mean_activation=("activation", "mean"), median_rank=("abundance_rank", "median"))
        .reset_index()
    )
    rank_profile["event_density"] = rank_profile["events"] / max(1, rank_profile["events"].sum())
    rank_profile["activation_mass_density"] = rank_profile["activation_mass"] / max(float(rank_profile["activation_mass"].sum()), 1e-12)
    rho_rank, p_rank = _spearman_safe(data["abundance_rank"], data["activation"])
    rho_abundance, p_abundance = _spearman_safe(data["abundance"], data["activation"])

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), constrained_layout=True)
    _title(fig, feature_id, unified, classification, final_validation, "abundance-rank trend test")
    x = np.arange(len(rank_profile))
    axes[0, 0].bar(x, rank_profile["event_density"], color="#CFCFCF", label="positive event share")
    axes[0, 0].plot(x, rank_profile["activation_mass_density"], color=COLORS["abundance"], marker="o", label="activation mass share")
    axes[0, 0].set_xticks(x[:: max(1, len(x) // 8)])
    axes[0, 0].set_xticklabels([f"{v:.0f}" for v in rank_profile["median_rank"].iloc[:: max(1, len(x) // 8)]], rotation=35, ha="right", fontsize=8)
    axes[0, 0].set_xlabel("Abundance rank bin median")
    axes[0, 0].set_ylabel("Fraction")
    axes[0, 0].set_title("Positive-event and activation-mass rank profile", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].grid(axis="y", color="#DDDDDD", alpha=0.7)

    axes[0, 1].plot(x, rank_profile["mean_activation"], color=COLORS["abundance"], marker="o")
    axes[0, 1].set_xticks(x[:: max(1, len(x) // 8)])
    axes[0, 1].set_xticklabels([f"{v:.0f}" for v in rank_profile["median_rank"].iloc[:: max(1, len(x) // 8)]], rotation=35, ha="right", fontsize=8)
    axes[0, 1].set_xlabel("Abundance rank bin median")
    axes[0, 1].set_ylabel("Mean activation")
    axes[0, 1].set_title("Mean activation by rank bin", loc="left", fontweight="bold")
    axes[0, 1].grid(color="#DDDDDD", alpha=0.7)

    colors = [ENV_COLORS.get(env, "#BAB0AC") for env in sample["primary_env"]]
    axes[1, 0].scatter(sample["abundance_rank"], sample["activation"], c=colors, s=12, alpha=0.45, linewidths=0)
    axes[1, 0].invert_xaxis()
    axes[1, 0].set_xlabel("Abundance rank (lower is more abundant)")
    axes[1, 0].set_ylabel("Activation")
    axes[1, 0].set_title("Activation scatter across ranks", loc="left", fontweight="bold")
    axes[1, 0].grid(color="#DDDDDD", alpha=0.7)

    corr = evidence.get("enrichment", {}).get("abundance_correlation", {})
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0,
        1,
        "Abundance/rank interpretation\n\n"
        + textwrap.fill(unified.get("abundance_context", ""), 82)
        + "\n\n"
        + f"evidence activation_vs_abundance={corr.get('activation_vs_abundance')}\n"
        + f"evidence activation_vs_abundance_rank={corr.get('activation_vs_abundance_rank')}\n\n"
        + f"Spearman abundance rho={rho_abundance:.3f}, p={_format_pvalue(p_abundance)}\n"
        + f"Spearman rank rho={rho_rank:.3f}, p={_format_pvalue(p_rank)}",
        va="top",
        fontsize=9,
        linespacing=1.35,
    )
    png, pdf = _save(fig, feature_dir, "01_rank_activation_trend_test")
    manifest.append({
        "figure": "01_rank_activation_trend_test",
        "png": str(png),
        "pdf": str(pdf),
        "purpose": "Tests whether activation increases or decreases with abundance rank among sparse-positive events.",
        "spearman_abundance_rho": rho_abundance,
        "spearman_abundance_pvalue": p_abundance,
        "spearman_rank_rho": rho_rank,
        "spearman_rank_pvalue": p_rank,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    _title(fig, feature_id, unified, classification, final_validation, "rank behavior by ecology and taxonomy")
    table = pd.pivot_table(
        data,
        index="primary_env",
        columns="rank_bin",
        values="activation",
        aggfunc="sum",
        fill_value=0.0,
        observed=False,
    )
    top_envs = data.groupby("primary_env")["activation"].sum().sort_values(ascending=False).head(5).index
    table = table.loc[table.index.intersection(top_envs)]
    if not table.empty:
        matrix = table.to_numpy(dtype=float)
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-12)
        im = axes[0].imshow(matrix, aspect="auto", cmap="YlOrBr")
        axes[0].set_yticks(np.arange(table.shape[0]))
        axes[0].set_yticklabels([_clean(x, 16) for x in table.index], fontsize=8)
        axes[0].set_xticks(np.arange(0, table.shape[1], max(1, table.shape[1] // 6)))
        axes[0].set_xticklabels([str(table.columns[i]) for i in range(0, table.shape[1], max(1, table.shape[1] // 6))], rotation=35, ha="right", fontsize=7)
        axes[0].set_title("Rank-bin activation by environment", loc="left", fontweight="bold")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    tax_data = data.copy()
    tax_data["phylum"] = tax_data["taxonomy"].map(lambda x: parse_taxonomy(str(x)).get("phylum", "unknown"))
    labels, values = _activation_mass_by(tax_data, "phylum", top_n)
    _barh(axes[1], labels, values, COLORS["taxonomy"], "Taxa carrying rank signal")
    png, pdf = _save(fig, feature_dir, "02_rank_signal_context")
    manifest.append({"figure": "02_rank_signal_context", "png": str(png), "pdf": str(pdf), "purpose": "Checks whether the rank signal is broad or restricted to particular environments/taxa."})
    return manifest


def _plot_unclear_story(root: Path, feature_id: int, out_dir: Path, top_n: int, background: dict, scatter_sample: int) -> list[dict]:
    evidence, unified, classification, final_validation, events = _load_feature(root, feature_id)
    feature_dir = out_dir / f"feature_{feature_id:06d}"
    manifest = []
    tax_diag = evidence.get("diagnostics", {}).get("taxonomy_signal", {})
    env_diag = evidence.get("diagnostics", {}).get("environment_enrichment", {})
    project_diag = evidence.get("diagnostics", {}).get("project_batch_artifact", {})
    tech_diag = evidence.get("diagnostics", {}).get("technology_bias", {})

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    _title(fig, feature_id, unified, classification, final_validation, "why no single interpretation dominates")
    labels = ["taxonomy", "environment", "project", "technology"]
    values = [
        float(tax_diag.get("best_fraction", 0.0)),
        float(env_diag.get("max_environment_fraction", 0.0)),
        float(project_diag.get("max_project_fraction", 0.0)),
        float(tech_diag.get("max_technology_fraction", 0.0)),
    ]
    axes[0].bar(labels, values, color=[COLORS["taxonomy"], COLORS["target"], COLORS["artifact"], COLORS["purple"]])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Competing axis strengths", loc="left", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", color="#DDDDDD", alpha=0.7)
    labels_env, values_env = _activation_mass_by(events, "primary_env", top_n)
    _barh(axes[1], labels_env, values_env, COLORS["unclear"], "Environment mixture")
    tax_data = events.copy()
    tax_data["phylum"] = tax_data["taxonomy"].map(lambda x: parse_taxonomy(str(x)).get("phylum", "unknown"))
    labels_tax, values_tax = _activation_mass_by(tax_data, "phylum", top_n)
    _barh(axes[2], labels_tax, values_tax, COLORS["unclear"], "Taxonomy mixture")
    png, pdf = _save(fig, feature_dir, "01_competing_axes_and_mixtures")
    manifest.append({"figure": "01_competing_axes_and_mixtures", "png": str(png), "pdf": str(pdf), "purpose": "Shows why the feature is marked unclear: multiple weak axes compete."})

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    _title(fig, feature_id, unified, classification, final_validation, "limitations from final validation")
    _story_card(ax, unified, final_validation)
    png, pdf = _save(fig, feature_dir, "02_unclear_interpretation_limits")
    manifest.append({"figure": "02_unclear_interpretation_limits", "png": str(png), "pdf": str(pdf), "purpose": "Records the exact limitations and overgeneralization risk for this unclear feature."})
    return manifest


def _plot_community_story(root: Path, feature_id: int, out_dir: Path, top_n: int, background: dict, scatter_sample: int) -> list[dict]:
    evidence, unified, classification, final_validation, events = _load_feature(root, feature_id)
    feature_dir = out_dir / f"feature_{feature_id:06d}"
    manifest = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    _title(fig, feature_id, unified, classification, final_validation, "community/co-occurrence evidence")
    co_table = _cooccurring_table(evidence, top_n)
    _barh_table(axes[0], co_table, "value", "fraction", COLORS["purple"], "Co-occurring taxa", "Fraction")
    tax_data = events.copy()
    tax_data["class_taxon"] = tax_data["taxonomy"].map(lambda x: parse_taxonomy(str(x)).get("class", "unknown"))
    labels, values = _activation_mass_by(tax_data, "class_taxon", top_n)
    _barh(axes[1], labels, values, COLORS["taxonomy"], "Class-level activation mixture")
    labels, values = _activation_mass_by(events, "primary_env", top_n)
    _barh(axes[2], labels, values, COLORS["target"], "Environment mixture")
    png, pdf = _save(fig, feature_dir, "01_community_cooccurrence_mixture")
    manifest.append({"figure": "01_community_cooccurrence_mixture", "png": str(png), "pdf": str(pdf), "purpose": "Shows the co-occurring taxa, mixed taxonomic carriers, and environmental spread supporting a community/co-occurrence interpretation."})

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    _title(fig, feature_id, unified, classification, final_validation, "why this is not a single-axis feature")
    labels = ["taxonomy", "environment", "project", "technology"]
    diag = evidence.get("diagnostics", {})
    values = [
        float(diag.get("taxonomy_signal", {}).get("best_fraction", 0.0)),
        float(diag.get("environment_enrichment", {}).get("max_environment_fraction", 0.0)),
        float(diag.get("project_batch_artifact", {}).get("max_project_fraction", 0.0)),
        float(diag.get("technology_bias", {}).get("max_technology_fraction", 0.0)),
    ]
    axes[0].bar(labels, values, color=[COLORS["taxonomy"], COLORS["target"], COLORS["artifact"], COLORS["purple"]])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Competing interpretation axes", loc="left", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", color="#DDDDDD", alpha=0.7)
    _story_card(axes[1], unified, final_validation)
    png, pdf = _save(fig, feature_dir, "02_community_interpretation_limits")
    manifest.append({"figure": "02_community_interpretation_limits", "png": str(png), "pdf": str(pdf), "purpose": "Documents why the feature is best treated as a mixed community signal rather than a single taxon, environment, or artifact."})
    return manifest


def _plot_disease_story(root: Path, feature_id: int, out_dir: Path, top_n: int, background: dict, scatter_sample: int) -> list[dict]:
    evidence, unified, classification, final_validation, events = _load_feature(root, feature_id)
    feature_dir = out_dir / f"feature_{feature_id:06d}"
    manifest = []
    clinical_terms = {term for terms in DISEASE_KEYWORD_GROUPS.values() for term in terms}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    _title(fig, feature_id, unified, classification, final_validation, "disease-associated context evidence")
    key_table = _keyword_activation_table(events, clinical_terms, top_n)
    if key_table.empty:
        key_table = _keyword_activation_table(events, None, top_n)
    _barh_table(axes[0], key_table, "keyword", "activation_mass", COLORS["artifact"], "Clinical/host keywords")
    labels, values = _activation_mass_by(events, "primary_env", top_n)
    _barh(axes[1], labels, values, COLORS["host"], "Primary environment distribution")
    tax_data = events.copy()
    tax_data["genus_taxon"] = tax_data["taxonomy"].map(lambda x: parse_taxonomy(str(x)).get("genus", "unknown"))
    labels, values = _activation_mass_by(tax_data, "genus_taxon", top_n)
    _barh(axes[2], labels, values, COLORS["taxonomy"], "Taxa observed in disease-context events")
    png, pdf = _save(fig, feature_dir, "01_disease_context_keywords_and_taxa")
    manifest.append({"figure": "01_disease_context_keywords_and_taxa", "png": str(png), "pdf": str(pdf), "purpose": "Shows clinical/host keywords, metadata context, and the taxa observed within disease-context activated events; the taxa panel is not evidence that those taxa are disease-specific."})

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    _title(fig, feature_id, unified, classification, final_validation, "disease interpretation caveats")
    labels, values = _activation_mass_by(events, "project", min(top_n, 8))
    _barh(axes[0], labels, values, COLORS["artifact"], "Top projects by activation mass")
    _story_card(axes[1], unified, final_validation)
    png, pdf = _save(fig, feature_dir, "02_disease_context_caveats")
    manifest.append({"figure": "02_disease_context_caveats", "png": str(png), "pdf": str(pdf), "purpose": "Checks artifact spread and records caveats for interpreting this as disease-associated context rather than a disease-specific marker."})

    keyword_stats = _keyword_statistics(events, DISEASE_KEYWORD_GROUPS)
    keyword_csv = feature_dir / "03_disease_keyword_statistics.csv"
    keyword_stats.to_csv(keyword_csv, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.3), constrained_layout=True)
    _title(fig, feature_id, unified, classification, final_validation, "explicit disease and clinical keyword statistics")
    disease_table = keyword_stats[keyword_stats["group"].isin(["disease", "clinical_infection"])]
    site_table = keyword_stats[keyword_stats["group"].isin(["respiratory_site", "oral_upper_airway", "host_surface"])]
    _plot_grouped_keyword_statistics(axes[0], disease_table, "Disease / infection keywords")
    _plot_grouped_keyword_statistics(axes[1], site_table, "Body-site and host-context keywords")
    png, pdf = _save(fig, feature_dir, "03_disease_keyword_statistics")
    manifest.append(
        {
            "figure": "03_disease_keyword_statistics",
            "png": str(png),
            "pdf": str(pdf),
            "purpose": "Explicitly reports disease and clinical keywords, including lower-frequency COPD/COVID terms that can be hidden by top activation-mass keyword ranking.",
            "csv": str(keyword_csv),
        }
    )
    return manifest


def _plot_perturbation_caveat_story(root: Path, feature_id: int, out_dir: Path, top_n: int) -> dict:
    evidence, unified, classification, final_validation, events = _load_feature(root, feature_id)
    feature_dir = out_dir / f"feature_{feature_id:06d}"
    terms = {
        "feed", "diet", "rumen", "weaning", "antibiotic", "treatment", "supplement", "manure", "raw", "fluid",
        "respiration", "degradation", "fertilization", "stress", "exposure", "transition",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    _title(fig, feature_id, unified, classification, final_validation, "perturbation/transition caveat")
    key_table = _keyword_activation_table(events, terms, top_n)
    if key_table.empty:
        key_table = _keyword_activation_table(events, None, top_n)
    _barh_table(axes[0], key_table, "keyword", "activation_mass", COLORS["abundance"], "Transition/perturbation-related keywords")
    labels, values = _activation_mass_by(events, "env_first2", top_n)
    _barh(axes[1], labels, values, COLORS["host"], "Host/environment subcontexts")
    tax_data = events.copy()
    tax_data["class_taxon"] = tax_data["taxonomy"].map(lambda x: parse_taxonomy(str(x)).get("class", "unknown"))
    labels, values = _activation_mass_by(tax_data, "class_taxon", top_n)
    _barh(axes[2], labels, values, COLORS["taxonomy"], "Taxa in transition examples")
    png, pdf = _save(fig, feature_dir, "03_perturbation_transition_caveat")
    return {
        "figure": "03_perturbation_transition_caveat",
        "png": str(png),
        "pdf": str(pdf),
        "purpose": "Covers Perturbation/transition as a secondary/caveated class by showing transition-related keywords and their host/taxonomic contexts.",
    }


def _plot_feature70_taxonomy_context(root: Path, out_dir: Path, top_n: int) -> dict:
    feature_id = 70
    evidence, unified, classification, final_validation, events = _load_feature(root, feature_id)
    feature_dir = out_dir / f"feature_{feature_id:06d}"
    ladder = _best_taxon_ladder(evidence)
    data = events.copy()
    data["phylum_taxon"] = data["taxonomy"].map(lambda x: parse_taxonomy(str(x)).get("phylum", "unknown"))
    phylum_table = (
        data.groupby("phylum_taxon", dropna=False)
        .agg(events=("activation", "size"), activation_mass=("activation", "sum"))
        .sort_values("activation_mass", ascending=False)
        .reset_index()
    )
    phylum_table["activation_mass_fraction"] = phylum_table["activation_mass"] / max(float(phylum_table["activation_mass"].sum()), 1e-12)
    phylum_csv = feature_dir / "04_phylum_activation_mass.csv"
    feature_dir.mkdir(parents=True, exist_ok=True)
    phylum_table.to_csv(phylum_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 4.9), constrained_layout=True)
    _title(fig, feature_id, unified, classification, final_validation, "taxonomy context for host-associated feature")
    _barh(
        axes[0],
        [f"{rank}: {value}" for rank, value, _ in ladder],
        [fraction for _, _, fraction in ladder],
        COLORS["taxonomy"],
        "Lineage-aware taxon ladder",
        "Fraction among top evidence examples",
        label_max_len=68,
    )
    top_phylum = phylum_table.head(top_n)
    _barh(
        axes[1],
        top_phylum["phylum_taxon"].astype(str).tolist(),
        top_phylum["activation_mass"].astype(float).tolist(),
        COLORS["abundance"],
        "Phylum activation mass",
        "Activation mass across sparse-positive events",
        label_max_len=44,
    )
    png, pdf = _save(fig, feature_dir, "04_taxon_ladder_and_phylum_activation")
    return {
        "figure": "04_taxon_ladder_and_phylum_activation",
        "png": str(png),
        "pdf": str(pdf),
        "purpose": "Shows feature 70's broad taxonomy context with a lineage-aware ladder and phylum-level activation mass; this is context rather than evidence for a taxonomy-primary interpretation.",
        "csv": str(phylum_csv),
    }


def _has_abundance_rank_secondary(classification: dict, evidence: dict) -> bool:
    if "Abundance/rank" not in (classification.get("secondary_classes", []) or []):
        return False
    corr = evidence.get("enrichment", {}).get("abundance_correlation", {})
    return abs(float(corr.get("activation_vs_abundance_rank", 0.0) or 0.0)) >= 0.25 or abs(float(corr.get("activation_vs_abundance", 0.0) or 0.0)) >= 0.25


def _top_fraction_text(rows: list[dict], n: int = 3) -> str:
    parts = []
    for row in rows[:n]:
        value = _clean(row.get("value", ""), 32)
        fraction = row.get("fraction", np.nan)
        try:
            parts.append(f"{value} ({float(fraction):.2f})")
        except (TypeError, ValueError):
            parts.append(value)
    return ", ".join(parts) if parts else "NA"


def _readme_feature_evidence(root: Path, feature_id: int) -> dict:
    evidence, unified, classification, final_validation, _ = _load_feature(root, feature_id)
    metadata = evidence.get("enrichment", {}).get("metadata", {})
    taxonomy = evidence.get("enrichment", {}).get("taxonomy", {})
    diagnostics = evidence.get("diagnostics", {})
    abundance = evidence.get("enrichment", {}).get("abundance_correlation", {})
    taxonomy_signal = diagnostics.get("taxonomy_signal", {})
    project_signal = diagnostics.get("project_batch_artifact", {})
    tech_signal = diagnostics.get("technology_bias", {})
    best_ladder = _best_taxon_ladder(evidence)
    best_ladder_text = ", ".join([f"{rank}:{_clean(value, 28)}={fraction:.2f}" for rank, value, fraction in best_ladder[:7]]) or "NA"
    return {
        "feature_id": feature_id,
        "short_name": unified.get("short_name", ""),
        "summary": unified.get("summary", ""),
        "primary_class": classification.get("primary_class", ""),
        "secondary_classes": "; ".join(classification.get("secondary_classes", []) or []),
        "confidence": final_validation.get("confidence", ""),
        "acceptance": final_validation.get("acceptance", ""),
        "specificity": final_validation.get("specificity", ""),
        "top_env": _top_fraction_text(metadata.get("primary_env", [])),
        "top_project": _top_fraction_text(metadata.get("project", []), n=2),
        "top_technology": _top_fraction_text(metadata.get("technology", []), n=2),
        "top_taxa": best_ladder_text,
        "best_taxon": f"{taxonomy_signal.get('best_rank', 'NA')}={taxonomy_signal.get('best_value', 'NA')} ({float(taxonomy_signal.get('best_fraction', 0.0) or 0.0):.2f})",
        "project_artifact": f"{project_signal.get('top_project', 'NA')} ({float(project_signal.get('max_project_fraction', 0.0) or 0.0):.2f})",
        "technology_bias": f"{tech_signal.get('top_technology', 'NA')} ({float(tech_signal.get('max_technology_fraction', 0.0) or 0.0):.2f})",
        "abundance_rank_corr": abundance.get("activation_vs_abundance_rank", "NA"),
        "taxonomy_ranks_seen": ", ".join([rank for rank, rows in taxonomy.items() if rows][:7]),
    }


def _write_readme(output_dir: Path, rows: list[dict], root: Path) -> Path:
    manifest_df = pd.DataFrame(rows)
    feature_ids = sorted(manifest_df["feature_id"].unique().tolist()) if not manifest_df.empty else []
    lines: list[str] = []
    lines.append("# MGM2 OTU SAE Feature Story Figures")
    lines.append("")
    lines.append("本目录保存用于展示的 feature-specific 图，而不是通用 evidence dump。每个 feature 的图都按解释类别选择，目标是验证解释是否由确定性统计支持。")
    lines.append("")
    lines.append("## 读取原则")
    lines.append("")
    lines.append("- `00_best_taxon_at_each_rank` 只要 feature 与 taxonomy 有关就生成；现在使用 lineage-aware ladder：如果诊断性低阶 taxon 出现在多个父级标签下，会合并父级标签并按合并后的 lineage 覆盖率作图。")
    lines.append("- environment/host 图使用 target group vs control/background，并在有 background metadata 时给出 enrichment 和 binomial p-value。")
    lines.append("- abundance/rank 图使用 rank trend 和 Spearman 检验，不再依赖单纯 activation density。")
    lines.append("- artifact 图先检验 project、technology 等非生物轴，再单独展示仍然存在的 biological caveat。")
    lines.append("- 所有统计都基于 sparse-topk positive lookup events；不等价于全体 zero-included token 分布。")
    lines.append("")
    lines.append("## Feature Summary")
    lines.append("")
    lines.append("| feature | class | confidence | key evidence | figure rationale |")
    lines.append("|---:|---|---|---|---|")
    for feature_id in feature_ids:
        evidence = _readme_feature_evidence(root, int(feature_id))
        sub = manifest_df[manifest_df["feature_id"].eq(feature_id)]
        figure_names = ", ".join(sub["figure"].astype(str).tolist())
        key_bits = [
            f"env: {evidence['top_env']}",
            f"taxon: {evidence['best_taxon']}",
        ]
        if evidence["primary_class"] == "Study/batch/artifact":
            key_bits.append(f"project: {evidence['project_artifact']}")
            key_bits.append(f"technology: {evidence['technology_bias']}")
        if evidence["primary_class"] == "Abundance/rank":
            key_bits.append(f"rank corr: {evidence['abundance_rank_corr']}")
        rationale = "; ".join(sub["purpose"].astype(str).tolist())
        lines.append(
            f"| {feature_id} | {evidence['primary_class']} | {evidence['confidence']} | "
            f"{'<br>'.join(key_bits)} | {figure_names}<br>{rationale} |"
        )
    lines.append("")
    lines.append("## Per-feature Notes")
    lines.append("")
    for feature_id in feature_ids:
        evidence = _readme_feature_evidence(root, int(feature_id))
        sub = manifest_df[manifest_df["feature_id"].eq(feature_id)]
        lines.append(f"### Feature {feature_id}: {evidence['short_name']}")
        lines.append("")
        lines.append(f"- Interpretation: {evidence['summary']}")
        lines.append(f"- Primary / secondary classes: {evidence['primary_class']} / {evidence['secondary_classes'] or 'NA'}")
        lines.append(f"- Best taxon evidence: {evidence['top_taxa']}")
        lines.append(f"- Metadata evidence: env {evidence['top_env']}; project {evidence['top_project']}; technology {evidence['top_technology']}.")
        lines.append(f"- Specificity note: {evidence['specificity'] or 'NA'}")
        lines.append("- Why these plots:")
        for _, row in sub.iterrows():
            stats = []
            if pd.notna(row.get("binomial_pvalue", np.nan)):
                stats.append(f"binomial p={_format_pvalue(float(row['binomial_pvalue']))}")
            if pd.notna(row.get("project_binomial_pvalue", np.nan)):
                stats.append(f"project binomial p={_format_pvalue(float(row['project_binomial_pvalue']))}")
            if pd.notna(row.get("technology_binomial_pvalue", np.nan)):
                stats.append(f"technology binomial p={_format_pvalue(float(row['technology_binomial_pvalue']))}")
            if pd.notna(row.get("spearman_rank_pvalue", np.nan)):
                stats.append(
                    f"rank Spearman rho={float(row.get('spearman_rank_rho', np.nan)):.3f}, p={_format_pvalue(float(row['spearman_rank_pvalue']))}"
                )
            stat_text = f" ({'; '.join(stats)})" if stats else ""
            lines.append(f"  - `{row['figure']}`: {row['purpose']}{stat_text}")
        lines.append("")
    readme = output_dir / "README.md"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme


def main():
    args = parse_args()
    config = load_config(args.config)
    root = Path(cfg(config, "output_dir"))
    output_dir = args.output_dir or root / "reports" / "feature_story_figures_refined"
    background = _load_json(Path(cfg(config, "background_metadata", root / "background_metadata.json")))
    plotters = {
        "Taxonomic identity": _plot_taxonomy_story,
        "Ecological niche": lambda *a, **k: _plot_environment_story(*a, **k, host_mode=False),
        "Host-associated": lambda *a, **k: _plot_environment_story(*a, **k, host_mode=True),
        "Study/batch/artifact": _plot_artifact_story,
        "Abundance/rank": _plot_abundance_story,
        "Community interaction/co-occurrence": _plot_community_story,
        "Disease-associated context": _plot_disease_story,
        "Unclear": _plot_unclear_story,
    }
    rows = []
    feature_ids = _feature_ids(args.feature_ids)
    for feature_id in feature_ids:
        evidence, unified, classification, final_validation, _ = _load_feature(root, feature_id)
        primary_class = classification.get("primary_class", "Unclear")
        if _is_taxonomy_related(classification, evidence):
            item = _plot_best_taxon_rank_figure(root, feature_id, output_dir, args.top_n)
            rows.append(
                {
                    "feature_id": feature_id,
                    "short_name": unified.get("short_name", ""),
                    "primary_class": primary_class,
                    "confidence": final_validation.get("confidence", ""),
                    "acceptance": final_validation.get("acceptance", ""),
                    **item,
                }
            )
        plotter = plotters.get(primary_class, _plot_unclear_story)
        for item in plotter(root, feature_id, output_dir, args.top_n, background, args.scatter_sample):
            rows.append(
                {
                    "feature_id": feature_id,
                    "short_name": unified.get("short_name", ""),
                    "primary_class": primary_class,
                    "confidence": final_validation.get("confidence", ""),
                    "acceptance": final_validation.get("acceptance", ""),
                    **item,
                }
            )
        if primary_class != "Abundance/rank" and _has_abundance_rank_secondary(classification, evidence):
            for item in _plot_abundance_story(root, feature_id, output_dir, args.top_n, background, args.scatter_sample):
                item = {**item, "purpose": "Secondary abundance/rank evidence: " + item.get("purpose", "")}
                rows.append(
                    {
                        "feature_id": feature_id,
                        "short_name": unified.get("short_name", ""),
                        "primary_class": primary_class,
                        "confidence": final_validation.get("confidence", ""),
                        "acceptance": final_validation.get("acceptance", ""),
                        **item,
                    }
                )
        if "Perturbation/transition" in (classification.get("secondary_classes", []) or []):
            item = _plot_perturbation_caveat_story(root, feature_id, output_dir, args.top_n)
            rows.append(
                {
                    "feature_id": feature_id,
                    "short_name": unified.get("short_name", ""),
                    "primary_class": primary_class,
                    "confidence": final_validation.get("confidence", ""),
                    "acceptance": final_validation.get("acceptance", ""),
                    **item,
                }
            )
        if feature_id == 70:
            item = _plot_feature70_taxonomy_context(root, output_dir, args.top_n)
            rows.append(
                {
                    "feature_id": feature_id,
                    "short_name": unified.get("short_name", ""),
                    "primary_class": primary_class,
                    "confidence": final_validation.get("confidence", ""),
                    "acceptance": final_validation.get("acceptance", ""),
                    **item,
                }
            )
    manifest = output_dir / "feature_story_figures_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    readme = _write_readme(output_dir, rows, root)
    print(json.dumps({"features": len(feature_ids), "figures": len(rows), "manifest": str(manifest), "readme": str(readme), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
