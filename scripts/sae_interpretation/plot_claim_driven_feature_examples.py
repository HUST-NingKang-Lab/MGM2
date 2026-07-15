#!/usr/bin/env python3
"""Plot claim-vs-control figures for selected OTU SAE feature interpretations."""

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

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.sae_interpretation.config import cfg, load_config
from src.sae_interpretation.taxonomy import parse_taxonomy


DEFAULT_FEATURES = (
    "2131,1821,3915,3435,"
    "686,758,2711,3602,"
    "9,25,2032,2647,"
    "973,1757,2400,"
    "3358,523,"
    "1974,4081"
)

COLORS = {
    "target": "#2A9D8F",
    "control": "#BDBDBD",
    "artifact": "#E15759",
    "taxonomy": "#4E79A7",
    "abundance": "#F28E2B",
    "unclear": "#8A8A8A",
}
ENV_COLORS = {
    "animal": "#4E79A7",
    "aquatic": "#2A9D8F",
    "soil": "#9C755F",
    "plant": "#59A14F",
    "<missing>": "#8A8A8A",
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
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--scatter-sample", type=int, default=1800)
    return parser.parse_args()


def _feature_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _clean(value: object, max_len: int = 34) -> str:
    text = "<missing>" if pd.isna(value) else str(value)
    for prefix in ["d__", "k__", "p__", "c__", "o__", "f__", "g__", "s__"]:
        text = text.replace(prefix, "")
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


def _background_map(background: dict, key: str) -> dict[str, float]:
    return {
        str(row.get("value", "")): float(row.get("fraction", 0.0))
        for row in background.get("distributions", {}).get(key, [])
    }


def _load_feature(root: Path, feature_id: int):
    evidence = _load_json(root / "evidence" / f"feature_{feature_id:06d}.evidence.json")
    unified = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "unified_description.json")
    classification = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "classification.json")
    final_validation = _load_json(root / "codex_outputs" / f"feature_{feature_id:06d}" / "final_validation.json")
    events = pd.read_csv(root / "feature_lookup" / f"feature_{feature_id:06d}.events.csv.gz")
    for col in ["activation", "abundance", "abundance_rank"]:
        if col in events:
            events[col] = pd.to_numeric(events[col], errors="coerce")
    events["primary_env"] = events.get("primary_env", "<missing>").fillna("<missing>").astype(str)
    events["env_first2"] = events.get("env_first2", "<missing>").fillna("<missing>").astype(str)
    events["project"] = events.get("project", "<missing>").fillna("<missing>").astype(str)
    events["technology"] = events.get("technology", "<missing>").fillna("<missing>").astype(str)
    events["taxonomy"] = events.get("taxonomy", "").fillna("").astype(str)
    return evidence, unified, classification, final_validation, events


def _save(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def _title(fig, feature_id: int, short_name: str, claim: str, confidence: str) -> None:
    fig.suptitle(
        f"Feature {feature_id}: {short_name}\nClaim test: {claim} | confidence={confidence}",
        fontsize=14,
        fontweight="bold",
    )


def _binary_summary(events: pd.DataFrame, mask: pd.Series | np.ndarray) -> dict:
    mask = pd.Series(mask, index=events.index).fillna(False).astype(bool)
    target = events[mask]
    control = events[~mask]
    total_activation = float(events["activation"].sum())
    return {
        "target_events": int(len(target)),
        "control_events": int(len(control)),
        "target_event_fraction": float(len(target) / max(1, len(events))),
        "target_activation_sum": float(target["activation"].sum()),
        "control_activation_sum": float(control["activation"].sum()),
        "target_activation_fraction": float(target["activation"].sum() / max(total_activation, 1e-12)),
        "target_mean_activation": float(target["activation"].mean()) if len(target) else 0.0,
        "control_mean_activation": float(control["activation"].mean()) if len(control) else 0.0,
    }


def _plot_binary_claim(
    axes,
    events: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    target_label: str,
    control_label: str,
    background_fraction: float | None,
    color: str,
) -> dict:
    summary = _binary_summary(events, mask)
    observed = [summary["target_event_fraction"], 1.0 - summary["target_event_fraction"]]
    activation = [summary["target_activation_fraction"], 1.0 - summary["target_activation_fraction"]]
    labels = [target_label, control_label]
    x = np.arange(2)
    axes[0].bar(x - 0.18, observed, width=0.36, color=[color, COLORS["control"]], label="positive event share")
    axes[0].bar(x + 0.18, activation, width=0.36, color=[color, COLORS["control"]], alpha=0.55, label="activation mass share")
    if background_fraction is not None:
        axes[0].scatter([0, 1], [background_fraction, 1.0 - background_fraction], color="#222222", s=38, zorder=3, label="background token share")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Fraction")
    axes[0].set_title("Claim group vs control", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].grid(axis="y", color="#DDDDDD", alpha=0.7)

    mask_series = pd.Series(mask, index=events.index).fillna(False).astype(bool)
    data = [
        events.loc[mask_series, "activation"].dropna().to_numpy(),
        events.loc[~mask_series, "activation"].dropna().to_numpy(),
    ]
    axes[1].boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, patch_color in zip(axes[1].artists, [color, COLORS["control"]]):
        patch.set_facecolor(patch_color)
        patch.set_alpha(0.6)
    axes[1].set_ylabel("Activation among positive events")
    axes[1].set_title("Activation magnitude", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#DDDDDD", alpha=0.7)
    axes[1].tick_params(axis="x", rotation=20)
    return summary


def _barh(ax, rows: list[dict], title: str, color: str, value_key: str = "fraction", top_n: int = 8) -> None:
    rows = rows[:top_n]
    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color="#666666")
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontweight="bold")
        return
    labels = [_clean(row.get("value", "")) for row in rows][::-1]
    values = [float(row.get(value_key, 0.0)) for row in rows][::-1]
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, alpha=0.86)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, max(1.0, max(values) * 1.15))
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", color="#DDDDDD", alpha=0.7)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)


def _taxonomy_mask(events: pd.DataFrame, rank: str, value: str) -> pd.Series:
    def matches(taxonomy: str) -> bool:
        return str(parse_taxonomy(taxonomy).get(rank, "")) == str(value)

    return events["taxonomy"].map(matches)


def _plot_text(ax, title: str, text: str) -> None:
    ax.axis("off")
    ax.text(0, 1, title + "\n\n" + textwrap.fill(text, 78), va="top", fontsize=9, linespacing=1.35)


def _plot_taxonomic(root, feature_id, output_dir, top_n, scatter_sample, background) -> tuple[list[Path], dict]:
    evidence, unified, cls, final, events = _load_feature(root, feature_id)
    signal = evidence.get("diagnostics", {}).get("taxonomy_signal", {})
    rank = signal.get("best_rank", "phylum")
    value = signal.get("best_value", "")
    mask = _taxonomy_mask(events, rank, value)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), constrained_layout=True)
    claim = f"{rank}={_clean(value)} positive events exceed other taxa"
    _title(fig, feature_id, unified.get("short_name", ""), claim, final.get("confidence", ""))
    summary = _plot_binary_claim(axes[0], events, mask, _clean(value, 22), "other taxa", None, COLORS["taxonomy"])
    _barh(axes[1, 0], evidence.get("enrichment", {}).get("taxonomy", {}).get(rank, []), f"{rank} composition", COLORS["taxonomy"], top_n=top_n)
    _plot_text(
        axes[1, 1],
        "Interpretation check",
        f"Primary class: {cls.get('primary_class')}. Taxonomy diagnostic reports best_fraction={signal.get('best_fraction')} strong={signal.get('strong')}. "
        f"Target activation mass fraction among lookup positives is {summary['target_activation_fraction']:.2f}, with event fraction {summary['target_event_fraction']:.2f}.",
    )
    paths = _save(fig, output_dir, f"feature_{feature_id:06d}_claim_taxon_vs_other")
    return paths, {"claim_test": claim, "target_group": f"{rank}={value}", "control_group": "other taxa"}


def _plot_environment_claim(root, feature_id, output_dir, top_n, scatter_sample, background, target_env: str | None = None, figure_type: str = "environment") -> tuple[list[Path], dict]:
    evidence, unified, cls, final, events = _load_feature(root, feature_id)
    env_diag = evidence.get("diagnostics", {}).get("environment_enrichment", {})
    target = target_env or env_diag.get("top_environment", "")
    bg = _background_map(background, "primary_env").get(str(target), None)
    mask = events["primary_env"].eq(str(target))
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), constrained_layout=True)
    claim = f"{target} positive events exceed non-{target} and background"
    _title(fig, feature_id, unified.get("short_name", ""), claim, final.get("confidence", ""))
    summary = _plot_binary_claim(axes[0], events, mask, str(target), f"non-{target}", bg, COLORS["target"])
    _barh(axes[1, 0], evidence.get("enrichment", {}).get("metadata", {}).get("env_first2", []), "Sub-environment support", "#59A14F", top_n=top_n)
    _plot_text(
        axes[1, 1],
        "Interpretation check",
        f"Primary class: {cls.get('primary_class')}. Background token share for {target} is {bg:.2f} when available. "
        f"Among feature positives, {target} event fraction is {summary['target_event_fraction']:.2f} and activation mass fraction is {summary['target_activation_fraction']:.2f}. "
        f"Environment diagnostic strong={env_diag.get('strong')} enrichment_ratio={env_diag.get('enrichment_ratio')}.",
    )
    paths = _save(fig, output_dir, f"feature_{feature_id:06d}_claim_{figure_type}_vs_control")
    return paths, {"claim_test": claim, "target_group": str(target), "control_group": f"non-{target}"}


def _plot_host(root, feature_id, output_dir, top_n, scatter_sample, background) -> tuple[list[Path], dict]:
    return _plot_environment_claim(root, feature_id, output_dir, top_n, scatter_sample, background, target_env="animal", figure_type="host")


def _plot_artifact(root, feature_id, output_dir, top_n, scatter_sample, background) -> tuple[list[Path], dict]:
    evidence, unified, cls, final, events = _load_feature(root, feature_id)
    project_diag = evidence.get("diagnostics", {}).get("project_batch_artifact", {})
    tech_diag = evidence.get("diagnostics", {}).get("technology_bias", {})
    target_project = str(project_diag.get("top_project", ""))
    target_tech = str(tech_diag.get("top_technology", ""))
    project_bg = _background_map(background, "project").get(target_project, None)
    tech_bg = _background_map(background, "technology").get(target_tech, None)
    project_mask = events["project"].eq(target_project)
    tech_mask = events["technology"].eq(target_tech)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), constrained_layout=True)
    claim = f"project/technology concentration: {target_project} and {target_tech}"
    _title(fig, feature_id, unified.get("short_name", ""), claim, final.get("confidence", ""))
    project_summary = _plot_binary_claim(axes[0], events, project_mask, _clean(target_project, 20), "other projects", project_bg, COLORS["artifact"])
    tech_summary = _plot_binary_claim(axes[1], events, tech_mask, _clean(target_tech, 20), "other technologies", tech_bg, "#B07AA1")
    axes[0, 0].set_title("Project claim vs background", loc="left", fontweight="bold")
    axes[1, 0].set_title("Technology claim vs background", loc="left", fontweight="bold")
    paths = _save(fig, output_dir, f"feature_{feature_id:06d}_claim_artifact_vs_background")
    return paths, {
        "claim_test": claim,
        "target_group": f"project={target_project}; technology={target_tech}",
        "control_group": "other projects/technologies",
        "project_activation_fraction": f"{project_summary['target_activation_fraction']:.3f}",
        "technology_activation_fraction": f"{tech_summary['target_activation_fraction']:.3f}",
    }


def _plot_abundance(root, feature_id, output_dir, top_n, scatter_sample, background) -> tuple[list[Path], dict]:
    evidence, unified, cls, final, events = _load_feature(root, feature_id)
    data = events.dropna(subset=["activation", "abundance_rank", "abundance"]).copy()
    if len(data) > scatter_sample:
        data = data.sample(scatter_sample, random_state=feature_id)
    threshold = float(data["abundance_rank"].quantile(0.25)) if len(data) else 0.0
    mask = data["abundance_rank"] <= threshold
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), constrained_layout=True)
    claim = f"activation differs by abundance rank; dominant rank <= {threshold:.0f}"
    _title(fig, feature_id, unified.get("short_name", ""), claim, final.get("confidence", ""))
    _plot_binary_claim(axes[0], data, mask, "dominant-rank positives", "other positives", None, COLORS["abundance"])
    x = np.log10(data["abundance"].clip(lower=1e-8))
    colors = [ENV_COLORS.get(env, "#BAB0AC") for env in data["primary_env"]]
    axes[1, 0].scatter(x, data["activation"], c=colors, s=12, alpha=0.5, linewidths=0)
    axes[1, 0].set_xlabel("log10(relative abundance)")
    axes[1, 0].set_ylabel("Activation")
    axes[1, 0].set_title("Activation vs abundance", loc="left", fontweight="bold")
    axes[1, 0].grid(color="#DDDDDD", alpha=0.7)
    corr = evidence.get("enrichment", {}).get("abundance_correlation", {})
    _plot_text(
        axes[1, 1],
        "Interpretation check",
        f"Primary class: {cls.get('primary_class')}. activation_vs_abundance={corr.get('activation_vs_abundance')}; "
        f"activation_vs_abundance_rank={corr.get('activation_vs_abundance_rank')}. "
        f"This panel checks whether the claimed abundance/rank behavior is visible among positive events.",
    )
    paths = _save(fig, output_dir, f"feature_{feature_id:06d}_claim_abundance_rank")
    return paths, {"claim_test": claim, "target_group": "dominant-rank positives", "control_group": "other positives"}


def _plot_unclear(root, feature_id, output_dir, top_n, scatter_sample, background) -> tuple[list[Path], dict]:
    evidence, unified, cls, final, events = _load_feature(root, feature_id)
    env_diag = evidence.get("diagnostics", {}).get("environment_enrichment", {})
    tax_diag = evidence.get("diagnostics", {}).get("taxonomy_signal", {})
    project_diag = evidence.get("diagnostics", {}).get("project_batch_artifact", {})
    tech_diag = evidence.get("diagnostics", {}).get("technology_bias", {})
    labels = ["taxonomy", "environment", "project", "technology"]
    values = [
        float(tax_diag.get("best_fraction", 0.0)),
        float(env_diag.get("max_environment_fraction", 0.0)),
        float(project_diag.get("max_project_fraction", 0.0)),
        float(tech_diag.get("max_technology_fraction", 0.0)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), constrained_layout=True)
    claim = "no single biological or artifact axis dominates"
    _title(fig, feature_id, unified.get("short_name", ""), claim, final.get("confidence", ""))
    axes[0, 0].bar(labels, values, color=[COLORS["taxonomy"], COLORS["target"], COLORS["artifact"], "#B07AA1"])
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("Competing axis strengths", loc="left", fontweight="bold")
    axes[0, 0].grid(axis="y", color="#DDDDDD", alpha=0.7)
    axes[0, 0].tick_params(axis="x", rotation=25)
    _barh(axes[0, 1], evidence.get("enrichment", {}).get("metadata", {}).get("primary_env", []), "Environment mixture", COLORS["unclear"], top_n=top_n)
    _barh(axes[1, 0], evidence.get("enrichment", {}).get("taxonomy", {}).get("phylum", []), "Taxonomy mixture", COLORS["unclear"], top_n=top_n)
    _plot_text(axes[1, 1], "Interpretation check", final.get("specificity", "") + " " + final.get("overgeneralization_risk", ""))
    paths = _save(fig, output_dir, f"feature_{feature_id:06d}_claim_unclear_axes")
    return paths, {"claim_test": claim, "target_group": "none", "control_group": "competing axes"}


def main():
    args = parse_args()
    config = load_config(args.config)
    root = Path(cfg(config, "output_dir"))
    output_dir = args.output_dir or root / "reports" / "claim_driven_feature_figures"
    background = _load_json(Path(cfg(config, "background_metadata", root / "background_metadata.json")))
    plotters = {
        "Taxonomic identity": _plot_taxonomic,
        "Ecological niche": _plot_environment_claim,
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
        paths, claim_info = plotter(root, feature_id, output_dir, args.top_n, args.scatter_sample, background)
        manifest.append(
            {
                "feature_id": feature_id,
                "short_name": unified.get("short_name", ""),
                "primary_class": primary,
                "confidence": final.get("confidence", ""),
                "acceptance": final.get("acceptance", ""),
                "figure_type": paths[0].stem.replace(f"feature_{feature_id:06d}_", ""),
                "png": str(paths[0]),
                "pdf": str(paths[1]),
                **claim_info,
            }
        )
    manifest_path = output_dir / "claim_driven_feature_figures_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(json.dumps({"figures": len(manifest), "manifest": str(manifest_path), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
