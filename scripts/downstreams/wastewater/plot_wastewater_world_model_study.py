#!/usr/bin/env python3
"""Plot manuscript-style figures for the wastewater MGM2 world-model study."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = 11
mpl.rcParams["axes.labelsize"] = 10
mpl.rcParams["xtick.labelsize"] = 9
mpl.rcParams["ytick.labelsize"] = 9
mpl.rcParams["legend.fontsize"] = 9


DEFAULT_RESULTS_DIR = Path("outputs_downstreams/wastewater/all_results")
DEFAULT_OUTPUT_DIR = Path("figures/wastewater_world_model")
SCALE_ORDER = ["scale_small", "scale_medium", "scale_large", "scale_xlarge"]
SCALE_SHORT = {
    "scale_small": "small",
    "scale_medium": "medium",
    "scale_large": "large",
    "scale_xlarge": "xlarge",
}
LABELS = {
    "context_mean": "Context mean",
    "persistence": "Persistence",
    "vector_gru": "Vector GRU",
    "vector_tcn": "Vector TCN",
    "hybrid_vector_gru_persistence_cutoff2": "GRU + persistence",
    "hybrid_vector_tcn_persistence_cutoff3": "TCN + persistence",
    "token_xattn_scale_small": "Token forecast small",
    "token_xattn_scale_medium": "Token forecast medium",
    "token_xattn_scale_large": "Token forecast large",
    "token_xattn_scale_xlarge": "Token forecast xlarge",
    "mae_scale_small": "MAE small",
    "mae_scale_medium": "MAE medium",
    "mae_scale_large": "MAE large",
    "mae_scale_xlarge": "MAE xlarge",
    "jepa_scale_small": "JEPA small",
    "jepa_scale_medium": "JEPA medium",
    "jepa_scale_large": "JEPA large",
    "jepa_scale_xlarge": "JEPA xlarge",
    "cls_scale_small": "CLS temporal small",
}
COLORS = {
    "context_mean": "#8A817C",
    "persistence": "#4D4D4D",
    "vector_gru": "#63A375",
    "vector_tcn": "#2E7D59",
    "hybrid_vector_gru_persistence_cutoff2": "#B36B00",
    "hybrid_vector_tcn_persistence_cutoff3": "#D8902F",
    "token_xattn_scale_small": "#386CB0",
    "token_xattn_scale_medium": "#2F5597",
    "token_xattn_scale_large": "#1F3A73",
    "token_xattn_scale_xlarge": "#102A54",
    "mae_scale_small": "#7CB5EC",
    "mae_scale_medium": "#4363D8",
    "mae_scale_large": "#1A237E",
    "mae_scale_xlarge": "#4A148C",
    "jepa_scale_small": "#E78AC3",
    "jepa_scale_medium": "#C94D9B",
    "jepa_scale_large": "#9C2D7A",
    "jepa_scale_xlarge": "#6A1B9A",
    "cls_scale_small": "#BDBDBD",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="*", default=["png", "pdf"])
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"required result table not found: {path}")
    return path


def label(run_label: str) -> str:
    return LABELS.get(run_label, run_label.replace("_", " "))


def color(run_label: str) -> str:
    return COLORS.get(run_label, "#666666")


def style_axis(ax, grid_axis: str = "y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.25, linewidth=0.8)


def save_figure(fig, output_dir: Path, stem: str, formats: Sequence[str]):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def add_panel_label(ax, text: str):
    ax.text(-0.08, 1.08, text, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def safe_rows(frame: pd.DataFrame, run_labels: Iterable[str]) -> pd.DataFrame:
    rows = frame[frame["run_label"].isin(list(run_labels))].copy()
    order = {run_label: idx for idx, run_label in enumerate(run_labels)}
    rows["plot_order"] = rows["run_label"].map(order)
    return rows.sort_values("plot_order")


def plot_main_performance(weighted: pd.DataFrame, output_dir: Path, formats: Sequence[str]):
    selected = [
        "context_mean",
        "persistence",
        "vector_tcn",
        "hybrid_vector_tcn_persistence_cutoff3",
        "token_xattn_scale_small",
        "jepa_scale_xlarge",
        "mae_scale_xlarge",
    ]
    rows = safe_rows(weighted, selected)
    rows = rows.sort_values("bray_curtis")
    metric_specs = [
        ("bray_curtis", "Bray-Curtis distance", "lower"),
        ("mae", "Mean absolute error", "lower"),
        ("rmse", "RMSE", "lower"),
        ("spearman", "Spearman correlation", "higher"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.4))
    axes = axes.flatten()
    y = np.arange(len(rows))
    for panel_idx, (ax, (metric, title, direction)) in enumerate(zip(axes, metric_specs)):
        ax.barh(y, rows[metric], color=[color(v) for v in rows["run_label"]], edgecolor="#333333", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([label(v) for v in rows["run_label"]] if panel_idx in (0, 2) else [])
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.set_title(f"{title} ({direction} is better)")
        for idx, value in enumerate(rows[metric]):
            text = f"{value:.3f}" if metric != "mae" else f"{value:.4f}"
            ax.text(value + max(rows[metric]) * 0.006, idx, text, va="center", fontsize=7.5)
        style_axis(ax, "x")
        add_panel_label(ax, chr(ord("A") + panel_idx))

    fig.suptitle("MGM2 wastewater world model improves multiple continuous abundance metrics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_dir, "main_figure_1_performance", formats)


def plot_main_horizon(horizon: pd.DataFrame, output_dir: Path, formats: Sequence[str]):
    selected = [
        "persistence",
        "vector_tcn",
        "token_xattn_scale_small",
        "jepa_scale_xlarge",
        "mae_scale_xlarge",
    ]
    rows = horizon[horizon["run_label"].isin(selected)].copy()
    metric_specs = [
        ("bray_curtis", "Bray-Curtis distance", "lower"),
        ("mae", "Mean absolute error", "lower"),
        ("spearman", "Spearman correlation", "higher"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    for panel_idx, (ax, (metric, ylabel, direction)) in enumerate(zip(axes, metric_specs)):
        for run_label in selected:
            sub = rows[rows["run_label"].eq(run_label)].sort_values("horizon")
            ax.plot(sub["horizon"], sub[metric], marker="o", linewidth=2.0, markersize=4, color=color(run_label), label=label(run_label))
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by horizon ({direction} is better)")
        ax.set_xticks(range(1, 11))
        if panel_idx == 0:
            ax.legend(frameon=False)
        style_axis(ax)
        add_panel_label(ax, chr(ord("A") + panel_idx))

    fig.suptitle("Horizon-specific performance across continuous abundance metrics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_dir, "main_figure_2_horizon", formats)


def plot_main_plant_robustness(plant_bc: pd.DataFrame, output_dir: Path, formats: Sequence[str]):
    plant_bc = plant_bc.set_index("plant")
    delta_persistence = (plant_bc["mae_scale_xlarge"] - plant_bc["persistence"]).sort_values()
    delta_token = plant_bc.loc[delta_persistence.index, "mae_scale_xlarge"] - plant_bc.loc[delta_persistence.index, "token_xattn_scale_small"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), gridspec_kw={"width_ratios": [1.0, 1.0]})

    ax = axes[0]
    y = np.arange(len(delta_persistence))
    colors = np.where(delta_persistence.values < 0, "#2E7D59", "#B23A48")
    ax.barh(y, delta_persistence.values, color=colors, edgecolor="#333333", linewidth=0.35)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(delta_persistence.index)
    ax.set_xlabel("MAE xlarge - persistence Bray-Curtis")
    ax.set_title("Per-plant gain vs persistence")
    ax.invert_yaxis()
    style_axis(ax, "x")
    add_panel_label(ax, "A")

    ax = axes[1]
    colors = np.where(delta_token.values < 0, "#2E7D59", "#B23A48")
    ax.barh(y, delta_token.values, color=colors, edgecolor="#333333", linewidth=0.35)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(delta_persistence.index)
    ax.set_xlabel("MAE xlarge - token forecast small Bray-Curtis")
    ax.set_title("Per-plant gain vs token forecast")
    ax.invert_yaxis()
    style_axis(ax, "x")
    add_panel_label(ax, "B")

    fig.suptitle("Best world model improves most plants, with a few site-specific exceptions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_dir, "main_figure_3_plant_robustness", formats)


def plot_supp_scale(weighted: pd.DataFrame, output_dir: Path, formats: Sequence[str]):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
    scale_x = np.arange(len(SCALE_ORDER))
    mode_specs = [("mae", "MAE world model"), ("jepa", "JEPA world model"), ("token_xattn", "Token forecast head")]
    for ax, (mode, title) in zip(axes, mode_specs):
        values = []
        for scale in SCALE_ORDER:
            if mode == "token_xattn":
                run_label = f"token_xattn_{scale}"
            else:
                run_label = f"{mode}_{scale}"
            row = weighted[weighted["run_label"].eq(run_label)]
            values.append(float(row["bray_curtis"].iloc[0]) if len(row) else np.nan)
        ax.plot(scale_x, values, marker="o", linewidth=2.0, color={"mae": "#4363D8", "jepa": "#C94D9B", "token_xattn": "#386CB0"}[mode])
        ax.set_xticks(scale_x)
        ax.set_xticklabels([SCALE_SHORT[s] for s in SCALE_ORDER], rotation=20)
        ax.set_ylabel("Test Bray-Curtis")
        ax.set_title(title)
        style_axis(ax)
    fig.suptitle("Scale ablation across MGM2-derived forecasters", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_dir, "supp_figure_1_scale_ablation", formats)


def plot_supp_metric_tradeoff(weighted: pd.DataFrame, output_dir: Path, formats: Sequence[str]):
    selected = [
        "persistence",
        "vector_tcn",
        "hybrid_vector_tcn_persistence_cutoff3",
        "token_xattn_scale_small",
        "mae_scale_xlarge",
        "jepa_scale_xlarge",
    ]
    rows = safe_rows(weighted, selected)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
    for ax, metric, title in zip(
        axes,
        ["mae", "rmse", "spearman"],
        ["Mean absolute error", "RMSE", "Rank correlation"],
    ):
        ax.scatter(rows["bray_curtis"], rows[metric], s=80, color=[color(v) for v in rows["run_label"]], edgecolor="#333333", linewidth=0.6)
        for _, row in rows.iterrows():
            ax.annotate(label(row["run_label"]), (row["bray_curtis"], row[metric]), xytext=(4, 3), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Bray-Curtis distance")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs Bray-Curtis")
        style_axis(ax)
    fig.suptitle("Continuous metric relationships across forecasting methods", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_dir, "supp_figure_2_metric_tradeoffs", formats)


def plot_supp_training(train_summary: pd.DataFrame, output_dir: Path, formats: Sequence[str]):
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    for ax, mode, metric, title in [
        (axes[0], "mae", "best_train_tail_holdout_loss", "MAE holdout reconstruction loss"),
        (axes[1], "jepa", "best_train_tail_holdout_loss", "JEPA latent holdout loss"),
        (axes[2], "jepa", "best_probe_holdout_loss", "JEPA probe holdout loss"),
    ]:
        rows = train_summary[train_summary["mode"].eq(mode)].copy()
        rows["scale_order"] = rows["run_label"].str.extract(r"(scale_[a-z]+)", expand=False).map({s: i for i, s in enumerate(SCALE_ORDER)})
        rows = rows.sort_values("scale_order")
        ax.plot(rows["scale_order"], rows[metric], marker="o", linewidth=2.0, color="#4A148C" if mode == "jepa" else "#4363D8")
        ax.set_xticks(range(len(SCALE_ORDER)))
        ax.set_xticklabels([SCALE_SHORT[s] for s in SCALE_ORDER], rotation=20)
        ax.set_title(title)
        ax.set_ylabel(metric.replace("_", " "))
        style_axis(ax)
    fig.suptitle("Training diagnostics for MAE and JEPA world models", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_dir, "supp_figure_3_training_diagnostics", formats)


def plot_supp_plant_heatmap(plant_bc: pd.DataFrame, output_dir: Path, formats: Sequence[str]):
    selected = [
        "persistence",
        "token_xattn_scale_small",
        "mae_scale_small",
        "mae_scale_medium",
        "mae_scale_large",
        "mae_scale_xlarge",
        "jepa_scale_small",
        "jepa_scale_medium",
        "jepa_scale_large",
        "jepa_scale_xlarge",
    ]
    frame = plant_bc.set_index("plant")
    selected = [col for col in selected if col in frame.columns]
    frame = frame[selected]
    order = (frame["mae_scale_xlarge"] - frame["persistence"]).sort_values().index
    frame = frame.loc[order]
    fig, ax = plt.subplots(figsize=(10.8, 7.5))
    image = ax.imshow(frame.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(frame.columns)))
    ax.set_xticklabels([label(col) for col in frame.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(frame.index)))
    ax.set_yticklabels(frame.index)
    ax.set_title("Per-plant Bray-Curtis distances")
    cbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Bray-Curtis distance")
    fig.tight_layout()
    return save_figure(fig, output_dir, "supp_figure_4_plant_heatmap", formats)


def write_readme(
    output_dir: Path,
    weighted: pd.DataFrame,
    wins: pd.DataFrame,
    train_summary: pd.DataFrame,
    generated: dict[str, list[Path]],
):
    best = weighted.iloc[0]
    persistence = float(weighted.loc[weighted["run_label"].eq("persistence"), "bray_curtis"].iloc[0])
    token = float(weighted.loc[weighted["run_label"].eq("token_xattn_scale_small"), "bray_curtis"].iloc[0])
    rel_persistence = (persistence - float(best["bray_curtis"])) / persistence * 100.0
    rel_token = (token - float(best["bray_curtis"])) / token * 100.0
    best_vs_persistence = wins[(wins["run_label"].eq(best["run_label"])) & (wins["reference"].eq("persistence"))].iloc[0]
    best_vs_token = wins[(wins["run_label"].eq(best["run_label"])) & (wins["reference"].eq("token_xattn_scale_small"))].iloc[0]

    top_table = weighted[
        ["run_label", "bray_curtis", "mae", "rmse", "spearman"]
    ].head(12)
    train_table = train_summary.sort_values("run_label").fillna("")

    lines = [
        "# Wastewater MGM2 World Model Figures",
        "",
        "This directory contains manuscript-oriented figures for the wastewater temporal forecasting study.",
        "",
        "## Result Summary",
        "",
        f"- Best overall model: **{label(best['run_label'])}** (`{best['run_label']}`).",
        f"- Test weighted Bray-Curtis: **{best['bray_curtis']:.6f}**.",
        f"- Relative Bray-Curtis improvement vs persistence: **{rel_persistence:.2f}%**.",
        f"- Relative Bray-Curtis improvement vs small token forecast: **{rel_token:.2f}%**.",
        f"- Per-plant wins vs persistence: **{int(best_vs_persistence['wins_lower_bray'])}/{int(best_vs_persistence['n_plants'])}**.",
        f"- Per-plant wins vs small token forecast: **{int(best_vs_token['wins_lower_bray'])}/{int(best_vs_token['n_plants'])}**.",
        "",
        "## Method Description",
        "",
        "### Task and data organization",
        "",
        "- The study uses organized wastewater microbiome time series from **24 wastewater treatment plants**.",
        "- Each plant is converted into causal forecasting windows with **10 context samples** followed by **10 future target samples**.",
        "- The **horizon** is the future step being predicted: horizon 1 is the next sample after the context window, and horizon 10 is the tenth future sample.",
        "- Each plant uses a plant-local abundance vector with **top 200 ASVs plus one `other` bucket** (`201` abundance positions). Abundances are normalized to relative abundance before training and evaluation.",
        "- The chronological split is plant-wise: train windows are used for fitting, train-tail windows are used for early stopping or probe selection, and val/test windows are held out for reporting.",
        "",
        "### Baselines and model variants",
        "",
        "- **Persistence baseline**: copies the last observed context abundance vector to every future horizon. It is a strong baseline because adjacent wastewater samples often have similar community composition.",
        "- **Context mean baseline**: averages the 10 context abundance vectors and repeats that mean vector across all future horizons.",
        "- **Vector GRU / Vector TCN**: uses only abundance vectors, without MGM2 token embeddings. These models encode the 10-step context sequence and predict a 10-step future abundance trajectory.",
        "- **Hybrid vector + persistence**: uses persistence for short horizons and switches to the vector model after a validation-selected horizon cutoff.",
        "- **MGM2 token forecast head**: freezes a pretrained MGM2 backbone, caches ASV-level token states for each wastewater sample, then trains a token cross-attention forecaster from context token states to future abundance. This head is trained separately for each plant.",
        "- **MAE world model**: uses cached MGM2 token states and trains one shared all-plant future predictor. The model encodes context token states, predicts future latent token states, decodes them to future abundance, and is trained directly with Bray-Curtis plus MSE abundance reconstruction loss.",
        "- **JEPA world model**: predicts future latent token states only. The target latent states come from an EMA target encoder with stop-gradient. No abundance reconstruction loss is used during JEPA pretraining; after pretraining, the world model is frozen and a small abundance probe is trained for downstream forecasting evaluation.",
        "",
        "### Evaluation metrics",
        "",
        "- **Bray-Curtis distance** is the primary metric; lower is better and directly measures community composition dissimilarity.",
        "- **MAE** and **RMSE** measure element-wise abundance error.",
        "- **Spearman correlation** measures whether ASV abundance ranks are preserved.",
        "- Reported test metrics are weighted by the number of test windows per plant.",
        "",
        "Top weighted test rows:",
        "",
        top_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "World-model training summary:",
        "",
        train_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Main Figure Organization",
        "",
        "- **Main Figure 1: Performance overview** (`main_figure_1_performance`). Panels compare representative methods by Bray-Curtis, MAE, RMSE, and Spearman correlation.",
        "- **Main Figure 2: Horizon behavior** (`main_figure_2_horizon`). Panels show Bray-Curtis, MAE, and Spearman correlation across forecast horizons.",
        "- **Main Figure 3: Plant robustness** (`main_figure_3_plant_robustness`). Per-plant Bray-Curtis deltas show where MAE xlarge improves or underperforms against persistence and token forecast.",
        "",
        "## Supplementary Figure Organization",
        "",
        "- **Supplementary Figure 1: Scale ablation** (`supp_figure_1_scale_ablation`). Compares small/medium/large/xlarge for MAE, JEPA, and token forecast heads.",
        "- **Supplementary Figure 2: Metric trade-offs** (`supp_figure_2_metric_tradeoffs`). Shows relationships among Bray-Curtis, MAE, RMSE, and Spearman across representative methods.",
        "- **Supplementary Figure 3: Training diagnostics** (`supp_figure_3_training_diagnostics`). Summarizes MAE reconstruction loss, JEPA latent loss, and JEPA probe loss across scales.",
        "- **Supplementary Figure 4: Plant heatmap** (`supp_figure_4_plant_heatmap`). Displays plant-level Bray-Curtis values for baselines and MGM2 world models.",
        "",
        "## Generated Files",
        "",
    ]
    for stem, paths in generated.items():
        file_list = ", ".join(path.name for path in paths)
        lines.append(f"- `{stem}`: {file_list}")
    lines.extend(
        [
            "",
            "## Simple Flowchart Prompt",
            "",
            "Use this prompt with a diagram-generation tool to draw a compact workflow figure:",
            "",
            "```text",
            "Draw a clean publication-style workflow diagram for a wastewater microbiome forecasting study. Keep it simple, with four left-to-right stages and no detailed equations.",
            "",
            "Stage 1: Data windows",
            "- 24 wastewater treatment plants",
            "- Chronological time series",
            "- 10 context samples -> 10 future horizons",
            "- Plant-local top 200 ASVs + other bucket",
            "",
            "Stage 2: Frozen MGM2 representation",
            "- Load pretrained MGM2 at small/medium/large/xlarge scales",
            "- Convert each sample into ASV token embeddings",
            "- Cache token states for downstream forecasting",
            "",
            "Stage 3: Forecasting models",
            "- Persistence baseline: repeat the last observed abundance",
            "- Vector GRU/TCN: abundance-only temporal models",
            "- Token forecast head: cross-attention from context token states",
            "- MAE world model: reconstruct future abundance from predicted future token states",
            "- JEPA world model: latent-only future prediction with EMA target encoder, then frozen abundance probe",
            "",
            "Stage 4: Evaluation",
            "- Test on held-out future windows",
            "- Report Bray-Curtis, MAE/RMSE, and Spearman correlation",
            "- Highlight best result: MGM2 MAE xlarge improves Bray-Curtis over persistence and token forecast",
            "",
            "Visual style: minimal boxes, arrows, muted scientific colors, one highlighted path for MAE xlarge, no crowded text.",
            "```",
            "",
            "## Source Tables",
            "",
            "- `outputs_downstreams/wastewater/all_results/test_weighted_summary_unique.csv`",
            "- `outputs_downstreams/wastewater/all_results/test_horizon_weighted_summary_unique.csv`",
            "- `outputs_downstreams/wastewater/all_results/test_bray_curtis_by_plant_unique.csv`",
            "- `outputs_downstreams/wastewater/all_results/pairwise_bray_wins_unique.csv`",
            "- `outputs_downstreams/wastewater/all_results/world_model_train_summary.csv`",
            "",
            "Regenerate with:",
            "",
            "```bash",
            "python scripts/downstreams/wastewater/plot_wastewater_world_model_study.py",
            "```",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir
    weighted = pd.read_csv(require_file(results_dir / "test_weighted_summary_unique.csv"))
    horizon = pd.read_csv(require_file(results_dir / "test_horizon_weighted_summary_unique.csv"))
    plant_bc = pd.read_csv(require_file(results_dir / "test_bray_curtis_by_plant_unique.csv"))
    wins = pd.read_csv(require_file(results_dir / "pairwise_bray_wins_unique.csv"))
    train_summary = pd.read_csv(require_file(results_dir / "world_model_train_summary.csv"))

    generated = {
        "main_figure_1_performance": plot_main_performance(weighted, output_dir, args.formats),
        "main_figure_2_horizon": plot_main_horizon(horizon, output_dir, args.formats),
        "main_figure_3_plant_robustness": plot_main_plant_robustness(plant_bc, output_dir, args.formats),
        "supp_figure_1_scale_ablation": plot_supp_scale(weighted, output_dir, args.formats),
        "supp_figure_2_metric_tradeoffs": plot_supp_metric_tradeoff(weighted, output_dir, args.formats),
        "supp_figure_3_training_diagnostics": plot_supp_training(train_summary, output_dir, args.formats),
        "supp_figure_4_plant_heatmap": plot_supp_plant_heatmap(plant_bc, output_dir, args.formats),
    }
    write_readme(output_dir, weighted, wins, train_summary, generated)
    print(f"Saved figures and README to {output_dir}")
    for stem, paths in generated.items():
        print(f"{stem}: {', '.join(str(path) for path in paths)}")


if __name__ == "__main__":
    main()
