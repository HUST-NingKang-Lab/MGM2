#!/usr/bin/env python3
"""Summarize MicroProphet wastewater forecasts and make final comparison plots."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics.wastewater import aggregate_forecast_metrics, compute_forecast_metrics, normalize_abundance


METRICS = ["bray_curtis", "mae", "rmse", "spearman", "top10_overlap", "top20_overlap", "top50_overlap"]
PLOT_METRICS = [
    ("bray_curtis", "Bray-Curtis", "lower"),
    ("mae", "MAE", "lower"),
    ("rmse", "RMSE", "lower"),
    ("spearman", "Spearman", "higher"),
]
META_COLUMNS = {"subject_id", "time", "original_time"}
DEFAULT_MICROPROPHET_ROOT = Path("data/downstreams/wastewater/microprophet_res")
DEFAULT_MAPPING = Path("data/downstreams/wastewater/microprophet_by_plant/subject_id_mapping.csv")
DEFAULT_REFERENCE = Path("outputs_downstreams/wastewater/all_results/wastewater_test_only_reassessment_summary.csv")
DEFAULT_TEMPORAL_REFERENCE = Path("outputs_downstreams/wastewater/all_results/test_weighted_summary_unique.csv")
DEFAULT_OUTPUT_DIR = Path("outputs_downstreams/wastewater_global_forecast/microprophet_comparison")
DEFAULT_FIGURE_DIR = Path("figures/wastewater_microprophet_comparison")


mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = 11
mpl.rcParams["axes.labelsize"] = 10
mpl.rcParams["xtick.labelsize"] = 9
mpl.rcParams["ytick.labelsize"] = 9
mpl.rcParams["legend.fontsize"] = 9


LABELS = {
    "baseline_context_mean": "Context mean",
    "baseline_persistence": "Persistence",
    "context_mean": "Context mean",
    "vector_gru": "GRU",
    "vector_tcn": "TCN",
    "mixed_ntv3_mae_small": "MGM2 small MAE",
    "mixed_ntv3_mae_medium": "MGM2 medium MAE",
    "mixed_ntv3_mae_large": "MGM2 large MAE",
    "mixed_ntv3_mae_xlarge": "MGM2 xlarge MAE",
    "mixed_ntv3_jepa_small": "MGM2 small JEPA",
    "mixed_ntv3_jepa_medium": "MGM2 medium JEPA",
    "mixed_ntv3_jepa_large": "MGM2 large JEPA",
    "mixed_ntv3_jepa_xlarge": "MGM2 xlarge JEPA",
    "mixed_fallback_mae_small": "MGM2 fallback MAE small",
    "mixed_fallback_mae_medium": "MGM2 fallback MAE medium",
    "mixed_fallback_mae_large": "MGM2 fallback MAE large",
    "mixed_fallback_mae_xlarge": "MGM2 fallback MAE xlarge",
    "byplant_ntv3_mae_medium": "MGM2 by-plant NTv3 MAE medium",
    "microprophet_context_mean": "Context mean",
    "microprophet_persistence": "Persistence",
    "microprophet": "MicroProphet",
}
COLORS = {
    "baseline_context_mean": "#8A817C",
    "baseline_persistence": "#4D4D4D",
    "context_mean": "#8A817C",
    "vector_gru": "#63A375",
    "vector_tcn": "#2E7D59",
    "mixed_ntv3_mae_small": "#7CB5EC",
    "mixed_ntv3_mae_medium": "#C84C31",
    "mixed_ntv3_mae_large": "#1A237E",
    "mixed_ntv3_mae_xlarge": "#4A148C",
    "mixed_ntv3_jepa_small": "#E78AC3",
    "mixed_ntv3_jepa_medium": "#C94D9B",
    "mixed_ntv3_jepa_large": "#9C2D7A",
    "mixed_ntv3_jepa_xlarge": "#6A1B9A",
    "mixed_fallback_mae_small": "#2B8CBE",
    "mixed_fallback_mae_medium": "#2B8CBE",
    "mixed_fallback_mae_large": "#2B8CBE",
    "mixed_fallback_mae_xlarge": "#2B8CBE",
    "byplant_ntv3_mae_medium": "#2E7D59",
    "microprophet_context_mean": "#8A817C",
    "microprophet_persistence": "#4D4D4D",
    "microprophet": "#D8902F",
}

MAIN_MODEL_ORDER = [
    "context_mean",
    "microprophet",
    "vector_gru",
    "vector_tcn",
    "mixed_ntv3_jepa_small",
    "mixed_ntv3_mae_small",
    "mixed_ntv3_jepa_medium",
    "mixed_ntv3_mae_medium",
    "mixed_ntv3_jepa_large",
    "mixed_ntv3_mae_large",
    "mixed_ntv3_jepa_xlarge",
    "mixed_ntv3_mae_xlarge",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microprophet_root", type=Path, default=DEFAULT_MICROPROPHET_ROOT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--reference_summary", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--temporal_reference_summary", type=Path, default=DEFAULT_TEMPORAL_REFERENCE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure_dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--formats", nargs="*", default=["png", "pdf"])
    return parser.parse_args()


def _label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " "))


def _color(name: str) -> str:
    return COLORS.get(name, "#666666")


def _style_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.25, linewidth=0.8)


def _save_figure(fig, figure_dir: Path, stem: str, formats: Sequence[str]) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = figure_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in META_COLUMNS]


def _read_prediction_pair(plant_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    true_path = plant_dir / "microprophet_last_subject_true.csv"
    pred_path = plant_dir / "microprophet_last_subject_pred.csv"
    if not true_path.exists() or not pred_path.exists():
        raise FileNotFoundError(f"missing MicroProphet pred/true pair in {plant_dir}")
    true = pd.read_csv(true_path)
    pred = pd.read_csv(pred_path)
    features = [column for column in _feature_columns(true) if column in pred.columns]
    if not features:
        raise ValueError(f"no shared abundance columns in {plant_dir}")
    true = true.sort_values(["subject_id", "time"]).reset_index(drop=True)
    pred = pred.sort_values(["subject_id", "time"]).reset_index(drop=True)
    return true[[*sorted(set(true.columns) & META_COLUMNS), *features]], pred[[*sorted(set(pred.columns) & META_COLUMNS), *features]]


def _baseline_arrays_for_last_subject(plant_dir: Path, feature_columns: list[str], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    test_path = plant_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"missing MicroProphet test.csv in {plant_dir}")
    test = pd.read_csv(test_path)
    missing = [column for column in feature_columns if column not in test.columns]
    if missing:
        raise ValueError(f"{plant_dir} test.csv missing {len(missing)} abundance columns")
    last_subject = test["subject_id"].max()
    subject = test[test["subject_id"].eq(last_subject)].sort_values("time")
    if len(subject) < horizon * 2:
        raise ValueError(f"{plant_dir} last subject has {len(subject)} rows, expected at least {horizon * 2}")
    context = normalize_abundance(subject.iloc[:horizon][feature_columns].to_numpy(dtype=np.float32))
    persistence = np.repeat(context[-1:, :], horizon, axis=0)
    context_mean = np.repeat(context.mean(axis=0, keepdims=True), horizon, axis=0)
    return context_mean, persistence


def _weighted(frame: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    keys = keys or []
    rows = []
    groups = [((), frame)] if not keys else frame.groupby(keys, dropna=False)
    for key, group in groups:
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(keys, key)}
        weights = pd.to_numeric(group["n_windows"], errors="coerce").fillna(0.0)
        row["n_windows"] = int(weights.sum())
        if "plant" in group.columns and "plant" not in keys:
            row["n_plants"] = int(group["plant"].nunique())
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            mask = values.notna() & (weights > 0)
            row[metric] = float((values[mask] * weights[mask]).sum() / weights[mask].sum()) if mask.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _load_mapping(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["plant_slug", "plant"])
    mapping = pd.read_csv(path)
    return mapping[["plant_slug", "plant"]].drop_duplicates()


def collect_microprophet(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping = _load_mapping(args.mapping)
    plant_names = dict(zip(mapping["plant_slug"], mapping["plant"]))
    plant_rows = []
    horizon_records = []

    for plant_dir in sorted(path for path in args.microprophet_root.iterdir() if path.is_dir() and path.name != "models"):
        slug = plant_dir.name
        true_frame, pred_frame = _read_prediction_pair(plant_dir)
        features = [column for column in _feature_columns(true_frame) if column in pred_frame.columns]
        true = true_frame[features].to_numpy(dtype=np.float32)
        pred = pred_frame[features].to_numpy(dtype=np.float32)
        if true.shape != pred.shape:
            raise ValueError(f"shape mismatch in {plant_dir}: true={true.shape}, pred={pred.shape}")
        horizon = true.shape[0]
        if horizon <= 0:
            raise ValueError(f"empty forecast horizon in {plant_dir}")

        context_mean, persistence = _baseline_arrays_for_last_subject(plant_dir, features, horizon)
        model_arrays = {
            "microprophet_context_mean": context_mean,
            "microprophet_persistence": persistence,
            "microprophet": pred,
        }
        plant_name = plant_names.get(slug, slug)
        true_3d = true[None, :, :]
        for model, pred_array in model_arrays.items():
            pred_3d = pred_array[None, :, :]
            metrics = compute_forecast_metrics(true_3d, pred_3d)
            plant_rows.append(
                {
                    "plant": plant_name,
                    "plant_slug": slug,
                    "model_group": model,
                    "source": "microprophet_last_subject",
                    "mode": model.replace("microprophet_", "") if model != "microprophet" else "microprophet",
                    "scale": "none",
                    "n_windows": 1,
                    "horizon": horizon,
                    "n_taxa": len(features),
                    **metrics,
                }
            )
            horizon_frame = aggregate_forecast_metrics(
                [{"plant": plant_name, "split": "microprophet_last_subject", "y_true": true_3d, "y_pred": pred_3d}]
            )
            horizon_frame["plant_slug"] = slug
            horizon_frame["model_group"] = model
            horizon_frame["source"] = "microprophet_last_subject"
            horizon_records.append(horizon_frame)

    if not plant_rows:
        raise FileNotFoundError(f"no MicroProphet plant prediction directories found in {args.microprophet_root}")

    plant_summary = pd.DataFrame(plant_rows)
    weighted = _weighted(plant_summary, ["model_group", "source", "mode", "scale"])
    horizon = pd.concat(horizon_records, ignore_index=True)
    horizon_weighted = _weighted(horizon, ["model_group", "source", "horizon"])
    return plant_summary, weighted, horizon_weighted


def _load_temporal_reference(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    temporal = pd.read_csv(path)
    if "run_label" not in temporal.columns:
        return pd.DataFrame()
    temporal = temporal[temporal["run_label"].isin(["context_mean", "vector_gru", "vector_tcn"])].copy()
    if temporal.empty:
        return temporal
    temporal["model_group"] = temporal["run_label"]
    temporal["source"] = temporal["mode"].map({"gru": "vector_temporal", "tcn": "vector_temporal", "baseline": "baseline"}).fillna("temporal_reference")
    temporal["mode"] = temporal["mode"].replace({"baseline": "context_mean"})
    temporal["scale"] = "none"
    temporal["evaluation_set"] = "full_test_windows"
    temporal["evaluation_note"] = "495 sliding windows across 24 plants"
    if "n_plants" not in temporal.columns:
        temporal["n_plants"] = 24
    return temporal


def build_model_comparison(micro_weighted: pd.DataFrame, reference_path: Path, temporal_reference_path: Path) -> pd.DataFrame:
    frames = []
    if reference_path.exists():
        reference = pd.read_csv(reference_path)
        reference = reference.rename(columns={"group": "model_group"})
        reference["evaluation_set"] = "full_test_windows"
        reference["evaluation_note"] = "495 sliding windows across 24 plants"
        reference = reference[reference["source"].isin(["mixed_ntv3"])].copy()
        frames.append(reference)
    temporal = _load_temporal_reference(temporal_reference_path)
    if not temporal.empty:
        frames.append(temporal)
    micro = micro_weighted.copy()
    micro = micro[micro["model_group"].eq("microprophet") | micro["model_group"].eq("microprophet_context_mean")].copy()
    micro["evaluation_set"] = "microprophet_last_subject"
    micro["evaluation_note"] = "one 10-step rollout per plant"
    frames.append(micro)
    comparison = pd.concat(frames, ignore_index=True, sort=False)
    comparison = comparison.drop_duplicates(subset=["model_group", "evaluation_set"], keep="first")
    return comparison


def _plot_metric_bars(
    rows: pd.DataFrame,
    model_order: list[str],
    title: str,
    stem: str,
    figure_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    rows = rows[rows["model_group"].isin(model_order)].copy()
    order = {model: idx for idx, model in enumerate(model_order)}
    rows["plot_order"] = rows["model_group"].map(order)
    rows = rows.sort_values("plot_order")
    fig_width = max(12.8, 0.88 * len(rows) + 3.2)
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 8.4))
    axes = axes.flatten()
    x = np.arange(len(rows))
    labels = [_label(model) for model in rows["model_group"]]
    for idx, (ax, (metric, metric_label, direction)) in enumerate(zip(axes, PLOT_METRICS)):
        values = pd.to_numeric(rows[metric], errors="coerce").to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            color=[_color(model) for model in rows["model_group"]],
            edgecolor="#333333",
            linewidth=0.55,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel(f"{metric_label} ({direction} is better)")
        max_value = np.nanmax(values)
        ax.set_ylim(0.0, max_value * 1.16 if max_value > 0 else 1.0)
        for row_idx, value in enumerate(values):
            text = f"{value:.4f}" if metric in {"mae", "rmse"} else f"{value:.3f}"
            ax.text(row_idx, value + max_value * 0.018, text, ha="center", va="bottom", fontsize=7.2, rotation=90)
        _style_axis(ax, "y")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, figure_dir, stem, formats)


def plot_microprophet_same_split(weighted: pd.DataFrame, figure_dir: Path, formats: Sequence[str]) -> list[Path]:
    return _plot_metric_bars(
        weighted,
        ["microprophet_context_mean", "microprophet"],
        "MicroProphet last-subject rollout: no-persistence reference",
        "final_microprophet_same_split_metrics",
        figure_dir,
        formats,
    )


def plot_final_external_comparison(comparison: pd.DataFrame, figure_dir: Path, formats: Sequence[str]) -> list[Path]:
    return _plot_metric_bars(
        comparison,
        MAIN_MODEL_ORDER,
        "Wastewater forecasting comparison with MicroProphet external rollout",
        "final_model_comparison_with_microprophet",
        figure_dir,
        formats,
    )


def plot_main_mixed_ntv3_same_test(comparison: pd.DataFrame, figure_dir: Path, formats: Sequence[str]) -> list[Path]:
    selected = [model for model in MAIN_MODEL_ORDER if model != "microprophet"]
    rows = comparison[comparison["evaluation_set"].eq("full_test_windows")].copy()
    return _plot_metric_bars(
        rows,
        selected,
        "Main same-test comparison: MGM2 mixed-NTv3 vs abundance temporal models",
        "main_mixed_ntv3_vs_temporal_same_test",
        figure_dir,
        formats,
    )


def plot_main_mixed_ntv3_external(comparison: pd.DataFrame, figure_dir: Path, formats: Sequence[str]) -> list[Path]:
    return _plot_metric_bars(
        comparison,
        MAIN_MODEL_ORDER,
        "Main comparison including MicroProphet external rollout",
        "main_mixed_ntv3_tcn_gru_microprophet",
        figure_dir,
        formats,
    )


def plot_main_mixed_ntv3_scale_variants(comparison: pd.DataFrame, figure_dir: Path, formats: Sequence[str]) -> list[Path]:
    rows = comparison[
        comparison["evaluation_set"].eq("full_test_windows")
        & comparison["source"].eq("mixed_ntv3")
        & comparison["mode"].isin(["mae", "jepa"])
    ].copy()
    scale_order = ["small", "medium", "large", "xlarge"]
    rows["scale_order"] = rows["scale"].map({scale: idx for idx, scale in enumerate(scale_order)})
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))
    for ax, (metric, metric_label, direction) in zip(axes, PLOT_METRICS[:3]):
        for mode, color, label in [("mae", "#C84C31", "MAE decoder"), ("jepa", "#9C2D7A", "JEPA latent")]:
            sub = rows[rows["mode"].eq(mode)].sort_values("scale_order")
            ax.plot(sub["scale"], sub[metric], marker="o", linewidth=2.2, markersize=5, color=color, label=label)
        best = rows[rows["model_group"].eq("mixed_ntv3_mae_medium")]
        if len(best):
            ax.scatter(best["scale"], best[metric], s=85, color="#C84C31", edgecolor="#111111", linewidth=1.0, zorder=3)
        ax.set_xlabel("MGM2 scale")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} by scale ({direction} is better)")
        _style_axis(ax)
    axes[0].legend(frameon=False)
    fig.suptitle("MGM2 mixed-NTv3 scale and objective comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, figure_dir, "main_mixed_ntv3_scale_variants", formats)


def plot_all_external_models(comparison: pd.DataFrame, figure_dir: Path, formats: Sequence[str]) -> list[Path]:
    allowed = {"context_mean", "vector_gru", "vector_tcn", "microprophet"}
    rows = comparison[
        comparison["source"].eq("mixed_ntv3") | comparison["model_group"].isin(allowed)
    ].copy()
    model_order = [model for model in MAIN_MODEL_ORDER if model in set(rows["model_group"])]
    order = {model: idx for idx, model in enumerate(model_order)}
    rows["plot_order"] = rows["model_group"].map(order)
    rows = rows.sort_values("plot_order")
    fig_width = max(13.8, 0.88 * len(rows) + 3.2)
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 8.4))
    axes = axes.flatten()
    x = np.arange(len(rows))
    labels = [_label(model) for model in rows["model_group"]]
    for idx, (ax, (metric, metric_label, direction)) in enumerate(zip(axes, PLOT_METRICS)):
        values = pd.to_numeric(rows[metric], errors="coerce").to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            color=[_color(model) for model in model_order],
            edgecolor="#333333",
            linewidth=0.35,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel(f"{metric_label} ({direction} is better)")
        max_value = np.nanmax(values)
        ax.set_ylim(0.0, max_value * 1.16 if max_value > 0 else 1.0)
        for row_idx, value in enumerate(values):
            text = f"{value:.4f}" if metric in {"mae", "rmse"} else f"{value:.3f}"
            ax.text(row_idx, value + max_value * 0.018, text, ha="center", va="bottom", fontsize=6.8, rotation=90)
        _style_axis(ax, "y")
    fig.suptitle("All available wastewater model summaries plus MicroProphet, excluding persistence", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, figure_dir, "main_all_models_no_persistence", formats)


def plot_horizon(horizon: pd.DataFrame, figure_dir: Path, formats: Sequence[str]) -> list[Path]:
    selected = ["microprophet_context_mean", "microprophet"]
    rows = horizon[horizon["model_group"].isin(selected)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))
    for ax, (metric, metric_label, direction) in zip(axes, PLOT_METRICS[:3]):
        for model in selected:
            sub = rows[rows["model_group"].eq(model)].sort_values("horizon")
            ax.plot(sub["horizon"], sub[metric], marker="o", linewidth=2.0, markersize=4, color=_color(model), label=_label(model))
        ax.set_xlabel("Horizon step")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} by horizon ({direction} is better)")
        ax.set_xticks(range(1, int(rows["horizon"].max()) + 1))
        _style_axis(ax)
    axes[0].legend(frameon=False)
    fig.suptitle("MicroProphet last-subject rollout by forecast horizon, excluding persistence", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, figure_dir, "final_microprophet_horizon_metrics", formats)


def plot_plant_bray(plant_summary: pd.DataFrame, figure_dir: Path, formats: Sequence[str]) -> list[Path]:
    pivot = plant_summary.pivot(index="plant", columns="model_group", values="bray_curtis")
    pivot = pivot.sort_values("microprophet")
    y = np.arange(len(pivot))
    height = 0.25
    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    for offset, model in zip([-height / 2, height / 2], ["microprophet_context_mean", "microprophet"]):
        ax.barh(y + offset, pivot[model], height=height, color=_color(model), edgecolor="#333333", linewidth=0.35, label=_label(model))
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.invert_yaxis()
    ax.set_xlabel("Bray-Curtis distance")
    ax.set_title("Per-plant MicroProphet rollout performance")
    ax.legend(frameon=False, loc="lower right")
    _style_axis(ax, "x")
    fig.tight_layout()
    return _save_figure(fig, figure_dir, "final_microprophet_plant_bray_curtis", formats)


def write_readme(
    output_dir: Path,
    figure_dir: Path,
    micro_weighted: pd.DataFrame,
    comparison: pd.DataFrame,
    figure_paths: list[Path],
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        table = frame.copy()
        for column in table.columns:
            if pd.api.types.is_float_dtype(table[column]):
                table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
        headers = list(table.columns)
        rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in table.iterrows():
            rows.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
        return "\n".join(rows)

    micro_display = micro_weighted[
        ~micro_weighted["model_group"].astype(str).str.contains("persistence", case=False, na=False)
    ].sort_values("bray_curtis")[
        ["model_group", "n_windows", "n_plants", "bray_curtis", "mae", "rmse", "spearman"]
    ].copy()
    def ordered_table(models: list[str], columns: list[str]) -> pd.DataFrame:
        table = comparison[comparison["model_group"].isin(models)][columns].copy()
        order = {model: idx for idx, model in enumerate(models)}
        table["plot_order"] = table["model_group"].map(order)
        return table.sort_values("plot_order").drop(columns=["plot_order"])

    comparison_selected = ordered_table(
        MAIN_MODEL_ORDER,
        ["model_group", "evaluation_set", "n_windows", "n_plants", "bray_curtis", "mae", "rmse", "spearman"],
    )
    main_same_test = ordered_table(
        [model for model in MAIN_MODEL_ORDER if model != "microprophet"],
        ["model_group", "evaluation_set", "n_windows", "n_plants", "bray_curtis", "mae", "rmse", "spearman", "top50_overlap"],
    )
    lines = [
        "# Wastewater Main Figure Plan: MGM2 Mixed-NTv3",
        "",
        "## Evaluation scope",
        "",
        "- MicroProphet outputs in `data/downstreams/wastewater/microprophet_res` contain one `microprophet_last_subject_pred.csv` and one matching truth table per plant.",
        "- The recomputed MicroProphet metrics therefore use 24 independent 10-step rollouts, one per plant.",
        "- Existing MGM2/reference rows from `outputs_downstreams/wastewater/all_results/wastewater_test_only_reassessment_summary.csv` use 495 sliding-window test examples across the same 24 plants.",
        "- The main model family is mixed-NTv3 MGM2. Fallback, by-plant NTv3, token-only, and persistence variants are excluded from the main comparison.",
        "- Bar charts use vertical bars. The x-axis order is context mean, MicroProphet, GRU, TCN, then MGM2 mixed-NTv3 by scale with JEPA and MAE alternating.",
        "- MicroProphet is shown as an external-reference model because its evaluation unit differs from the full-test MGM2/GRU/TCN rows.",
        "",
        "## Main Figure Organization",
        "",
        "- `main_mixed_ntv3_vs_temporal_same_test`: same-test comparison on 495 windows, using context mean, GRU, TCN, and mixed-NTv3 MGM2 variants.",
        "- `main_mixed_ntv3_scale_variants`: mixed-NTv3 scale/objective comparison, showing MAE decoder and JEPA latent variants.",
        "- `main_mixed_ntv3_tcn_gru_microprophet`: external-reference figure adding MicroProphet to context mean, GRU, TCN, and MGM2 mixed-NTv3.",
        "- `main_all_models_no_persistence`: compact summary containing only mixed-NTv3 MGM2 rows plus context mean, GRU, TCN, and MicroProphet.",
        "",
        "## Same-test Main Comparison",
        "",
        markdown_table(main_same_test),
        "",
        "## MicroProphet External Metrics",
        "",
        markdown_table(micro_display),
        "",
        "## External Model Comparison",
        "",
        markdown_table(comparison_selected),
        "",
        "## Figures",
        "",
        *[f"- `{path}`" for path in figure_paths],
        "",
        "## Output tables",
        "",
        f"- `{output_dir / 'microprophet_plant_summary.csv'}`",
        f"- `{output_dir / 'microprophet_weighted_summary.csv'}`",
        f"- `{output_dir / 'microprophet_horizon_summary.csv'}`",
        f"- `{output_dir / 'model_comparison_summary.csv'}`",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    plant_summary, micro_weighted, horizon = collect_microprophet(args)
    comparison = build_model_comparison(micro_weighted, args.reference_summary, args.temporal_reference_summary)

    plant_summary.sort_values(["model_group", "plant"]).to_csv(args.output_dir / "microprophet_plant_summary.csv", index=False)
    micro_weighted.sort_values("bray_curtis").to_csv(args.output_dir / "microprophet_weighted_summary.csv", index=False)
    horizon.sort_values(["model_group", "horizon"]).to_csv(args.output_dir / "microprophet_horizon_summary.csv", index=False)
    comparison.to_csv(args.output_dir / "model_comparison_summary.csv", index=False)

    figure_paths: list[Path] = []
    figure_paths.extend(plot_main_mixed_ntv3_same_test(comparison, args.figure_dir, args.formats))
    figure_paths.extend(plot_main_mixed_ntv3_scale_variants(comparison, args.figure_dir, args.formats))
    figure_paths.extend(plot_main_mixed_ntv3_external(comparison, args.figure_dir, args.formats))
    figure_paths.extend(plot_microprophet_same_split(micro_weighted, args.figure_dir, args.formats))
    figure_paths.extend(plot_final_external_comparison(comparison, args.figure_dir, args.formats))
    figure_paths.extend(plot_all_external_models(comparison, args.figure_dir, args.formats))
    figure_paths.extend(plot_horizon(horizon, args.figure_dir, args.formats))
    figure_paths.extend(plot_plant_bray(plant_summary, args.figure_dir, args.formats))
    write_readme(args.output_dir, args.figure_dir, micro_weighted, comparison, figure_paths)

    print(f"Saved MicroProphet plant summary: {args.output_dir / 'microprophet_plant_summary.csv'}")
    print(f"Saved MicroProphet weighted summary: {args.output_dir / 'microprophet_weighted_summary.csv'}")
    print(f"Saved MicroProphet horizon summary: {args.output_dir / 'microprophet_horizon_summary.csv'}")
    print(f"Saved model comparison summary: {args.output_dir / 'model_comparison_summary.csv'}")
    print(f"Saved README: {args.output_dir / 'README.md'}")
    micro_print = micro_weighted[~micro_weighted["model_group"].astype(str).str.contains("persistence", case=False, na=False)]
    print("MicroProphet external summary, excluding persistence:")
    print(micro_print.sort_values("bray_curtis")[["model_group", "n_windows", "n_plants", "bray_curtis", "mae", "rmse", "spearman"]].to_string(index=False))


if __name__ == "__main__":
    main()
