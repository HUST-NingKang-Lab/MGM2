#!/usr/bin/env python3
"""Plot final last-10-step statistics for the base V2 token model.

All model metrics in this script are recomputed from prediction tables on the
same evaluation unit: each plant's final test window, with 10 target steps.
This matches the MicroProphet last-subject setup and avoids mixing 495-window
test summaries with last-window forecasts.
"""

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

from src.metrics.wastewater import compute_forecast_metrics, normalize_abundance


DEFAULT_OUTPUT_DIR = Path("figures/wastewater_v2/base_token_statistics")
DEFAULT_TABLE_DIR = Path("outputs_downstreams/wastewater_v2/base_token_statistics")
DEFAULT_MICROPROPHET_ROOT = Path("data/downstreams/wastewater/microprophet_res")
DEFAULT_MAPPING = Path("data/downstreams/wastewater/microprophet_by_plant/subject_id_mapping.csv")
DEFAULT_TOKEN_RESULT_ROOT = Path("outputs_downstreams/wastewater_v2/token_asv_delta_transformer")

BASE_SOURCES = [
    {
        "run_label": "context_mean",
        "label": "Context mean",
        "path": Path("outputs_downstreams/wastewater/baseline/baseline_top200/predictions.csv.gz"),
        "model": "context_mean",
        "color": "#8A817C",
    },
    {
        "run_label": "microprophet",
        "label": "MicroProphet",
        "path": None,
        "model": "microprophet",
        "color": "#D8902F",
    },
    {
        "run_label": "gru",
        "label": "GRU",
        "path": Path("outputs_downstreams/wastewater/vector_temporal/gru_top200/predictions.csv.gz"),
        "model": "vector_gru",
        "color": "#74A57F",
    },
    {
        "run_label": "tcn",
        "label": "TCN",
        "path": Path("outputs_downstreams/wastewater/vector_temporal/tcn_top200/predictions.csv.gz"),
        "model": "vector_tcn",
        "color": "#246A73",
    },
]

TOKEN_SCALE_SPECS = [
    ("small", "#F2A65A"),
    ("medium", "#D96C45"),
    ("large", "#C4382B"),
    ("xlarge", "#7F1D1D"),
]

METRICS = [
    ("bray_curtis", "Bray-Curtis", "lower"),
    ("mae", "MAE", "lower"),
    ("rmse", "RMSE", "lower"),
    ("spearman", "Spearman", "higher"),
]
TREND_METRICS = [
    ("mean_asv_trend_pearson", "Mean ASV trend Pearson", "higher"),
    ("top_variable_asv_trend_pearson", "Variable-ASV trend Pearson", "higher"),
    ("centered_temporal_pearson", "Centered temporal Pearson", "higher"),
]
META_COLUMNS = {"subject_id", "time", "original_time"}

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["legend.fontsize"] = 8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microprophet_root", type=Path, default=DEFAULT_MICROPROPHET_ROOT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--token_result_root", type=Path, default=DEFAULT_TOKEN_RESULT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table_dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--formats", nargs="*", default=["png", "pdf"])
    return parser.parse_args()


def _style_axis(ax, grid_axis: str = "y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.24, linewidth=0.8)


def _save(fig, output_dir: Path, stem: str, formats: Sequence[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in META_COLUMNS]


def _read_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    mapping = pd.read_csv(path)
    return dict(zip(mapping["plant_slug"], mapping["plant"]))


def build_prediction_sources(token_result_root: Path) -> list[dict]:
    sources = list(BASE_SOURCES)
    missing_scales = []
    for scale_name, color in TOKEN_SCALE_SPECS:
        path = token_result_root / f"scale_{scale_name}" / "predictions.csv.gz"
        if not path.exists():
            missing_scales.append(scale_name)
            continue
        sources.append(
            {
                "run_label": f"token_{scale_name}",
                "label": f"MGM2 {scale_name}",
                "path": path,
                "model": "v2_token_asv_delta_transformer",
                "scale": f"scale_{scale_name}",
                "color": color,
            }
        )
    if missing_scales:
        print("[warn] missing last10 prediction tables for token scales: " + ", ".join(missing_scales))
        print("[warn] these scales are skipped instead of mixing incompatible summary metrics")
    return sources


def load_standard_prediction(spec: dict) -> pd.DataFrame:
    path = spec["path"]
    if path is None or not path.exists():
        raise FileNotFoundError(path)
    columns = ["model", "plant", "split", "window_idx", "horizon", "sample_id", "date", "asv_id", "y_true", "y_pred"]
    if "scale" in spec:
        columns.append("scale")
    frame = pd.read_csv(path, usecols=columns)
    frame = frame[frame["model"].eq(spec["model"]) & frame["split"].eq("test")].copy()
    if "scale" in spec:
        frame = frame[frame["scale"].eq(spec["scale"])].copy()
    if frame.empty:
        raise ValueError(f"no test predictions found for {spec['run_label']}: {path}")
    max_window = frame.groupby("plant")["window_idx"].transform("max")
    frame = frame[frame["window_idx"].eq(max_window)].copy()
    frame["run_label"] = spec["run_label"]
    frame["label"] = spec["label"]
    return frame[["run_label", "label", "plant", "window_idx", "horizon", "sample_id", "date", "asv_id", "y_true", "y_pred"]]


def load_microprophet_predictions(root: Path, mapping_path: Path) -> pd.DataFrame:
    plant_names = _read_mapping(mapping_path)
    rows = []
    for plant_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "models"):
        true_path = plant_dir / "microprophet_last_subject_true.csv"
        pred_path = plant_dir / "microprophet_last_subject_pred.csv"
        if not true_path.exists() or not pred_path.exists():
            continue
        true = pd.read_csv(true_path).sort_values(["subject_id", "time"]).reset_index(drop=True)
        pred = pd.read_csv(pred_path).sort_values(["subject_id", "time"]).reset_index(drop=True)
        features = [column for column in _feature_columns(true) if column in pred.columns]
        plant = plant_names.get(plant_dir.name, plant_dir.name)
        for step_idx, (_, true_row) in enumerate(true.iterrows(), start=1):
            pred_row = pred.iloc[step_idx - 1]
            for asv_id in features:
                rows.append(
                    {
                        "run_label": "microprophet",
                        "label": "MicroProphet",
                        "plant": plant,
                        "window_idx": -1,
                        "horizon": step_idx,
                        "sample_id": f"microprophet_subject{int(true_row['subject_id'])}_step{step_idx}",
                        "date": int(true_row["original_time"]) if "original_time" in true.columns else step_idx,
                        "asv_id": asv_id,
                        "y_true": float(true_row[asv_id]),
                        "y_pred": float(pred_row[asv_id]),
                    }
                )
    if not rows:
        raise FileNotFoundError(f"no MicroProphet prediction pairs found under {root}")
    return pd.DataFrame(rows)


def load_all_predictions(args, sources: Sequence[dict]) -> pd.DataFrame:
    frames = []
    for spec in sources:
        if spec["run_label"] == "microprophet":
            frames.append(load_microprophet_predictions(args.microprophet_root, args.mapping))
        else:
            frames.append(load_standard_prediction(spec))
    predictions = pd.concat(frames, ignore_index=True)
    plants_by_run = predictions.groupby("run_label")["plant"].nunique()
    if plants_by_run.min() != plants_by_run.max():
        print("[warn] model plant counts differ:")
        print(plants_by_run.to_string())
    return predictions


def _pivot_group(group: pd.DataFrame, value_col: str) -> pd.DataFrame:
    table = group.pivot_table(index="horizon", columns="asv_id", values=value_col, aggfunc="first")
    table = table.sort_index(axis=0).sort_index(axis=1)
    return table


def _as_arrays(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    true = _pivot_group(group, "y_true")
    pred = _pivot_group(group, "y_pred")
    shared_cols = true.columns.intersection(pred.columns)
    true = true.loc[:, shared_cols]
    pred = pred.loc[true.index, shared_cols]
    return true.to_numpy(dtype=np.float32), pred.to_numpy(dtype=np.float32)


def _safe_corr(x: np.ndarray, y: np.ndarray, min_std: float = 1e-8) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    x = x[valid]
    y = y[valid]
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std < min_std:
        return float("nan")
    if y_std < min_std:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return float(corr) if np.isfinite(corr) else float("nan")


def trend_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true = normalize_abundance(y_true)
    pred = normalize_abundance(y_pred)
    per_asv = []
    true_std = np.std(true, axis=0)
    for idx in range(true.shape[1]):
        corr = _safe_corr(true[:, idx], pred[:, idx])
        if np.isfinite(corr):
            per_asv.append((idx, true_std[idx], corr))
    if per_asv:
        top = sorted(per_asv, key=lambda item: item[1], reverse=True)[: min(50, len(per_asv))]
        mean_corr = float(np.mean([item[2] for item in per_asv]))
        top_corr = float(np.mean([item[2] for item in top]))
    else:
        mean_corr = float("nan")
        top_corr = float("nan")
    true_step = np.diff(true, axis=0).reshape(-1)
    pred_step = np.diff(pred, axis=0).reshape(-1)
    mask = np.abs(true_step) > 1e-4
    direction = float(np.mean((true_step[mask] > 0) == (pred_step[mask] > 0))) if np.any(mask) else float("nan")
    centered_true = true - np.mean(true, axis=0, keepdims=True)
    centered_pred = pred - np.mean(pred, axis=0, keepdims=True)
    return {
        "mean_asv_trend_pearson": mean_corr,
        "top_variable_asv_trend_pearson": top_corr,
        "centered_temporal_pearson": _safe_corr(centered_true.reshape(-1), centered_pred.reshape(-1)),
        "step_direction_accuracy": direction,
        "n_asv_trend_pearson": int(len(per_asv)),
        "step_direction_n": int(mask.sum()),
    }


def compute_plant_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run_label, plant), group in predictions.groupby(["run_label", "plant"], sort=False):
        y_true, y_pred = _as_arrays(group)
        metrics = compute_forecast_metrics(y_true[None, :, :], y_pred[None, :, :])
        metrics.update(trend_metrics(y_true, y_pred))
        rows.append(
            {
                "run_label": run_label,
                "label": group["label"].iloc[0],
                "plant": plant,
                "n_windows": 1,
                "horizon": int(y_true.shape[0]),
                "n_asvs": int(y_true.shape[1]),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def compute_weighted_summary(plant_summary: pd.DataFrame) -> pd.DataFrame:
    metric_names = [m[0] for m in [*METRICS, *TREND_METRICS]]
    rows = []
    for run_label, group in plant_summary.groupby("run_label", sort=False):
        weights = group["n_windows"].astype(float)
        row = {
            "run_label": run_label,
            "label": group["label"].iloc[0],
            "n_windows": int(weights.sum()),
            "n_plants": int(group["plant"].nunique()),
        }
        for metric in metric_names:
            values = pd.to_numeric(group[metric], errors="coerce")
            mask = values.notna() & (weights > 0)
            row[metric] = float((values[mask] * weights[mask]).sum() / weights[mask].sum()) if mask.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_horizon_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run_label, horizon_idx, plant), group in predictions.groupby(["run_label", "horizon", "plant"], sort=False):
        true = group.sort_values("asv_id")["y_true"].to_numpy(dtype=np.float32)
        pred = group.sort_values("asv_id")["y_pred"].to_numpy(dtype=np.float32)
        metrics = compute_forecast_metrics(true[None, None, :], pred[None, None, :])
        rows.append(
            {
                "run_label": run_label,
                "label": group["label"].iloc[0],
                "horizon": int(horizon_idx),
                "plant": plant,
                "n_windows": 1,
                **metrics,
            }
        )
    per_plant = pd.DataFrame(rows)
    weighted_rows = []
    for (run_label, horizon_idx), group in per_plant.groupby(["run_label", "horizon"], sort=False):
        weights = group["n_windows"].astype(float)
        row = {
            "run_label": run_label,
            "label": group["label"].iloc[0],
            "horizon": int(horizon_idx),
            "n_windows": int(weights.sum()),
        }
        for metric, _, _ in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            mask = values.notna() & (weights > 0)
            row[metric] = float((values[mask] * weights[mask]).sum() / weights[mask].sum()) if mask.any() else np.nan
        weighted_rows.append(row)
    return pd.DataFrame(weighted_rows)


def pairwise_wins(plant_summary: pd.DataFrame, candidate_label: str = "token_large") -> pd.DataFrame:
    candidate = plant_summary[plant_summary["run_label"].eq(candidate_label)].copy()
    rows = []
    for ref_label, ref in plant_summary[~plant_summary["run_label"].eq(candidate_label)].groupby("run_label"):
        joined = candidate.set_index("plant").join(ref.set_index("plant"), lsuffix="_candidate", rsuffix="_reference", how="inner")
        for metric, _, direction in [*METRICS, *TREND_METRICS]:
            cand = joined[f"{metric}_candidate"]
            refv = joined[f"{metric}_reference"]
            if direction == "lower":
                delta = refv - cand
                wins = cand < refv
            else:
                delta = cand - refv
                wins = cand > refv
            weights = joined["n_windows_candidate"].astype(float)
            rows.append(
                {
                    "reference": ref_label,
                    "reference_label": ref["label"].iloc[0],
                    "metric": metric,
                    "wins": int(wins.sum()),
                    "n": int(len(joined)),
                    "weighted_delta": float((delta * weights).sum() / weights.sum()),
                    "median_delta": float(delta.median()),
                }
            )
    return pd.DataFrame(rows)


def _source_order(sources: Sequence[dict]) -> list[str]:
    return [spec["run_label"] for spec in sources]


def _source_color(run_label: str, sources: Sequence[dict]) -> str:
    return next(spec["color"] for spec in sources if spec["run_label"] == run_label)


def plot_main_metrics(weighted: pd.DataFrame, sources: Sequence[dict], output_dir: Path, formats: Sequence[str]) -> list[Path]:
    rows = weighted.set_index("run_label").loc[_source_order(sources)].reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.0))
    axes = axes.flatten()
    x = np.arange(len(rows))
    colors = [_source_color(run, sources) for run in rows["run_label"]]
    for ax, (metric, label, direction) in zip(axes, METRICS):
        ax.bar(x, rows[metric], color=colors, edgecolor="#333333", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(rows["label"], rotation=35, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"{label} ({direction} is better)")
        for idx, value in enumerate(rows[metric]):
            text = f"{value:.3f}" if metric != "mae" else f"{value:.4f}"
            ax.text(idx, value + max(rows[metric]) * 0.012, text, ha="center", va="bottom", fontsize=7.5)
        _style_axis(ax)
    fig.suptitle("Last-10-step wastewater forecasting on the same final test window", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, output_dir, "base_token_last10_main_metrics", formats)


def plot_trend_metrics(weighted: pd.DataFrame, sources: Sequence[dict], output_dir: Path, formats: Sequence[str]) -> list[Path]:
    rows = weighted.set_index("run_label").loc[_source_order(sources)].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))
    for ax, (metric, label, direction) in zip(axes, TREND_METRICS):
        plot_rows = rows.copy()
        if metric in {"mean_asv_trend_pearson", "top_variable_asv_trend_pearson", "centered_temporal_pearson"}:
            plot_rows = plot_rows[~plot_rows["run_label"].eq("context_mean")].copy()
        x = np.arange(len(plot_rows))
        colors = [_source_color(run, sources) for run in plot_rows["run_label"]]
        ax.bar(x, plot_rows[metric], color=colors, edgecolor="#333333", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_rows["label"], rotation=35, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"{label} ({direction})")
        _style_axis(ax)
    fig.tight_layout()
    return _save(fig, output_dir, "base_token_last10_trend_metrics", formats)


def plot_horizon(weighted_h: pd.DataFrame, sources: Sequence[dict], output_dir: Path, formats: Sequence[str]) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    plot_metrics = [("bray_curtis", "Bray-Curtis"), ("mae", "MAE"), ("spearman", "Spearman")]
    for ax, (metric, label) in zip(axes, plot_metrics):
        for spec in sources:
            sub = weighted_h[weighted_h["run_label"].eq(spec["run_label"])].sort_values("horizon")
            ax.plot(sub["horizon"], sub[metric], marker="o", markersize=3.5, linewidth=1.8, color=spec["color"], label=spec["label"])
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(label)
        ax.set_xticks(range(1, 11))
        ax.set_title(f"{label} by horizon")
        _style_axis(ax)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return _save(fig, output_dir, "base_token_last10_horizon_metrics", formats)


def plot_plant_improvements(plant_summary: pd.DataFrame, output_dir: Path, formats: Sequence[str]) -> list[Path]:
    token = plant_summary[plant_summary["run_label"].eq("token_large")].set_index("plant")
    refs = ["context_mean", "microprophet", "gru", "tcn"]
    fig, axes = plt.subplots(1, 4, figsize=(20.5, 6.4), sharey=True)
    for ax, ref_label in zip(axes, refs):
        ref = plant_summary[plant_summary["run_label"].eq(ref_label)].set_index("plant")
        joined = token.join(ref, lsuffix="_token", rsuffix="_ref", how="inner")
        delta = (joined["bray_curtis_ref"] - joined["bray_curtis_token"]).sort_values()
        y = np.arange(len(delta))
        ax.barh(y, delta.values, color=np.where(delta.values > 0, "#2E7D59", "#B23A48"), edgecolor="#333333", linewidth=0.35)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(delta.index)
        ax.set_xlabel("Reference - base token Bray-Curtis")
        ref_name = plant_summary[plant_summary["run_label"].eq(ref_label)]["label"].iloc[0]
        ax.set_title(f"Per-plant improvement vs {ref_name}")
        _style_axis(ax, "x")
    fig.tight_layout()
    return _save(fig, output_dir, "base_token_last10_plant_improvements", formats)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    sources = build_prediction_sources(args.token_result_root)
    predictions = load_all_predictions(args, sources)
    plant_summary = compute_plant_summary(predictions)
    weighted = compute_weighted_summary(plant_summary)
    weighted_h = compute_horizon_summary(predictions)
    wins = pairwise_wins(plant_summary)

    predictions.to_csv(args.table_dir / "base_token_last10_predictions_long.csv.gz", index=False, compression="gzip")
    plant_summary.to_csv(args.table_dir / "base_token_last10_plant_summary.csv", index=False)
    weighted.to_csv(args.table_dir / "base_token_last10_weighted_summary.csv", index=False)
    weighted_h.to_csv(args.table_dir / "base_token_last10_horizon_summary.csv", index=False)
    wins.to_csv(args.table_dir / "base_token_last10_pairwise_wins.csv", index=False)

    figure_paths = []
    figure_paths.extend(plot_main_metrics(weighted, sources, args.output_dir, args.formats))
    figure_paths.extend(plot_trend_metrics(weighted, sources, args.output_dir, args.formats))
    figure_paths.extend(plot_horizon(weighted_h, sources, args.output_dir, args.formats))
    if "token_large" in set(plant_summary["run_label"]):
        figure_paths.extend(plot_plant_improvements(plant_summary, args.output_dir, args.formats))

    print("Last10 weighted summary:")
    print(weighted[["label", "n_windows", "n_plants", "bray_curtis", "mae", "rmse", "spearman", "mean_asv_trend_pearson", "top_variable_asv_trend_pearson", "centered_temporal_pearson"]].sort_values("bray_curtis").to_string(index=False))
    print("\nPairwise wins for base token large:")
    print(wins[wins["metric"].isin(["bray_curtis", "mae", "spearman", "top_variable_asv_trend_pearson"])].to_string(index=False))
    for path in figure_paths:
        print(f"Saved figure: {path}")
    print(f"Saved tables under: {args.table_dir}")


if __name__ == "__main__":
    main()
