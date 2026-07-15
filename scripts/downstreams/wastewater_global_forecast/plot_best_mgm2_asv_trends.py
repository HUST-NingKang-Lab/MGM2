#!/usr/bin/env python3
"""Plot final-10-step ASV trends selected only by MGM2 trend correlation."""

from __future__ import annotations

import argparse
import os
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


DEFAULT_OUTPUT_DIR = Path("figures/wastewater_microprophet_comparison/asv_trends")
DEFAULT_TABLE_DIR = Path("outputs_downstreams/wastewater_global_forecast/microprophet_comparison/asv_trends")
DEFAULT_MICROPROPHET_ROOT = Path("data/downstreams/wastewater/microprophet_res")
DEFAULT_MAPPING = Path("data/downstreams/wastewater/microprophet_by_plant/subject_id_mapping.csv")

STANDARD_MODEL_SPECS = {
    "context_mean": {
        "path": Path("outputs_downstreams/wastewater/baseline/baseline_top200/predictions.csv.gz"),
        "model": "context_mean",
    },
    "gru": {
        "path": Path("outputs_downstreams/wastewater/vector_temporal/gru_top200/predictions.csv.gz"),
        "model": "vector_gru",
    },
    "tcn": {
        "path": Path("outputs_downstreams/wastewater/vector_temporal/tcn_top200/predictions.csv.gz"),
        "model": "vector_tcn",
    },
    "mgm2_medium_mae": {
        "path": Path("outputs_downstreams/wastewater/mgm2_world_model_ntv3_asv/mae_scale_medium_top200/predictions.csv.gz"),
        "model": "mgm2_world_mae",
    },
    "mgm2_xlarge_mae": {
        "path": Path("outputs_downstreams/wastewater/mgm2_world_model_ntv3_asv/mae_scale_xlarge_top200/predictions.csv.gz"),
        "model": "mgm2_world_mae",
    },
}

PLOT_ORDER = ["context_mean", "microprophet", "gru", "tcn", "mgm2_medium_mae", "mgm2_xlarge_mae"]
LABELS = {
    "truth": "True abundance",
    "context_mean": "Context mean",
    "microprophet": "MicroProphet",
    "gru": "GRU",
    "tcn": "TCN",
    "mgm2_medium_mae": "MGM2 medium MAE",
    "mgm2_xlarge_mae": "MGM2 xlarge MAE",
}
COLORS = {
    "truth": "#111111",
    "context_mean": "#8A817C",
    "microprophet": "#D8902F",
    "gru": "#63A375",
    "tcn": "#2E7D59",
    "mgm2_medium_mae": "#C84C31",
    "mgm2_xlarge_mae": "#4A148C",
}

META_COLUMNS = {"subject_id", "time", "original_time"}


mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["legend.fontsize"] = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microprophet_root", type=Path, default=DEFAULT_MICROPROPHET_ROOT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table_dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--top_n", type=int, default=6)
    parser.add_argument("--min_mean_abundance", type=float, default=0.002)
    parser.add_argument("--min_range", type=float, default=0.004)
    parser.add_argument("--min_std", type=float, default=1e-6, help="Minimum true/predicted standard deviation for correlation scoring")
    parser.add_argument("--formats", nargs="*", default=["png", "pdf"])
    return parser.parse_args()


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)


def _read_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    mapping = pd.read_csv(path)
    return dict(zip(mapping["plant_slug"], mapping["plant"]))


def _last_test_window(frame: pd.DataFrame) -> pd.DataFrame:
    test = frame[frame["split"].eq("test")].copy()
    max_window = test.groupby("plant")["window_idx"].transform("max")
    return test[test["window_idx"].eq(max_window)].copy()


def load_standard_prediction(run_label: str, path: Path, model_filter: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    columns = ["model", "plant", "split", "window_idx", "horizon", "sample_id", "date", "asv_id", "y_true", "y_pred"]
    frame = pd.read_csv(path, usecols=columns)
    frame = frame[frame["model"].eq(model_filter)].copy()
    frame = _last_test_window(frame)
    frame["run_label"] = run_label
    return frame[["run_label", "plant", "window_idx", "horizon", "sample_id", "date", "asv_id", "y_true", "y_pred"]]


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in META_COLUMNS]


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


def load_all_predictions(args: argparse.Namespace) -> pd.DataFrame:
    frames = [
        load_standard_prediction(run_label, spec["path"], spec["model"])
        for run_label, spec in STANDARD_MODEL_SPECS.items()
    ]
    frames.append(load_microprophet_predictions(args.microprophet_root, args.mapping))
    return pd.concat(frames, ignore_index=True)


def _pearson_corr(group: pd.DataFrame, min_std: float) -> float:
    true = group["y_true"].to_numpy(dtype=float)
    pred = group["y_pred"].to_numpy(dtype=float)
    if len(true) < 3 or np.nanstd(true) < min_std or np.nanstd(pred) < min_std:
        return np.nan
    corr = np.corrcoef(true, pred)[0, 1]
    return float(corr) if np.isfinite(corr) else np.nan


def select_best_mgm2_asvs(
    predictions: pd.DataFrame,
    top_n: int,
    min_mean_abundance: float,
    min_range: float,
    min_std: float,
) -> pd.DataFrame:
    mgm2 = predictions[predictions["run_label"].isin(["mgm2_medium_mae", "mgm2_xlarge_mae"])].copy()
    mgm2["abs_error"] = (mgm2["y_pred"] - mgm2["y_true"]).abs()
    truth = (
        mgm2.drop_duplicates(["plant", "asv_id", "horizon"])[["plant", "asv_id", "horizon", "y_true"]]
        .groupby(["plant", "asv_id"], as_index=False)
        .agg(
            mean_true=("y_true", "mean"),
            max_true=("y_true", "max"),
            min_true=("y_true", "min"),
            std_true=("y_true", "std"),
            n_horizons=("horizon", "nunique"),
        )
    )
    truth["range_true"] = truth["max_true"] - truth["min_true"]
    mae_scores = (
        mgm2.groupby(["plant", "asv_id", "run_label"], as_index=False)
        .agg(model_mae=("abs_error", "mean"))
        .pivot(index=["plant", "asv_id"], columns="run_label", values="model_mae")
        .reset_index()
        .rename(columns={"mgm2_medium_mae": "mgm2_medium_mae_abs_error", "mgm2_xlarge_mae": "mgm2_xlarge_mae_abs_error"})
    )
    corr_scores = (
        mgm2.groupby(["plant", "asv_id", "run_label"])
        .apply(lambda group: _pearson_corr(group.sort_values("horizon"), min_std), include_groups=False)
        .rename("pearson")
        .reset_index()
        .pivot(index=["plant", "asv_id"], columns="run_label", values="pearson")
        .reset_index()
        .rename(columns={"mgm2_medium_mae": "mgm2_medium_corr", "mgm2_xlarge_mae": "mgm2_xlarge_corr"})
    )
    scores = corr_scores.merge(mae_scores, on=["plant", "asv_id"], how="left")
    scores["mgm2_mean_corr"] = scores[["mgm2_medium_corr", "mgm2_xlarge_corr"]].mean(axis=1)
    scores["mgm2_min_corr"] = scores[["mgm2_medium_corr", "mgm2_xlarge_corr"]].min(axis=1)
    scores["mgm2_mean_abs_error"] = scores[["mgm2_medium_mae_abs_error", "mgm2_xlarge_mae_abs_error"]].mean(axis=1)
    scores = scores.merge(truth, on=["plant", "asv_id"], how="left")
    scores = scores[scores["n_horizons"].eq(10)].copy()
    filtered = scores[(scores["mean_true"] >= min_mean_abundance) & (scores["range_true"] >= min_range)].copy()
    if len(filtered) < top_n:
        filtered = scores[(scores["mean_true"] >= min_mean_abundance / 2.0) & (scores["range_true"] >= min_range / 2.0)].copy()
    if len(filtered) < top_n:
        filtered = scores.copy()
    filtered = filtered.dropna(subset=["mgm2_mean_corr"]).copy()
    filtered = filtered.sort_values(["mgm2_mean_corr", "mgm2_min_corr", "range_true"], ascending=[False, False, False])
    return filtered.head(top_n).reset_index(drop=True)


def subset_for_selected(predictions: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    key = selected[["plant", "asv_id"]].drop_duplicates()
    subset = predictions.merge(key, on=["plant", "asv_id"], how="inner")
    return subset[subset["run_label"].isin([*PLOT_ORDER])].copy()


def plot_selected_trends(trends: pd.DataFrame, selected: pd.DataFrame, output_dir: Path, formats: Sequence[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(selected)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.8, max(3.4, 3.25 * n_rows)), squeeze=False)
    axes = axes.flatten()

    for idx, row in selected.iterrows():
        ax = axes[idx]
        plant = row["plant"]
        asv_id = row["asv_id"]
        sub = trends[(trends["plant"].eq(plant)) & (trends["asv_id"].eq(asv_id))].copy()
        truth = (
            sub[sub["run_label"].eq("mgm2_medium_mae")][["horizon", "y_true"]]
            .drop_duplicates()
            .sort_values("horizon")
        )
        ax.plot(
            truth["horizon"],
            truth["y_true"],
            color=COLORS["truth"],
            linewidth=2.4,
            marker="o",
            markersize=4,
            label=LABELS["truth"],
        )
        for run_label in PLOT_ORDER:
            model = sub[sub["run_label"].eq(run_label)].sort_values("horizon")
            if model.empty:
                continue
            ax.plot(
                model["horizon"],
                model["y_pred"],
                color=COLORS[run_label],
                linewidth=1.65 if run_label.startswith("mgm2") else 1.25,
                linestyle="-" if run_label.startswith("mgm2") else "--",
                alpha=0.92,
                marker=None,
                label=LABELS[run_label],
            )
        ax.set_title(
            f"{plant} | {asv_id} | MGM2 r={row['mgm2_mean_corr']:.2f}",
            fontsize=9,
            fontweight="bold",
        )
        ax.set_xlabel("Last target step")
        ax.set_ylabel("Relative abundance")
        ax.set_xticks(range(1, 11))
        _style_axis(ax)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Best MGM2-correlated ASV trends over the last 10 target steps", fontsize=13, fontweight="bold")
    fig.subplots_adjust(bottom=0.12, hspace=0.38, wspace=0.22)
    paths = []
    for fmt in formats:
        path = output_dir / f"best_mgm2_asv_trends.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_all_predictions(args)
    selected = select_best_mgm2_asvs(predictions, args.top_n, args.min_mean_abundance, args.min_range, args.min_std)
    trends = subset_for_selected(predictions, selected)
    selected.to_csv(args.table_dir / "best_mgm2_asv_selection.csv", index=False)
    trends.to_csv(args.table_dir / "best_mgm2_asv_trends_long.csv", index=False)
    paths = plot_selected_trends(trends, selected, args.output_dir, args.formats)
    print(f"Saved selection: {args.table_dir / 'best_mgm2_asv_selection.csv'}")
    print(f"Saved trend table: {args.table_dir / 'best_mgm2_asv_trends_long.csv'}")
    for path in paths:
        print(f"Saved figure: {path}")
    print(
        selected[
            [
                "plant",
                "asv_id",
                "mgm2_medium_corr",
                "mgm2_xlarge_corr",
                "mgm2_mean_corr",
                "mgm2_mean_abs_error",
                "mean_true",
                "range_true",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
