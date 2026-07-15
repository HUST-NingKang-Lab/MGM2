#!/usr/bin/env python3
"""Plot high-variation ASV trajectories for the base V2 token model."""

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


DEFAULT_OUTPUT_DIR = Path("figures/wastewater_v2/token_asv_trends")
DEFAULT_TABLE_DIR = Path("outputs_downstreams/wastewater_v2/token_asv_trends")
DEFAULT_MICROPROPHET_ROOT = Path("data/downstreams/wastewater/microprophet_res")
DEFAULT_MAPPING = Path("data/downstreams/wastewater/microprophet_by_plant/subject_id_mapping.csv")

MODEL_SPECS = {
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
    "token_large": {
        "path": Path("outputs_downstreams/wastewater_v2/token_asv_delta_transformer/scale_large/predictions.csv.gz"),
        "model": "v2_token_asv_delta_transformer",
        "scale": "scale_large",
    },
    "token_xlarge": {
        "path": Path("outputs_downstreams/wastewater_v2/token_asv_delta_transformer/scale_xlarge/predictions.csv.gz"),
        "model": "v2_token_asv_delta_transformer",
        "scale": "scale_xlarge",
    },
}

PLOT_ORDER = [
    "context_mean",
    "microprophet",
    "gru",
    "tcn",
    "token_large",
    "token_xlarge",
]

LABELS = {
    "truth": "True abundance",
    "context_mean": "Context mean",
    "microprophet": "MicroProphet",
    "gru": "GRU",
    "tcn": "TCN",
    "token_large": "MGM2 large",
    "token_xlarge": "MGM2 xlarge",
}

COLORS = {
    "truth": "#111111",
    "context_mean": "#8A817C",
    "microprophet": "#D8902F",
    "gru": "#74A57F",
    "tcn": "#246A73",
    "token_large": "#C4382B",
    "token_xlarge": "#1F4E9A",
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
    parser.add_argument("--examples_per_bin", type=int, default=2)
    parser.add_argument("--min_mean_abundance", type=float, default=0.002)
    parser.add_argument("--min_range", type=float, default=0.004)
    parser.add_argument("--min_std", type=float, default=1e-6)
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


def load_prediction(run_label: str, spec: dict) -> pd.DataFrame:
    path = spec["path"]
    if not path.exists():
        raise FileNotFoundError(path)
    columns = ["model", "plant", "split", "window_idx", "horizon", "sample_id", "date", "asv_id", "y_true", "y_pred"]
    if "scale" in spec:
        columns.append("scale")
    frame = pd.read_csv(path, usecols=columns)
    frame = frame[frame["model"].eq(spec["model"])].copy()
    if "scale" in spec:
        frame = frame[frame["scale"].eq(spec["scale"])].copy()
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
    frames = [load_prediction(label, spec) for label, spec in MODEL_SPECS.items()]
    frames.append(load_microprophet_predictions(args.microprophet_root, args.mapping))
    return pd.concat(frames, ignore_index=True)


def _pearson_corr(group: pd.DataFrame, min_std: float) -> float:
    true = group["y_true"].to_numpy(dtype=float)
    pred = group["y_pred"].to_numpy(dtype=float)
    if len(true) < 3 or np.nanstd(true) < min_std or np.nanstd(pred) < min_std:
        return np.nan
    corr = np.corrcoef(true, pred)[0, 1]
    return float(corr) if np.isfinite(corr) else np.nan


def select_best_token_asvs(
    predictions: pd.DataFrame,
    top_n: int,
    min_mean_abundance: float,
    min_range: float,
    min_std: float,
    examples_per_bin: int = 2,
) -> pd.DataFrame:
    token = predictions[predictions["run_label"].isin(["token_large", "token_xlarge"])].copy()
    token["abs_error"] = (token["y_pred"] - token["y_true"]).abs()
    truth = (
        token.drop_duplicates(["plant", "asv_id", "horizon"])[["plant", "asv_id", "horizon", "y_true"]]
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
    corr_scores = (
        token.groupby(["plant", "asv_id", "run_label"])
        .apply(lambda group: _pearson_corr(group.sort_values("horizon"), min_std), include_groups=False)
        .rename("pearson")
        .reset_index()
        .pivot(index=["plant", "asv_id"], columns="run_label", values="pearson")
        .reset_index()
        .rename(columns={"token_large": "token_large_corr", "token_xlarge": "token_xlarge_corr"})
    )
    mae_scores = (
        token.groupby(["plant", "asv_id", "run_label"], as_index=False)
        .agg(model_mae=("abs_error", "mean"))
        .pivot(index=["plant", "asv_id"], columns="run_label", values="model_mae")
        .reset_index()
        .rename(columns={"token_large": "token_large_abs_error", "token_xlarge": "token_xlarge_abs_error"})
    )
    scores = corr_scores.merge(mae_scores, on=["plant", "asv_id"], how="left").merge(truth, on=["plant", "asv_id"], how="left")
    scores = scores[scores["n_horizons"].eq(10)].copy()
    filtered = scores[(scores["mean_true"] >= min_mean_abundance) & (scores["range_true"] >= min_range)].copy()
    if len(filtered) < top_n:
        filtered = scores[(scores["mean_true"] >= min_mean_abundance / 2.0) & (scores["range_true"] >= min_range / 2.0)].copy()
    if len(filtered) < top_n:
        filtered = scores.copy()
    filtered = filtered.sort_values(["mean_true", "range_true", "std_true"], ascending=[True, False, False]).reset_index(drop=True)
    filtered["abundance_bin"] = pd.qcut(
        filtered["mean_true"].rank(method="first"),
        q=3,
        labels=["Low", "Medium", "High"],
    )
    selected_frames = []
    for abundance_bin in ["Low", "Medium", "High"]:
        bin_frame = filtered[filtered["abundance_bin"].eq(abundance_bin)].copy()
        bin_frame = bin_frame.sort_values(["range_true", "std_true", "mean_true"], ascending=[False, False, False])
        selected_frames.append(bin_frame.head(examples_per_bin))
    selected = pd.concat(selected_frames, ignore_index=True)
    if len(selected) < top_n:
        missing = top_n - len(selected)
        used = set(zip(selected["plant"], selected["asv_id"]))
        fallback = filtered[~filtered.apply(lambda row: (row["plant"], row["asv_id"]) in used, axis=1)].copy()
        fallback = fallback.sort_values(["range_true", "std_true", "mean_true"], ascending=[False, False, False])
        selected = pd.concat([selected, fallback.head(missing)], ignore_index=True)
    selected["abundance_bin"] = pd.Categorical(selected["abundance_bin"], categories=["Low", "Medium", "High"], ordered=True)
    return selected.sort_values(["abundance_bin", "range_true", "std_true"], ascending=[True, False, False]).reset_index(drop=True)


def subset_for_selected(predictions: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    key = selected[["plant", "asv_id"]].drop_duplicates()
    subset = predictions.merge(key, on=["plant", "asv_id"], how="inner")
    return subset[subset["run_label"].isin(PLOT_ORDER)].copy()


def plot_selected_trends(trends: pd.DataFrame, selected: pd.DataFrame, output_dir: Path, formats: Sequence[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(selected)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.2, max(3.5, 3.35 * n_rows)), squeeze=False)
    axes = axes.flatten()

    for idx, row in selected.iterrows():
        ax = axes[idx]
        plant = row["plant"]
        asv_id = row["asv_id"]
        sub = trends[(trends["plant"].eq(plant)) & (trends["asv_id"].eq(asv_id))].copy()
        truth = (
            sub[sub["run_label"].eq("token_large")][["horizon", "y_true"]]
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
            is_token = run_label.startswith("token")
            ax.plot(
                model["horizon"],
                model["y_pred"],
                color=COLORS[run_label],
                linewidth=1.9 if is_token else 1.15,
                linestyle="-" if is_token else "--",
                alpha=0.95 if is_token else 0.8,
                label=LABELS[run_label],
            )
        ax.set_title(
            f"{row.get('abundance_bin', '')} | {plant} | {asv_id} | range={row['range_true']:.3f}",
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
    fig.suptitle("High-variation ASV trend examples stratified by relative abundance", fontsize=13, fontweight="bold")
    fig.subplots_adjust(bottom=0.13, hspace=0.4, wspace=0.24)
    paths = []
    for fmt in formats:
        path = output_dir / f"base_high_variation_abundance_stratified_asv_trends.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_all_predictions(args)
    selected = select_best_token_asvs(
        predictions,
        args.top_n,
        args.min_mean_abundance,
        args.min_range,
        args.min_std,
        args.examples_per_bin,
    )
    trends = subset_for_selected(predictions, selected)
    selected.to_csv(args.table_dir / "base_high_variation_abundance_stratified_asv_selection.csv", index=False)
    trends.to_csv(args.table_dir / "base_high_variation_abundance_stratified_asv_trends_long.csv", index=False)
    paths = plot_selected_trends(trends, selected, args.output_dir, args.formats)
    print(f"Saved selection: {args.table_dir / 'base_high_variation_abundance_stratified_asv_selection.csv'}")
    print(f"Saved trend table: {args.table_dir / 'base_high_variation_abundance_stratified_asv_trends_long.csv'}")
    for path in paths:
        print(f"Saved figure: {path}")
    print(
        selected[
            [
                "plant",
                "asv_id",
                "abundance_bin",
                "token_large_corr",
                "token_xlarge_corr",
                "token_large_abs_error",
                "token_xlarge_abs_error",
                "mean_true",
                "range_true",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
