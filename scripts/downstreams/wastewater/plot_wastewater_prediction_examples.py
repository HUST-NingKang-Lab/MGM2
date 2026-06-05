#!/usr/bin/env python3
"""Plot a single ASV's forecast across all plants — truth vs model predictions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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

DEFAULT_OUTPUT_DIR = Path("figures/wastewater_world_model")
MODEL_PATHS = {
    "persistence":       Path("outputs_downstreams/wastewater/baseline/baseline_top200/predictions.csv.gz"),
    "context_mean":      Path("outputs_downstreams/wastewater/baseline/baseline_top200/predictions.csv.gz"),
    "vector_tcn":        Path("outputs_downstreams/wastewater/vector_temporal/tcn_top200/predictions.csv.gz"),
    "hybrid_tcn":        Path("outputs_downstreams/wastewater/hybrid/vector_tcn_persistence_top200/predictions.csv.gz"),
    "token_xattn_small": Path("outputs_downstreams/wastewater/mgm2_token_cross_attention/scale_small_top200/predictions.csv.gz"),
    "mae_xlarge":        Path("outputs_downstreams/wastewater/mgm2_world_model_ntv3_asv/mae_scale_xlarge_top200/predictions.csv.gz"),
    "jepa_xlarge":       Path("outputs_downstreams/wastewater/mgm2_world_model_ntv3_asv/jepa_scale_xlarge_top200/predictions.csv.gz"),
}
MODEL_NAME_FILTERS = {
    "persistence":       "persistence",
    "context_mean":      "context_mean",
    "vector_tcn":        "vector_tcn",
    "hybrid_tcn":        "hybrid_vector_tcn_persistence_cutoff3",
    "token_xattn_small": "mgm2_token_cross_attention",
    "mae_xlarge":        "mgm2_world_mae",
    "jepa_xlarge":       "mgm2_world_jepa",
}
MODEL_LABELS = {
    "persistence":       "Persistence",
    "context_mean":      "Context mean",
    "vector_tcn":        "Vector TCN",
    "hybrid_tcn":        "TCN + persistence",
    "token_xattn_small": "Token forecast small",
    "mae_xlarge":        "MAE xlarge",
    "jepa_xlarge":       "JEPA xlarge",
}
MODEL_COLORS = {
    "truth":             "#111111",
    "persistence":       "#6A6A6A",
    "context_mean":      "#9A8C7A",
    "vector_tcn":        "#2E7D59",
    "hybrid_tcn":        "#D8902F",
    "token_xattn_small": "#386CB0",
    "mae_xlarge":        "#4A148C",
    "jepa_xlarge":       "#C94D9B",
}
MODEL_ORDER = ["persistence", "vector_tcn", "token_xattn_small", "jepa_xlarge", "mae_xlarge"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asv_id", type=str, default="asv2", help="ASV to plot across plants")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="*", default=["png", "pdf"])
    return parser.parse_args()


def load_predictions() -> pd.DataFrame:
    frames = []
    columns = ["model", "plant", "split", "window_idx", "horizon", "sample_id", "date", "asv_id", "y_true", "y_pred"]
    for run_label, path in MODEL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, usecols=columns)
        frame = frame[(frame["split"] == "test") & (frame["model"] == MODEL_NAME_FILTERS[run_label])].copy()
        frame["run_label"] = run_label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_asv_across_plants(predictions: pd.DataFrame, asv_id: str, output_dir: Path, formats):
    # Filter to the target ASV (case-insensitive match)
    df = predictions[predictions["asv_id"].str.upper().eq(asv_id.upper())].copy()

    # Collect truth from any model's y_true (all share same ground truth)
    truth = df[df["run_label"].eq("mae_xlarge")][["plant", "window_idx", "horizon", "y_true"]].drop_duplicates()

    # Build plant×window table with all model predictions
    rows = truth.copy()
    for run_label in MODEL_ORDER:
        if run_label == "persistence":
            pred = df[df["run_label"].eq(run_label)][["plant", "window_idx", "horizon", "y_pred"]].rename(
                columns={"y_pred": run_label}
            )
        else:
            pred = df[df["run_label"].eq(run_label)][["plant", "window_idx", "horizon", "y_pred"]].rename(
                columns={"y_pred": run_label}
            )
        rows = rows.merge(pred, on=["plant", "window_idx", "horizon"], how="left")

    plants = sorted(rows["plant"].unique())
    n_plants = len(plants)
    n_cols = 4
    n_rows = int(np.ceil(n_plants / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.2, n_rows * 3.0), squeeze=False)
    axes = axes.flatten()

    for idx, plant in enumerate(plants):
        ax = axes[idx]
        # Pick first test window for this plant
        plant_data = rows[rows["plant"].eq(plant)]
        first_window = int(plant_data["window_idx"].min())
        sub = plant_data[plant_data["window_idx"].eq(first_window)].sort_values("horizon")

        ax.plot(sub["horizon"], sub["y_true"], color=MODEL_COLORS["truth"],
                marker="o", linewidth=2.4, label="Truth")
        for run_label in MODEL_ORDER:
            ax.plot(sub["horizon"], sub[run_label],
                    color=MODEL_COLORS[run_label], linewidth=1.4, alpha=0.85,
                    label=MODEL_LABELS[run_label])

        ax.set_title(f"{plant} (w={first_window})", fontsize=9, fontweight="bold")
        ax.set_xlabel("Horizon", fontsize=8)
        ax.set_ylabel("Rel. abundance", fontsize=8)
        ax.set_xticks([1, 5, 10])
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    # Hide unused subplots
    for idx in range(n_plants, len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle(f"{asv_id} — true and predicted abundance across all plants (test, 1st window)",
                 fontsize=14, fontweight="bold")
    fig.subplots_adjust(bottom=0.10)

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"asv_{asv_id}_all_plants.{fmt}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Save table
    rows.to_csv(output_dir / f"asv_{asv_id}_all_plants.csv", index=False)
    print(f"Saved {output_dir}/asv_{asv_id}_all_plants.{{png,pdf,csv}} ({n_plants} plants)")


def main():
    args = parse_args()
    predictions = load_predictions()
    plot_asv_across_plants(predictions, args.asv_id, args.output_dir, args.formats)


if __name__ == "__main__":
    main()
