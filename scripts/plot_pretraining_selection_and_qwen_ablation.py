#!/usr/bin/env python3
"""Plot Stage-1 search and Stage-2 Qwen ablation at early-stopping checkpoints.

Each run contributes the complete validation row with minimum ``val/total_loss``,
the metric monitored by EarlyStopping and ModelCheckpoint.

Example:
    python scripts/plot_pretraining_selection_and_qwen_ablation.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#24313a",
        "xtick.color": "#42515c",
        "ytick.color": "#42515c",
    }
)

SELECTED_COLOR = "#d95f02"
QWEN_COLOR = "#00897b"
NO_QWEN_COLOR = "#aab7bf"
GRID_COLOR = "#dce3e7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1_summary",
        type=Path,
        default=Path("outputs_grid/stage1/20260426_230440/summary.csv"),
    )
    parser.add_argument(
        "--qwen_summary",
        type=Path,
        default=Path("outputs_grid/stage2/20260505_094404/summary.csv"),
    )
    parser.add_argument(
        "--qwen0_summary",
        type=Path,
        default=Path("outputs_grid/stage2_qwen0/20260615_021958/summary.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("figures/pretraining_evidence"),
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def format_lr(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-")


def metrics_version_key(path: Path) -> tuple[int, float]:
    match = re.fullmatch(r"version_(\d+)", path.parent.name)
    return (int(match.group(1)) if match else -1, path.stat().st_mtime)


def find_metrics_csv(summary_path: Path, run_name: str) -> Path:
    candidates = list(
        (summary_path.parent / run_name / "logs").glob("*/version_*/metrics.csv")
    )
    if not candidates:
        raise FileNotFoundError(f"No metrics.csv found for {run_name}")
    return max(candidates, key=metrics_version_key)


def load_early_stopping_metrics(
    summary_path: Path, run_names: pd.Series
) -> pd.DataFrame:
    """Return all metrics from the row selected by minimum validation total loss."""
    records = []
    required = [
        "epoch",
        "val/total_loss",
        "val/bin_accuracy",
        "val/regression_mse",
    ]
    for run_name in run_names.astype(str):
        metrics_path = find_metrics_csv(summary_path, run_name)
        metrics = pd.read_csv(metrics_path)
        require_columns(metrics, required, metrics_path)
        validation = metrics.dropna(subset=required)
        if validation.empty:
            raise ValueError(f"No complete validation rows in {metrics_path}")
        row = validation.loc[validation["val/total_loss"].idxmin()]
        records.append(
            {
                "run_name": run_name,
                "early_stopping_best_epoch": int(row["epoch"]),
                "early_stopping_val_total_loss": float(row["val/total_loss"]),
                "early_stopping_val_bin_accuracy": float(row["val/bin_accuracy"]),
                "early_stopping_val_regression_mse": float(
                    row["val/regression_mse"]
                ),
                "last_validation_epoch": int(validation["epoch"].max()),
                "metrics_csv": str(metrics_path),
            }
        )
    return pd.DataFrame(records)


def effective_qwen_weight(frame: pd.DataFrame) -> pd.Series:
    """Prefer run names because one historical summary row contains a stale value."""
    parsed = pd.to_numeric(
        frame["run_name"].str.extract(r"_qw([0-9.eE+-]+)$", expand=False),
        errors="coerce",
    )
    reported = pd.to_numeric(frame["qwen3_loss_weight"], errors="raise")
    return parsed.fillna(reported)


def load_stage1(summary_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(summary_path)
    columns = ["run_name", "learning_rate", "mask_prob", "qwen3_loss_weight"]
    require_columns(frame, columns, summary_path)
    frame["qwen3_loss_weight_reported"] = frame["qwen3_loss_weight"]
    frame["qwen3_loss_weight"] = effective_qwen_weight(frame)
    frame = frame[columns + ["qwen3_loss_weight_reported"]].merge(
        load_early_stopping_metrics(summary_path, frame["run_name"]),
        on="run_name",
        validate="one_to_one",
    )
    keys = ["learning_rate", "mask_prob", "qwen3_loss_weight"]
    if frame.duplicated(keys).any():
        raise ValueError("Stage-1 summary contains duplicate hyperparameter combinations")
    frame["selected"] = (
        np.isclose(frame["learning_rate"], 5e-5)
        & np.isclose(frame["mask_prob"], 0.15)
        & np.isclose(frame["qwen3_loss_weight"], 0.1)
    )
    if frame["selected"].sum() != 1:
        raise ValueError("Could not identify the selected Stage-1 configuration")
    return frame.sort_values(keys).reset_index(drop=True)


def annotate_heatmap(
    ax: plt.Axes,
    subset: pd.DataFrame,
    matrix: np.ndarray,
    learning_rates: list[float],
    mask_probs: list[float],
    metric: str,
) -> None:
    midpoint = (float(np.nanmin(matrix)) + float(np.nanmax(matrix))) / 2
    for y, learning_rate in enumerate(learning_rates):
        for x, mask_prob in enumerate(mask_probs):
            cell = subset[
                np.isclose(subset["learning_rate"], learning_rate)
                & np.isclose(subset["mask_prob"], mask_prob)
            ]
            if len(cell) != 1:
                continue
            row = cell.iloc[0]
            value = float(row[metric])
            ax.text(
                x,
                y,
                f"{value:.4f}\nepoch {int(row['early_stopping_best_epoch'])}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white" if value < midpoint else "#17242c",
            )
            if bool(row["selected"]):
                ax.add_patch(
                    Rectangle(
                        (x - 0.48, y - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor=SELECTED_COLOR,
                        linewidth=3.2,
                    )
                )


def plot_stage1(frame: pd.DataFrame, output_dir: Path) -> None:
    weights = sorted(frame["qwen3_loss_weight"].unique())
    learning_rates = sorted(frame["learning_rate"].unique(), reverse=True)
    mask_probs = sorted(frame["mask_prob"].unique())
    if len(weights) < 2:
        raise ValueError("Expected all searched Qwen weights in Stage-1 summary")

    metric_specs = [
        ("early_stopping_val_bin_accuracy", "Bin accuracy", "YlGnBu", "higher"),
        ("early_stopping_val_regression_mse", "Regression MSE", "YlOrRd_r", "lower"),
    ]
    fig, axes = plt.subplots(
        len(metric_specs),
        len(weights),
        figsize=(5.1 * len(weights), 7.9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row_index, (metric, metric_name, cmap, direction) in enumerate(metric_specs):
        limits = frame[metric].to_numpy(float)
        images = []
        for column_index, weight in enumerate(weights):
            ax = axes[row_index, column_index]
            subset = frame[np.isclose(frame["qwen3_loss_weight"], weight)]
            matrix = np.full((len(learning_rates), len(mask_probs)), np.nan)
            for y, learning_rate in enumerate(learning_rates):
                for x, mask_prob in enumerate(mask_probs):
                    cell = subset[
                        np.isclose(subset["learning_rate"], learning_rate)
                        & np.isclose(subset["mask_prob"], mask_prob)
                    ]
                    if len(cell) == 1:
                        matrix[y, x] = cell.iloc[0][metric]
            image = ax.imshow(
                matrix,
                cmap=cmap,
                vmin=float(limits.min()),
                vmax=float(limits.max()),
                aspect="auto",
            )
            images.append(image)
            annotate_heatmap(
                ax, subset, matrix, learning_rates, mask_probs, metric
            )
            ax.set_title(f"Qwen weight = {weight:g}", fontsize=12, pad=9)
            ax.set_xticks(
                range(len(mask_probs)), [f"{value:.2f}" for value in mask_probs]
            )
            ax.set_yticks(
                range(len(learning_rates)),
                [format_lr(value) for value in learning_rates],
            )
            ax.set_xlabel("Mask probability")
            if column_index == 0:
                ax.set_ylabel(f"{metric_name}\nLearning rate")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("white")
                spine.set_linewidth(1.5)
        colorbar = fig.colorbar(
            images[-1], ax=axes[row_index, :], shrink=0.82, pad=0.025
        )
        colorbar.set_label(f"{metric_name} ({direction} is better)")

    selected_mask = frame["selected"]
    selected = frame.loc[selected_mask].iloc[0]
    bin_rank = int(
        frame["early_stopping_val_bin_accuracy"]
        .rank(method="min", ascending=False)[selected_mask]
        .iloc[0]
    )
    mse_rank = int(
        frame["early_stopping_val_regression_mse"]
        .rank(method="min", ascending=True)[selected_mask]
        .iloc[0]
    )
    fig.suptitle(
        "Stage 1 factorial search at early-stopping best checkpoints",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        (
            "Orange outline: selected lr=5e-5, mask=0.15, Qwen weight=0.1 "
            f"(bin rank {bin_rank}/{len(frame)}, MSE rank {mse_rank}/{len(frame)}; "
            f"best epoch {int(selected['early_stopping_best_epoch'])}). "
            "Total loss is omitted because Qwen weight changes its scale."
        ),
        ha="center",
        va="top",
        fontsize=9.5,
        color="#37474f",
    )
    save_figure(fig, output_dir, "figure1_stage1_hyperparameter_selection")
    frame.to_csv(
        output_dir / "figure1_stage1_hyperparameter_selection.csv", index=False
    )


def load_qwen_ablation(qwen_path: Path, qwen0_path: Path) -> pd.DataFrame:
    columns = [
        "run_name",
        "learning_rate",
        "mask_prob",
        "lambda_regression",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "qwen3_loss_weight",
    ]
    frames = []
    for path, suffix in ((qwen_path, "_qwen"), (qwen0_path, "_qwen0")):
        frame = pd.read_csv(path)
        require_columns(frame, columns, path)
        frame = frame[columns].merge(
            load_early_stopping_metrics(path, frame["run_name"]),
            on="run_name",
            validate="one_to_one",
        )
        frames.append(frame.add_suffix(suffix))
    merged = frames[0].merge(
        frames[1],
        left_on="run_name_qwen",
        right_on="run_name_qwen0",
        validate="one_to_one",
    )
    for field in columns[1:-1]:
        if not np.allclose(merged[f"{field}_qwen"], merged[f"{field}_qwen0"]):
            raise ValueError(f"Qwen ablation is not matched for {field}")
    if not np.allclose(merged["qwen3_loss_weight_qwen0"], 0):
        raise ValueError("Qwen-disabled runs have a non-zero alignment weight")
    order = {
        "scale_small": 0,
        "scale_medium": 1,
        "scale_large": 2,
        "scale_xlarge": 3,
    }
    merged["order"] = merged["run_name_qwen"].map(order)
    if merged["order"].isna().any() or len(merged) != 4:
        raise ValueError("Expected four matched scale pairs")
    return merged.sort_values("order").reset_index(drop=True)


def plot_dumbbells(
    ax: plt.Axes,
    baseline: np.ndarray,
    qwen: np.ndarray,
    scales: list[str],
    title: str,
    higher_is_better: bool,
) -> None:
    y = np.arange(len(scales))
    for index, (start, end) in enumerate(zip(baseline, qwen)):
        ax.plot([start, end], [index, index], color=GRID_COLOR, linewidth=4)
    ax.scatter(
        baseline,
        y,
        s=85,
        color=NO_QWEN_COLOR,
        edgecolor="white",
        label="Qwen weight = 0",
        zorder=2,
    )
    ax.scatter(
        qwen,
        y,
        s=95,
        color=QWEN_COLOR,
        edgecolor="white",
        label="Qwen weight = 0.1",
        zorder=3,
    )
    ax.set_yticks(y, scales)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=13, pad=12)
    ax.text(
        0.5,
        1.01,
        "Higher is better" if higher_is_better else "Lower is better",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#65747d",
    )
    values = np.concatenate([baseline, qwen])
    span = max(float(np.ptp(values)), abs(float(values.max())) * 0.04, 1e-4)
    ax.set_xlim(float(values.min()) - 0.25 * span, float(values.max()) + 0.42 * span)
    for index, (start, end) in enumerate(zip(baseline, qwen)):
        delta = end - start
        improved = delta > 0 if higher_is_better else delta < 0
        ax.text(
            max(start, end) + 0.035 * span,
            index,
            f"{delta:+.4f}",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=QWEN_COLOR if improved else "#b23a48",
        )
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=350,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def plot_qwen_ablation(frame: pd.DataFrame, output_dir: Path) -> None:
    scales = [
        name.replace("scale_", "").capitalize() for name in frame["run_name_qwen"]
    ]
    qwen_acc = frame["early_stopping_val_bin_accuracy_qwen"].to_numpy(float)
    no_qwen_acc = frame["early_stopping_val_bin_accuracy_qwen0"].to_numpy(float)
    qwen_mse = frame["early_stopping_val_regression_mse_qwen"].to_numpy(float)
    no_qwen_mse = frame["early_stopping_val_regression_mse_qwen0"].to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.3))
    plot_dumbbells(axes[0], no_qwen_acc, qwen_acc, scales, "Bin accuracy", True)
    plot_dumbbells(axes[1], no_qwen_mse, qwen_mse, scales, "Regression MSE", False)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")

    acc_delta = qwen_acc - no_qwen_acc
    mse_delta = qwen_mse - no_qwen_mse
    fig.suptitle(
        "Qwen alignment ablation at early-stopping best checkpoints",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.005,
        (
            f"Mean bin change {acc_delta.mean():+.4f} "
            f"({np.sum(acc_delta > 0)}/4 improve); mean MSE change "
            f"{mse_delta.mean():+.4f} ({np.sum(mse_delta < 0)}/4 improve). "
            "Every pair uses all metrics from its selected epoch."
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#37474f",
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.98))
    save_figure(fig, output_dir, "figure2_qwen_alignment_ablation")

    output = frame.copy()
    output["bin_accuracy_delta_qwen_minus_zero"] = acc_delta
    output["regression_mse_delta_qwen_minus_zero"] = mse_delta
    output.to_csv(
        output_dir / "figure2_qwen_alignment_ablation.csv", index=False
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_stage1(load_stage1(args.stage1_summary), args.output_dir)
    plot_qwen_ablation(
        load_qwen_ablation(args.qwen_summary, args.qwen0_summary),
        args.output_dir,
    )
    print(f"Saved figures and source tables to {args.output_dir}")


if __name__ == "__main__":
    main()
