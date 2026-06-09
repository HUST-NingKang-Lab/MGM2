#!/usr/bin/env python3
"""Visualize MGM2 xlarge OTU BatchTopK SAE grid-search results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mgm2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mgm2-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=Path("outputs_sae/mgm2_xlarge_otu_batchtopk_grid_d4096"),
        help="Directory containing one subdirectory per SAE grid run.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures/sae/otu_grid"))
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument(
        "--exclude-top-k",
        type=int,
        nargs="*",
        default=[],
        help="Top-k values to exclude from the plotted summary and curves.",
    )
    return parser.parse_args()


def _load_grid(grid_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    curve_rows = []
    for run_dir in sorted(grid_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.csv"
        config_path = run_dir / "config_summary.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        metrics = pd.read_csv(metrics_path)
        if metrics.empty:
            continue
        metrics["run"] = run_dir.name
        metrics["top_k"] = int(config["top_k"])
        metrics["aux_penalty"] = float(config["aux_penalty"])
        metrics["dict_size"] = int(config["dict_size"])
        curve_rows.append(metrics)

        best_idx = metrics["val_loss"].idxmin()
        best = metrics.loc[best_idx]
        last = metrics.iloc[-1]
        summary_rows.append(
            {
                "run": run_dir.name,
                "top_k": int(config["top_k"]),
                "aux_penalty": float(config["aux_penalty"]),
                "dict_size": int(config["dict_size"]),
                "best_step": int(best["step"]),
                "best_val_loss": float(best["val_loss"]),
                "best_val_mse": float(best["val_mse"]),
                "best_val_explained_variance": float(best["val_explained_variance"]),
                "best_dead_features": int(best["dead_features"]),
                "last_step": int(last["step"]),
                "last_val_loss": float(last["val_loss"]),
                "last_val_explained_variance": float(last["val_explained_variance"]),
                "last_dead_features": int(last["dead_features"]),
                "early_stop": bool(last.get("early_stop", False)),
                "eval_count": int(len(metrics)),
            }
        )
    if not summary_rows:
        raise FileNotFoundError(f"No SAE grid metrics found under {grid_dir}")
    summary = pd.DataFrame(summary_rows).sort_values(["top_k", "aux_penalty"]).reset_index(drop=True)
    curves = pd.concat(curve_rows, ignore_index=True)
    return summary, curves


def _run_label(row: pd.Series) -> str:
    return f"k={int(row['top_k'])}, aux={row['aux_penalty']:.2f}"


def _save(fig: plt.Figure, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_best_summary(summary: pd.DataFrame, output_dir: Path):
    df = summary.copy()
    df["label"] = df.apply(_run_label, axis=1)
    df = df.sort_values("best_val_loss")
    colors = sns.color_palette("viridis", n_colors=len(df))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    specs = [
        ("best_val_loss", "Best val loss", False),
        ("best_val_explained_variance", "Best explained variance", True),
        ("best_val_mse", "Best val MSE", False),
        ("best_dead_features", "Dead features at best", False),
    ]
    for ax, (col, title, higher_better) in zip(axes.flat, specs):
        ordered = df.sort_values(col, ascending=not higher_better)
        ax.barh(ordered["label"], ordered[col], color=colors, alpha=0.88)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="x", color="#DDDDDD", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, value in enumerate(ordered[col]):
            if "dead" in col:
                text = f"{int(value)}"
            else:
                text = f"{value:.4f}"
            ax.text(value, i, f" {text}", va="center", fontsize=8)
    fig.suptitle("MGM2 xlarge OTU BatchTopK SAE grid: best validation metrics", fontsize=15, fontweight="bold")
    _save(fig, output_dir, "otu_sae_grid_best_summary")


def plot_heatmaps(summary: pd.DataFrame, output_dir: Path):
    metrics = [
        ("best_val_loss", "Best val loss", "mako_r", ".4f"),
        ("best_val_explained_variance", "Best explained variance", "viridis", ".4f"),
        ("best_dead_features", "Dead features", "rocket_r", ".0f"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, (col, title, cmap, fmt) in zip(axes, metrics):
        table = summary.pivot(index="top_k", columns="aux_penalty", values=col).sort_index(ascending=False)
        sns.heatmap(table, annot=True, fmt=fmt, cmap=cmap, linewidths=0.5, linecolor="white", ax=ax, cbar=True)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("aux penalty")
        ax.set_ylabel("top k")
    fig.suptitle("SAE grid heatmaps", fontsize=15, fontweight="bold")
    _save(fig, output_dir, "otu_sae_grid_heatmaps")


def plot_tradeoff(summary: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7.8, 5.6), constrained_layout=True)
    scatter = ax.scatter(
        summary["best_dead_features"],
        summary["best_val_explained_variance"],
        s=summary["top_k"] * 8,
        c=summary["aux_penalty"],
        cmap="coolwarm",
        edgecolor="#222222",
        linewidth=0.8,
        alpha=0.9,
    )
    for _, row in summary.iterrows():
        ax.text(
            row["best_dead_features"] + 12,
            row["best_val_explained_variance"],
            _run_label(row),
            fontsize=8,
            va="center",
        )
    ax.set_xlabel("Dead features at best checkpoint")
    ax.set_ylabel("Best explained variance")
    ax.set_title("Reconstruction vs feature usage trade-off", fontsize=13, fontweight="bold")
    ax.grid(color="#DDDDDD", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("aux penalty")
    _save(fig, output_dir, "otu_sae_grid_tradeoff")


def plot_curves(curves: pd.DataFrame, output_dir: Path, smooth_window: int):
    df = curves.copy().sort_values(["top_k", "aux_penalty", "step"])
    df["label"] = df.apply(_run_label, axis=1)
    smooth_window = max(1, int(smooth_window))
    for col in ["val_loss", "val_explained_variance", "dead_features"]:
        smoothed = []
        for _, group in df.groupby("run", sort=False):
            values = group[col].rolling(smooth_window, min_periods=1, center=True).mean()
            smoothed.append(values)
        df[f"{col}_smooth"] = pd.concat(smoothed).sort_index()

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    specs = [
        ("val_loss_smooth", "Val loss", "lower is better"),
        ("val_explained_variance_smooth", "Explained variance", "higher is better"),
        ("dead_features_smooth", "Dead features", "lower is better"),
    ]
    palette = sns.color_palette("tab10", n_colors=df["run"].nunique())
    for ax, (col, title, subtitle) in zip(axes, specs):
        for color, (label, group) in zip(palette, df.groupby("label", sort=False)):
            ax.plot(group["step"], group[col], label=label, linewidth=2, color=color)
        ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Step")
        ax.grid(color="#DDDDDD", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Metric value")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("OTU SAE validation curves", fontsize=15, fontweight="bold")
    _save(fig, output_dir, "otu_sae_grid_curves")


def main():
    args = parse_args()
    summary, curves = _load_grid(args.grid_dir)
    if args.exclude_top_k:
        excluded = set(args.exclude_top_k)
        summary = summary[~summary["top_k"].isin(excluded)].reset_index(drop=True)
        curves = curves[~curves["top_k"].isin(excluded)].reset_index(drop=True)
        if summary.empty or curves.empty:
            raise ValueError(f"No runs remain after excluding top_k values: {sorted(excluded)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "otu_sae_grid_summary.csv", index=False)
    plot_best_summary(summary, args.output_dir)
    plot_heatmaps(summary, args.output_dir)
    plot_tradeoff(summary, args.output_dir)
    plot_curves(curves, args.output_dir, args.smooth_window)
    print(f"Saved summary and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
