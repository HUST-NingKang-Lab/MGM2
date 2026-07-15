#!/usr/bin/env python3
"""Plot ASV trend correlations stratified by true abundance."""

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

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover - scipy is available in the mwm env, but keep the script portable.
    wilcoxon = None


DEFAULT_PREDICTIONS = Path("outputs_downstreams/wastewater_v2/base_token_statistics/base_token_last10_predictions_long.csv.gz")
DEFAULT_OUTPUT_DIR = Path("figures/wastewater_v2/base_token_statistics")
DEFAULT_TABLE_DIR = Path("outputs_downstreams/wastewater_v2/base_token_statistics")

RUN_ORDER = ["microprophet", "gru", "tcn", "token_small", "token_medium", "token_large", "token_xlarge"]
BIN_ORDER = ["Low", "Medium", "High"]
LABELS = {
    "microprophet": "MicroProphet",
    "gru": "GRU",
    "tcn": "TCN",
    "token_small": "MGM2 small",
    "token_medium": "MGM2 medium",
    "token_large": "MGM2 large",
    "token_xlarge": "MGM2 xlarge",
}
COLORS = {
    "microprophet": "#D8902F",
    "gru": "#8AA64B",
    "tcn": "#347A50",
    "token_small": "#8A90AD",
    "token_medium": "#7D8EC7",
    "token_large": "#586FB4",
    "token_xlarge": "#3156AA",
}

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 9
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["legend.fontsize"] = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table_dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--formats", nargs="*", default=["pdf", "png"])
    parser.add_argument("--min_std", type=float, default=1e-8)
    parser.add_argument("--include_other", action="store_true", help="Include the aggregated __OTHER__ bucket.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    return parser.parse_args()


def _pearson(true: np.ndarray, pred: np.ndarray, min_std: float) -> float:
    if len(true) < 3 or np.nanstd(true) < min_std or np.nanstd(pred) < min_std:
        return float("nan")
    corr = np.corrcoef(true, pred)[0, 1]
    return float(corr) if np.isfinite(corr) else float("nan")


def _bootstrap_mean_ci(values: np.ndarray, seed: int, n_bootstrap: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _bh_fdr(p_values: Sequence[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q.tolist()
    valid_indices = np.where(valid)[0]
    ordered = valid_indices[np.argsort(p[valid])]
    ranked = p[ordered] * len(ordered) / np.arange(1, len(ordered) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[ordered] = np.clip(ranked, 0.0, 1.0)
    return q.tolist()


def load_predictions(path: Path, include_other: bool) -> pd.DataFrame:
    columns = ["run_label", "plant", "horizon", "asv_id", "y_true", "y_pred"]
    frame = pd.read_csv(path, usecols=columns)
    frame = frame[frame["run_label"].isin(RUN_ORDER)].copy()
    if not include_other:
        frame = frame[~frame["asv_id"].eq("__OTHER__")].copy()
    return frame


def assign_abundance_bins(predictions: pd.DataFrame) -> pd.DataFrame:
    truth = predictions[["plant", "asv_id", "horizon", "y_true"]].drop_duplicates()
    abundance = (
        truth.groupby(["plant", "asv_id"], as_index=False)
        .agg(mean_true=("y_true", "mean"), max_true=("y_true", "max"), n_horizons=("horizon", "nunique"))
    )
    abundance = abundance[abundance["n_horizons"].ge(3)].copy()
    abundance["abundance_bin"] = pd.qcut(
        abundance["mean_true"].rank(method="first"),
        q=3,
        labels=BIN_ORDER,
    )
    abundance["abundance_bin"] = pd.Categorical(abundance["abundance_bin"], categories=BIN_ORDER, ordered=True)
    return abundance[["plant", "asv_id", "mean_true", "max_true", "abundance_bin"]]


def compute_asv_correlations(predictions: pd.DataFrame, abundance: pd.DataFrame, min_std: float) -> pd.DataFrame:
    rows = []
    for (run_label, plant, asv_id), group in predictions.groupby(["run_label", "plant", "asv_id"], sort=False):
        group = group.sort_values("horizon")
        rows.append(
            {
                "run_label": run_label,
                "plant": plant,
                "asv_id": asv_id,
                "trend_pearson": _pearson(
                    group["y_true"].to_numpy(dtype=float),
                    group["y_pred"].to_numpy(dtype=float),
                    min_std,
                ),
            }
        )
    corr = pd.DataFrame(rows).dropna(subset=["trend_pearson"])
    corr = corr.merge(abundance, on=["plant", "asv_id"], how="inner")
    return corr


def summarize_plant_bins(correlations: pd.DataFrame, seed: int, n_bootstrap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    plant_bin = (
        correlations.groupby(["run_label", "plant", "abundance_bin"], observed=True, as_index=False)
        .agg(plant_bin_mean_r=("trend_pearson", "mean"), n_asvs=("asv_id", "nunique"))
    )
    summary_rows = []
    for idx, ((run_label, abundance_bin), group) in enumerate(
        plant_bin.groupby(["run_label", "abundance_bin"], observed=True, sort=False)
    ):
        values = group["plant_bin_mean_r"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_mean_ci(values, seed + idx, n_bootstrap)
        summary_rows.append(
            {
                "run_label": run_label,
                "label": LABELS.get(run_label, run_label),
                "abundance_bin": abundance_bin,
                "n_plants": int(group["plant"].nunique()),
                "n_asvs": int(group["n_asvs"].sum()),
                "mean_r": float(np.nanmean(values)),
                "median_r": float(np.nanmedian(values)),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["run_label"] = pd.Categorical(summary["run_label"], categories=RUN_ORDER, ordered=True)
    summary["abundance_bin"] = pd.Categorical(summary["abundance_bin"], categories=BIN_ORDER, ordered=True)
    return plant_bin, summary.sort_values(["abundance_bin", "run_label"]).reset_index(drop=True)


def compute_pairwise_tests(plant_bin: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wide = plant_bin.pivot_table(
        index=["plant", "abundance_bin"],
        columns="run_label",
        values="plant_bin_mean_r",
        observed=True,
    )
    for abundance_bin in BIN_ORDER:
        sub = wide.xs(abundance_bin, level="abundance_bin", drop_level=False) if abundance_bin in wide.index.get_level_values("abundance_bin") else pd.DataFrame()
        for reference in [run for run in RUN_ORDER if run != "token_xlarge"]:
            if sub.empty or "token_xlarge" not in sub.columns or reference not in sub.columns:
                rows.append({"abundance_bin": abundance_bin, "reference": reference, "n_plants": 0, "mean_delta": np.nan, "p_value": np.nan})
                continue
            pair = sub[["token_xlarge", reference]].dropna()
            delta = pair["token_xlarge"] - pair[reference]
            p_value = float("nan")
            if wilcoxon is not None and len(delta) >= 3 and np.nanstd(delta.to_numpy(dtype=float)) > 0:
                p_value = float(wilcoxon(delta, alternative="greater").pvalue)
            rows.append(
                {
                    "abundance_bin": abundance_bin,
                    "reference": reference,
                    "n_plants": int(len(delta)),
                    "mean_delta": float(delta.mean()) if len(delta) else np.nan,
                    "median_delta": float(delta.median()) if len(delta) else np.nan,
                    "fraction_positive_delta": float((delta > 0).mean()) if len(delta) else np.nan,
                    "p_value": p_value,
                }
            )
    tests = pd.DataFrame(rows)
    tests["fdr_bh"] = _bh_fdr(tests["p_value"].tolist())
    return tests


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)


def plot_abundance_strata(plant_bin: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, formats: Sequence[str], seed: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    x_base = {name: idx for idx, name in enumerate(BIN_ORDER)}
    offsets = np.linspace(-0.28, 0.28, len(RUN_ORDER))
    offset_map = dict(zip(RUN_ORDER, offsets))

    for run_label in RUN_ORDER:
        sub = plant_bin[plant_bin["run_label"].eq(run_label)].copy()
        for abundance_bin in BIN_ORDER:
            points = sub[sub["abundance_bin"].astype(str).eq(abundance_bin)]
            if points.empty:
                continue
            x = x_base[abundance_bin] + offset_map[run_label]
            jitter = rng.uniform(-0.025, 0.025, size=len(points))
            ax.scatter(
                np.full(len(points), x) + jitter,
                points["plant_bin_mean_r"],
                s=16,
                color=COLORS[run_label],
                alpha=0.42,
                linewidths=0,
                rasterized=False,
            )
            row = summary[(summary["run_label"].astype(str).eq(run_label)) & (summary["abundance_bin"].astype(str).eq(abundance_bin))]
            if row.empty:
                continue
            row = row.iloc[0]
            ax.errorbar(
                x,
                row["mean_r"],
                yerr=[[row["mean_r"] - row["ci_low"]], [row["ci_high"] - row["mean_r"]]],
                fmt="D",
                markersize=4.3,
                color="#222222",
                ecolor="#222222",
                elinewidth=1.1,
                capsize=2.5,
                zorder=5,
            )

    handles = [
        mpl.lines.Line2D([0], [0], marker="o", linestyle="", color=COLORS[run], label=LABELS[run], markersize=5)
        for run in RUN_ORDER
    ]
    ax.legend(handles=handles, frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=0.9)
    ax.set_xticks(range(len(BIN_ORDER)))
    ax.set_xticklabels(BIN_ORDER)
    ax.set_xlabel("ASV abundance stratum")
    ax.set_ylabel("Plant mean ASV trend Pearson r")
    ax.set_title("ASV trend correlation by abundance stratum")
    style_axis(ax)
    fig.tight_layout()

    paths = []
    for fmt in formats:
        path = output_dir / f"base_asv_trend_r_by_abundance_stratum.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_abundance_strata_slim(summary: pd.DataFrame, output_dir: Path, formats: Sequence[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_summary = summary.copy()
    plot_summary["run_label"] = plot_summary["run_label"].astype(str)
    plot_summary["abundance_bin"] = plot_summary["abundance_bin"].astype(str)
    y_pos = {run: idx for idx, run in enumerate(reversed(RUN_ORDER))}
    x_pos = {abundance_bin: idx for idx, abundance_bin in enumerate(BIN_ORDER)}

    fig, ax = plt.subplots(figsize=(2.9, 4.6))
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(0.18, float(plot_summary["mean_r"].max())))
    cmap = mpl.cm.Blues

    for _, row in plot_summary.iterrows():
        run_label = row["run_label"]
        abundance_bin = row["abundance_bin"]
        if run_label not in y_pos or abundance_bin not in x_pos:
            continue
        value = float(row["mean_r"])
        ax.scatter(
            x_pos[abundance_bin],
            y_pos[run_label],
            s=260,
            color=cmap(norm(value)),
            edgecolor="#333333",
            linewidth=0.5,
            zorder=3,
        )
        text_color = "white" if value >= 0.11 else "#222222"
        ax.text(
            x_pos[abundance_bin],
            y_pos[run_label],
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=7.5,
            color=text_color,
            zorder=4,
        )

    ax.set_xticks(range(len(BIN_ORDER)))
    ax.set_xticklabels(["Low", "Med.", "High"], fontsize=7.5)
    ax.set_yticks([y_pos[run] for run in RUN_ORDER])
    ax.set_yticklabels([LABELS[run] for run in RUN_ORDER], fontsize=8)
    ax.set_xlim(-0.55, len(BIN_ORDER) - 0.45)
    ax.set_ylim(-0.65, len(RUN_ORDER) - 0.35)
    ax.set_xlabel("ASV abundance", fontsize=8)
    ax.set_title("Trend r by abundance", fontsize=9, fontweight="bold", pad=6)
    ax.grid(color="#D8D8D8", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Mean r", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)
    fig.tight_layout()

    paths = []
    for fmt in formats:
        path = output_dir / f"base_asv_trend_r_by_abundance_stratum_slim.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_abundance_strata_rotated(
    plant_bin: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    seed: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(3.9, 7.5))
    y_base = {name: idx for idx, name in enumerate(BIN_ORDER)}
    offsets = np.linspace(-0.31, 0.31, len(RUN_ORDER))
    offset_map = dict(zip(RUN_ORDER, offsets))

    for run_label in RUN_ORDER:
        sub = plant_bin[plant_bin["run_label"].eq(run_label)].copy()
        for abundance_bin in BIN_ORDER:
            points = sub[sub["abundance_bin"].astype(str).eq(abundance_bin)]
            if points.empty:
                continue
            y = y_base[abundance_bin] + offset_map[run_label]
            jitter = rng.uniform(-0.025, 0.025, size=len(points))
            ax.scatter(
                points["plant_bin_mean_r"],
                np.full(len(points), y) + jitter,
                s=15,
                color=COLORS[run_label],
                alpha=0.44,
                linewidths=0,
                rasterized=False,
            )
            row = summary[(summary["run_label"].astype(str).eq(run_label)) & (summary["abundance_bin"].astype(str).eq(abundance_bin))]
            if row.empty:
                continue
            row = row.iloc[0]
            ax.errorbar(
                row["mean_r"],
                y,
                xerr=[[row["mean_r"] - row["ci_low"]], [row["ci_high"] - row["mean_r"]]],
                fmt="D",
                markersize=4.0,
                color="#222222",
                ecolor="#222222",
                elinewidth=1.0,
                capsize=2.2,
                zorder=5,
            )

    handles = [
        mpl.lines.Line2D([0], [0], marker="o", linestyle="", color=COLORS[run], label=LABELS[run], markersize=4.8)
        for run in RUN_ORDER
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.31),
        columnspacing=0.9,
        handletextpad=0.35,
        fontsize=7.5,
    )
    ax.axvline(0, color="#777777", linestyle="--", linewidth=0.9)
    ax.set_yticks(range(len(BIN_ORDER)))
    ax.set_yticklabels(BIN_ORDER)
    ax.set_xlabel("Plant mean ASV trend Pearson r", fontsize=8.5, labelpad=5)
    ax.set_ylabel("ASV abundance stratum", fontsize=9)
    ax.set_title("ASV trend correlation by abundance stratum", fontsize=9.5, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.22, linewidth=0.8)
    fig.subplots_adjust(bottom=0.32)

    paths = []
    for fmt in formats:
        path = output_dir / f"base_asv_trend_r_by_abundance_stratum_rotated_scatter.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(args.predictions, args.include_other)
    abundance = assign_abundance_bins(predictions)
    correlations = compute_asv_correlations(predictions, abundance, args.min_std)
    plant_bin, summary = summarize_plant_bins(correlations, args.seed, args.n_bootstrap)
    tests = compute_pairwise_tests(plant_bin)
    paths = plot_abundance_strata(plant_bin, summary, args.output_dir, args.formats, args.seed)
    paths.extend(plot_abundance_strata_slim(summary, args.output_dir, args.formats))
    paths.extend(plot_abundance_strata_rotated(plant_bin, summary, args.output_dir, args.formats, args.seed))

    abundance.to_csv(args.table_dir / "base_asv_abundance_strata.csv", index=False)
    correlations.to_csv(args.table_dir / "base_asv_trend_r_by_abundance_asv.csv", index=False)
    plant_bin.to_csv(args.table_dir / "base_asv_trend_r_by_abundance_plant.csv", index=False)
    summary.to_csv(args.table_dir / "base_asv_trend_r_by_abundance_summary.csv", index=False)
    tests.to_csv(args.table_dir / "base_asv_trend_r_by_abundance_pairwise_tests.csv", index=False)

    print("Saved abundance strata:", args.table_dir / "base_asv_abundance_strata.csv")
    print("Saved plant-bin trend table:", args.table_dir / "base_asv_trend_r_by_abundance_plant.csv")
    print("Saved summary:", args.table_dir / "base_asv_trend_r_by_abundance_summary.csv")
    print("Saved pairwise tests:", args.table_dir / "base_asv_trend_r_by_abundance_pairwise_tests.csv")
    for path in paths:
        print("Saved figure:", path)
    print(summary.sort_values(["abundance_bin", "run_label"]).to_string(index=False))


if __name__ == "__main__":
    main()
