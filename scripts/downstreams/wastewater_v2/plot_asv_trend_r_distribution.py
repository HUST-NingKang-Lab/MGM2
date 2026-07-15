#!/usr/bin/env python3
"""Plot per-ASV trend Pearson distributions for base wastewater forecasts."""

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
except ImportError:  # pragma: no cover - scipy is available in the project env.
    wilcoxon = None


DEFAULT_PREDICTIONS = Path("outputs_downstreams/wastewater_v2/base_token_statistics/base_token_last10_predictions_long.csv.gz")
DEFAULT_OUTPUT_DIR = Path("figures/wastewater_v2/base_token_statistics")
DEFAULT_TABLE_DIR = Path("outputs_downstreams/wastewater_v2/base_token_statistics")

RUN_ORDER = ["microprophet", "gru", "tcn", "token_small", "token_medium", "token_large", "token_xlarge"]
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
    "gru": "#74A57F",
    "tcn": "#246A73",
    "token_small": "#F2A65A",
    "token_medium": "#D96C45",
    "token_large": "#C4382B",
    "token_xlarge": "#1F4E9A",
}

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
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table_dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--min_std", type=float, default=1e-8)
    parser.add_argument("--jitter_width", type=float, default=0.16)
    parser.add_argument("--point_size", type=float, default=3.0)
    parser.add_argument("--point_alpha", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--formats", nargs="*", default=["png", "pdf"])
    return parser.parse_args()


def _safe_corr(true: np.ndarray, pred: np.ndarray, min_std: float = 1e-8) -> float:
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(true) & np.isfinite(pred)
    true = true[valid]
    pred = pred[valid]
    if true.size < 3 or np.std(true) < min_std or np.std(pred) < min_std:
        return float("nan")
    corr = np.corrcoef(true, pred)[0, 1]
    return float(corr) if np.isfinite(corr) else float("nan")


def compute_asv_trend_correlations(predictions: pd.DataFrame, min_std: float = 1e-8) -> pd.DataFrame:
    rows = []
    for (run_label, plant, asv_id), group in predictions.groupby(["run_label", "plant", "asv_id"], sort=False):
        group = group.sort_values("horizon")
        corr = _safe_corr(group["y_true"].to_numpy(), group["y_pred"].to_numpy(), min_std=min_std)
        if not np.isfinite(corr):
            continue
        true = group["y_true"].to_numpy(dtype=np.float64)
        pred = group["y_pred"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "run_label": run_label,
                "label": group["label"].iloc[0] if "label" in group.columns else LABELS.get(run_label, run_label),
                "plant": plant,
                "asv_id": asv_id,
                "trend_pearson": corr,
                "mean_true": float(np.mean(true)),
                "range_true": float(np.max(true) - np.min(true)),
                "std_true": float(np.std(true)),
                "mean_abs_error": float(np.mean(np.abs(pred - true))),
                "n_horizons": int(group["horizon"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def summarize_correlations(correlations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_label, group in correlations.groupby("run_label", sort=False):
        rows.append(
            {
                "run_label": run_label,
                "label": LABELS.get(run_label, group["label"].iloc[0]),
                "n_plant_asv": int(len(group)),
                "mean_r": float(group["trend_pearson"].mean()),
                "median_r": float(group["trend_pearson"].median()),
                "q25_r": float(group["trend_pearson"].quantile(0.25)),
                "q75_r": float(group["trend_pearson"].quantile(0.75)),
                "fraction_positive": float((group["trend_pearson"] > 0).mean()),
                "fraction_r_gt_0_5": float((group["trend_pearson"] > 0.5).mean()),
            }
        )
    return pd.DataFrame(rows)


def compute_box_statistics(correlations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_label, group in correlations.groupby("run_label", sort=False):
        values = group["trend_pearson"].to_numpy(dtype=float)
        rows.append(
            {
                "run_label": run_label,
                "label": LABELS.get(run_label, group["label"].iloc[0]),
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "q10": float(np.quantile(values, 0.10)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "q90": float(np.quantile(values, 0.90)),
            }
        )
    return pd.DataFrame(rows)


def compute_plant_mean_correlations(correlations: pd.DataFrame) -> pd.DataFrame:
    return (
        correlations.groupby(["run_label", "label", "plant"], as_index=False)
        .agg(
            plant_mean_r=("trend_pearson", "mean"),
            n_asv=("trend_pearson", "size"),
        )
    )


def _bh_fdr(p_values: Sequence[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return adjusted.tolist()
    valid_idx = np.where(valid)[0]
    values = p[valid]
    order = np.argsort(values)
    ranked = values[order]
    m = len(ranked)
    raw = ranked * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    adjusted[valid_idx[order]] = monotone
    return adjusted.tolist()


def _paired_wilcoxon_greater(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=np.float64)
    delta = delta[np.isfinite(delta)]
    delta = delta[np.abs(delta) > 1e-12]
    if delta.size == 0 or wilcoxon is None:
        return float("nan")
    try:
        return float(wilcoxon(delta, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def _bootstrap_mean_ci(delta_by_unit: pd.Series, seed: int, n_bootstrap: int = 10000) -> tuple[float, float]:
    values = delta_by_unit.dropna().to_numpy(dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def compute_pairwise_tests(
    correlations: pd.DataFrame,
    candidate: str = "token_xlarge",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare candidate against each model using plant-level and ASV-level paired tests."""
    plant_mean = (
        correlations.groupby(["plant", "run_label"], as_index=False)
        .agg(trend_pearson=("trend_pearson", "mean"))
        .pivot(index="plant", columns="run_label", values="trend_pearson")
    )
    asv_values = correlations.pivot_table(index=["plant", "asv_id"], columns="run_label", values="trend_pearson", aggfunc="first")
    references = [run for run in RUN_ORDER if run != candidate and run in set(correlations["run_label"])]

    plant_rows = []
    asv_rows = []
    for idx, reference in enumerate(references):
        plant_pair = plant_mean[[candidate, reference]].dropna()
        plant_delta = plant_pair[candidate] - plant_pair[reference]
        ci_low, ci_high = _bootstrap_mean_ci(plant_delta, seed=seed + idx)
        plant_rows.append(
            {
                "candidate": candidate,
                "candidate_label": LABELS[candidate],
                "reference": reference,
                "reference_label": LABELS[reference],
                "level": "plant_mean",
                "n_pairs": int(len(plant_delta)),
                "mean_delta": float(plant_delta.mean()),
                "median_delta": float(plant_delta.median()),
                "bootstrap_mean_delta_ci_low": ci_low,
                "bootstrap_mean_delta_ci_high": ci_high,
                "fraction_positive_delta": float((plant_delta > 0).mean()),
                "wilcoxon_greater_p": _paired_wilcoxon_greater(plant_delta.to_numpy()),
            }
        )

        asv_pair = asv_values[[candidate, reference]].dropna()
        asv_delta = asv_pair[candidate] - asv_pair[reference]
        asv_rows.append(
            {
                "candidate": candidate,
                "candidate_label": LABELS[candidate],
                "reference": reference,
                "reference_label": LABELS[reference],
                "level": "plant_asv",
                "n_pairs": int(len(asv_delta)),
                "mean_delta": float(asv_delta.mean()),
                "median_delta": float(asv_delta.median()),
                "fraction_positive_delta": float((asv_delta > 0).mean()),
                "wilcoxon_greater_p": _paired_wilcoxon_greater(asv_delta.to_numpy()),
            }
        )

    plant_tests = pd.DataFrame(plant_rows)
    asv_tests = pd.DataFrame(asv_rows)
    if not plant_tests.empty:
        plant_tests["wilcoxon_greater_fdr"] = _bh_fdr(plant_tests["wilcoxon_greater_p"].tolist())
    if not asv_tests.empty:
        asv_tests["wilcoxon_greater_fdr"] = _bh_fdr(asv_tests["wilcoxon_greater_p"].tolist())
    return plant_tests, asv_tests


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)


def compute_plant_delta_points(correlations: pd.DataFrame, candidate: str = "token_xlarge") -> pd.DataFrame:
    plant_mean = (
        correlations.groupby(["plant", "run_label"], as_index=False)
        .agg(trend_pearson=("trend_pearson", "mean"))
        .pivot(index="plant", columns="run_label", values="trend_pearson")
    )
    rows = []
    references = [run for run in RUN_ORDER if run != candidate and run in plant_mean.columns]
    for reference in references:
        pair = plant_mean[[candidate, reference]].dropna()
        for plant, row in pair.iterrows():
            rows.append(
                {
                    "reference": reference,
                    "reference_label": LABELS[reference],
                    "plant": plant,
                    "candidate_r": float(row[candidate]),
                    "reference_r": float(row[reference]),
                    "delta_r": float(row[candidate] - row[reference]),
                }
            )
    return pd.DataFrame(rows)


def plot_plant_delta(
    delta_points: pd.DataFrame,
    plant_tests: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    jitter_width: float,
    point_size: float,
    seed: int,
) -> list[Path]:
    references = [run for run in RUN_ORDER if run in set(delta_points["reference"])]
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    rng = np.random.default_rng(seed)
    positions = np.arange(1, len(references) + 1)
    for position, reference in zip(positions, references):
        sub = delta_points[delta_points["reference"].eq(reference)].copy()
        jitter = rng.uniform(-jitter_width, jitter_width, size=len(sub))
        ax.scatter(
            np.full(len(sub), position) + jitter,
            sub["delta_r"],
            s=point_size,
            color=COLORS[reference],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
        )
        test = plant_tests[plant_tests["reference"].eq(reference)].iloc[0]
        ax.errorbar(
            position,
            test["mean_delta"],
            yerr=[
                [test["mean_delta"] - test["bootstrap_mean_delta_ci_low"]],
                [test["bootstrap_mean_delta_ci_high"] - test["mean_delta"]],
            ],
            fmt="D",
            markersize=5,
            color="#111111",
            ecolor="#111111",
            elinewidth=1.2,
            capsize=3,
            zorder=10,
        )
    ax.axhline(0, color="#333333", linewidth=0.9, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[run] for run in references], rotation=25, ha="right")
    ax.set_ylabel("Plant mean ASV trend r: MGM2 xlarge - reference")
    ax.set_title("Plant-level paired improvement in ASV temporal correlation")
    _style_axis(ax)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_dir / f"base_asv_trend_r_plant_delta.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_plant_mean_r(
    plant_mean: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    jitter_width: float,
    point_size: float,
    seed: int,
) -> list[Path]:
    plot_runs = [run for run in RUN_ORDER if run in set(plant_mean["run_label"])]
    fig, ax = plt.subplots(figsize=(9.6, 4.7))
    rng = np.random.default_rng(seed)
    positions = np.arange(1, len(plot_runs) + 1)
    for idx, run in enumerate(plot_runs):
        position = positions[idx]
        sub = plant_mean[plant_mean["run_label"].eq(run)].copy()
        jitter = rng.uniform(-jitter_width, jitter_width, size=len(sub))
        ax.scatter(
            np.full(len(sub), position) + jitter,
            sub["plant_mean_r"],
            s=point_size,
            color=COLORS[run],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.35,
        )
        mean = float(sub["plant_mean_r"].mean())
        ci_low, ci_high = _bootstrap_mean_ci(sub["plant_mean_r"], seed=seed + idx)
        ax.errorbar(
            position,
            mean,
            yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="D",
            markersize=5,
            color="#111111",
            ecolor="#111111",
            elinewidth=1.2,
            capsize=3,
            zorder=10,
        )
    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[run] for run in plot_runs], rotation=25, ha="right")
    ax.set_ylabel("Plant mean ASV trend Pearson r")
    ax.set_title("Plant-level ASV temporal correlation by model")
    _style_axis(ax)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_dir / f"base_asv_trend_r_by_plant_model.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_asv_point_summary(
    correlations: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    jitter_width: float,
    point_size: float,
    point_alpha: float,
    seed: int,
) -> list[Path]:
    plot_runs = [run for run in RUN_ORDER if run in set(correlations["run_label"])]
    fig, ax = plt.subplots(figsize=(9.6, 4.7))
    positions = np.arange(1, len(plot_runs) + 1)
    rng = np.random.default_rng(seed)
    for idx, run in enumerate(plot_runs):
        position = positions[idx]
        sub = correlations[correlations["run_label"].eq(run)].copy()
        jitter = rng.uniform(-jitter_width, jitter_width, size=len(sub))
        ax.scatter(
            np.full(len(sub), position) + jitter,
            sub["trend_pearson"],
            s=point_size,
            color=COLORS[run],
            alpha=point_alpha,
            edgecolor="none",
            linewidth=0,
        )
        mean = float(sub["trend_pearson"].mean())
        ci_low, ci_high = _bootstrap_mean_ci(sub["trend_pearson"], seed=seed + idx)
        ax.errorbar(
            position,
            mean,
            yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="D",
            markersize=5,
            color="#111111",
            ecolor="#111111",
            elinewidth=1.2,
            capsize=3,
            zorder=10,
        )
    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[run] for run in plot_runs], rotation=25, ha="right")
    ax.set_ylabel("ASV-level trend Pearson r")
    ax.set_title("ASV-level temporal correlation by model")
    ax.set_ylim(-1.05, 1.05)
    _style_axis(ax)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_dir / f"base_asv_trend_r_by_asv_model.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_ecdf(correlations: pd.DataFrame, output_dir: Path, formats: Sequence[str]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    for run in [run for run in RUN_ORDER if run in set(correlations["run_label"])]:
        values = np.sort(correlations.loc[correlations["run_label"].eq(run), "trend_pearson"].to_numpy(dtype=float))
        y = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, y, color=COLORS[run], linewidth=1.8, label=LABELS[run])
    ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Per plant-ASV trend Pearson r")
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Cumulative distribution of ASV-level temporal correlation")
    ax.legend(frameon=False, loc="lower right")
    _style_axis(ax)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_dir / f"base_asv_trend_r_ecdf.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_violin_jitter(correlations: pd.DataFrame, output_dir: Path, formats: Sequence[str], jitter_width: float, point_size: float, point_alpha: float, seed: int) -> list[Path]:
    plot_runs = [run for run in RUN_ORDER if run in set(correlations["run_label"])]
    data = [correlations.loc[correlations["run_label"].eq(run), "trend_pearson"].to_numpy() for run in plot_runs]
    labels = [LABELS[run] for run in plot_runs]
    colors = [COLORS[run] for run in plot_runs]

    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    positions = np.arange(1, len(plot_runs) + 1)
    violin = ax.violinplot(data, positions=positions, widths=0.74, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.34)
        body.set_linewidth(0.7)

    rng = np.random.default_rng(seed)
    for position, values, color in zip(positions, data, colors):
        jitter = rng.uniform(-jitter_width, jitter_width, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=point_size,
            color=color,
            alpha=point_alpha,
            linewidths=0,
        )
        median = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        ax.plot([position - 0.22, position + 0.22], [median, median], color="#111111", linewidth=1.4)
        ax.plot([position, position], [q25, q75], color="#111111", linewidth=1.0)

    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Per plant-ASV trend Pearson r")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Distribution of ASV-level temporal correlation across all plants")
    _style_axis(ax)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_dir / f"base_asv_trend_r_violin_jitter.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    predictions = predictions[predictions["run_label"].isin(RUN_ORDER)].copy()
    correlations = compute_asv_trend_correlations(predictions, min_std=args.min_std)
    summary = summarize_correlations(correlations)
    box_stats = compute_box_statistics(correlations)
    plant_tests, asv_tests = compute_pairwise_tests(correlations, candidate="token_xlarge", seed=args.seed)
    plant_mean = compute_plant_mean_correlations(correlations)
    delta_points = compute_plant_delta_points(correlations, candidate="token_xlarge")

    correlations.to_csv(args.table_dir / "base_asv_trend_r_by_plant_asv.csv", index=False)
    summary.to_csv(args.table_dir / "base_asv_trend_r_summary.csv", index=False)
    box_stats.to_csv(args.table_dir / "base_asv_trend_r_box_statistics.csv", index=False)
    plant_tests.to_csv(args.table_dir / "base_asv_trend_r_pairwise_tests_by_plant.csv", index=False)
    asv_tests.to_csv(args.table_dir / "base_asv_trend_r_pairwise_tests_by_plant_asv.csv", index=False)
    plant_mean.to_csv(args.table_dir / "base_asv_trend_r_by_plant_model.csv", index=False)
    delta_points.to_csv(args.table_dir / "base_asv_trend_r_plant_delta_points.csv", index=False)
    paths = []
    paths.extend(plot_plant_mean_r(plant_mean, args.output_dir, args.formats, args.jitter_width, max(args.point_size * 5.0, 14.0), args.seed))
    paths.extend(plot_ecdf(correlations, args.output_dir, args.formats))

    print("Saved correlation table:", args.table_dir / "base_asv_trend_r_by_plant_asv.csv")
    print("Saved summary table:", args.table_dir / "base_asv_trend_r_summary.csv")
    print("Saved box statistics:", args.table_dir / "base_asv_trend_r_box_statistics.csv")
    print("Saved plant-level tests:", args.table_dir / "base_asv_trend_r_pairwise_tests_by_plant.csv")
    print("Saved plant-ASV-level tests:", args.table_dir / "base_asv_trend_r_pairwise_tests_by_plant_asv.csv")
    print("Saved plant model means:", args.table_dir / "base_asv_trend_r_by_plant_model.csv")
    print("Saved plant delta points:", args.table_dir / "base_asv_trend_r_plant_delta_points.csv")
    print(summary.sort_values("median_r", ascending=False).to_string(index=False))
    print("\nPlant-level paired Wilcoxon tests: MGM2 xlarge > reference")
    print(plant_tests.sort_values("wilcoxon_greater_fdr").to_string(index=False))
    for path in paths:
        print("Saved figure:", path)


if __name__ == "__main__":
    main()
