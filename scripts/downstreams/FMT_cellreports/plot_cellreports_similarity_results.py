#!/usr/bin/env python3
"""Plot Cell Reports donor-similarity classification results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

SCALE_ORDER = ["scale_small", "scale_medium", "scale_large", "scale_xlarge"]
SCALE_LABELS = {
    "scale_small": "MGM2-small",
    "scale_medium": "MGM2-medium",
    "scale_large": "MGM2-large",
    "scale_xlarge": "MGM2-xlarge",
}
COLORS = {
    "MOZAIC-repro": "#2A9D8F",
    "MGM2-small": "#7CB5EC",
    "MGM2-medium": "#4363D8",
    "MGM2-large": "#1A237E",
    "MGM2-xlarge": "#4A148C",
}
METRIC_LABELS = {
    "roc_auc": "ROC AUC",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro F1",
    "average_precision": "Average precision",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("outputs_downstreams/FMT_cellreports/community_similarity_random"),
        help="Directory containing scale_*/split_*/metrics.json files.",
    )
    parser.add_argument(
        "--mozaic_results_dir",
        type=Path,
        default=Path("outputs_downstreams/FMT_cellreports/mozaic_similarity_random"),
        help="Optional reproduced MOZAIC split_*/metrics.json directory.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("figures/FMT_cellreports"))
    parser.add_argument("--metric", type=str, default="roc_auc", choices=sorted(METRIC_LABELS))
    parser.add_argument(
        "--display_p",
        choices=[
            "wilcoxon_greater",
            "wilcoxon_two_sided",
            "paired_t_greater",
            "paired_t_two_sided",
            "signflip_greater",
            "signflip_two_sided",
            "holm",
        ],
        default="wilcoxon_greater",
        help="P value used for figure annotations.",
    )
    return parser.parse_args()


def parse_split(path: Path) -> int:
    match = re.search(r"split_(\d+)", str(path))
    return int(match.group(1)) if match else -1


def read_metrics(results_dir: Path, mozaic_results_dir: Path | None = None) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for scale in SCALE_ORDER:
        for path in sorted((results_dir / scale).glob("split_*/metrics.json")):
            with path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            row: Dict[str, object] = {
                "method": SCALE_LABELS[scale],
                "scale": scale,
                "split": parse_split(path),
                "metrics_file": str(path),
            }
            for metric in METRIC_LABELS:
                row[metric] = float(metrics.get(metric, np.nan))
            row["train_events"] = int(metrics.get("train_events", 0))
            row["val_events"] = int(metrics.get("val_events", 0))
            row["test_events"] = int(metrics.get("test_events", 0))
            rows.append(row)
    if mozaic_results_dir is not None and mozaic_results_dir.exists():
        for path in sorted(mozaic_results_dir.glob("split_*/metrics.json")):
            with path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            row = {
                "method": "MOZAIC-repro",
                "scale": "mozaic_repro",
                "split": parse_split(path),
                "metrics_file": str(path),
            }
            for metric in METRIC_LABELS:
                row[metric] = float(metrics.get(metric, np.nan))
            row["train_events"] = int(metrics.get("train_events", 0))
            row["val_events"] = int(metrics.get("val_events", 0))
            row["test_events"] = int(metrics.get("test_events", 0))
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {results_dir}")
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = list(METRIC_LABELS)
    summary = (
        metrics.groupby(["method", "scale"], observed=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns]
    order = {"MOZAIC-repro": 0}
    order.update({SCALE_LABELS[scale]: idx + 1 for idx, scale in enumerate(SCALE_ORDER)})
    summary["method_order"] = summary["method"].map(order)
    return summary.sort_values("method_order").drop(columns=["method_order"]).reset_index(drop=True)


def set_tight_ylim(ax, values: np.ndarray) -> None:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    ymin = float(finite.min())
    ymax = float(finite.max())
    span = max(ymax - ymin, 0.08)
    pad = span * 0.22
    ax.set_ylim(max(0.0, ymin - pad), min(1.02, ymax + pad))


def paired_signflip_pvalue(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return float("nan")
    observed = abs(float(differences.mean()))
    signs = np.asarray(np.meshgrid(*([[-1.0, 1.0]] * differences.size))).T.reshape(-1, differences.size)
    null_values = np.abs((signs * differences).mean(axis=1))
    return float((np.sum(null_values >= observed - 1e-12) + 1.0) / (len(null_values) + 1.0))


def paired_signflip_greater_pvalue(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return float("nan")
    observed = float(differences.mean())
    signs = np.asarray(np.meshgrid(*([[-1.0, 1.0]] * differences.size))).T.reshape(-1, differences.size)
    null_values = (signs * differences).mean(axis=1)
    return float((np.sum(null_values >= observed - 1e-12) + 1.0) / (len(null_values) + 1.0))


def paired_t_pvalues(differences: np.ndarray) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if differences.size < 2:
        return float("nan"), float("nan")
    two_sided = float(stats.ttest_1samp(differences, popmean=0.0, alternative="two-sided").pvalue)
    greater = float(stats.ttest_1samp(differences, popmean=0.0, alternative="greater").pvalue)
    return two_sided, greater


def wilcoxon_pvalues(differences: np.ndarray) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if differences.size < 2 or np.allclose(differences, 0.0):
        return float("nan"), float("nan")
    two_sided = float(stats.wilcoxon(differences, alternative="two-sided", zero_method="wilcox").pvalue)
    greater = float(stats.wilcoxon(differences, alternative="greater", zero_method="wilcox").pvalue)
    return two_sided, greater


def holm_adjust(p_values: List[float]) -> List[float]:
    indexed = [(idx, float(p)) for idx, p in enumerate(p_values)]
    valid = [(idx, p) for idx, p in indexed if np.isfinite(p)]
    adjusted = [float("nan")] * len(p_values)
    if not valid:
        return adjusted
    valid_sorted = sorted(valid, key=lambda item: item[1])
    running_max = 0.0
    m = len(valid_sorted)
    for rank, (idx, p_value) in enumerate(valid_sorted):
        candidate = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max
    return adjusted


def significance_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def paired_tests(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    baseline = metrics[metrics["method"] == "MOZAIC-repro"][["split", metric]].rename(columns={metric: "mozaic"})
    rows = []
    for method in [SCALE_LABELS[scale] for scale in SCALE_ORDER]:
        other = metrics[metrics["method"] == method][["split", metric]].rename(columns={metric: "method_value"})
        merged = baseline.merge(other, on="split", how="inner")
        if merged.empty:
            rows.append(
                {
                    "comparison": f"{method} - MOZAIC-repro",
                    "method": method,
                    "metric": metric,
                    "n_pairs": 0,
                    "mean_difference": float("nan"),
                    "p_signflip_two_sided": float("nan"),
                    "p_signflip_greater": float("nan"),
                    "p_t_two_sided": float("nan"),
                    "p_t_greater": float("nan"),
                    "p_wilcoxon_two_sided": float("nan"),
                    "p_wilcoxon_greater": float("nan"),
                }
            )
            continue
        differences = merged["method_value"].to_numpy(dtype=float) - merged["mozaic"].to_numpy(dtype=float)
        p_t_two_sided, p_t_greater = paired_t_pvalues(differences)
        p_wilcoxon_two_sided, p_wilcoxon_greater = wilcoxon_pvalues(differences)
        rows.append(
            {
                "comparison": f"{method} - MOZAIC-repro",
                "method": method,
                "metric": metric,
                "n_pairs": int(len(differences)),
                "mean_difference": float(np.mean(differences)),
                "p_signflip_two_sided": paired_signflip_pvalue(differences),
                "p_signflip_greater": paired_signflip_greater_pvalue(differences),
                "p_t_two_sided": p_t_two_sided,
                "p_t_greater": p_t_greater,
                "p_wilcoxon_two_sided": p_wilcoxon_two_sided,
                "p_wilcoxon_greater": p_wilcoxon_greater,
            }
        )
    tests = pd.DataFrame(rows)
    tests["p_holm_signflip_two_sided"] = holm_adjust(tests["p_signflip_two_sided"].tolist())
    return tests


def p_column_for_display(mode: str) -> str:
    return {
        "wilcoxon_greater": "p_wilcoxon_greater",
        "wilcoxon_two_sided": "p_wilcoxon_two_sided",
        "paired_t_greater": "p_t_greater",
        "paired_t_two_sided": "p_t_two_sided",
        "signflip_greater": "p_signflip_greater",
        "signflip_two_sided": "p_signflip_two_sided",
        "holm": "p_holm_signflip_two_sided",
    }[mode]


def test_label_for_display(mode: str) -> str:
    return {
        "wilcoxon_greater": "paired one-sided Wilcoxon signed-rank test",
        "wilcoxon_two_sided": "paired two-sided Wilcoxon signed-rank test",
        "paired_t_greater": "paired one-sided t-test",
        "paired_t_two_sided": "paired two-sided t-test",
        "signflip_greater": "paired one-sided sign-flip test",
        "signflip_two_sided": "paired two-sided sign-flip test",
        "holm": "paired two-sided sign-flip test, Holm adjusted",
    }[mode]


def add_significance_annotations(
    ax,
    methods: List[str],
    means: List[float],
    stds: List[float],
    tests: pd.DataFrame,
    display_p: str,
) -> None:
    if "MOZAIC-repro" not in methods:
        return
    baseline_idx = methods.index("MOZAIC-repro")
    finite_top = np.asarray(means, dtype=float) + np.nan_to_num(np.asarray(stds, dtype=float), nan=0.0)
    y_start = float(np.nanmax(finite_top)) + 0.018
    y_step = 0.018
    p_column = p_column_for_display(display_p)
    for row_idx, row in tests.iterrows():
        method = str(row["method"])
        if method not in methods:
            continue
        method_idx = methods.index(method)
        y = y_start + row_idx * y_step
        p_value = float(row[p_column])
        ax.plot([baseline_idx, baseline_idx, method_idx, method_idx], [y - 0.004, y, y, y - 0.004], color="#333333", linewidth=0.9)
        ax.text(
            (baseline_idx + method_idx) / 2,
            y + 0.002,
            f"{significance_label(p_value)} p={p_value:.3g}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#333333",
        )
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, max(ymax, y_start + max(len(tests), 1) * y_step + 0.018))


def plot_metric(metrics: pd.DataFrame, output_dir: Path, metric: str, tests: pd.DataFrame, display_p: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(metrics)
    methods = summary["method"].tolist()
    means = summary[f"{metric}_mean"].to_numpy(dtype=float).tolist()
    stds = summary[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float).tolist()
    counts = summary[f"{metric}_count"].fillna(0).astype(int).tolist()

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10.8, 5.9))
    bar_colors = [COLORS.get(method, "#777777") for method in methods]
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        color=bar_colors,
        edgecolor="#222222",
        linewidth=1.1,
        width=0.62,
        capsize=4,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )

    for idx, (mean, count) in enumerate(zip(means, counts)):
        ax.text(idx, mean + 0.012, f"{mean:.3f}\nn={count}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#222222")

    for method_idx, method in enumerate(methods):
        split_values = metrics.loc[metrics["method"] == method, metric].dropna().to_numpy(dtype=float)
        if split_values.size == 0:
            continue
        jitter = np.linspace(-0.16, 0.16, split_values.size)
        ax.scatter(
            np.full(split_values.size, method_idx) + jitter,
            split_values,
            s=26,
            color="white",
            edgecolor="#222222",
            linewidth=0.65,
            zorder=4,
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11, fontweight="bold", rotation=20, ha="right")
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
    ax.set_title("Cell Reports FMT Donor-Similarity Classification", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    set_tight_ylim(ax, np.asarray(means, dtype=float))
    add_significance_annotations(ax, methods, means, stds, tests, display_p)
    if metric == "roc_auc":
        ax.set_ylim(0.5, 1.0)
    ax.text(
        0.01,
        0.02,
        f"Disease-stratified 72/8/20 splits; error bars show SD. Figure p values: {test_label_for_display(display_p)} vs MOZAIC-repro.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"cellreports_similarity_{metric}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics = read_metrics(args.results_dir, args.mozaic_results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "cellreports_similarity_split_metrics.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(args.output_dir / "cellreports_similarity_summary.csv", index=False)
    tests = paired_tests(metrics, args.metric)
    tests.to_csv(args.output_dir / f"cellreports_similarity_{args.metric}_paired_tests.csv", index=False)
    plot_metric(metrics, args.output_dir, args.metric, tests, args.display_p)
    print(summary[["method", "roc_auc_mean", "roc_auc_std", "roc_auc_count"]].to_string(index=False))
    print(tests.to_string(index=False))
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
