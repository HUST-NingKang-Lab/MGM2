#!/usr/bin/env python3
"""Summarize and plot FMT community prediction results."""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.FMTDataset import FMTCommunityPredictionDataset
from train_mlp_baseline import build_arrays, bray_curtis, cosine, finalize_raw_predictions

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

SCALE_ORDER = ["scale_small", "scale_medium", "scale_large", "scale_xlarge"]
SCALE_LABELS = {
    "scale_small": "MGM2-small",
    "scale_medium": "MGM2-medium",
    "scale_large": "MGM2-large",
    "scale_xlarge": "MGM2-xlarge",
}
METHOD_ORDER = ["Recipient", "Mean-post", "MLP", "MGM2-small", "MGM2-medium", "MGM2-large", "MGM2-xlarge"]
SCALE_COLORS = {
    "MGM2-small": "#7CB5EC",
    "MGM2-medium": "#4363D8",
    "MGM2-large": "#1A237E",
    "MGM2-xlarge": "#4A148C",
}
COLORS = {
    "Recipient": "#777777",
    "Mean-post": "#B06B3A",
    "MLP": "#2A9D8F",
    **SCALE_COLORS,
}


def bootstrap_ci(values: Iterable[float], n_boot: int = 2000, seed: int = 7) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        means[idx] = rng.choice(arr, size=arr.size, replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def set_tight_ylim(ax, values: Iterable[float], min_span: float = 0.02, pad_frac: float = 0.18) -> None:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    ymin = float(arr.min())
    ymax = float(arr.max())
    span = max(ymax - ymin, min_span)
    pad = span * pad_frac
    ax.set_ylim(ymin - pad, ymax + pad)


def load_mgm2_metrics(root: Path) -> pd.DataFrame:
    frames = []
    for scale in SCALE_ORDER:
        scale_dir = root / scale
        if not scale_dir.exists():
            continue
        for metrics_path in sorted(scale_dir.glob("seed_*/predicted_abundance.metrics.csv")):
            seed = metrics_path.parent.name.replace("seed_", "")
            frame = pd.read_csv(metrics_path)
            frame["scale"] = scale
            frame["seed"] = int(seed) if seed.isdigit() else seed
            frame["method"] = SCALE_LABELS.get(scale, scale)
            frame["bray_curtis"] = frame["mgm2_bray_curtis"].astype(float)
            frame["cosine"] = frame["mgm2_cosine"].astype(float)
            frame["recipient_bray_curtis"] = frame["recipient_baseline_bray_curtis"].astype(float)
            frame["recipient_cosine"] = frame["recipient_baseline_cosine"].astype(float)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No MGM2 metrics found under {root}")
    return pd.concat(frames, ignore_index=True)


def parse_test_metrics_from_log(log_path: Path) -> Dict[str, float]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metric_patterns = {
        "bray_curtis": r"test/raw_bray_curtis\s*│\s*([0-9.eE+-]+)",
        "cosine": r"test/raw_cosine\s*│\s*([0-9.eE+-]+)",
        "recipient_bray_curtis": r"test/recipient_baseline_bray_curtis\s*│\s*([0-9.eE+-]+)",
    }
    parsed = {}
    for key, pattern in metric_patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            parsed[key] = float(matches[-1])
    return parsed


def load_mgm2_run_summary(root: Path, prediction_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scale in SCALE_ORDER:
        scale_dir = root / scale
        if not scale_dir.exists():
            continue
        for seed_dir in sorted(scale_dir.glob("seed_*")):
            metrics_path = seed_dir / "predicted_abundance.metrics.csv"
            if not metrics_path.exists():
                continue
            seed_raw = seed_dir.name.replace("seed_", "")
            seed = int(seed_raw) if seed_raw.isdigit() else seed_raw
            parsed = parse_test_metrics_from_log(seed_dir / "train.log")
            source = "trainer_test_log" if {"bray_curtis", "cosine"}.issubset(parsed) else "prediction_csv"
            if source == "prediction_csv":
                frame = prediction_metrics[(prediction_metrics["scale"] == scale) & (prediction_metrics["seed"] == seed)]
                parsed = {
                    "bray_curtis": float(frame["bray_curtis"].mean()),
                    "cosine": float(frame["cosine"].mean()),
                    "recipient_bray_curtis": float(frame["recipient_bray_curtis"].mean()),
                }
            rows.append(
                {
                    "method": SCALE_LABELS.get(scale, scale),
                    "scale": scale,
                    "seed": seed,
                    "bray_curtis": parsed["bray_curtis"],
                    "cosine": parsed["cosine"],
                    "recipient_bray_curtis": parsed.get("recipient_bray_curtis", np.nan),
                    "source": source,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No MGM2 run summaries could be built under {root}")
    return pd.DataFrame(rows)


def _load_one_mlp_metrics(path: Path, seed: int | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["method"] = "MLP"
    frame["bray_curtis"] = frame["mlp_bray_curtis"].astype(float)
    frame["cosine"] = frame["mlp_cosine"].astype(float)
    frame["recipient_bray_curtis"] = frame["recipient_baseline_bray_curtis"].astype(float)
    frame["recipient_cosine"] = frame["recipient_baseline_cosine"].astype(float)
    frame["scale"] = "mlp"
    frame["seed"] = seed
    return frame


def load_mlp_metrics(root: Path, legacy_path: Path | None = None) -> pd.DataFrame:
    seed_paths = sorted(root.glob("seed_*/predicted_abundance.metrics.csv")) if root.exists() else []
    frames = []
    if seed_paths:
        for path in seed_paths:
            seed_raw = path.parent.name.replace("seed_", "")
            seed = int(seed_raw) if seed_raw.isdigit() else seed_raw
            frames.append(_load_one_mlp_metrics(path, seed))
    elif legacy_path is not None and legacy_path.exists():
        frames.append(_load_one_mlp_metrics(legacy_path, -1))
    elif root.is_file():
        frames.append(_load_one_mlp_metrics(root, -1))
    if not frames:
        searched = f"{root}/seed_*/predicted_abundance.metrics.csv"
        if legacy_path is not None:
            searched += f" or {legacy_path}"
        raise FileNotFoundError(f"No MLP metrics found at {searched}")
    return pd.concat(frames, ignore_index=True)


def compute_mean_baseline(args) -> pd.DataFrame:
    train_dataset = FMTCommunityPredictionDataset(
        args.train_data_file,
        args.train_metadata_file,
        args.ntv3_embedding_path,
    )
    test_dataset = FMTCommunityPredictionDataset(
        args.test_data_file,
        args.test_metadata_file,
        args.test_ntv3_embedding_path,
    )
    train_arrays = build_arrays(train_dataset)
    test_arrays = build_arrays(test_dataset)
    mean_raw = train_arrays.post_raw.mean(axis=0, keepdims=True)
    mean_raw = finalize_raw_predictions(mean_raw, raw_zero_threshold=0.0, renormalize=True)
    mean_pred = np.repeat(mean_raw, test_arrays.post_raw.shape[0], axis=0)
    frame = test_arrays.metadata.copy()
    frame["method"] = "Mean-post"
    frame["bray_curtis"] = bray_curtis(mean_pred, test_arrays.post_raw)
    frame["cosine"] = cosine(mean_pred, test_arrays.post_raw)
    frame["recipient_bray_curtis"] = bray_curtis(test_arrays.recipient_raw, test_arrays.post_raw)
    frame["recipient_cosine"] = cosine(test_arrays.recipient_raw, test_arrays.post_raw)
    frame["scale"] = "mean"
    frame["seed"] = -1
    return frame


def build_plot_tables(mgm2: pd.DataFrame, mlp: pd.DataFrame, mean_base: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid_targets = set(mlp["target_id"].astype(str))
    mgm2 = mgm2[mgm2["target_id"].astype(str).isin(valid_targets)].copy()
    metadata_cols = [
        "target_id",
        "fmt_id",
        "study",
        "response_label",
        "donor_sample_id",
        "recipient_sample_id",
        "post_sample_id",
        "timepoint_fmt",
    ]
    value_cols = metadata_cols + ["method", "scale", "seed", "bray_curtis", "cosine", "recipient_bray_curtis", "recipient_cosine"]

    recipient = mlp[metadata_cols + ["recipient_bray_curtis", "recipient_cosine"]].drop_duplicates("target_id").copy()
    recipient["method"] = "Recipient"
    recipient["scale"] = "recipient"
    recipient["seed"] = -1
    recipient["bray_curtis"] = recipient["recipient_bray_curtis"]
    recipient["cosine"] = recipient["recipient_cosine"]

    all_runs = pd.concat(
        [
            recipient[value_cols],
            mean_base[value_cols],
            mlp[value_cols],
            mgm2[value_cols],
        ],
        ignore_index=True,
    )
    grouped = []
    for method, frame in all_runs.groupby("method", sort=False):
        if method.startswith("MGM2-"):
            averaged = (
                frame.groupby(metadata_cols, dropna=False)[["bray_curtis", "cosine", "recipient_bray_curtis", "recipient_cosine"]]
                .mean()
                .reset_index()
            )
            averaged["method"] = method
            averaged["scale"] = frame["scale"].iloc[0]
            averaged["seed"] = "mean_seed"
            grouped.append(averaged[value_cols])
        else:
            grouped.append(frame[value_cols])
    per_target = pd.concat(grouped, ignore_index=True)
    return all_runs, per_target


def summarize_methods(per_target: pd.DataFrame, run_level: pd.DataFrame, mgm2_test_runs: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        frame = per_target[per_target["method"] == method]
        run_frame = run_level[run_level["method"] == method]
        if method.startswith("MGM2-") and mgm2_test_runs is not None:
            test_run_frame = mgm2_test_runs[mgm2_test_runs["method"] == method]
        else:
            test_run_frame = pd.DataFrame()
        if frame.empty and test_run_frame.empty:
            continue
        if method.startswith("MGM2-") and not test_run_frame.empty:
            bray_values = test_run_frame["bray_curtis"]
            cosine_values = test_run_frame["cosine"]
            bray_mean = float(bray_values.mean())
            cosine_mean = float(cosine_values.mean())
            bray_low, bray_high = bootstrap_ci(bray_values, n_boot=2000)
            cos_low, cos_high = bootstrap_ci(cosine_values, n_boot=2000)
            recipient_bray_mean = float(test_run_frame["recipient_bray_curtis"].mean())
            recipient_cosine_mean = float(per_target[per_target["method"] == "Recipient"]["cosine"].mean())
            n_runs = int(test_run_frame.shape[0])
            n_targets = int(per_target[per_target["method"] == "Recipient"]["target_id"].nunique())
        else:
            bray_mean = float(frame["bray_curtis"].mean())
            cosine_mean = float(frame["cosine"].mean())
            bray_low, bray_high = bootstrap_ci(frame["bray_curtis"])
            cos_low, cos_high = bootstrap_ci(frame["cosine"])
            recipient_bray_mean = float(frame["recipient_bray_curtis"].mean())
            recipient_cosine_mean = float(frame["recipient_cosine"].mean())
            n_runs = int(run_frame.groupby("seed").ngroups)
            n_targets = int(frame["target_id"].nunique())
        rows.append(
            {
                "method": method,
                "n_targets": n_targets,
                "n_runs": n_runs,
                "bray_curtis_mean": bray_mean,
                "bray_curtis_ci_low": bray_low,
                "bray_curtis_ci_high": bray_high,
                "cosine_mean": cosine_mean,
                "cosine_ci_low": cos_low,
                "cosine_ci_high": cos_high,
                "mean_bray_improvement_vs_recipient": recipient_bray_mean - bray_mean,
                "mean_cosine_improvement_vs_recipient": cosine_mean - recipient_cosine_mean,
                "summary_source": "trainer_test_log" if method.startswith("MGM2-") and not test_run_frame.empty else "prediction_csv",
            }
        )
    return pd.DataFrame(rows)


def save_bar_with_ci(summary: pd.DataFrame, metric: str, ylabel: str, output: Path, lower_is_better: bool) -> None:
    frame = summary.copy()
    frame["method"] = pd.Categorical(frame["method"], METHOD_ORDER, ordered=True)
    frame = frame.sort_values("method")
    means = frame[f"{metric}_mean"].to_numpy()
    ci_low = frame[f"{metric}_ci_low"].to_numpy()
    ci_high = frame[f"{metric}_ci_high"].to_numpy()
    labels = frame["method"].tolist()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.bar(x, means, color=[COLORS.get(label, "#777777") for label in labels], edgecolor="#333333", linewidth=1.2, width=0.55)
    ax.errorbar(x, means, yerr=[means - ci_low, ci_high - means], fmt="none", ecolor="#222222", elinewidth=1.2, capsize=4)
    for i, value in enumerate(means):
        ax.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    set_tight_ylim(ax, np.concatenate([ci_low, ci_high, means]), min_span=0.025 if metric == "bray_curtis" else 0.05)
    style_axis(ax)
    direction = "lower is better" if lower_is_better else "higher is better"
    ax.set_title(f"FMT post-community prediction ({direction})", fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_scale_trend(run_level: pd.DataFrame, output: Path) -> None:
    mgm2 = run_level[run_level["method"].str.startswith("MGM2-")].copy()
    mgm2["scale"] = pd.Categorical(mgm2["scale"], SCALE_ORDER, ordered=True)
    mgm2 = mgm2.sort_values("scale")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "bray_curtis", "Bray-Curtis"),
        (axes[1], "cosine", "Cosine"),
    ]:
        summary = mgm2.groupby("scale", observed=False)[metric].agg(["mean", "std"]).reindex(SCALE_ORDER)
        x = np.arange(len(summary))
        for idx, scale in enumerate(SCALE_ORDER):
            vals = mgm2.loc[mgm2["scale"] == scale, metric].to_numpy()
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) else []
            ax.scatter(np.full(len(vals), idx) + jitter, vals, s=40, color=COLORS[SCALE_LABELS[scale]], edgecolor="#333333", linewidth=0.5, alpha=0.9)
        ax.plot(x, summary["mean"], color="#222222", linewidth=2.0)
        ax.errorbar(x, summary["mean"], yerr=summary["std"], fmt="none", ecolor="#222222", capsize=4, linewidth=1.0)
        for idx, value in enumerate(summary["mean"]):
            if np.isfinite(value):
                ax.text(idx, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#222222")
        ax.set_xticks(x)
        ax.set_xticklabels([SCALE_LABELS[scale].replace("MGM2-", "").replace("xlarge", "XLarge").title() for scale in SCALE_ORDER], fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        set_tight_ylim(ax, mgm2[metric], min_span=0.012 if metric == "bray_curtis" else 0.035)
        style_axis(ax)
    axes[0].set_title("Scale trend: Bray-Curtis ↓", fontsize=12, fontweight="bold")
    axes[1].set_title("Scale trend: Cosine ↑", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_improvement_box(per_target: pd.DataFrame, output: Path) -> None:
    frame = per_target[per_target["method"].isin(METHOD_ORDER[1:])].copy()
    frame["bray_improvement"] = frame["recipient_bray_curtis"] - frame["bray_curtis"]
    frame["cosine_improvement"] = frame["cosine"] - frame["recipient_cosine"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "bray_improvement", "Recipient Bray - method Bray"),
        (axes[1], "cosine_improvement", "Method cosine - recipient cosine"),
    ]:
        labels = [method for method in METHOD_ORDER[1:] if method in set(frame["method"])]
        values = [frame.loc[frame["method"] == method, metric].dropna().to_numpy() for method in labels]
        positions = np.arange(1, len(labels) + 1)
        box = ax.boxplot(values, positions=positions, patch_artist=True, showfliers=False)
        for patch, label in zip(box["boxes"], labels):
            patch.set_facecolor(COLORS.get(label, "#777777"))
            patch.set_alpha(0.75)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(1.0)
        ax.axhline(0.0, color="#222222", linewidth=1.0, linestyle="--")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        set_tight_ylim(ax, frame[metric], min_span=0.08)
        style_axis(ax)
    axes[0].set_title("Positive values improve Bray-Curtis", fontsize=12, fontweight="bold")
    axes[1].set_title("Positive values improve cosine", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_best_vs_mlp(per_target: pd.DataFrame, summary: pd.DataFrame, output: Path) -> str:
    mgm2_summary = summary[summary["method"].str.startswith("MGM2-")].sort_values("bray_curtis_mean")
    best_method = str(mgm2_summary.iloc[0]["method"])
    mlp = (
        per_target[per_target["method"] == "MLP"]
        .groupby("target_id", dropna=False)[["bray_curtis", "cosine"]]
        .mean()
        .reset_index()
        .rename(
            columns={"bray_curtis": "mlp_bray", "cosine": "mlp_cosine"}
        )
    )
    best = per_target[per_target["method"] == best_method][["target_id", "bray_curtis", "cosine", "study"]].rename(
        columns={"bray_curtis": "best_bray", "cosine": "best_cosine"}
    )
    merged = best.merge(mlp, on="target_id", how="inner")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    axes[0].scatter(merged["mlp_bray"], merged["best_bray"], s=30, alpha=0.75, color=COLORS[best_method], edgecolor="#202020", linewidth=0.3)
    lims = [
        min(merged["mlp_bray"].min(), merged["best_bray"].min()),
        max(merged["mlp_bray"].max(), merged["best_bray"].max()),
    ]
    axes[0].plot(lims, lims, color="#202020", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("MLP Bray-Curtis")
    axes[0].set_ylabel(f"{best_method} Bray-Curtis")
    axes[0].set_title("Below diagonal favors MGM2", fontsize=12, fontweight="bold")
    set_tight_ylim(axes[0], np.concatenate([merged["mlp_bray"].to_numpy(), merged["best_bray"].to_numpy()]), min_span=0.2)
    axes[0].set_xlim(axes[0].get_ylim())
    style_axis(axes[0])

    axes[1].scatter(merged["mlp_cosine"], merged["best_cosine"], s=30, alpha=0.75, color=COLORS[best_method], edgecolor="#202020", linewidth=0.3)
    lims = [
        min(merged["mlp_cosine"].min(), merged["best_cosine"].min()),
        max(merged["mlp_cosine"].max(), merged["best_cosine"].max()),
    ]
    axes[1].plot(lims, lims, color="#202020", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("MLP cosine")
    axes[1].set_ylabel(f"{best_method} cosine")
    axes[1].set_title("Above diagonal favors MGM2", fontsize=12, fontweight="bold")
    set_tight_ylim(axes[1], np.concatenate([merged["mlp_cosine"].to_numpy(), merged["best_cosine"].to_numpy()]), min_span=0.25)
    axes[1].set_xlim(axes[1].get_ylim())
    style_axis(axes[1])
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return best_method


def summarize_prediction_coverage(mgm2: pd.DataFrame, mlp: pd.DataFrame) -> pd.DataFrame:
    reference = set(mlp["target_id"].astype(str))
    rows = []
    for (method, scale, seed), frame in mgm2.groupby(["method", "scale", "seed"]):
        targets = set(frame["target_id"].astype(str))
        rows.append(
            {
                "method": method,
                "scale": scale,
                "seed": seed,
                "rows": int(len(frame)),
                "unique_targets": int(len(targets)),
                "reference_targets": int(len(reference)),
                "coverage_fraction": len(targets & reference) / len(reference),
                "extra_targets": int(len(targets - reference)),
                "missing_targets": int(len(reference - targets)),
            }
        )
    return pd.DataFrame(rows).sort_values(["scale", "seed"])


def save_per_study(per_target: pd.DataFrame, output: Path) -> None:
    methods = METHOD_ORDER
    frame = per_target[per_target["method"].isin(methods)].copy()
    study_counts = frame[frame["method"] == "Recipient"]["study"].value_counts()
    studies = study_counts.index.tolist()

    fig, axes = plt.subplots(2, 1, figsize=(max(12.5, len(studies) * 0.85), 9.0), sharex=True)
    for ax, metric, ylabel, title in [
        (axes[0], "bray_curtis", "Mean Bray-Curtis", "Per-study prediction: Bray-Curtis ↓"),
        (axes[1], "cosine", "Mean cosine", "Per-study prediction: Cosine ↑"),
    ]:
        pivot = frame.groupby(["study", "method"])[metric].mean().unstack("method").reindex(studies)
        pivot = pivot[[method for method in methods if method in pivot.columns]]
        x = np.arange(len(pivot))
        width = 0.86 / len(pivot.columns)
        for idx, method in enumerate(pivot.columns):
            ax.bar(
                x + (idx - (len(pivot.columns) - 1) / 2) * width,
                pivot[method],
                width=width,
                label=method,
                color=COLORS.get(method, "#777777"),
                edgecolor="#333333",
                linewidth=0.45,
            )
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        set_tight_ylim(ax, pivot.to_numpy().ravel(), min_span=0.08 if metric == "bray_curtis" else 0.18)
        style_axis(ax)
    axes[1].set_xticks(np.arange(len(studies)))
    axes[1].set_xticklabels([f"{study}\n(n={study_counts[study]})" for study in studies], rotation=35, ha="right", fontsize=9)
    axes[0].legend(frameon=False, ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.24))
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot FMT MGM2, MLP, recipient, and mean-post baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_dir = Path("data/downstreams/FMT")
    parser.add_argument("--community_prediction_dir", type=Path, default=Path("outputs_downstreams/FMT/community_prediction"))
    parser.add_argument("--mlp_dir", type=Path, default=Path("outputs_downstreams/FMT/mlp_baseline"))
    parser.add_argument("--mlp_metrics", type=Path, default=Path("outputs_downstreams/FMT/mlp_baseline/predicted_abundance.metrics.csv"))
    parser.add_argument("--train_data_file", type=Path, default=data_dir / "community_train.pkl")
    parser.add_argument("--train_metadata_file", type=Path, default=data_dir / "post_train_meta_split.csv")
    parser.add_argument("--test_data_file", type=Path, default=data_dir / "community_test.pkl")
    parser.add_argument("--test_metadata_file", type=Path, default=data_dir / "post_test_meta_split.csv")
    parser.add_argument("--ntv3_embedding_path", type=Path, default=data_dir / "community_train.resolved_embeddings.h5")
    parser.add_argument("--test_ntv3_embedding_path", type=Path, default=data_dir / "community_test.resolved_embeddings.h5")
    parser.add_argument("--output_dir", type=Path, default=Path("figures/FMT"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mgm2 = load_mgm2_metrics(args.community_prediction_dir)
    mlp = load_mlp_metrics(args.mlp_dir, args.mlp_metrics)
    mean_base = compute_mean_baseline(args)
    mgm2_test_runs = load_mgm2_run_summary(args.community_prediction_dir, mgm2)
    coverage = summarize_prediction_coverage(mgm2, mlp)
    run_level, per_target = build_plot_tables(mgm2, mlp, mean_base)

    prediction_run_summary = (
        run_level.groupby(["method", "scale", "seed"], dropna=False)[["bray_curtis", "cosine", "recipient_bray_curtis", "recipient_cosine"]]
        .mean()
        .reset_index()
    )
    run_summary = pd.concat(
        [
            prediction_run_summary[~prediction_run_summary["method"].str.startswith("MGM2-")].assign(source="prediction_csv"),
            mgm2_test_runs.assign(recipient_cosine=np.nan),
        ],
        ignore_index=True,
        sort=False,
    )
    summary = summarize_methods(per_target, run_level, mgm2_test_runs=mgm2_test_runs)
    summary["method"] = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)
    summary = summary.sort_values("method").reset_index(drop=True)

    per_target.to_csv(args.output_dir / "fmt_per_target_metrics.csv", index=False)
    run_summary.to_csv(args.output_dir / "fmt_run_summary.csv", index=False)
    prediction_run_summary.to_csv(args.output_dir / "fmt_prediction_csv_run_summary.csv", index=False)
    coverage.to_csv(args.output_dir / "fmt_prediction_coverage.csv", index=False)
    summary.to_csv(args.output_dir / "fmt_method_summary.csv", index=False)

    save_bar_with_ci(summary, "bray_curtis", "Mean Bray-Curtis distance", args.output_dir / "fmt_bray_curtis_by_method.png", lower_is_better=True)
    save_bar_with_ci(summary, "cosine", "Mean cosine similarity", args.output_dir / "fmt_cosine_by_method.png", lower_is_better=False)
    save_scale_trend(run_summary, args.output_dir / "fmt_mgm2_scale_trend.png")
    save_improvement_box(per_target, args.output_dir / "fmt_improvement_vs_recipient.png")
    best_method = save_best_vs_mlp(per_target, summary, args.output_dir / "fmt_best_mgm2_vs_mlp_per_sample.png")
    save_per_study(per_target, args.output_dir / "fmt_by_study_all_methods.png")
    save_per_study(per_target, args.output_dir / "fmt_bray_by_study.png")

    available = mgm2.groupby("scale")["seed"].nunique().reindex(SCALE_ORDER).fillna(0).astype(int)
    print("MGM2 runs by scale:")
    for scale, n_runs in available.items():
        print(f"  {scale}: {n_runs} seed runs")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Best MGM2 by mean Bray-Curtis: {best_method}")
    print(f"Wrote FMT figures and tables to {args.output_dir}")


if __name__ == "__main__":
    main()
