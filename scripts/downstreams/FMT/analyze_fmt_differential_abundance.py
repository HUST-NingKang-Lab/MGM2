#!/usr/bin/env python3
"""Analyze pre/post-FMT differential abundance recovery from prediction files."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is expected in the mwm env.
    stats = None

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

SCALE_LABELS = {
    "scale_small": "MGM2-small",
    "scale_medium": "MGM2-medium",
    "scale_large": "MGM2-large",
    "scale_xlarge": "MGM2-xlarge",
}
METHOD_ORDER = ["MLP", "MGM2-small", "MGM2-medium", "MGM2-large", "MGM2-xlarge"]
COLORS = {
    "MLP": "#2A9D8F",
    "MGM2-small": "#7CB5EC",
    "MGM2-medium": "#4363D8",
    "MGM2-large": "#1A237E",
    "MGM2-xlarge": "#4A148C",
}


def decode_array(values: np.ndarray) -> List[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def infer_method(path: Path) -> Tuple[str, str, int]:
    text = str(path)
    if "mlp_baseline" in text:
        seed = -1
        for part in path.parts:
            match = re.match(r"seed_(\d+)$", part)
            if match:
                seed = int(match.group(1))
                break
        return "MLP", "mlp", seed
    scale = next((name for name in SCALE_LABELS if name in path.parts), "unknown")
    seed = -1
    for part in path.parts:
        match = re.match(r"seed_(\d+)$", part)
        if match:
            seed = int(match.group(1))
            break
    return SCALE_LABELS.get(scale, scale), scale, seed


def find_prediction_files(community_prediction_dir: Path, mlp_prediction_dir: Path, mlp_prediction_file: Path) -> List[Path]:
    paths = []
    mlp_seed_paths = []
    if mlp_prediction_dir.exists():
        mlp_seed_paths = sorted(mlp_prediction_dir.glob("seed_*/predicted_abundance.h5"))
    if mlp_seed_paths:
        paths.extend(mlp_seed_paths)
    elif mlp_prediction_file.exists():
        paths.append(mlp_prediction_file)
    for scale in ["scale_small", "scale_medium", "scale_large", "scale_xlarge"]:
        paths.extend(sorted((community_prediction_dir / scale).glob("seed_*/predicted_abundance.h5")))
    return sorted(dict.fromkeys(paths))


def read_prediction_file(path: Path) -> Dict[str, Any]:
    method, scale, seed = infer_method(path)
    with h5py.File(path, "r") as handle:
        feature_ids = decode_array(handle["feature_ids"][:])
        metadata = {key: handle["metadata"][key][:] for key in handle["metadata"].keys()}
        metadata = {
            key: decode_array(value) if value.dtype.kind in {"O", "S", "U"} else value
            for key, value in metadata.items()
        }
        pred_group = handle["predictions"]
        pred_key = None
        for key in pred_group.keys():
            if key not in {"true_post_raw", "recipient_raw", "donor_raw"} and key.endswith("_post_raw"):
                pred_key = key
                break
        if pred_key is None and "pred_post_raw" in pred_group:
            pred_key = "pred_post_raw"
        if pred_key is None:
            raise KeyError(f"No predicted post-FMT abundance key found in {path}")
        arrays = {
            "recipient_raw": pred_group["recipient_raw"][:].astype(np.float64),
            "true_post_raw": pred_group["true_post_raw"][:].astype(np.float64),
            "pred_post_raw": pred_group[pred_key][:].astype(np.float64),
        }
    meta_frame = pd.DataFrame(metadata)
    meta_frame["method"] = method
    meta_frame["scale"] = scale
    meta_frame["seed"] = seed
    meta_frame["prediction_file"] = str(path)
    return {
        "method": method,
        "scale": scale,
        "seed": seed,
        "path": path,
        "feature_ids": feature_ids,
        "metadata": meta_frame,
        **arrays,
    }


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=np.float64)
    q = np.full_like(p, np.nan, dtype=np.float64)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    finite_idx = np.where(finite)[0]
    order = finite_idx[np.argsort(p[finite])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


def paired_wilcoxon_p(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    p_values = np.ones(before.shape[1], dtype=np.float64)
    if stats is None:
        return np.full(before.shape[1], np.nan, dtype=np.float64)
    for idx in range(before.shape[1]):
        diff = after[:, idx] - before[:, idx]
        if np.allclose(diff, 0.0):
            p_values[idx] = 1.0
            continue
        try:
            p_values[idx] = float(stats.wilcoxon(after[:, idx], before[:, idx], zero_method="wilcox").pvalue)
        except ValueError:
            p_values[idx] = 1.0
    return p_values


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    return correlation_stats(x, y)["spearman_r"]


def correlation_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    result = {
        "n": int(mask.sum()),
        "spearman_r": float("nan"),
        "spearman_p": float("nan"),
        "pearson_r": float("nan"),
        "pearson_p": float("nan"),
    }
    if mask.sum() < 3 or stats is None:
        return result
    x_valid = x[mask]
    y_valid = y[mask]
    if np.allclose(x_valid, x_valid[0]) or np.allclose(y_valid, y_valid[0]):
        return result
    spearman = stats.spearmanr(x_valid, y_valid)
    pearson = stats.pearsonr(x_valid, y_valid)
    result.update(
        {
            "spearman_r": float(spearman.correlation),
            "spearman_p": float(spearman.pvalue),
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
        }
    )
    return result


def cosine_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    numerator = np.sum(x * y, axis=1)
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    mask = denom > 0
    out[mask] = numerator[mask] / denom[mask]
    return out


def short_taxon_label(feature_id: str) -> str:
    parts = [part for part in str(feature_id).split(";") if part]
    if not parts:
        return str(feature_id)
    if len(parts) >= 2:
        return parts[-2] if parts[-2].lower() not in {"recipient", "donor"} else parts[-3]
    return parts[-1]


def summarize_taxa(record: Dict[str, Any], pseudocount: float) -> pd.DataFrame:
    recipient = record["recipient_raw"]
    true_post = record["true_post_raw"]
    pred_post = record["pred_post_raw"]
    feature_ids = record["feature_ids"]

    observed_delta = true_post - recipient
    predicted_delta = pred_post - recipient
    observed_p = paired_wilcoxon_p(recipient, true_post)
    predicted_p = paired_wilcoxon_p(recipient, pred_post)

    frame = pd.DataFrame(
        {
            "method": record["method"],
            "scale": record["scale"],
            "seed": record["seed"],
            "feature_id": feature_ids,
            "taxon_label": [short_taxon_label(x) for x in feature_ids],
            "mean_pre_recipient": recipient.mean(axis=0),
            "mean_true_post": true_post.mean(axis=0),
            "mean_pred_post": pred_post.mean(axis=0),
            "observed_delta_mean": observed_delta.mean(axis=0),
            "predicted_delta_mean": predicted_delta.mean(axis=0),
            "delta_mae": np.mean(np.abs(predicted_delta - observed_delta), axis=0),
            "pre_prevalence": (recipient > 0).mean(axis=0),
            "true_post_prevalence": (true_post > 0).mean(axis=0),
            "pred_post_prevalence": (pred_post > 0).mean(axis=0),
            "observed_wilcoxon_p": observed_p,
            "predicted_wilcoxon_p": predicted_p,
        }
    )
    frame["observed_wilcoxon_q"] = benjamini_hochberg(frame["observed_wilcoxon_p"])
    frame["predicted_wilcoxon_q"] = benjamini_hochberg(frame["predicted_wilcoxon_p"])
    frame["observed_log2fc"] = np.log2((frame["mean_true_post"] + pseudocount) / (frame["mean_pre_recipient"] + pseudocount))
    frame["predicted_log2fc"] = np.log2((frame["mean_pred_post"] + pseudocount) / (frame["mean_pre_recipient"] + pseudocount))
    frame["delta_sign_match"] = np.sign(frame["observed_delta_mean"]) == np.sign(frame["predicted_delta_mean"])
    return frame


def summarize_samples(record: Dict[str, Any], min_abs_delta: float) -> pd.DataFrame:
    observed_delta = record["true_post_raw"] - record["recipient_raw"]
    predicted_delta = record["pred_post_raw"] - record["recipient_raw"]
    meta = record["metadata"].copy()
    changed = np.abs(observed_delta) >= min_abs_delta
    matches = np.sign(observed_delta) == np.sign(predicted_delta)
    agreement = np.full(observed_delta.shape[0], np.nan, dtype=np.float64)
    for idx in range(observed_delta.shape[0]):
        if changed[idx].any():
            agreement[idx] = matches[idx, changed[idx]].mean()
    meta["observed_delta_l1"] = np.abs(observed_delta).sum(axis=1)
    meta["predicted_delta_l1"] = np.abs(predicted_delta).sum(axis=1)
    meta["delta_mae"] = np.abs(predicted_delta - observed_delta).mean(axis=1)
    meta["delta_cosine"] = cosine_rows(observed_delta, predicted_delta)
    meta["changed_taxa_sign_agreement"] = agreement
    meta["n_observed_changed_taxa"] = changed.sum(axis=1)
    return meta


def summarize_runs(taxon_stats: pd.DataFrame, sample_stats: pd.DataFrame, alpha: float, min_abs_log2fc: float, top_k: int) -> pd.DataFrame:
    rows = []
    for (method, scale, seed), frame in taxon_stats.groupby(["method", "scale", "seed"], sort=False):
        sample_frame = sample_stats[
            (sample_stats["method"] == method)
            & (sample_stats["scale"] == scale)
            & (sample_stats["seed"] == seed)
        ]
        observed_top = set(frame.nlargest(top_k, "observed_delta_mean", keep="all")["feature_id"])
        observed_top |= set(frame.nsmallest(top_k, "observed_delta_mean", keep="all")["feature_id"])
        predicted_top = set(frame.nlargest(top_k, "predicted_delta_mean", keep="all")["feature_id"])
        predicted_top |= set(frame.nsmallest(top_k, "predicted_delta_mean", keep="all")["feature_id"])
        changed = (frame["observed_wilcoxon_q"] <= alpha) & (frame["observed_log2fc"].abs() >= min_abs_log2fc)
        delta_corr = correlation_stats(frame["observed_delta_mean"].to_numpy(), frame["predicted_delta_mean"].to_numpy())
        log2fc_corr = correlation_stats(frame["observed_log2fc"].to_numpy(), frame["predicted_log2fc"].to_numpy())
        rows.append(
            {
                "method": method,
                "scale": scale,
                "seed": seed,
                "n_targets": int(sample_frame["target_id"].nunique()) if "target_id" in sample_frame else int(sample_frame.shape[0]),
                "n_features": int(frame.shape[0]),
                "n_observed_da_taxa": int(changed.sum()),
                "mean_delta_mae_per_taxon": float(frame["delta_mae"].mean()),
                "mean_delta_mae_per_sample": float(sample_frame["delta_mae"].mean()),
                "mean_delta_cosine_per_sample": float(sample_frame["delta_cosine"].mean()),
                "mean_changed_taxa_sign_agreement_per_sample": float(sample_frame["changed_taxa_sign_agreement"].mean()),
                "taxon_delta_spearman": delta_corr["spearman_r"],
                "taxon_delta_spearman_p": delta_corr["spearman_p"],
                "taxon_delta_pearson": delta_corr["pearson_r"],
                "taxon_delta_pearson_p": delta_corr["pearson_p"],
                "taxon_log2fc_spearman": log2fc_corr["spearman_r"],
                "taxon_log2fc_spearman_p": log2fc_corr["spearman_p"],
                "taxon_log2fc_pearson": log2fc_corr["pearson_r"],
                "taxon_log2fc_pearson_p": log2fc_corr["pearson_p"],
                "top_delta_taxa_overlap_fraction": len(observed_top & predicted_top) / max(1, len(observed_top)),
                "observed_da_taxa_sign_agreement": float(frame.loc[changed, "delta_sign_match"].mean()) if changed.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def aggregate_method_taxa(taxon_stats: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = taxon_stats.select_dtypes(include=[np.number, bool]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "seed"]
    grouped = (
        taxon_stats.groupby(["method", "feature_id", "taxon_label"], sort=False)[numeric_cols]
        .mean()
        .reset_index()
    )
    return grouped


def summarize_method_correlations(method_taxa: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        frame = method_taxa[method_taxa["method"] == method]
        if frame.empty:
            continue
        log2fc_corr = correlation_stats(frame["observed_log2fc"].to_numpy(), frame["predicted_log2fc"].to_numpy())
        delta_corr = correlation_stats(frame["observed_delta_mean"].to_numpy(), frame["predicted_delta_mean"].to_numpy())
        rows.append(
            {
                "method": method,
                "n_taxa": log2fc_corr["n"],
                "log2fc_spearman_r": log2fc_corr["spearman_r"],
                "log2fc_spearman_p": log2fc_corr["spearman_p"],
                "log2fc_pearson_r": log2fc_corr["pearson_r"],
                "log2fc_pearson_p": log2fc_corr["pearson_p"],
                "delta_spearman_r": delta_corr["spearman_r"],
                "delta_spearman_p": delta_corr["spearman_p"],
                "delta_pearson_r": delta_corr["pearson_r"],
                "delta_pearson_p": delta_corr["pearson_p"],
            }
        )
    return pd.DataFrame(rows)


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.28)


def format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 1e-300:
        return "<1e-300"
    if value < 1e-3:
        return f"{value:.1e}"
    return f"{value:.3f}"


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def save_summary_bars(run_summary: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("taxon_log2fc_spearman", "Taxon log2FC Spearman"),
        ("mean_delta_cosine_per_sample", "Sample delta cosine"),
        ("top_delta_taxa_overlap_fraction", "Top delta taxa overlap"),
    ]
    methods = [method for method in METHOD_ORDER if method in set(run_summary["method"])]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14.5, 4.8), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        values = []
        for method in methods:
            vals = run_summary.loc[run_summary["method"] == method, metric].dropna().to_numpy()
            values.append(vals)
        means = np.asarray([vals.mean() if vals.size else np.nan for vals in values])
        stds = np.asarray([vals.std(ddof=1) if vals.size > 1 else 0.0 for vals in values])
        x = np.arange(len(methods))
        ax.bar(x, means, yerr=stds, color=[COLORS.get(m, "#777777") for m in methods], edgecolor="#333333", linewidth=1.0, capsize=4)
        for idx, vals in enumerate(values):
            if vals.size:
                jitter = np.linspace(-0.08, 0.08, vals.size)
                ax.scatter(np.full(vals.size, idx) + jitter, vals, s=28, color="white", edgecolor="#222222", linewidth=0.65, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel)
        style_axis(ax)
    fig.suptitle("Recovery of pre-to-post FMT abundance changes", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_delta_pearson_scatter(ax, frame: pd.DataFrame, method: str, method_correlations: pd.DataFrame) -> None:
    frame = frame.copy()
    if frame.empty:
        return
    corr_lookup = method_correlations.set_index("method") if not method_correlations.empty else pd.DataFrame()
    if method in corr_lookup.index:
        pearson_r = float(corr_lookup.loc[method, "delta_pearson_r"])
        pearson_p = float(corr_lookup.loc[method, "delta_pearson_p"])
        n_taxa = int(corr_lookup.loc[method, "n_taxa"])
    else:
        corr = correlation_stats(frame["observed_delta_mean"].to_numpy(), frame["predicted_delta_mean"].to_numpy())
        pearson_r = corr["pearson_r"]
        pearson_p = corr["pearson_p"]
        n_taxa = corr["n"]

    ax.scatter(
        frame["observed_delta_mean"],
        frame["predicted_delta_mean"],
        s=17,
        alpha=0.50,
        color=COLORS.get(method, "#777777"),
        edgecolor="none",
    )
    lim = float(np.nanmax(np.abs(np.r_[frame["observed_delta_mean"], frame["predicted_delta_mean"]])))
    lim = max(lim, 1e-4)
    ax.plot([-lim, lim], [-lim, lim], color="#222222", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.axvline(0.0, color="#777777", linewidth=0.8)
    ax.text(
        0.04,
        0.96,
        f"Pearson r = {pearson_r:.3f}\np = {format_p_value(pearson_p)}\nn = {n_taxa}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.90},
    )
    ax.set_xlim(-lim * 1.08, lim * 1.08)
    ax.set_ylim(-lim * 1.08, lim * 1.08)
    ax.set_aspect("equal", adjustable="box")
    try:
        ax.set_box_aspect(1)
    except AttributeError:
        pass
    ax.set_title(method, fontsize=11, fontweight="bold")
    style_axis(ax)


def save_delta_pearson_summary(method_taxa: pd.DataFrame, method_correlations: pd.DataFrame, output: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in set(method_taxa["method"])]
    if not methods:
        return
    n_cols = min(3, len(methods))
    n_rows = int(math.ceil(len(methods) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.7 * n_cols, 4.7 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, method in zip(axes.ravel(), methods):
        ax.set_visible(True)
        frame = method_taxa[method_taxa["method"] == method]
        draw_delta_pearson_scatter(ax, frame, method, method_correlations)

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if not ax.get_visible():
                continue
            if row_idx == n_rows - 1:
                ax.set_xlabel("Observed mean delta")
            else:
                ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel("Predicted mean delta")
            else:
                ax.set_ylabel("")
    fig.suptitle("Taxon-level recovery of mean pre-to-post abundance delta", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_delta_scatter(ax, frame: pd.DataFrame, method: str, top_n: int, annotate_taxa: bool = True) -> None:
    frame = frame.copy()
    if frame.empty:
        return
    frame["rank_score"] = frame["observed_log2fc"].abs()
    top = frame.nlargest(top_n, "rank_score")
    corr = correlation_stats(frame["observed_log2fc"].to_numpy(), frame["predicted_log2fc"].to_numpy())
    ax.scatter(frame["observed_log2fc"], frame["predicted_log2fc"], s=18, alpha=0.45, color=COLORS.get(method, "#4A148C"), edgecolor="none")
    lim = float(np.nanmax(np.abs(np.r_[frame["observed_log2fc"], frame["predicted_log2fc"]])))
    lim = max(lim, 0.5)
    ax.plot([-lim, lim], [-lim, lim], color="#222222", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.axvline(0.0, color="#777777", linewidth=0.8)
    if annotate_taxa:
        for _, row in top.iterrows():
            ax.text(row["observed_log2fc"], row["predicted_log2fc"], row["taxon_label"], fontsize=7.5, alpha=0.9)
    ax.text(
        0.04,
        0.96,
        f"Spearman rho = {corr['spearman_r']:.3f}\np = {format_p_value(corr['spearman_p'])}\nn = {corr['n']}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88},
    )
    ax.set_xlim(-lim * 1.08, lim * 1.08)
    ax.set_ylim(-lim * 1.08, lim * 1.08)
    ax.set_aspect("equal", adjustable="box")
    try:
        ax.set_box_aspect(1)
    except AttributeError:
        pass
    ax.set_xlabel("Observed pre-to-post log2 fold change")
    ax.set_ylabel("Predicted pre-to-post log2 fold change")
    ax.set_title(f"{method}: predicted vs observed abundance shifts", fontsize=12, fontweight="bold")
    style_axis(ax)


def save_delta_scatter(method_taxa: pd.DataFrame, focus_method: str, top_n: int, output: Path) -> None:
    frame = method_taxa[method_taxa["method"] == focus_method].copy()
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    draw_delta_scatter(ax, frame, focus_method, top_n, annotate_taxa=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_delta_scatter_by_method(method_taxa: pd.DataFrame, top_n: int, output_dir: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in set(method_taxa["method"])]
    if not methods:
        return
    for method in methods:
        output = output_dir / f"fmt_da_delta_scatter_{safe_filename(method)}.png"
        save_delta_scatter(method_taxa, method, top_n, output)

    n_cols = min(3, len(methods))
    n_rows = int(math.ceil(len(methods) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.8 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, method in zip(axes.ravel(), methods):
        ax.set_visible(True)
        frame = method_taxa[method_taxa["method"] == method]
        draw_delta_scatter(ax, frame, method, top_n=0, annotate_taxa=False)
    fig.tight_layout()
    output = output_dir / "fmt_da_delta_scatter_all_methods.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_delta_scatter_all_methods(method_taxa: pd.DataFrame, output: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in set(method_taxa["method"])]
    if not methods:
        return
    n_cols = min(3, len(methods))
    n_rows = int(math.ceil(len(methods) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.9 * n_cols, 4.9 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, method in zip(axes.ravel(), methods):
        ax.set_visible(True)
        frame = method_taxa[method_taxa["method"] == method]
        draw_delta_scatter(ax, frame, method, top_n=0, annotate_taxa=False)
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if not ax.get_visible():
                continue
            if row_idx == n_rows - 1:
                ax.set_xlabel("Observed pre-to-post log2 fold change")
            else:
                ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel("Predicted pre-to-post log2 fold change")
            else:
                ax.set_ylabel("")
    fig.suptitle("Taxon-level recovery of pre-to-post log2 fold changes", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def select_top_observed_taxa(method_taxa: pd.DataFrame, top_n: int) -> pd.DataFrame:
    reference = (
        method_taxa.groupby(["feature_id", "taxon_label"], sort=False)[["observed_log2fc"]]
        .mean()
        .reset_index()
    )
    reference["rank_score"] = reference["observed_log2fc"].abs()
    return reference.nlargest(top_n, "rank_score").sort_values("observed_log2fc").reset_index(drop=True)


def save_top_taxa_bars(method_taxa: pd.DataFrame, focus_method: str, top_n: int, output: Path) -> pd.DataFrame:
    top_reference = select_top_observed_taxa(method_taxa, top_n)
    if top_reference.empty:
        return top_reference
    methods = [method for method in METHOD_ORDER if method in set(method_taxa["method"])]
    if not methods:
        return top_reference

    y = np.arange(len(top_reference))
    height = 0.36
    fig, axes = plt.subplots(
        len(methods),
        1,
        figsize=(9.2, max(7.0, 0.30 * len(top_reference) * len(methods) + 1.9)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    plot_rows = []
    for ax, method in zip(axes, methods):
        frame = method_taxa[method_taxa["method"] == method].copy()
        top = top_reference[["feature_id", "taxon_label", "observed_log2fc"]].merge(
            frame[["feature_id", "predicted_log2fc"]],
            on="feature_id",
            how="left",
        )
        top["method"] = method
        plot_rows.append(top)
        ax.barh(
            y - height / 2,
            top["observed_log2fc"],
            height=height,
            label="Observed post vs pre",
            color="#555555",
            edgecolor="#222222",
            linewidth=0.55,
            alpha=0.82,
        )
        ax.barh(
            y + height / 2,
            top["predicted_log2fc"],
            height=height,
            label="Predicted post vs pre",
            color=COLORS.get(method, "#777777"),
            edgecolor="#222222",
            linewidth=0.55,
            alpha=0.88,
        )
        ax.axvline(0.0, color="#222222", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(top_reference["taxon_label"], fontsize=8.0)
        ax.set_title(method, fontsize=10.5, fontweight="bold", loc="left", pad=2)
        style_axis(ax)
    axes[-1].set_xlabel("log2 fold change relative to pre-FMT recipient")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.suptitle("Top observed pre/post abundance changes across models", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    all_top = pd.concat(plot_rows, ignore_index=True)
    focus = all_top[all_top["method"] == focus_method].copy()
    if focus.empty:
        focus = all_top.copy()
    return focus


def save_focus_top_taxa_bars(method_taxa: pd.DataFrame, focus_method: str, top_n: int, output: Path) -> pd.DataFrame:
    frame = method_taxa[method_taxa["method"] == focus_method].copy()
    if frame.empty:
        return frame
    frame["rank_score"] = frame["observed_log2fc"].abs()
    top = frame.nlargest(top_n, "rank_score").sort_values("observed_log2fc")
    y = np.arange(len(top))
    height = 0.38
    fig, ax = plt.subplots(figsize=(8.0, max(5.0, 0.32 * len(top) + 1.8)))
    ax.barh(y - height / 2, top["observed_log2fc"], height=height, label="Observed post vs pre", color="#555555", edgecolor="#222222", linewidth=0.7)
    ax.barh(y + height / 2, top["predicted_log2fc"], height=height, label="Predicted post vs pre", color=COLORS.get(focus_method, "#4A148C"), edgecolor="#222222", linewidth=0.7)
    ax.axvline(0.0, color="#222222", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(top["taxon_label"], fontsize=8.5)
    ax.set_xlabel("log2 fold change relative to pre-FMT recipient")
    ax.set_title(f"{focus_method}: top observed pre/post abundance changes", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return top


def build_top_taxa_by_method_table(method_taxa: pd.DataFrame, top_n: int) -> pd.DataFrame:
    top_reference = select_top_observed_taxa(method_taxa, top_n)
    if top_reference.empty:
        return top_reference
    rows = []
    for method in [method for method in METHOD_ORDER if method in set(method_taxa["method"])]:
        frame = method_taxa[method_taxa["method"] == method]
        top = top_reference[["feature_id", "taxon_label", "observed_log2fc"]].merge(
            frame[["feature_id", "predicted_log2fc", "observed_delta_mean", "predicted_delta_mean"]],
            on="feature_id",
            how="left",
        )
        top["method"] = method
        rows.append(top)
    return pd.concat(rows, ignore_index=True) if rows else top_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether FMT community predictors recover taxa-level pre/post abundance changes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--community_prediction_dir", type=Path, default=Path("outputs_downstreams/FMT/community_prediction"))
    parser.add_argument("--mlp_prediction_dir", type=Path, default=Path("outputs_downstreams/FMT/mlp_baseline"))
    parser.add_argument("--mlp_prediction_file", type=Path, default=Path("outputs_downstreams/FMT/mlp_baseline/predicted_abundance.h5"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures/FMT/differential_abundance"))
    parser.add_argument("--focus_method", default="MGM2-xlarge")
    parser.add_argument("--top_taxa", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min_abs_log2fc", type=float, default=0.25)
    parser.add_argument("--min_sample_delta", type=float, default=1e-4)
    parser.add_argument("--min_targets", type=int, default=10, help="Skip readable but incomplete prediction files below this target count.")
    parser.add_argument("--pseudocount", type=float, default=1e-6)
    parser.add_argument("--strict_h5", action="store_true", help="Fail instead of skipping unreadable prediction HDF5 files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_files = find_prediction_files(args.community_prediction_dir, args.mlp_prediction_dir, args.mlp_prediction_file)
    if not prediction_files:
        raise FileNotFoundError("No predicted_abundance.h5 files found")

    taxon_frames = []
    sample_frames = []
    skipped_rows = []
    for path in prediction_files:
        try:
            record = read_prediction_file(path)
        except OSError as exc:
            if args.strict_h5:
                raise
            method, scale, seed = infer_method(path)
            skipped_rows.append(
                {
                    "prediction_file": str(path),
                    "method": method,
                    "scale": scale,
                    "seed": seed,
                    "reason": repr(exc),
                }
            )
            print(f"Skipping unreadable HDF5 file: {path} ({exc})")
            continue
        target_ids = record["metadata"].get("target_id", pd.Series(dtype=str)).astype(str)
        n_targets = int((target_ids.str.len() > 0).sum())
        n_unique_targets = int(target_ids[target_ids.str.len() > 0].nunique())
        if n_targets < args.min_targets:
            method, scale, seed = infer_method(path)
            skipped_rows.append(
                {
                    "prediction_file": str(path),
                    "method": method,
                    "scale": scale,
                    "seed": seed,
                    "reason": f"incomplete prediction metadata: n_nonempty_targets={n_targets} < min_targets={args.min_targets}",
                }
            )
            print(f"Skipping incomplete prediction file: {path} (n_nonempty_targets={n_targets})")
            continue
        if n_unique_targets < args.min_targets:
            method, scale, seed = infer_method(path)
            skipped_rows.append(
                {
                    "prediction_file": str(path),
                    "method": method,
                    "scale": scale,
                    "seed": seed,
                    "reason": f"incomplete prediction metadata: n_unique_targets={n_unique_targets} < min_targets={args.min_targets}",
                }
            )
            print(f"Skipping incomplete prediction file: {path} (n_unique_targets={n_unique_targets})")
            continue
        taxon_frames.append(summarize_taxa(record, args.pseudocount))
        sample_frames.append(summarize_samples(record, args.min_sample_delta))
    if not taxon_frames:
        raise RuntimeError("No readable prediction files remained after HDF5 validation")

    taxon_stats = pd.concat(taxon_frames, ignore_index=True)
    sample_stats = pd.concat(sample_frames, ignore_index=True)
    run_summary = summarize_runs(taxon_stats, sample_stats, args.alpha, args.min_abs_log2fc, args.top_taxa)
    method_taxa = aggregate_method_taxa(taxon_stats)
    method_correlations = summarize_method_correlations(method_taxa)
    top_taxa = save_focus_top_taxa_bars(method_taxa, args.focus_method, args.top_taxa, args.output_dir / "fmt_da_top_taxa_log2fc.png")

    taxon_stats.to_csv(args.output_dir / "fmt_da_taxon_stats_by_run.csv", index=False)
    sample_stats.to_csv(args.output_dir / "fmt_da_sample_delta_stats.csv", index=False)
    run_summary.to_csv(args.output_dir / "fmt_da_run_summary.csv", index=False)
    method_taxa.to_csv(args.output_dir / "fmt_da_taxon_stats_by_method.csv", index=False)
    method_correlations.to_csv(args.output_dir / "fmt_da_method_correlations.csv", index=False)
    top_taxa.to_csv(args.output_dir / "fmt_da_focus_top_taxa.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(args.output_dir / "fmt_da_skipped_prediction_files.csv", index=False)

    save_summary_bars(run_summary, args.output_dir / "fmt_da_recovery_summary.png")
    save_delta_pearson_summary(method_taxa, method_correlations, args.output_dir / "fmt_da_delta_pearson_by_method.png")
    save_delta_scatter_all_methods(method_taxa, args.output_dir / "fmt_da_delta_scatter_all_methods.png")

    display = run_summary.copy()
    display["method"] = pd.Categorical(display["method"], METHOD_ORDER, ordered=True)
    display = display.sort_values(["method", "seed"])
    print(display.to_string(index=False))
    print()
    print(method_correlations.to_string(index=False))
    print(f"Wrote differential abundance analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
