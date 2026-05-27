#!/usr/bin/env python3
"""Summarize and plot standard FMT outer 5-fold results."""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")
matplotlib.set_loglevel("warning")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fmt_5fold_utils import nested_grouped_cv_indices
from src.data.FMTDataset import FMTCommunityPredictionDataset
from train_mlp_baseline import build_arrays, bray_curtis, cosine, finalize_raw_predictions

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
logging.getLogger("fontTools").setLevel(logging.WARNING)

SCALE_ORDER = ["scale_small", "scale_medium", "scale_large", "scale_xlarge"]
SCALE_LABELS = {
    "scale_small": "MGM2-small",
    "scale_medium": "MGM2-medium",
    "scale_large": "MGM2-large",
    "scale_xlarge": "MGM2-xlarge",
}
COMMUNITY_METHOD_ORDER = ["Recipient", "Mean-post", "MLP", "MGM2-small", "MGM2-medium", "MGM2-large", "MGM2-xlarge"]
CLASS_METHOD_ORDER = ["Majority", "MLP", "MGM2-small", "MGM2-medium", "MGM2-large", "MGM2-xlarge"]
COLORS = {
    "Recipient": "#777777",
    "Mean-post": "#B06B3A",
    "Majority": "#777777",
    "MLP": "#2A9D8F",
    "MGM2-small": "#7CB5EC",
    "MGM2-medium": "#4363D8",
    "MGM2-large": "#1A237E",
    "MGM2-xlarge": "#4A148C",
}
COMMUNITY_METRIC_LABELS = {
    "bray_curtis": "Mean Bray-Curtis distance",
    "cosine": "Mean cosine similarity",
}
CLASS_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro F1",
    "roc_auc": "ROC AUC",
    "average_precision": "Average precision",
}


def parse_fold(path: Path) -> int:
    match = re.search(r"fold_(\d+)", str(path))
    return int(match.group(1)) if match else -1


def parse_seed(path: Path) -> int:
    match = re.search(r"seed_(\d+)", str(path))
    return int(match.group(1)) if match else -1


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.28)


def set_tight_ylim(ax, values: Iterable[float], min_span: float, pad_frac: float = 0.18, bounded01: bool = False) -> None:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    ymin = float(arr.min())
    ymax = float(arr.max())
    span = max(ymax - ymin, min_span)
    pad = span * pad_frac
    low = ymin - pad
    high = ymax + pad
    if bounded01:
        low = max(0.0, low)
        high = min(1.02, high)
    ax.set_ylim(low, high)


def read_json(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_community_frame(path: Path, method: str, scale: str, seed: int, fold: int, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "target_id",
        "fmt_id",
        "study",
        "response_label",
        "donor_sample_id",
        "recipient_sample_id",
        "post_sample_id",
        "timepoint_fmt",
        f"{prefix}_bray_curtis",
        f"{prefix}_cosine",
        "recipient_baseline_bray_curtis",
        "recipient_baseline_cosine",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    out = frame.copy()
    out["method"] = method
    out["scale"] = scale
    out["seed"] = seed
    out["fold"] = fold
    out["bray_curtis"] = out[f"{prefix}_bray_curtis"].astype(float)
    out["cosine"] = out[f"{prefix}_cosine"].astype(float)
    out["recipient_bray_curtis"] = out["recipient_baseline_bray_curtis"].astype(float)
    out["recipient_cosine"] = out["recipient_baseline_cosine"].astype(float)
    return out


def load_community_prediction_metrics(mgm2_dir: Path, mlp_dir: Path, seeds: List[int] | None = None) -> pd.DataFrame:
    frames = []
    seed_set = set(seeds) if seeds else None
    for scale in SCALE_ORDER:
        for path in sorted((mgm2_dir / scale).glob("seed_*/fold_*/fold_validation_predictions.metrics.csv")):
            if seed_set and parse_seed(path) not in seed_set:
                continue
            frames.append(
                normalize_community_frame(
                    path,
                    method=SCALE_LABELS[scale],
                    scale=scale,
                    seed=parse_seed(path),
                    fold=parse_fold(path),
                    prefix="mgm2",
                )
            )
    for path in sorted(mlp_dir.glob("seed_*/fold_*/predicted_abundance.metrics.csv")):
        if seed_set and parse_seed(path) not in seed_set:
            continue
        frames.append(
            normalize_community_frame(
                path,
                method="MLP",
                scale="mlp",
                seed=parse_seed(path),
                fold=parse_fold(path),
                prefix="mlp",
            )
        )
    if not frames:
        raise FileNotFoundError(f"No FMT 5-fold community metrics found under {mgm2_dir} or {mlp_dir}")
    return pd.concat(frames, ignore_index=True, sort=False)


def add_recipient_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    reference = metrics[metrics["method"] == "MLP"].copy()
    if reference.empty:
        reference = metrics.groupby(["target_id", "seed", "fold"], dropna=False).first().reset_index()
    recipient = reference.copy()
    recipient["method"] = "Recipient"
    recipient["scale"] = "recipient"
    recipient["bray_curtis"] = recipient["recipient_bray_curtis"]
    recipient["cosine"] = recipient["recipient_cosine"]
    return pd.concat([recipient, metrics], ignore_index=True, sort=False)


def compute_mean_post_baseline(args: argparse.Namespace, reference: pd.DataFrame) -> pd.DataFrame:
    dataset = FMTCommunityPredictionDataset(args.data_file, args.post_metadata_file, args.ntv3_embedding_path)
    arrays = build_arrays(dataset)
    rows = []
    split_keys = reference[["seed", "fold"]].drop_duplicates().sort_values(["seed", "fold"])
    for _, split in split_keys.iterrows():
        seed = int(split["seed"])
        fold = int(split["fold"])
        train_idx, _, test_idx = nested_grouped_cv_indices(
            arrays.metadata,
            fold_index=fold,
            num_folds=args.num_folds,
            val_ratio=args.val_ratio,
            seed=seed,
            group_column=args.group_column,
            label_column=args.label_column,
        )
        mean_raw = arrays.post_raw[train_idx].mean(axis=0, keepdims=True)
        mean_raw = finalize_raw_predictions(mean_raw, raw_zero_threshold=0.0, renormalize=True)
        pred = np.repeat(mean_raw, len(test_idx), axis=0)
        meta = arrays.metadata.iloc[test_idx].reset_index(drop=True).copy()
        true = arrays.post_raw[test_idx]
        recipient = arrays.recipient_raw[test_idx]
        meta["method"] = "Mean-post"
        meta["scale"] = "mean"
        meta["seed"] = seed
        meta["fold"] = fold
        meta["bray_curtis"] = bray_curtis(pred, true)
        meta["cosine"] = cosine(pred, true)
        meta["recipient_bray_curtis"] = bray_curtis(recipient, true)
        meta["recipient_cosine"] = cosine(recipient, true)
        rows.append(meta)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def summarize_by_fold(per_sample: pd.DataFrame, metrics: List[str], method_order: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fold_summary = (
        per_sample.groupby(["method", "scale", "seed", "fold"], dropna=False, observed=False)[metrics]
        .mean()
        .reset_index()
    )
    rows = []
    for method in method_order:
        frame = fold_summary[fold_summary["method"] == method]
        if frame.empty:
            continue
        row = {"method": method, "n_folds": int(frame[["seed", "fold"]].drop_duplicates().shape[0])}
        for metric in metrics:
            values = frame[metric].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    method_summary = pd.DataFrame(rows)
    method_summary["method"] = pd.Categorical(method_summary["method"], method_order, ordered=True)
    method_summary = method_summary.sort_values("method").reset_index(drop=True)
    return fold_summary, method_summary


def save_metric_bars(summary: pd.DataFrame, metric: str, labels: Dict[str, str], order: List[str], output: Path, bounded01: bool) -> None:
    frame = summary[summary["method"].isin(order)].copy()
    frame["method"] = pd.Categorical(frame["method"], order, ordered=True)
    frame = frame.sort_values("method")
    x = np.arange(len(frame))
    means = frame[f"{metric}_mean"].astype(float).to_numpy()
    stds = frame[f"{metric}_std"].astype(float).fillna(0.0).to_numpy()
    names = frame["method"].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(max(8.5, 1.25 * len(names)), 5.0))
    ax.bar(x, means, yerr=stds, capsize=4, color=[COLORS[name] for name in names], edgecolor="#333333", linewidth=1.1, width=0.58)
    for idx, value in enumerate(means):
        ax.text(idx, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel(labels[metric], fontsize=11)
    set_tight_ylim(ax, np.concatenate([means - stds, means + stds]), min_span=0.025 if not bounded01 else 0.08, bounded01=bounded01)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_community_scale_trend(fold_summary: pd.DataFrame, output: Path) -> None:
    mgm2 = fold_summary[fold_summary["scale"].isin(SCALE_ORDER)].copy()
    mgm2["scale"] = pd.Categorical(mgm2["scale"], SCALE_ORDER, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True)
    for ax, metric in zip(axes, ["bray_curtis", "cosine"]):
        grouped = mgm2.groupby("scale", observed=False)[metric].agg(["mean", "std"]).reindex(SCALE_ORDER)
        x = np.arange(len(SCALE_ORDER))
        for idx, scale in enumerate(SCALE_ORDER):
            vals = mgm2.loc[mgm2["scale"] == scale, metric].dropna().to_numpy()
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) else []
            ax.scatter(np.full(len(vals), idx) + jitter, vals, s=38, color=COLORS[SCALE_LABELS[scale]], edgecolor="#333333", linewidth=0.5, alpha=0.9)
        ax.plot(x, grouped["mean"], color="#222222", linewidth=2.0)
        ax.errorbar(x, grouped["mean"], yerr=grouped["std"].fillna(0.0), fmt="none", ecolor="#222222", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(["Small", "Medium", "Large", "XLarge"], fontsize=10, fontweight="bold")
        ax.set_ylabel(COMMUNITY_METRIC_LABELS[metric])
        ax.set_title(f"MGM2 scale trend: {COMMUNITY_METRIC_LABELS[metric]}", fontsize=12, fontweight="bold")
        set_tight_ylim(ax, mgm2[metric], min_span=0.015 if metric == "bray_curtis" else 0.04)
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_community_improvement(per_sample: pd.DataFrame, output: Path) -> None:
    frame = per_sample[per_sample["method"].isin(COMMUNITY_METHOD_ORDER[1:])].copy()
    frame["bray_improvement"] = frame["recipient_bray_curtis"] - frame["bray_curtis"]
    frame["cosine_improvement"] = frame["cosine"] - frame["recipient_cosine"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "bray_improvement", "Recipient Bray - method Bray"),
        (axes[1], "cosine_improvement", "Method cosine - recipient cosine"),
    ]:
        labels = [m for m in COMMUNITY_METHOD_ORDER[1:] if m in set(frame["method"])]
        values = [frame.loc[frame["method"] == method, metric].dropna().to_numpy() for method in labels]
        box = ax.boxplot(values, positions=np.arange(1, len(labels) + 1), patch_artist=True, showfliers=False)
        for patch, label in zip(box["boxes"], labels):
            patch.set_facecolor(COLORS[label])
            patch.set_alpha(0.75)
            patch.set_edgecolor("#333333")
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel)
        set_tight_ylim(ax, frame[metric], min_span=0.08)
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_best_mgm2_vs_mlp(per_sample: pd.DataFrame, method_summary: pd.DataFrame, output: Path) -> str:
    mgm2 = method_summary[method_summary["method"].astype(str).str.startswith("MGM2-")].sort_values("bray_curtis_mean")
    best_method = str(mgm2.iloc[0]["method"])
    best = per_sample[per_sample["method"] == best_method][["target_id", "bray_curtis", "cosine", "study"]].rename(
        columns={"bray_curtis": "best_bray", "cosine": "best_cosine"}
    )
    mlp = per_sample[per_sample["method"] == "MLP"][["target_id", "bray_curtis", "cosine"]].rename(
        columns={"bray_curtis": "mlp_bray", "cosine": "mlp_cosine"}
    )
    merged = best.merge(mlp, on="target_id", how="inner")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for ax, x_col, y_col, xlabel, ylabel, favors in [
        (axes[0], "mlp_bray", "best_bray", "MLP Bray-Curtis", f"{best_method} Bray-Curtis", "Below diagonal favors MGM2"),
        (axes[1], "mlp_cosine", "best_cosine", "MLP cosine", f"{best_method} cosine", "Above diagonal favors MGM2"),
    ]:
        ax.scatter(merged[x_col], merged[y_col], s=30, alpha=0.75, color=COLORS[best_method], edgecolor="#202020", linewidth=0.3)
        low = float(min(merged[x_col].min(), merged[y_col].min()))
        high = float(max(merged[x_col].max(), merged[y_col].max()))
        ax.plot([low, high], [low, high], color="#202020", linestyle="--", linewidth=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(favors, fontsize=12, fontweight="bold")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return best_method


def save_per_study_community(per_sample: pd.DataFrame, output: Path) -> None:
    studies = per_sample[per_sample["method"] == "Recipient"]["study"].value_counts().index.tolist()
    fig, axes = plt.subplots(2, 1, figsize=(max(12.5, len(studies) * 0.85), 9.0), sharex=True)
    for ax, metric in zip(axes, ["bray_curtis", "cosine"]):
        pivot = per_sample.groupby(["study", "method"], dropna=False, observed=False)[metric].mean().unstack("method").reindex(studies)
        pivot = pivot[[m for m in COMMUNITY_METHOD_ORDER if m in pivot.columns]]
        x = np.arange(len(pivot))
        width = 0.86 / max(len(pivot.columns), 1)
        for idx, method in enumerate(pivot.columns):
            ax.bar(x + (idx - (len(pivot.columns) - 1) / 2) * width, pivot[method], width=width, color=COLORS[method], edgecolor="#333333", linewidth=0.4, label=method)
        ax.set_ylabel(COMMUNITY_METRIC_LABELS[metric])
        ax.set_title(f"Per-study community prediction: {COMMUNITY_METRIC_LABELS[metric]}", fontsize=12, fontweight="bold")
        set_tight_ylim(ax, pivot.to_numpy().ravel(), min_span=0.08 if metric == "bray_curtis" else 0.16)
        style_axis(ax)
    counts = per_sample[per_sample["method"] == "Recipient"]["study"].value_counts()
    axes[1].set_xticks(np.arange(len(studies)))
    axes[1].set_xticklabels([f"{study}\n(n={counts[study]})" for study in studies], rotation=35, ha="right", fontsize=9)
    axes[0].legend(frameon=False, ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    unique_scores, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if len(unique_scores) != len(scores):
        rank_sums = np.zeros(len(unique_scores), dtype=np.float64)
        np.add.at(rank_sums, inverse, ranks)
        ranks = rank_sums[inverse] / counts[inverse]
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    precision = tp / (np.arange(len(labels)) + 1)
    return float((precision * (sorted_labels == 1)).sum() / n_pos)


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    labels = labels.astype(int)
    preds = (scores >= threshold).astype(int)
    recalls = []
    f1_values = []
    for class_id in [0, 1]:
        pred_mask = preds == class_id
        label_mask = labels == class_id
        tp = float((pred_mask & label_mask).sum())
        fp = float((pred_mask & ~label_mask).sum())
        fn = float((~pred_mask & label_mask).sum())
        if label_mask.any():
            recalls.append(tp / max(tp + fn, 1.0))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        if (tp + fp) > 0 or (tp + fn) > 0:
            f1_values.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    return {
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_f1": float(np.mean(f1_values)) if f1_values else float("nan"),
        "roc_auc": binary_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
    }


def load_prediction_csv(path: Path, method: str, scale: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "prob_responder" not in frame.columns:
        raise ValueError(f"{path} is missing prob_responder")
    frame["method"] = method
    frame["scale"] = scale
    frame["seed"] = parse_seed(path)
    frame["fold"] = parse_fold(path)
    frame["source"] = str(path)
    return frame


def load_classification_results(mgm2_dir: Path, mlp_dir: Path, seeds: List[int] | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_frames = []
    seed_set = set(seeds) if seeds else None
    for scale in SCALE_ORDER:
        for path in sorted((mgm2_dir / scale).glob("seed_*/fold_*/metrics.json")):
            if seed_set and parse_seed(path) not in seed_set:
                continue
            metrics = read_json(path)
            metric_rows.append({"method": SCALE_LABELS[scale], "scale": scale, "seed": parse_seed(path), "fold": parse_fold(path), "source": str(path), **metrics})
            pred_path = path.parent / "fold_validation_predictions.csv"
            if pred_path.exists():
                prediction_frames.append(load_prediction_csv(pred_path, SCALE_LABELS[scale], scale))
    for path in sorted(mlp_dir.glob("seed_*/fold_*/metrics.json")):
        if seed_set and parse_seed(path) not in seed_set:
            continue
        metrics = read_json(path)
        metric_rows.append({"method": "MLP", "scale": "mlp", "seed": parse_seed(path), "fold": parse_fold(path), "source": str(path), **metrics})
        pred_path = path.parent / "test_predictions.csv"
        if pred_path.exists():
            prediction_frames.append(load_prediction_csv(pred_path, "MLP", "mlp"))
    if not metric_rows:
        raise FileNotFoundError(f"No FMT 5-fold classification metrics found under {mgm2_dir} or {mlp_dir}")
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True, sort=False)


def response_metadata_for_majority(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path).copy()
    if "TargetID" not in metadata.columns:
        metadata["TargetID"] = metadata.get("fmt_id", metadata.index).astype(str)
    metadata["response_label"] = metadata["response_label"].astype(str)
    metadata = metadata[metadata["response_label"].isin(["non-responder", "responder"])].copy()
    metadata = metadata.sort_values("TargetID").drop_duplicates("fmt_id", keep="first").reset_index(drop=True)
    return metadata


def add_majority_baseline(args: argparse.Namespace, metrics: pd.DataFrame, predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metadata = response_metadata_for_majority(args.response_metadata_file)
    label_map = {"non-responder": 0, "responder": 1}
    labels = metadata["response_label"].map(label_map).astype(int).to_numpy()
    metric_rows = []
    prediction_rows = []
    for _, split in metrics[["seed", "fold"]].drop_duplicates().sort_values(["seed", "fold"]).iterrows():
        seed = int(split["seed"])
        fold = int(split["fold"])
        train_idx, _, test_idx = nested_grouped_cv_indices(
            metadata,
            fold_index=fold,
            num_folds=args.num_folds,
            val_ratio=args.val_ratio,
            seed=seed,
            group_column=args.group_column,
            label_column=args.label_column,
        )
        train_counts = np.bincount(labels[train_idx], minlength=2)
        majority = int(train_counts.argmax())
        score = float(majority)
        test_labels = labels[test_idx]
        fold_metrics = classification_metrics(test_labels, np.full(len(test_idx), score, dtype=np.float32))
        metric_rows.append({"method": "Majority", "scale": "majority", "seed": seed, "fold": fold, **fold_metrics})
        frame = metadata.iloc[test_idx].copy()
        frame["target_id"] = frame["TargetID"].astype(str)
        frame["label_id"] = test_labels
        frame["pred_id"] = majority
        frame["pred_label"] = "responder" if majority == 1 else "non-responder"
        frame["prob_responder"] = score
        frame["method"] = "Majority"
        frame["scale"] = "majority"
        frame["seed"] = seed
        frame["fold"] = fold
        prediction_rows.append(frame)
    metrics = pd.concat([pd.DataFrame(metric_rows), metrics], ignore_index=True, sort=False)
    predictions = pd.concat([pd.concat(prediction_rows, ignore_index=True, sort=False), predictions], ignore_index=True, sort=False)
    return metrics, predictions


def save_class_scale_trend(fold_metrics: pd.DataFrame, output: Path) -> None:
    metrics = ["roc_auc", "average_precision", "balanced_accuracy"]
    mgm2 = fold_metrics[fold_metrics["scale"].isin(SCALE_ORDER)].copy()
    mgm2["scale"] = pd.Categorical(mgm2["scale"], SCALE_ORDER, ordered=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharex=True)
    for ax, metric in zip(axes, metrics):
        grouped = mgm2.groupby("scale", observed=False)[metric].agg(["mean", "std"]).reindex(SCALE_ORDER)
        x = np.arange(len(SCALE_ORDER))
        for idx, scale in enumerate(SCALE_ORDER):
            vals = mgm2.loc[mgm2["scale"] == scale, metric].astype(float).dropna().to_numpy()
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) else []
            ax.scatter(np.full(len(vals), idx) + jitter, vals, s=38, color=COLORS[SCALE_LABELS[scale]], edgecolor="#333333", linewidth=0.5)
        ax.plot(x, grouped["mean"], color="#222222", linewidth=2.0)
        ax.errorbar(x, grouped["mean"], yerr=grouped["std"].fillna(0.0), fmt="none", ecolor="#222222", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(["Small", "Medium", "Large", "XLarge"], fontsize=10, fontweight="bold")
        ax.set_ylabel(CLASS_METRIC_LABELS[metric])
        ax.set_title(f"Response scale trend: {CLASS_METRIC_LABELS[metric]}", fontsize=12, fontweight="bold")
        set_tight_ylim(ax, mgm2[metric], min_span=0.08, bounded01=True)
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def curve_points(labels: np.ndarray, scores: np.ndarray, curve: str) -> Tuple[np.ndarray, np.ndarray]:
    labels = labels.astype(int)
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    if curve == "roc":
        x = np.concatenate([[0.0], fp / max(int((labels == 0).sum()), 1), [1.0]])
        y = np.concatenate([[0.0], tp / max(int((labels == 1).sum()), 1), [1.0]])
    else:
        x = np.concatenate([[0.0], tp / max(int((labels == 1).sum()), 1)])
        y = np.concatenate([[1.0], tp / np.clip(tp + fp, 1, None)])
    return x, y


def save_roc_pr_curves(predictions: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.1))
    for method in CLASS_METHOD_ORDER:
        frame = predictions[predictions["method"] == method]
        if frame.empty:
            continue
        labels = frame["label_id"].astype(int).to_numpy()
        scores = frame["prob_responder"].astype(float).to_numpy()
        fpr, tpr = curve_points(labels, scores, "roc")
        recall, precision = curve_points(labels, scores, "pr")
        axes[0].plot(fpr, tpr, color=COLORS[method], linewidth=2.0, label=f"{method} ({binary_auc(labels, scores):.3f})")
        axes[1].plot(recall, precision, color=COLORS[method], linewidth=2.0, label=f"{method} ({average_precision(labels, scores):.3f})")
    axes[0].plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("Outer-fold pooled ROC curves", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Outer-fold pooled precision-recall curves", fontsize=12, fontweight="bold")
    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        style_axis(ax)
        ax.legend(frameon=False, fontsize=8.2)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_probability_strip(predictions: pd.DataFrame, output: Path) -> None:
    methods = [m for m in CLASS_METHOD_ORDER if m in set(predictions["method"])]
    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    for idx, method in enumerate(methods):
        frame = predictions[predictions["method"] == method].copy()
        colors = np.where(frame["label_id"].astype(int).to_numpy() == 1, "#D55E00", "#0072B2")
        jitter = np.linspace(-0.16, 0.16, len(frame)) if len(frame) else []
        ax.scatter(frame["prob_responder"].astype(float), np.full(len(frame), idx) + jitter, c=colors, edgecolor="#333333", linewidth=0.35, s=38, alpha=0.9)
    ax.axvline(0.5, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=10, fontweight="bold")
    ax.set_xlabel("Probability of responder")
    ax.set_title("Response probability distribution by true label", fontsize=12, fontweight="bold")
    ax.scatter([], [], c="#D55E00", edgecolor="#333333", label="True responder")
    ax.scatter([], [], c="#0072B2", edgecolor="#333333", label="True non-responder")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_per_study_classification(predictions: pd.DataFrame, output: Path) -> None:
    frame = predictions.copy()
    frame["correct"] = (frame["pred_id"].astype(int) == frame["label_id"].astype(int)).astype(float)
    counts = frame[frame["method"] == "MLP"]["study"].value_counts()
    studies = counts.index.tolist()
    pivot = frame.groupby(["study", "method"], dropna=False, observed=False)["correct"].mean().unstack("method").reindex(studies)
    pivot = pivot[[m for m in CLASS_METHOD_ORDER if m in pivot.columns]]
    fig, ax = plt.subplots(figsize=(max(11.5, len(studies) * 0.9), 5.2))
    x = np.arange(len(pivot))
    width = 0.86 / max(len(pivot.columns), 1)
    for idx, method in enumerate(pivot.columns):
        ax.bar(x + (idx - (len(pivot.columns) - 1) / 2) * width, pivot[method], width=width, color=COLORS[method], edgecolor="#333333", linewidth=0.45, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{study}\n(n={counts[study]})" for study in studies], rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-study FMT response classification", fontsize=12, fontweight="bold")
    set_tight_ylim(ax, pivot.to_numpy().ravel(), min_span=0.35, bounded01=True)
    style_axis(ax)
    ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_community_plots(args: argparse.Namespace, seeds: List[int] | None = None) -> None:
    raw_metrics = load_community_prediction_metrics(args.community_prediction_dir, args.mlp_community_dir, seeds)
    per_sample = add_recipient_rows(raw_metrics)
    mean_base = compute_mean_post_baseline(args, raw_metrics)
    per_sample = pd.concat([per_sample, mean_base], ignore_index=True, sort=False)
    per_sample["method"] = pd.Categorical(per_sample["method"], COMMUNITY_METHOD_ORDER, ordered=True)
    per_sample = per_sample.sort_values(["method", "seed", "fold", "target_id"]).reset_index(drop=True)
    fold_summary, method_summary = summarize_by_fold(per_sample, ["bray_curtis", "cosine"], COMMUNITY_METHOD_ORDER)

    per_sample.to_csv(args.output_dir / "fmt_5fold_community_per_sample_metrics.csv", index=False)
    fold_summary.to_csv(args.output_dir / "fmt_5fold_community_fold_metrics.csv", index=False)
    method_summary.to_csv(args.output_dir / "fmt_5fold_community_method_summary.csv", index=False)
    save_metric_bars(method_summary, "bray_curtis", COMMUNITY_METRIC_LABELS, COMMUNITY_METHOD_ORDER, args.output_dir / "fmt_5fold_community_bray_curtis_by_method.png", bounded01=False)
    save_metric_bars(method_summary, "cosine", COMMUNITY_METRIC_LABELS, COMMUNITY_METHOD_ORDER, args.output_dir / "fmt_5fold_community_cosine_by_method.png", bounded01=True)
    save_community_scale_trend(fold_summary, args.output_dir / "fmt_5fold_community_scale_trend.png")
    save_community_improvement(per_sample, args.output_dir / "fmt_5fold_community_improvement_vs_recipient.png")
    best_method = save_best_mgm2_vs_mlp(per_sample, method_summary, args.output_dir / "fmt_5fold_community_best_mgm2_vs_mlp.png")
    save_per_study_community(per_sample, args.output_dir / "fmt_5fold_community_per_study.png")
    print("Community prediction 5-fold summary:")
    print(method_summary.to_string(index=False))
    print(f"Best MGM2 community method by Bray-Curtis: {best_method}")


def run_classification_plots(args: argparse.Namespace, seeds: List[int] | None = None) -> None:
    metrics, predictions = load_classification_results(args.community_classification_dir, args.mlp_classification_dir, seeds)
    metrics, predictions = add_majority_baseline(args, metrics, predictions)
    metrics["method"] = pd.Categorical(metrics["method"], CLASS_METHOD_ORDER, ordered=True)
    metrics = metrics.sort_values(["method", "seed", "fold"]).reset_index(drop=True)
    predictions["method"] = pd.Categorical(predictions["method"], CLASS_METHOD_ORDER, ordered=True)
    predictions = predictions.sort_values(["method", "seed", "fold", "target_id"]).reset_index(drop=True)
    summary_rows = []
    for method in CLASS_METHOD_ORDER:
        frame = metrics[metrics["method"].astype(str) == method]
        if frame.empty:
            continue
        row = {"method": method, "n_folds": int(frame[["seed", "fold"]].drop_duplicates().shape[0])}
        for metric in CLASS_METRIC_LABELS:
            values = frame[metric].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary["method"] = pd.Categorical(summary["method"], CLASS_METHOD_ORDER, ordered=True)
    summary = summary.sort_values("method").reset_index(drop=True)

    metrics.to_csv(args.output_dir / "fmt_5fold_response_fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "fmt_5fold_response_method_summary.csv", index=False)
    predictions.to_csv(args.output_dir / "fmt_5fold_response_predictions.csv", index=False)
    save_metric_bars(summary, "roc_auc", CLASS_METRIC_LABELS, CLASS_METHOD_ORDER, args.output_dir / "fmt_5fold_response_roc_auc_by_method.png", bounded01=True)
    save_metric_bars(summary, "average_precision", CLASS_METRIC_LABELS, CLASS_METHOD_ORDER, args.output_dir / "fmt_5fold_response_average_precision_by_method.png", bounded01=True)
    save_metric_bars(summary, "balanced_accuracy", CLASS_METRIC_LABELS, CLASS_METHOD_ORDER, args.output_dir / "fmt_5fold_response_balanced_accuracy_by_method.png", bounded01=True)
    save_class_scale_trend(metrics, args.output_dir / "fmt_5fold_response_scale_trend.png")
    save_roc_pr_curves(predictions, args.output_dir / "fmt_5fold_response_roc_pr_curves.png")
    save_probability_strip(predictions, args.output_dir / "fmt_5fold_response_probability_strip.png")
    save_per_study_classification(predictions, args.output_dir / "fmt_5fold_response_per_study_accuracy.png")
    print()
    print("Response classification 5-fold summary:")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot standard FMT 5-fold community prediction and response classification results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    root = Path("outputs_downstreams/FMT_5folds")
    data_dir = Path("data/downstreams/FMT_5folds")
    parser.add_argument("--community_prediction_dir", type=Path, default=root / "community_prediction")
    parser.add_argument("--community_classification_dir", type=Path, default=root / "community_classification")
    parser.add_argument("--mlp_community_dir", type=Path, default=root / "mlp_community_prediction")
    parser.add_argument("--mlp_classification_dir", type=Path, default=root / "mlp_classification")
    parser.add_argument("--data_file", type=Path, default=data_dir / "community_all.pkl")
    parser.add_argument("--post_metadata_file", type=Path, default=data_dir / "post_all_meta.csv")
    parser.add_argument("--response_metadata_file", type=Path, default=data_dir / "response_all_meta.csv")
    parser.add_argument("--ntv3_embedding_path", type=Path, default=data_dir / "community_all.motu_embeddings.h5")
    parser.add_argument("--output_dir", type=Path, default=Path("figures/FMT_5folds"))
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--group_column", type=str, default="fmt_id")
    parser.add_argument("--label_column", type=str, default="response_label")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Filter by seed(s), e.g. --seeds 11 22")
    parser.add_argument("--skip_community", action="store_true")
    parser.add_argument("--skip_classification", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds
    if not args.skip_community:
        run_community_plots(args, seeds)
    if not args.skip_classification:
        run_classification_plots(args, seeds)
    print()
    print(f"Wrote FMT 5-fold figures and tables to {args.output_dir}")


if __name__ == "__main__":
    main()
