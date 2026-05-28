#!/usr/bin/env python3
"""Summarize and plot FMT response classification results."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

SCALE_ORDER = ["scale_small", "scale_medium", "scale_large", "scale_xlarge"]
SCALE_LABELS = {
    "scale_small": "MGM2-Small",
    "scale_medium": "MGM2-Medium",
    "scale_large": "MGM2-Large",
    "scale_xlarge": "MGM2-XLarge",
}
METHOD_ORDER = ["Majority", "MLP", "MGM2-small", "MGM2-medium", "MGM2-large", "MGM2-xlarge"]
COLORS = {
    "Majority": "#777777",
    "MLP": "#2A9D8F",
    "MGM2-small": "#7CB5EC",
    "MGM2-medium": "#4363D8",
    "MGM2-large": "#1A237E",
    "MGM2-xlarge": "#4A148C",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro F1",
    "roc_auc": "ROC AUC",
    "average_precision": "Average precision",
}


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.28)


def set_tight_ylim(ax, values: Iterable[float], min_span: float = 0.08, pad_frac: float = 0.18) -> None:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    ymin = float(arr.min())
    ymax = float(arr.max())
    span = max(ymax - ymin, min_span)
    pad = span * pad_frac
    ax.set_ylim(max(0.0, ymin - pad), min(1.02, ymax + pad))


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


def roc_curve_points(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = labels.astype(int)
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    n_pos = max(int((labels == 1).sum()), 1)
    n_neg = max(int((labels == 0).sum()), 1)
    tpr = np.concatenate([[0.0], np.cumsum(sorted_labels == 1) / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], np.cumsum(sorted_labels == 0) / n_neg, [1.0]])
    return fpr, tpr


def pr_curve_points(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = labels.astype(int)
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    recall = tp / max(int((labels == 1).sum()), 1)
    precision = tp / np.clip(tp + fp, 1, None)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return recall, precision


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    labels = labels.astype(int)
    preds = (scores >= threshold).astype(int)
    accuracy = float((preds == labels).mean())
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
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_f1": float(np.mean(f1_values)) if f1_values else float("nan"),
        "roc_auc": binary_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
    }


def majority_metrics(labels: np.ndarray) -> Dict[str, float]:
    majority = int(pd.Series(labels).value_counts().idxmax())
    scores = np.full(labels.shape[0], float(majority), dtype=np.float32)
    metrics = compute_metrics(labels, scores, threshold=0.5)
    metrics["majority_class"] = majority
    return metrics


def read_metrics_json(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_fold(path: Path) -> int:
    match = re.search(r"fold_(\d+)", str(path))
    if not match:
        return -1
    return int(match.group(1))


def parse_seed(path: Path) -> int:
    match = re.search(r"seed_(\d+)", str(path))
    if not match:
        return -1
    return int(match.group(1))


def load_mgm2_fold_metrics(root: Path) -> pd.DataFrame:
    rows = []
    for scale in SCALE_ORDER:
        for path in sorted((root / scale).glob("seed_*/fold_*/metrics.json")):
            metrics = read_metrics_json(path)
            row = {
                "method": SCALE_LABELS[scale],
                "scale": scale,
                "seed": parse_seed(path),
                "fold": parse_fold(path),
                "source": str(path),
            }
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def load_mlp_fold_metrics(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("seed_*/fold_*/metrics.json")):
        metrics = read_metrics_json(path)
        row = {
            "method": "MLP",
            "scale": "mlp",
            "seed": parse_seed(path),
            "fold": parse_fold(path),
            "source": str(path),
        }
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def load_predictions(root: Path, method: str, scale: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("seed_*/fold_*/test_predictions.csv")):
        frame = pd.read_csv(path)
        frame["method"] = method
        frame["scale"] = scale
        frame["seed"] = parse_seed(path)
        frame["fold"] = parse_fold(path)
        frame["source"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_mgm2_predictions(root: Path) -> pd.DataFrame:
    frames = []
    for scale in SCALE_ORDER:
        frames.append(load_predictions(root / scale, SCALE_LABELS[scale], scale))
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_all_predictions(mgm2_root: Path, mlp_root: Path) -> pd.DataFrame:
    frames = [load_predictions(mlp_root, "MLP", "mlp"), load_mgm2_predictions(mgm2_root)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise FileNotFoundError("No FMT classification prediction CSV files were found")
    data = pd.concat(frames, ignore_index=True)
    required = {"target_id", "label_id", "response_label", "prob_responder"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Prediction tables are missing columns: {missing}")
    return data


def build_ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [
        "target_id",
        "fmt_id",
        "study",
        "indication",
        "response_label",
        "donor_sample_id",
        "recipient_sample_id",
        "label_id",
    ]
    rows = []
    for (method, scale), frame in predictions.groupby(["method", "scale"], sort=False):
        grouped = frame.groupby("target_id", dropna=False)
        meta = grouped[metadata_cols].first().reset_index(drop=True)
        averaged = grouped["prob_responder"].mean().reset_index(drop=True)
        out = meta.copy()
        out["method"] = method
        out["scale"] = scale
        out["prob_responder"] = averaged.astype(float)
        out["pred_id"] = (out["prob_responder"] >= 0.5).astype(int)
        out["pred_label"] = np.where(out["pred_id"] == 1, "responder", "non-responder")
        out["n_models_ensembled"] = grouped.size().to_numpy()
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def add_majority_rows(fold_metrics: pd.DataFrame, ensemble_predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reference = ensemble_predictions[ensemble_predictions["method"] == "MLP"].copy()
    if reference.empty:
        reference = ensemble_predictions.groupby("target_id", dropna=False).first().reset_index()
    labels = reference["label_id"].astype(int).to_numpy()
    majority = int(pd.Series(labels).value_counts().idxmax())
    majority_fold = {
        "method": "Majority",
        "scale": "majority",
        "seed": -1,
        "fold": -1,
        **majority_metrics(labels),
    }
    fold_metrics = pd.concat([pd.DataFrame([majority_fold]), fold_metrics], ignore_index=True, sort=False)
    majority_pred = reference.copy()
    majority_pred["method"] = "Majority"
    majority_pred["scale"] = "majority"
    majority_pred["prob_responder"] = float(majority)
    majority_pred["pred_id"] = majority
    majority_pred["pred_label"] = "responder" if majority == 1 else "non-responder"
    majority_pred["n_models_ensembled"] = 1
    ensemble_predictions = pd.concat([majority_pred, ensemble_predictions], ignore_index=True, sort=False)
    return fold_metrics, ensemble_predictions


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        frame = fold_metrics[fold_metrics["method"] == method]
        if frame.empty:
            continue
        row = {"method": method, "n_folds": int(frame["fold"].nunique())}
        for metric in METRIC_LABELS:
            values = frame[metric].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_ensemble(ensemble_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        frame = ensemble_predictions[ensemble_predictions["method"] == method]
        if frame.empty:
            continue
        metrics = compute_metrics(frame["label_id"].astype(int).to_numpy(), frame["prob_responder"].astype(float).to_numpy())
        rows.append(
            {
                "method": method,
                "n_events": int(frame["target_id"].nunique()),
                "n_ensembled_models": int(frame["n_models_ensembled"].median()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def save_metric_bars(fold_summary: pd.DataFrame, ensemble_summary: pd.DataFrame, metrics: List[str], output: Path) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 4.8), sharex=False)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        frame = fold_summary[fold_summary["method"].isin(METHOD_ORDER)].copy()
        frame["method"] = pd.Categorical(frame["method"], METHOD_ORDER, ordered=True)
        frame = frame.sort_values("method")
        labels = frame["method"].tolist()
        values = frame[f"{metric}_mean"].astype(float).to_numpy()
        std = frame[f"{metric}_std"].astype(float).fillna(0.0).to_numpy()
        x = np.arange(len(labels))
        ax.bar(x, values, yerr=std, capsize=4, color=[COLORS[label] for label in labels], edgecolor="#333333", linewidth=1.0, width=0.58)
        ensemble_lookup = dict(zip(ensemble_summary["method"], ensemble_summary[metric]))
        for idx, label in enumerate(labels):
            ens_value = ensemble_lookup.get(label, np.nan)
            if np.isfinite(ens_value):
                ax.scatter(idx, ens_value, color="white", edgecolor="#111111", linewidth=1.0, s=42, zorder=4)
                ax.text(idx, max(values[idx], ens_value), f"{ens_value:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5, fontweight="bold")
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        ax.set_title(f"{METRIC_LABELS[metric]} by method", fontsize=12, fontweight="bold")
        set_tight_ylim(ax, np.concatenate([values - std, values + std, ensemble_summary[metric].to_numpy()]), min_span=0.1)
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_scale_trend(fold_metrics: pd.DataFrame, ensemble_summary: pd.DataFrame, output: Path) -> None:
    metrics = ["roc_auc", "average_precision", "balanced_accuracy"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharex=True)
    for ax, metric in zip(axes, metrics):
        frame = fold_metrics[fold_metrics["scale"].isin(SCALE_ORDER)].copy()
        frame["scale"] = pd.Categorical(frame["scale"], SCALE_ORDER, ordered=True)
        means = frame.groupby("scale", observed=False)[metric].mean().reindex(SCALE_ORDER)
        stds = frame.groupby("scale", observed=False)[metric].std().reindex(SCALE_ORDER).fillna(0.0)
        x = np.arange(len(SCALE_ORDER))
        for idx, scale in enumerate(SCALE_ORDER):
            vals = frame.loc[frame["scale"] == scale, metric].astype(float).dropna().to_numpy()
            jitter = np.linspace(-0.09, 0.09, len(vals)) if len(vals) else []
            ax.scatter(np.full(len(vals), idx) + jitter, vals, s=38, color=COLORS[SCALE_LABELS[scale]], edgecolor="#333333", linewidth=0.5, alpha=0.9)
        ax.plot(x, means, color="#222222", linewidth=2.0)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#222222", capsize=4, linewidth=1.0)
        ensemble_lookup = dict(zip(ensemble_summary["method"], ensemble_summary[metric]))
        for idx, scale in enumerate(SCALE_ORDER):
            value = ensemble_lookup.get(SCALE_LABELS[scale], np.nan)
            if np.isfinite(value):
                ax.scatter(idx, value, color="white", edgecolor="#111111", linewidth=1.0, s=48, zorder=4)
                ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SCALE_LABELS[scale].replace("MGM2-", "").replace("xlarge", "XLarge").title() for scale in SCALE_ORDER], fontsize=10, fontweight="bold")
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        ax.set_title(f"Scale trend: {METRIC_LABELS[metric]}", fontsize=12, fontweight="bold")
        set_tight_ylim(ax, np.concatenate([frame[metric].to_numpy(), ensemble_summary[metric].to_numpy()]), min_span=0.1)
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_roc_pr_curves(ensemble_predictions: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.1))
    for method in METHOD_ORDER:
        frame = ensemble_predictions[ensemble_predictions["method"] == method]
        if frame.empty:
            continue
        labels = frame["label_id"].astype(int).to_numpy()
        scores = frame["prob_responder"].astype(float).to_numpy()
        fpr, tpr = roc_curve_points(labels, scores)
        recall, precision = pr_curve_points(labels, scores)
        auc = binary_auc(labels, scores)
        ap = average_precision(labels, scores)
        axes[0].plot(fpr, tpr, color=COLORS[method], linewidth=2.0, label=f"{method} ({auc:.3f})")
        axes[1].plot(recall, precision, color=COLORS[method], linewidth=2.0, label=f"{method} ({ap:.3f})")
    axes[0].plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("Ensemble ROC curves", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Ensemble precision-recall curves", fontsize=12, fontweight="bold")
    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        style_axis(ax)
        ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_probability_strip(ensemble_predictions: pd.DataFrame, output: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in set(ensemble_predictions["method"])]
    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    y_positions = np.arange(len(methods))
    for idx, method in enumerate(methods):
        frame = ensemble_predictions[ensemble_predictions["method"] == method].copy()
        colors = np.where(frame["label_id"].astype(int).to_numpy() == 1, "#D55E00", "#0072B2")
        jitter = np.linspace(-0.16, 0.16, len(frame)) if len(frame) else []
        ax.scatter(
            frame["prob_responder"].astype(float),
            np.full(len(frame), idx) + jitter,
            c=colors,
            edgecolor="#333333",
            linewidth=0.35,
            s=42,
            alpha=0.9,
        )
    ax.axvline(0.5, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(methods, fontsize=10, fontweight="bold")
    ax.set_xlabel("Ensembled probability of responder", fontsize=11)
    ax.set_title("Response probability distribution by true label", fontsize=12, fontweight="bold")
    ax.scatter([], [], c="#D55E00", edgecolor="#333333", label="True responder")
    ax.scatter([], [], c="#0072B2", edgecolor="#333333", label="True non-responder")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_per_study_accuracy(ensemble_predictions: pd.DataFrame, output: Path) -> None:
    frame = ensemble_predictions.copy()
    frame["correct"] = (frame["pred_id"].astype(int) == frame["label_id"].astype(int)).astype(float)
    counts = frame[frame["method"] == "MLP"]["study"].value_counts()
    studies = counts.index.tolist()
    pivot = frame.groupby(["study", "method"])["correct"].mean().unstack("method").reindex(studies)
    pivot = pivot[[method for method in METHOD_ORDER if method in pivot.columns]]
    fig, ax = plt.subplots(figsize=(max(11.0, len(studies) * 0.9), 5.2))
    x = np.arange(len(pivot))
    width = 0.86 / max(len(pivot.columns), 1)
    for idx, method in enumerate(pivot.columns):
        ax.bar(
            x + (idx - (len(pivot.columns) - 1) / 2) * width,
            pivot[method],
            width=width,
            color=COLORS[method],
            edgecolor="#333333",
            linewidth=0.45,
            label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{study}\n(n={counts[study]})" for study in studies], rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Ensemble accuracy")
    ax.set_title("Per-study FMT response classification", fontsize=12, fontweight="bold")
    set_tight_ylim(ax, pivot.to_numpy().ravel(), min_span=0.35)
    style_axis(ax)
    ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot FMT response classification results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mgm2_dir", type=Path, default=Path("outputs_downstreams/FMT/community_classification"))
    parser.add_argument("--mlp_dir", type=Path, default=Path("outputs_downstreams/FMT/mlp_classification_baseline"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures/FMT"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = pd.concat(
        [
            load_mlp_fold_metrics(args.mlp_dir),
            load_mgm2_fold_metrics(args.mgm2_dir),
        ],
        ignore_index=True,
        sort=False,
    )
    predictions = load_all_predictions(args.mgm2_dir, args.mlp_dir)
    ensemble_predictions = build_ensemble_predictions(predictions)
    fold_metrics, ensemble_predictions = add_majority_rows(fold_metrics, ensemble_predictions)
    fold_summary = summarize_fold_metrics(fold_metrics)
    ensemble_summary = summarize_ensemble(ensemble_predictions)

    fold_metrics["method"] = pd.Categorical(fold_metrics["method"], METHOD_ORDER, ordered=True)
    fold_metrics = fold_metrics.sort_values(["method", "seed", "fold"]).reset_index(drop=True)
    fold_summary["method"] = pd.Categorical(fold_summary["method"], METHOD_ORDER, ordered=True)
    fold_summary = fold_summary.sort_values("method").reset_index(drop=True)
    ensemble_summary["method"] = pd.Categorical(ensemble_summary["method"], METHOD_ORDER, ordered=True)
    ensemble_summary = ensemble_summary.sort_values("method").reset_index(drop=True)

    fold_metrics.to_csv(args.output_dir / "fmt_response_fold_metrics.csv", index=False)
    fold_summary.to_csv(args.output_dir / "fmt_response_fold_summary.csv", index=False)
    ensemble_predictions.to_csv(args.output_dir / "fmt_response_ensemble_predictions.csv", index=False)
    ensemble_summary.to_csv(args.output_dir / "fmt_response_ensemble_summary.csv", index=False)

    save_metric_bars(
        fold_summary,
        ensemble_summary,
        ["roc_auc", "average_precision"],
        args.output_dir / "fmt_response_auc_ap_by_method.png",
    )
    save_metric_bars(
        fold_summary,
        ensemble_summary,
        ["accuracy", "balanced_accuracy", "macro_f1"],
        args.output_dir / "fmt_response_accuracy_by_method.png",
    )
    save_scale_trend(fold_metrics, ensemble_summary, args.output_dir / "fmt_response_scale_trend.png")
    save_roc_pr_curves(ensemble_predictions, args.output_dir / "fmt_response_roc_pr_curves.png")
    save_probability_strip(ensemble_predictions, args.output_dir / "fmt_response_probability_strip.png")
    save_per_study_accuracy(ensemble_predictions, args.output_dir / "fmt_response_per_study_accuracy.png")

    print("Fold-level mean +/- std:")
    print(fold_summary.to_string(index=False))
    print()
    print("Ensemble metrics:")
    print(ensemble_summary.to_string(index=False))
    print()
    print(f"Wrote FMT response classification figures and tables to {args.output_dir}")


if __name__ == "__main__":
    main()
