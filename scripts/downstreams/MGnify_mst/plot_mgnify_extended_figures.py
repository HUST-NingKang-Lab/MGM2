#!/usr/bin/env python3
"""Additional MGnify MST figures for main/supplementary panels.

The script reads existing prediction CSVs and produces:
  * MGM2 scale x hierarchy heatmaps.
  * Per-class frequency vs AUROC scatter.
  * Normalized confusion matrices.
  * Confidence/calibration diagnostics.
  * Optional xlarge frozen-backbone CLS embedding dimensionality reduction.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

LEVELS = ["env_level_1", "env_level_2", "env_level_3", "env_level_4", "env_level_5"]
LEVEL_LABELS = ["L1", "L2", "L3", "L4", "L5"]
SEEDS = [11, 22, 33, 44, 55]
MGM2_SCALES = ["scale_small", "scale_medium", "scale_large", "scale_xlarge"]
SCALE_LABELS = {
    "scale_small": "small",
    "scale_medium": "medium",
    "scale_large": "large",
    "scale_xlarge": "xlarge",
}
SCALE_COLORS = {
    "scale_small": "#4363D8",
    "scale_medium": "#1A237E",
    "scale_large": "#00897B",
    "scale_xlarge": "#6A1B9A",
}
MODEL_ORDER = [
    "RF",
    "Expert",
    "DeepPhylo",
    "MGM1",
    "MGM2-small",
    "MGM2-medium",
    "MGM2-large",
    "MGM2-xlarge",
]
MGM2_MODEL_ORDER = ["MGM2-small", "MGM2-medium", "MGM2-large", "MGM2-xlarge"]
MODEL_COLORS = {
    "RF": "#D4782B",
    "Expert": "#4E7B48",
    "DeepPhylo": "#6D2A1F",
    "MGM1": "#7CB5EC",
    "MGM2-small": "#4363D8",
    "MGM2-medium": "#1A237E",
    "MGM2-large": "#00897B",
    "MGM2-xlarge": "#6A1B9A",
}
MODEL_MARKERS = {
    "RF": "o",
    "Expert": "s",
    "DeepPhylo": "^",
    "MGM1": "D",
    "MGM2-small": "P",
    "MGM2-medium": "X",
    "MGM2-large": "v",
    "MGM2-xlarge": "*",
}
LEVEL_COLORS = {
    "Engineered": "#C44E52",
    "Environmental": "#4C72B0",
    "Host-associated": "#55A868",
}
RF_ROOT = Path("outputs_downstreams/MGnify_mst_rf")
MGM1_ROOT = Path("/home/zhanghaohong/data6/projects/MGM2Benchmarks/MGnify_mst/MGM/scratch_outputs")
DP_ROOT = Path("/home/zhanghaohong/data6/projects/MGM2Benchmarks/MGnify_mst/deepphylo/results/benchmark_raw")
EXPERT_ROOT = Path("/home/zhanghaohong/data6/projects/MGM2Benchmarks/MGnify_mst/expert/outputs")
EXPERT_TRAIN_META = Path("/home/zhanghaohong/data6/projects/MGM2Benchmarks/MGnify_mst/expert/train_meta.csv")
EXPERT_TEST_META = Path("/home/zhanghaohong/data6/projects/MGM2Benchmarks/MGnify_mst/expert/test_meta.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate extended MGnify MST figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", type=Path, default=Path("figures/mgnify_extended"))
    parser.add_argument("--mgnify_root", type=Path, default=Path("outputs_downstreams/MGnify_mst"))
    parser.add_argument("--stage2_root", type=Path, default=Path("outputs_grid/stage2/20260505_094404"))
    parser.add_argument("--metrics_json", type=Path, default=Path("figures/mgnify_comparison/all_metrics.json"))
    parser.add_argument("--test_metadata", type=Path, default=Path("data/downstreams/MGnify_mst/test_meta_split.csv"))
    parser.add_argument("--train_metadata", type=Path, default=Path("data/downstreams/MGnify_mst/train_meta_split.csv"))
    parser.add_argument("--seed", type=int, default=11, help="Representative seed for class/confusion/embedding panels")
    parser.add_argument("--scatter_scales", nargs="+", default=MGM2_SCALES)
    parser.add_argument("--scatter_models", nargs="+", default=MGM2_MODEL_ORDER)
    parser.add_argument("--scatter_levels", nargs="+", default=LEVELS)
    parser.add_argument("--embedding_scale", default="scale_xlarge")
    parser.add_argument("--embedding_level", default="env_level_1")
    parser.add_argument("--embedding_max_samples", type=int, default=5000)
    parser.add_argument("--embedding_batch_size", type=int, default=512)
    parser.add_argument("--embedding_num_workers", type=int, default=2)
    parser.add_argument("--embedding_gpu", type=int, default=0)
    parser.add_argument("--skip_embedding", action="store_true")
    parser.add_argument("--only_embedding", action="store_true")
    parser.add_argument("--only_class_frequency", action="store_true")
    return parser.parse_args()


def normalize_label(label) -> Optional[str]:
    if pd.isna(label):
        return None
    value = str(label).strip()
    if not value:
        return None
    return value.replace("root:", "")


def is_known_label(label) -> bool:
    value = normalize_label(label)
    return value is not None and value.lower() != "unknown"


def known_set(values: Iterable) -> set[str]:
    return {normalize_label(value) for value in values if is_known_label(value)}


def shared_known_labels(train_meta: pd.DataFrame, test_meta: pd.DataFrame, level: str) -> List[str]:
    return sorted(known_set(train_meta[level]) & known_set(test_meta[level]))


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("low_memory", False)
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError:
        return pd.read_csv(path, on_bad_lines="skip", **kwargs)


def load_label_to_id(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return {normalize_label(k): int(v) for k, v in json.load(handle).items()}


def prediction_usecols(path: Path, include_probs: bool) -> List[str]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    wanted = {"SampleID", "label", "label_id", "prediction", "pred_id", "probability", "correct"}
    if include_probs:
        wanted.update(col for col in header if re.match(r"^prob_\d+", col))
    return [col for col in header if col in wanted]


def prob_columns_with_labels(df: pd.DataFrame, label_to_id: Dict[str, int]) -> Tuple[List[str], List[str]]:
    parsed = []
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    for col in df.columns:
        match = re.match(r"^prob_(\d+)(?:_(.+))?$", col)
        if not match:
            continue
        class_id = int(match.group(1))
        fallback = normalize_label(match.group(2)) if match.group(2) else None
        parsed.append((class_id, id_to_label.get(class_id, fallback), col))
    parsed.sort(key=lambda item: item[0])
    labels = [label for _, label, _ in parsed]
    columns = [col for _, _, col in parsed]
    return labels, columns


def numeric_prob_columns(df: pd.DataFrame) -> List[str]:
    parsed = []
    for col in df.columns:
        match = re.match(r"^prob_(\d+)$", col)
        if match:
            parsed.append((int(match.group(1)), col))
    parsed.sort(key=lambda item: item[0])
    return [col for _, col in parsed]


def labels_from_id_columns(df: pd.DataFrame, n_prob_cols: int) -> List[Optional[str]]:
    id_to_label = {}
    for id_col, label_col in [("label_id", "label"), ("pred_id", "prediction")]:
        if id_col not in df.columns or label_col not in df.columns:
            continue
        pairs = df[[id_col, label_col]].dropna().drop_duplicates()
        for _, row in pairs.iterrows():
            try:
                label_id = int(row[id_col])
            except (TypeError, ValueError):
                continue
            label = normalize_label(row[label_col])
            if label is not None:
                id_to_label[label_id] = label
    return [id_to_label.get(idx) for idx in range(n_prob_cols)]


def parse_expert_label(env_str, layer: int) -> Optional[str]:
    parts = str(env_str).split(":")
    if len(parts) < layer:
        return None
    return normalize_label(":".join(parts[:layer]))


def representative_expert_seed(seed: int) -> int:
    if seed in SEEDS:
        return SEEDS.index(seed)
    return 0


def load_deepphylo_labels(seed: int, level_index: int) -> List[str]:
    metadata_path = DP_ROOT / "all_seeds.csv"
    if not metadata_path.exists():
        return []
    rows = pd.read_csv(metadata_path)
    layer = f"l{level_index + 1}"
    match = rows[(rows["seed"] == seed) & (rows["layer"] == layer)]
    if match.empty:
        return []
    raw = match.iloc[0]["label_names"]
    try:
        labels = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return []
    return [normalize_label(label) for label in labels]


def load_mgm2_predictions(root: Path, scale: str, level: str, seed: int, test_meta: pd.DataFrame, include_probs: bool = True) -> Optional[pd.DataFrame]:
    run_dir = root / scale / level / f"seed_{seed}"
    pred_path = run_dir / "test_predictions.csv"
    if not pred_path.exists():
        return None
    df = read_csv(pred_path, dtype={"SampleID": str}, usecols=prediction_usecols(pred_path, include_probs))
    df["SampleID"] = df["SampleID"].astype(str).str.strip()
    merged = test_meta[["SampleID", level]].merge(df, on="SampleID", how="inner")
    merged = merged.rename(columns={level: "true_label"})
    return merged


def per_class_auc(df: pd.DataFrame, prob_labels: List[str], prob_cols: List[str], eval_labels: List[str]) -> pd.DataFrame:
    rows = []
    y_true = df["true_label"].map(normalize_label).to_numpy(dtype=object)
    eval_labels = [label for label in eval_labels if label is not None]
    mask = np.isin(y_true, eval_labels)
    if int(mask.sum()) < 2:
        return pd.DataFrame(rows)
    y_eval = y_true[mask]
    masked_scores = df.loc[mask, prob_cols]
    label_to_col = {}
    for label, col in zip(prob_labels, prob_cols):
        label = normalize_label(label)
        if label is not None and label not in label_to_col:
            label_to_col[label] = col
    for label in eval_labels:
        if label not in label_to_col:
            continue
        col = label_to_col[label]
        binary = (y_eval == label).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue
        try:
            class_auc = roc_auc_score(binary, masked_scores[col].to_numpy(float))
        except ValueError:
            continue
        rows.append(
            {
                "label": label,
                "n_test": int(binary.sum()),
                "auc": float(class_auc),
                "frequency": float(binary.mean()),
            }
        )
    return pd.DataFrame(rows)


def macro_auc_for_run(root: Path, scale: str, level: str, seed: int, train_meta: pd.DataFrame, test_meta: pd.DataFrame) -> Optional[Dict]:
    df = load_mgm2_predictions(root, scale, level, seed, test_meta)
    if df is None or df.empty:
        return None
    labels = shared_known_labels(train_meta, test_meta, level)
    label_to_id = load_label_to_id(root / scale / level / f"seed_{seed}" / "label_to_id.json")
    prob_labels, prob_cols = prob_columns_with_labels(df, label_to_id)
    class_scores = per_class_auc(df, prob_labels, prob_cols, labels)
    if class_scores.empty:
        return None
    known_mask = df["true_label"].map(lambda x: normalize_label(x) in set(labels))
    return {
        "scale": scale,
        "level": level,
        "seed": seed,
        "macro_auc": float(class_scores["auc"].mean()),
        "n_classes": int(class_scores.shape[0]),
        "n_samples": int(known_mask.sum()),
        "prediction_rows": int(df.shape[0]),
        "complete_prediction": bool(df["SampleID"].nunique() >= len(test_meta) * 0.95),
    }


def savefig(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_scale_heatmaps(args: argparse.Namespace, train_meta: pd.DataFrame, test_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if args.metrics_json.exists():
        with args.metrics_json.open() as handle:
            all_metrics = json.load(handle)
        model_to_scale = {
            "MGM2-small": "scale_small",
            "MGM2-medium": "scale_medium",
            "MGM2-large": "scale_large",
            "MGM2-xlarge": "scale_xlarge",
        }
        for model_name, scale in model_to_scale.items():
            for level in LEVELS:
                row = all_metrics.get(model_name, {}).get(level)
                if not row:
                    continue
                rows.append(
                    {
                        "scale": scale,
                        "level": level,
                        "seed": "mean",
                        "macro_auc": float(row["roc_mean"]),
                        "n_classes": float(row.get("n_eval_classes_mean", np.nan)),
                        "n_samples": float(row.get("n_eval_samples_mean", np.nan)),
                        "prediction_rows": np.nan,
                        "complete_prediction": bool(row.get("n_eval_samples_mean", 0) > len(test_meta) * 0.75 or level in {"env_level_4", "env_level_5"}),
                    }
                )
    else:
        for scale in MGM2_SCALES:
            for level in LEVELS:
                row = macro_auc_for_run(args.mgnify_root, scale, level, args.seed, train_meta, test_meta)
                if row is not None:
                    rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "mgnify_mgm2_scale_level_metrics.csv", index=False)
    if metrics.empty:
        return metrics

    pivot = metrics.pivot_table(index="scale", columns="level", values="macro_auc", aggfunc="mean")
    pivot = pivot.reindex(MGM2_SCALES)[LEVELS]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    image = ax.imshow(pivot.to_numpy(float), cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(LEVELS)))
    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_yticks(np.arange(len(MGM2_SCALES)))
    ax.set_yticklabels([SCALE_LABELS[s] for s in MGM2_SCALES])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.74 else "black", fontsize=9)
    ax.set_title("MGM2 scale performance across MGnify hierarchy")
    ax.set_xlabel("Hierarchy level")
    ax.set_ylabel("Model scale")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Macro one-vs-rest AUROC")
    fig.tight_layout()
    savefig(fig, args.output_dir, "main_mgm2_scale_level_auc_heatmap")

    samples = metrics.pivot_table(index="scale", columns="level", values="n_samples", aggfunc="mean").reindex(MGM2_SCALES)[LEVELS]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    image = ax.imshow(samples.to_numpy(float), cmap="mako" if "mako" in plt.colormaps() else "Blues", aspect="auto")
    ax.set_xticks(np.arange(len(LEVELS)))
    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_yticks(np.arange(len(MGM2_SCALES)))
    ax.set_yticklabels([SCALE_LABELS[s] for s in MGM2_SCALES])
    for i in range(samples.shape[0]):
        for j in range(samples.shape[1]):
            value = samples.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{int(round(value)):,}", ha="center", va="center", fontsize=8)
    ax.set_title("Evaluated known-label samples")
    ax.set_xlabel("Hierarchy level")
    ax.set_ylabel("Model scale")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label("Samples")
    fig.tight_layout()
    savefig(fig, args.output_dir, "supp_mgm2_scale_level_sample_coverage")
    return metrics


def plot_class_frequency_scatter(args: argparse.Namespace, train_meta: pd.DataFrame, test_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    requested_models = [model for model in MGM2_MODEL_ORDER if model in set(args.scatter_models)]
    exp_train = exp_test = None
    if {"Expert", "DeepPhylo"} & set(requested_models) and EXPERT_TRAIN_META.exists() and EXPERT_TEST_META.exists():
        exp_train = read_csv(EXPERT_TRAIN_META)
        exp_test = read_csv(EXPERT_TEST_META, dtype={"SampleID": str})
        exp_test["SampleID"] = exp_test["SampleID"].astype(str).str.strip()

    for level_index, level in enumerate(args.scatter_levels):
        print(f"Per-class AUC: {level}", flush=True)
        local_eval_labels = shared_known_labels(train_meta, test_meta, level)

        if "RF" in requested_models:
            print("  RF", flush=True)
            pred_path = RF_ROOT / level / f"seed_{args.seed}" / "test_predictions.csv"
            if pred_path.exists():
                df = read_csv(pred_path, dtype={"SampleID": str})
                df["SampleID"] = df["SampleID"].astype(str).str.strip()
                merged = test_meta[["SampleID", level]].merge(df, on="SampleID", how="inner")
                merged = merged.rename(columns={level: "true_label"})
                prob_cols = numeric_prob_columns(merged)
                if prob_cols:
                    class_scores = per_class_auc(
                        merged,
                        labels_from_id_columns(merged, len(prob_cols)),
                        prob_cols,
                        local_eval_labels,
                    )
                    if not class_scores.empty:
                        class_scores["model"] = "RF"
                        class_scores["level"] = level
                        rows.append(class_scores)

        if "MGM1" in requested_models:
            print("  MGM1", flush=True)
            pred_path = MGM1_ROOT / f"predictions_layer{level_index + 1}_seed{args.seed}" / "y_score.csv"
            if pred_path.exists():
                df = read_csv(pred_path)
                prob_cols = [col for col in df.columns if col != "Unnamed: 0"]
                if prob_cols:
                    score_df = pd.concat(
                        [
                            pd.DataFrame({"true_label": test_meta[level].map(normalize_label).values}),
                            df[prob_cols].reset_index(drop=True),
                        ],
                        axis=1,
                    )
                    class_scores = per_class_auc(score_df, prob_cols, prob_cols, local_eval_labels)
                    if not class_scores.empty:
                        class_scores["model"] = "MGM1"
                        class_scores["level"] = level
                        rows.append(class_scores)

        if "DeepPhylo" in requested_models:
            print("  DeepPhylo", flush=True)
            pred_path = DP_ROOT / f"seed_{args.seed}" / f"l{level_index + 1}" / "predictions.npy"
            prob_labels = load_deepphylo_labels(args.seed, level_index)
            if pred_path.exists() and prob_labels and exp_train is not None and exp_test is not None:
                expert_level = level_index + 2
                train_labels = exp_train["Env"].apply(lambda value: parse_expert_label(value, expert_level))
                test_labels = exp_test["Env"].apply(lambda value: parse_expert_label(value, expert_level))
                eval_labels = shared_known_labels(pd.DataFrame({"label": train_labels}), pd.DataFrame({"label": test_labels}), "label")
                scores_array = np.load(pred_path)
                n = min(len(test_labels), scores_array.shape[0])
                prob_cols = [f"dp_prob_{idx}" for idx in range(scores_array.shape[1])]
                score_df = pd.concat(
                    [
                        pd.DataFrame({"true_label": test_labels.iloc[:n].map(normalize_label).values}),
                        pd.DataFrame(scores_array[:n, :], columns=prob_cols),
                    ],
                    axis=1,
                )
                class_scores = per_class_auc(score_df, prob_labels, prob_cols, eval_labels)
                if not class_scores.empty:
                    class_scores["model"] = "DeepPhylo"
                    class_scores["level"] = level
                    rows.append(class_scores)

        if "Expert" in requested_models and exp_train is not None and exp_test is not None:
            print("  Expert", flush=True)
            expert_level = level_index + 2
            train_labels = exp_train["Env"].apply(lambda value: parse_expert_label(value, expert_level))
            test_labels = exp_test["Env"].apply(lambda value: parse_expert_label(value, expert_level))
            eval_labels = shared_known_labels(pd.DataFrame({"label": train_labels}), pd.DataFrame({"label": test_labels}), "label")
            seed_id = representative_expert_seed(args.seed)
            pred_path = EXPERT_ROOT / f"seed{seed_id}_predictions" / f"layer-{expert_level}.csv"
            if pred_path.exists():
                preds = read_csv(pred_path, dtype={"Unnamed: 0": str})
                preds.columns = ["SampleID"] + [normalize_label(col) for col in preds.columns[1:]]
                preds["SampleID"] = preds["SampleID"].astype(str).str.strip()
                prob_cols = [col for col in preds.columns if col != "SampleID"]
                labeled = exp_test[["SampleID"]].copy()
                labeled["true_label"] = test_labels.values
                merged = labeled.merge(preds, on="SampleID", how="inner")
                class_scores = per_class_auc(merged, prob_cols, prob_cols, eval_labels)
                if not class_scores.empty:
                    class_scores["model"] = "Expert"
                    class_scores["level"] = level
                    rows.append(class_scores)

        for scale in args.scatter_scales:
            model_name = f"MGM2-{SCALE_LABELS.get(scale, scale.replace('scale_', ''))}"
            if model_name not in requested_models:
                continue
            print(f"  {model_name}", flush=True)
            df = load_mgm2_predictions(args.mgnify_root, scale, level, args.seed, test_meta)
            if df is None:
                continue
            label_to_id = load_label_to_id(args.mgnify_root / scale / level / f"seed_{args.seed}" / "label_to_id.json")
            prob_labels, prob_cols = prob_columns_with_labels(df, label_to_id)
            class_scores = per_class_auc(df, prob_labels, prob_cols, local_eval_labels)
            if class_scores.empty:
                continue
            class_scores["model"] = model_name
            class_scores["level"] = level
            rows.append(class_scores)
    scores = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    scores.to_csv(args.output_dir / "mgnify_per_class_auc.csv", index=False)
    if scores.empty:
        return scores

    levels = [level for level in LEVELS if level in set(args.scatter_levels)]
    level_label_map = dict(zip(LEVELS, LEVEL_LABELS))
    fig, axes = plt.subplots(1, len(levels), figsize=(max(5, 3.4 * len(levels)), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, level in zip(axes, levels):
        level_label = level_label_map[level]
        sub = scores[scores["level"] == level]
        for model_name in requested_models:
            model_sub = sub[sub["model"] == model_name]
            if model_sub.empty:
                continue
            ax.scatter(
                model_sub["n_test"],
                model_sub["auc"],
                s=np.clip(np.sqrt(model_sub["n_test"]) * 2.1, 16, 135),
                alpha=0.68,
                color=MODEL_COLORS.get(model_name, "#777777"),
                marker="o",
                label=model_name if level == levels[0] else None,
                edgecolor="white",
                linewidth=0.5,
            )
            if model_sub["n_test"].nunique() >= 2:
                fit_data = model_sub.sort_values("n_test")
                ax.plot(
                    fit_data["n_test"],
                    fit_data["auc"],
                    color=MODEL_COLORS.get(model_name, "#777777"),
                    linewidth=1.5,
                    alpha=0.9,
                )
        ax.set_xscale("log")
        ax.set_title(level_label)
        ax.set_xlabel("Class test samples")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Per-class AUROC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Class frequency vs MGM2 per-class separability", y=1.04, fontweight="bold")
    fig.tight_layout()
    savefig(fig, args.output_dir, "main_class_frequency_vs_auc")
    return scores


def plot_confusion_matrices(args: argparse.Namespace, test_meta: pd.DataFrame, scale: str = "scale_xlarge") -> None:
    levels = LEVELS[:3]
    fig, axes = plt.subplots(1, len(levels), figsize=(15, 4.8))
    for ax, level, level_label in zip(axes, levels, LEVEL_LABELS[:3]):
        df = load_mgm2_predictions(args.mgnify_root, scale, level, args.seed, test_meta, include_probs=False)
        if df is None or df.empty:
            ax.axis("off")
            continue
        y_true = df["true_label"].map(normalize_label)
        y_pred = df["prediction"].map(normalize_label)
        labels = [label for label in y_true.value_counts().index.tolist() if is_known_label(label)][:12]
        labels = sorted(labels)
        mask = y_true.isin(labels) & y_pred.isin(labels)
        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=labels, normalize="true")
        image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{SCALE_LABELS[scale]} {level_label}")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        short = [label[:18] + ("..." if len(label) > 18 else "") for label in labels]
        ax.set_xticklabels(short, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(short, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    if "image" in locals():
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02).set_label("Row-normalized fraction")
    fig.suptitle("Representative MGM2 confusion matrices", y=1.03, fontweight="bold")
    savefig(fig, args.output_dir, "supp_xlarge_confusion_matrices_l1_l3")


def plot_confidence_calibration(args: argparse.Namespace, test_meta: pd.DataFrame, scale: str = "scale_xlarge") -> pd.DataFrame:
    rows = []
    for level in LEVELS:
        df = load_mgm2_predictions(args.mgnify_root, scale, level, args.seed, test_meta, include_probs=False)
        if df is None or df.empty or "probability" not in df:
            continue
        known = df["true_label"].map(is_known_label)
        sub = df[known].copy()
        sub["level"] = level
        rows.append(sub[["level", "probability", "correct"]])
    data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    data.to_csv(args.output_dir / "mgnify_xlarge_confidence.csv", index=False)
    if data.empty:
        return data

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for correct, color, label in [(True, "#4C72B0", "correct"), (False, "#C44E52", "incorrect")]:
        sub = data[data["correct"].astype(bool) == correct]
        axes[0].hist(sub["probability"], bins=np.linspace(0, 1, 31), density=True, histtype="step", linewidth=2, color=color, label=label)
    axes[0].set_xlabel("Max predicted probability")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Confidence separates correct and incorrect predictions")
    axes[0].legend(frameon=False)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    bins = np.linspace(0, 1, 11)
    calibration_rows = []
    for level in LEVELS:
        sub = data[data["level"] == level]
        if sub.empty:
            continue
        bin_ids = np.digitize(sub["probability"], bins, right=True) - 1
        for idx in range(len(bins) - 1):
            mask = bin_ids == idx
            if not mask.any():
                continue
            calibration_rows.append(
                {
                    "level": level,
                    "confidence": float(sub.loc[mask, "probability"].mean()),
                    "accuracy": float(sub.loc[mask, "correct"].astype(bool).mean()),
                    "n": int(mask.sum()),
                }
            )
    cal = pd.DataFrame(calibration_rows)
    cal.to_csv(args.output_dir / "mgnify_xlarge_calibration.csv", index=False)
    axes[1].plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    for level, level_label in zip(LEVELS, LEVEL_LABELS):
        sub = cal[cal["level"] == level]
        if sub.empty:
            continue
        axes[1].plot(sub["confidence"], sub["accuracy"], marker="o", linewidth=1.5, label=level_label)
    axes[1].set_xlabel("Mean confidence")
    axes[1].set_ylabel("Empirical accuracy")
    axes[1].set_title("Reliability by hierarchy level")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, args.output_dir, "supp_xlarge_confidence_calibration")
    return data


def extract_embedding_panel(args: argparse.Namespace, test_meta: pd.DataFrame) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
        from src.configs.Config import MGM2Config
        from src.data.CommunityClassificationDataset import (
            create_community_classification_dataloaders,
            create_community_classification_test_loader,
        )
        from src.data.Dataset import BatchCollatorMGM2
        from src.models import MGM2Model
    except Exception as exc:
        print(f"Skipping embedding plot: imports failed: {exc}")
        return

    stage2_dir = args.stage2_root / args.embedding_scale
    ckpt = stage2_dir / "checkpoints" / "last.ckpt"
    config_path = stage2_dir / "config.json"
    label_path = args.mgnify_root / args.embedding_scale / args.embedding_level / f"seed_{args.seed}" / "label_to_id.json"
    if not ckpt.exists() or not config_path.exists() or not label_path.exists():
        print(f"Skipping embedding plot: missing stage2 checkpoint/config or downstream label map for {args.embedding_scale}")
        return

    config = MGM2Config.load(str(config_path))
    with label_path.open() as handle:
        label_to_id = {str(label): int(idx) for label, idx in json.load(handle).items()}
    data_dir = Path("data/downstreams/MGnify_mst")
    train_loader, _, _ = create_community_classification_dataloaders(
        data_path=data_dir / "train.pkl",
        metadata_path=data_dir / "train_meta_split.csv",
        ntv3_embedding_path=data_dir / "train.resolved_embeddings.h5",
        abundance_bin_config=config.abundance.__dict__,
        batch_size=args.embedding_batch_size,
        num_workers=args.embedding_num_workers,
        sample_id_column="SampleID",
        label_column=args.embedding_level,
        val_ratio=0.1,
        seed=config.seed,
    )
    test_loader_full = create_community_classification_test_loader(
        data_path=data_dir / "test.pkl",
        metadata_path=data_dir / "test_meta_split.csv",
        ntv3_embedding_path=data_dir / "test.resolved_embeddings.h5",
        abundance_bin_config=config.abundance.__dict__,
        label_to_id=label_to_id,
        batch_size=args.embedding_batch_size,
        num_workers=args.embedding_num_workers,
        sample_id_column="SampleID",
        label_column=args.embedding_level,
        abundance_preprocessor=train_loader.dataset.abundance_preprocessor,
        bin_manager=train_loader.dataset.bin_manager,
    )
    labels_by_sample = test_meta.set_index("SampleID")[args.embedding_level].map(normalize_label).to_dict()
    rng = np.random.default_rng(0)
    indices = np.arange(len(test_loader_full.dataset))
    if len(indices) > args.embedding_max_samples:
        indices = rng.choice(indices, size=args.embedding_max_samples, replace=False)
    subset = Subset(test_loader_full.dataset, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=args.embedding_batch_size,
        shuffle=False,
        num_workers=args.embedding_num_workers,
        collate_fn=BatchCollatorMGM2(),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.embedding_num_workers > 0,
    )
    model = MGM2Model(
        config=config.model.__dict__,
        num_bins=config.abundance.num_bins,
        enable_regression=config.abundance.enable_regression,
        enable_qwen3_alignment=config.qwen3_alignment.enabled,
        qwen3_projection_dim=config.qwen3_alignment.projection_dim,
        mlp_dropout=config.model.mlp_dropout,
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    if any(key.startswith("model.backbone.") for key in state_dict):
        backbone_state = {
            key[len("model.backbone."):]: value
            for key, value in state_dict.items()
            if key.startswith("model.backbone.")
        }
    elif any(key.startswith("backbone.") for key in state_dict):
        backbone_state = {
            key[len("backbone."):]: value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
    else:
        backbone_state = state_dict
    model.backbone.load_state_dict(backbone_state, strict=True)
    device = torch.device(f"cuda:{args.embedding_gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.embedding_gpu)
    model.to(device)
    model.eval()
    embeddings = []
    sample_ids = []
    true_labels = []
    with torch.no_grad():
        for batch in loader:
            model_batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=model_batch["input_ids"],
                    embeddings=model_batch["embeddings"],
                    abundance_values=model_batch["abundance_values"],
                    attention_mask=model_batch.get("attention_mask"),
                )
            embeddings.append(outputs["cls_hidden"].detach().float().cpu().numpy())
            batch_ids = [str(item) for item in batch["sample_id"]]
            sample_ids.extend(batch_ids)
            true_labels.extend([labels_by_sample.get(sample_id, "unknown") for sample_id in batch_ids])

    matrix = np.concatenate(embeddings, axis=0)
    pca = PCA(n_components=min(50, matrix.shape[1]), random_state=42)
    reduced = pca.fit_transform(matrix)
    method = "PCA"
    coords = reduced[:, :2]
    try:
        import umap

        reducer = umap.UMAP(n_neighbors=30, min_dist=0.08, metric="cosine", random_state=42)
        coords = reducer.fit_transform(reduced)
        method = "UMAP"
    except Exception as exc:
        print(f"UMAP unavailable, using PCA embedding plot: {exc}")

    coord_df = pd.DataFrame({"SampleID": sample_ids, "label": true_labels, "x": coords[:, 0], "y": coords[:, 1]})
    coord_df.to_csv(args.output_dir / "mgnify_xlarge_embedding_coordinates.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, count in coord_df["label"].value_counts().head(8).items():
        sub = coord_df[coord_df["label"] == label]
        ax.scatter(
            sub["x"],
            sub["y"],
            s=7,
            alpha=0.65,
            label=f"{label} ({count})",
            color=LEVEL_COLORS.get(label),
            linewidths=0,
        )
    rest = coord_df[~coord_df["label"].isin(coord_df["label"].value_counts().head(8).index)]
    if not rest.empty:
        ax.scatter(rest["x"], rest["y"], s=5, alpha=0.25, color="#999999", label="other")
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(f"{SCALE_LABELS.get(args.embedding_scale, args.embedding_scale)} frozen-backbone CLS embeddings")
    ax.legend(frameon=False, fontsize=8, markerscale=2, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, args.output_dir, "main_xlarge_sample_embedding_umap")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_meta = read_csv(args.test_metadata, dtype={"SampleID": str})
    train_meta = read_csv(args.train_metadata, dtype={"SampleID": str})
    test_meta["SampleID"] = test_meta["SampleID"].astype(str).str.strip()
    train_meta["SampleID"] = train_meta["SampleID"].astype(str).str.strip()

    if args.only_embedding:
        print(
            "Embedding UMAP/AvgBIO has moved to "
            "scripts/downstreams/MGnify_mst/plot_mgnify_embedding_umap.py"
        )
        return

    if args.only_class_frequency:
        class_scores = plot_class_frequency_scatter(args, train_meta, test_meta)
        print(f"Per-class scores: {len(class_scores)} rows")
    elif not args.only_embedding:
        metrics = plot_scale_heatmaps(args, train_meta, test_meta)
        print(f"Scale metrics: {len(metrics)} rows")
        class_scores = plot_class_frequency_scatter(args, train_meta, test_meta)
        print(f"Per-class scores: {len(class_scores)} rows")
        diagnostic_scale = args.embedding_scale
        plot_confusion_matrices(args, test_meta, scale=diagnostic_scale)
        confidence = plot_confidence_calibration(args, test_meta, scale=diagnostic_scale)
        print(f"Confidence rows: {len(confidence)}")
    print(f"Done. {args.output_dir}")


if __name__ == "__main__":
    main()
