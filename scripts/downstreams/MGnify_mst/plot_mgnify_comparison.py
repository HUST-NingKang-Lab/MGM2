#!/usr/bin/env python3
"""
MGnify MST comparison across models and hierarchy levels.

Metrics are computed independently for each model:
  * unknown labels are excluded.
  * only labels shared by that model's train and test label sets are scored.
  * probability columns are matched to that model's own class order.
  * class spaces are not forced to align across different models.

Usage:
  python scripts/downstreams/MGnify_mst/plot_mgnify_comparison.py
"""

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

OUTPUT_DIR = Path("figures/mgnify_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 22, 33, 44, 55]
LEVELS = ["env_level_1", "env_level_2", "env_level_3", "env_level_4", "env_level_5"]
LEVEL_LABELS = ["L1", "L2", "L3", "L4", "L5"]

LOCAL_TRAIN_META = Path("data/downstreams/MGnify_mst/train_meta_split.csv")
LOCAL_TEST_META = Path("data/downstreams/MGnify_mst/test_meta_split.csv")
MGM1_DIR = Path("/home/zhanghaohong/data6/projects/MGM2Benchmarks/MGnify_mst/MGM/finetune_outputs")
DP_DIR = Path("/home/zhanghaohong/data6/projects/MGM2Benchmarks/MGnify_mst/deepphylo/results/benchmark_raw")
EXPERT_DIR = Path("/home/zhanghaohong/data6/projects/MGM2Benchmarks/MGnify_mst/expert/outputs")
EXPERT_TRAIN_META = Path("/home/zhanghaohong/data6/projects/MGM2Benchmarks/MGnify_mst/expert/train_meta.csv")
EXPERT_TEST_META = Path("/home/zhanghaohong/data6/projects/MGM2Benchmarks/MGnify_mst/expert/test_meta.csv")

train_meta = pd.read_csv(LOCAL_TRAIN_META)
test_meta = pd.read_csv(LOCAL_TEST_META)
test_meta["SampleID"] = test_meta["SampleID"].astype(str).str.strip()


def normalize_label(label):
    if pd.isna(label):
        return None
    value = str(label).strip()
    if not value:
        return None
    return value.replace("root:", "")


def is_known_label(label):
    value = normalize_label(label)
    return value is not None and value.lower() != "unknown"


def known_label_set(values):
    return {normalize_label(v) for v in values if is_known_label(v)}


def shared_known_labels(train_values, test_values):
    return sorted(known_label_set(train_values) & known_label_set(test_values))


def local_shared_classes(level):
    return shared_known_labels(train_meta[level], test_meta[level])


def score_multiclass(y_true, probs, prob_labels, eval_labels):
    """Return macro ROC-AUC and macro F-max on a model-specific label space."""
    y_true = np.array([normalize_label(y) for y in y_true], dtype=object)
    prob_labels = [normalize_label(label) for label in prob_labels]
    probs = np.asarray(probs, dtype=float)

    available = {
        label
        for label in prob_labels
        if label is not None and is_known_label(label)
    }
    target = set(eval_labels) & known_label_set(y_true) & available
    labels = [label for label in eval_labels if label in target]
    if len(labels) < 2:
        return None

    col_index = {}
    for idx, label in enumerate(prob_labels):
        if label in labels and label not in col_index:
            col_index[label] = idx
    labels = [label for label in labels if label in col_index]
    if len(labels) < 2:
        return None

    mask = np.isin(y_true, labels)
    if int(mask.sum()) < 2:
        return None

    y_eval = y_true[mask]
    p_eval = probs[mask][:, [col_index[label] for label in labels]]
    mapping = {label: idx for idx, label in enumerate(labels)}
    y_id = np.array([mapping[label] for label in y_eval], dtype=int)

    if len(np.unique(y_id)) < 2:
        return None

    one_hot = np.eye(len(labels))[y_id]
    try:
        roc = roc_auc_score(one_hot, p_eval, multi_class="ovr", average="macro")
    except ValueError:
        roc = float("nan")

    fmax_values = []
    for class_id in range(len(labels)):
        binary_y = (y_id == class_id).astype(int)
        if binary_y.sum() == 0 or binary_y.sum() == len(binary_y):
            continue
        precision, recall, _ = precision_recall_curve(binary_y, p_eval[:, class_id])
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
        fmax_values.append(float(np.nanmax(f1)))

    fmax = float(np.mean(fmax_values)) if fmax_values else float("nan")
    return {
        "roc": float(roc),
        "fmax": fmax,
        "n_eval_samples": int(mask.sum()),
        "n_eval_classes": len(labels),
        "labels": labels,
    }


def summarize_scores(scores):
    scores = [score for score in scores if score is not None and not np.isnan(score["roc"])]
    if not scores:
        return None
    rocs = [score["roc"] for score in scores]
    fmaxs = [score["fmax"] for score in scores]
    class_counts = [score["n_eval_classes"] for score in scores]
    sample_counts = [score["n_eval_samples"] for score in scores]
    labels_by_seed = [score["labels"] for score in scores]
    return {
        "roc_mean": float(np.mean(rocs)),
        "roc_std": float(np.std(rocs)),
        "fmax_mean": float(np.mean(fmaxs)),
        "fmax_std": float(np.std(fmaxs)),
        "n": len(scores),
        "n_eval_classes_mean": float(np.mean(class_counts)),
        "n_eval_classes_min": int(np.min(class_counts)),
        "n_eval_classes_max": int(np.max(class_counts)),
        "n_eval_samples_mean": float(np.mean(sample_counts)),
        "n_eval_samples_min": int(np.min(sample_counts)),
        "n_eval_samples_max": int(np.max(sample_counts)),
        "rocs": rocs,
        "fmaxs": fmaxs,
        "labels_by_seed": labels_by_seed,
    }


def read_csv_robust(path, **kwargs):
    kwargs.setdefault("low_memory", False)
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError:
        return pd.read_csv(path, on_bad_lines="skip", **kwargs)


def named_prob_columns(df):
    """Parse columns like prob_0_Engineered into class-ordered labels."""
    parsed = []
    for col in df.columns:
        match = re.match(r"^prob_(\d+)_(.+)$", col)
        if match:
            parsed.append((int(match.group(1)), normalize_label(match.group(2)), col))
    if not parsed:
        return [], []
    parsed.sort(key=lambda item: item[0])
    return [label for _, label, _ in parsed], [col for _, _, col in parsed]


def numeric_prob_columns(df):
    parsed = []
    for col in df.columns:
        match = re.match(r"^prob_(\d+)$", col)
        if match:
            parsed.append((int(match.group(1)), col))
    parsed.sort(key=lambda item: item[0])
    return [col for _, col in parsed]


def prob_columns_with_ids(df):
    parsed = []
    for col in df.columns:
        match = re.match(r"^prob_(\d+)(?:_(.+))?$", col)
        if match:
            fallback_label = normalize_label(match.group(2)) if match.group(2) else None
            parsed.append((int(match.group(1)), fallback_label, col))
    parsed.sort(key=lambda item: item[0])
    return parsed


def load_label_to_id(path):
    if not path.exists():
        return {}
    with path.open() as handle:
        raw = json.load(handle)
    return {normalize_label(label): int(idx) for label, idx in raw.items()}


def labels_from_id_columns(df, n_prob_cols):
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


def result_for_level(scores):
    return summarize_scores(scores)


# ===================================================================
# MGM2
# ===================================================================
def eval_mgm2(scale):
    results = {}
    for level in LEVELS:
        eval_labels = local_shared_classes(level)
        scores = []
        for seed in SEEDS:
            run_dir = Path("outputs_downstreams/MGnify_mst") / scale / level / f"seed_{seed}"
            pred_path = run_dir / "test_predictions.csv"
            if not pred_path.exists():
                continue
            df = read_csv_robust(pred_path, dtype={"SampleID": str})
            df["SampleID"] = df["SampleID"].astype(str).str.strip()
            merged = test_meta[["SampleID", level]].merge(df, on="SampleID", how="inner")
            if merged.empty:
                continue

            parsed_probs = prob_columns_with_ids(merged)
            if not parsed_probs:
                continue
            label_to_id = load_label_to_id(run_dir / "label_to_id.json")
            id_to_label = {idx: label for label, idx in label_to_id.items()}
            prob_labels = [
                id_to_label.get(class_id, fallback_label)
                for class_id, fallback_label, _ in parsed_probs
            ]
            prob_cols = [col for _, _, col in parsed_probs]

            score = score_multiclass(
                y_true=merged[level].values,
                probs=merged[prob_cols].values,
                prob_labels=prob_labels,
                eval_labels=eval_labels,
            )
            scores.append(score)
        summary = result_for_level(scores)
        if summary:
            results[level] = summary
    return results


# ===================================================================
# MGM1
# ===================================================================
def eval_mgm1():
    results = {}
    for li, level in enumerate(LEVELS):
        eval_labels = local_shared_classes(level)
        mgm1_level = f"layer{li + 1}"
        scores = []
        for seed in SEEDS:
            pred_path = MGM1_DIR / f"predictions_{mgm1_level}_seed{seed}" / "y_score.csv"
            if not pred_path.exists():
                continue
            df = pd.read_csv(pred_path)
            prob_cols = [col for col in df.columns if col != "Unnamed: 0"]
            score = score_multiclass(
                y_true=test_meta[level].values,
                probs=df[prob_cols].values,
                prob_labels=prob_cols,
                eval_labels=eval_labels,
            )
            scores.append(score)
        summary = result_for_level(scores)
        if summary:
            results[level] = summary
    return results


# ===================================================================
# DeepPhylo
# ===================================================================
def eval_deepphylo():
    results = {}
    for li, level in enumerate(LEVELS):
        dp_level = f"l{li + 1}"
        rocs = []
        fmaxs = []
        class_counts = []
        sample_counts = []
        labels_by_seed = []
        for seed in SEEDS:
            metrics_path = DP_DIR / f"seed_{seed}" / dp_level / "metrics.json"
            if not metrics_path.exists():
                continue
            with metrics_path.open() as handle:
                metrics = json.load(handle)
            if "roc_auc_macro" not in metrics or "f1_macro" not in metrics:
                continue
            rocs.append(float(metrics["roc_auc_macro"]))
            # DeepPhylo benchmark stores macro-F1 rather than F-max. Keep the
            # shared plot slot for comparison without recomputing predictions.
            fmaxs.append(float(metrics["f1_macro"]))
            class_counts.append(int(metrics.get("n_classes", 0)))
            sample_counts.append(int(metrics.get("n_test", 0)))
            labels_by_seed.append([normalize_label(label) for label in metrics.get("label_names", [])])
        if rocs:
            results[level] = {
                "roc_mean": float(np.mean(rocs)),
                "roc_std": float(np.std(rocs)),
                "fmax_mean": float(np.mean(fmaxs)),
                "fmax_std": float(np.std(fmaxs)),
                "n": len(rocs),
                "n_eval_classes_mean": float(np.mean(class_counts)) if class_counts else 0.0,
                "n_eval_classes_min": int(np.min(class_counts)) if class_counts else 0,
                "n_eval_classes_max": int(np.max(class_counts)) if class_counts else 0,
                "n_eval_samples_mean": float(np.mean(sample_counts)) if sample_counts else 0.0,
                "n_eval_samples_min": int(np.min(sample_counts)) if sample_counts else 0,
                "n_eval_samples_max": int(np.max(sample_counts)) if sample_counts else 0,
                "rocs": rocs,
                "fmaxs": fmaxs,
                "labels_by_seed": labels_by_seed,
                "fmax_source": "deepphylo_metrics_f1_macro",
            }
    return results


# ===================================================================
# Expert
# ===================================================================
def parse_expert_label(env_str, layer):
    parts = str(env_str).split(":")
    if len(parts) < layer:
        return None
    return normalize_label(":".join(parts[:layer]))


def eval_expert():
    results = {}
    if not EXPERT_TRAIN_META.exists() or not EXPERT_TEST_META.exists():
        return results
    exp_train = pd.read_csv(EXPERT_TRAIN_META)
    exp_test = pd.read_csv(EXPERT_TEST_META)
    exp_test["SampleID"] = exp_test["SampleID"].astype(str).str.strip()
    seed_dirs = sorted(glob.glob(str(EXPERT_DIR / "seed*_predictions")))

    for li, level in enumerate(LEVELS):
        expert_level = li + 2
        train_labels = exp_train["Env"].apply(lambda value: parse_expert_label(value, expert_level))
        test_labels = exp_test["Env"].apply(lambda value: parse_expert_label(value, expert_level))
        eval_labels = shared_known_labels(train_labels, test_labels)
        if len(eval_labels) < 2:
            continue
        y_true_map = dict(zip(exp_test["SampleID"], test_labels))

        scores = []
        pred_file = f"layer-{expert_level}.csv"
        for seed_dir in seed_dirs[: len(SEEDS)]:
            pred_path = Path(seed_dir) / pred_file
            if not pred_path.exists():
                continue
            preds = pd.read_csv(pred_path)
            preds.columns = ["SampleID"] + [normalize_label(col) for col in preds.columns[1:]]
            preds["SampleID"] = preds["SampleID"].astype(str).str.strip()
            prob_cols = [col for col in preds.columns if col != "SampleID"]
            common = sorted(set(preds["SampleID"]) & set(y_true_map))
            if not common:
                continue
            score = score_multiclass(
                y_true=[y_true_map[sid] for sid in common],
                probs=preds.set_index("SampleID").loc[common][prob_cols].values,
                prob_labels=prob_cols,
                eval_labels=eval_labels,
            )
            scores.append(score)
        summary = result_for_level(scores)
        if summary:
            results[level] = summary
    return results


# ===================================================================
# RF
# ===================================================================
def eval_rf():
    results = {}
    for level in LEVELS:
        eval_labels = local_shared_classes(level)
        scores = []
        for seed in SEEDS:
            pred_path = Path("outputs_downstreams/MGnify_mst_rf") / level / f"seed_{seed}" / "test_predictions.csv"
            if not pred_path.exists():
                continue
            df = pd.read_csv(pred_path, dtype={"SampleID": str})
            df["SampleID"] = df["SampleID"].astype(str).str.strip()
            merged = test_meta[["SampleID", level]].merge(df, on="SampleID", how="inner")
            if merged.empty:
                continue
            prob_cols = numeric_prob_columns(merged)
            if not prob_cols:
                continue
            prob_labels = labels_from_id_columns(merged, len(prob_cols))
            score = score_multiclass(
                y_true=merged[level].values,
                probs=merged[prob_cols].values,
                prob_labels=prob_labels,
                eval_labels=eval_labels,
            )
            scores.append(score)
        summary = result_for_level(scores)
        if summary:
            results[level] = summary
    return results


def available_mgm2_scales():
    scales = [
        ("scale_small", "MGM2-small"),
        ("scale_medium", "MGM2-medium"),
        ("scale_large", "MGM2-large"),
        ("scale_xlarge", "MGM2-xlarge"),
    ]
    available = []
    for scale_dir, label in scales:
        if any((Path("outputs_downstreams/MGnify_mst") / scale_dir).glob("env_level_*/seed_*/test_predictions.csv")):
            available.append((scale_dir, label))
    return available


# ===================================================================
# Collect
# ===================================================================
MODELS = {
    "RF": eval_rf(),
    "Expert": eval_expert(),
    "DeepPhylo": eval_deepphylo(),
    "MGM1": eval_mgm1(),
    "MGM2-small": eval_mgm2("scale_small"),
    "MGM2-medium": eval_mgm2("scale_medium"),
    "MGM2-large": eval_mgm2("scale_large"),
}

print(
    f"{'Model':>15} {'Level':>6} {'Seeds':>6} {'ROC-AUC':>16} "
    f"{'F-max':>16} {'Classes':>10} {'Samples':>10}"
)
print("-" * 92)
for model_name, model_results in MODELS.items():
    for level in LEVELS:
        row = model_results.get(level)
        if not row:
            continue
        print(
            f"{model_name:>15} {level[-1:]:>6} {row['n']:>6} "
            f"{row['roc_mean']:>8.4f}+/-{row['roc_std']:<6.4f} "
            f"{row['fmax_mean']:>8.4f}+/-{row['fmax_std']:<6.4f} "
            f"{row['n_eval_classes_min']:>3}-{row['n_eval_classes_max']:<3} "
            f"{row['n_eval_samples_min']:>4}-{row['n_eval_samples_max']:<4}"
        )

all_serializable = {}
for model_name, model_results in MODELS.items():
    all_serializable[model_name] = {}
    for level, row in model_results.items():
        all_serializable[model_name][level] = {
            "roc_mean": row["roc_mean"],
            "roc_std": row["roc_std"],
            "fmax_mean": row["fmax_mean"],
            "fmax_std": row["fmax_std"],
            "n_seeds": row["n"],
            "n_eval_classes_mean": row["n_eval_classes_mean"],
            "n_eval_classes_min": row["n_eval_classes_min"],
            "n_eval_classes_max": row["n_eval_classes_max"],
            "n_eval_samples_mean": row["n_eval_samples_mean"],
            "n_eval_samples_min": row["n_eval_samples_min"],
            "n_eval_samples_max": row["n_eval_samples_max"],
            "rocs": [float(value) for value in row["rocs"]],
            "fmaxs": [float(value) for value in row["fmaxs"]],
            "labels_by_seed": row["labels_by_seed"],
        }
        if "fmax_source" in row:
            all_serializable[model_name][level]["fmax_source"] = row["fmax_source"]

with (OUTPUT_DIR / "all_metrics.json").open("w") as handle:
    json.dump(all_serializable, handle, indent=2)


# ===================================================================
# Plot
# ===================================================================
plot_colors = {
    "RF": "#D4782B",
    "MGM1": "#7CB5EC",
    "DeepPhylo": "#6D2A1F",
    "Expert": "#4E7B48",
    "MGM2-small": "#4363D8",
    "MGM2-medium": "#1A237E",
    "MGM2-large": "#00897B",
    "MGM2-xlarge": "#6A1B9A",
}
plot_order = ["RF", "Expert", "DeepPhylo", "MGM1", "MGM2-small", "MGM2-medium", "MGM2-large"]

x = np.arange(len(LEVELS))
n_models = max(len(plot_order), 1)
bar_width = 0.8 / n_models
rng = np.random.default_rng(42)
jitter = bar_width * 0.25

for metric_key, ylabel in [("roc", "ROC-AUC"), ("fmax", "F-max")]:
    fig, ax = plt.subplots(figsize=(16, 6))
    for model_idx, model_name in enumerate(plot_order):
        model_results = MODELS[model_name]
        means = []
        stds = []
        seed_points = []
        for level_idx, level in enumerate(LEVELS):
            row = model_results.get(level)
            if row:
                means.append(row[f"{metric_key}_mean"])
                stds.append(row[f"{metric_key}_std"])
                for value in row[f"{metric_key}s"]:
                    seed_points.append((level_idx, value))
            else:
                means.append(np.nan)
                stds.append(0.0)
        if all(np.isnan(value) for value in means):
            continue

        offset = (model_idx - n_models / 2 + 0.5) * bar_width
        for level_idx, mean_value in enumerate(means):
            if np.isnan(mean_value):
                continue
            yerr = [[0], [stds[level_idx]]] if stds[level_idx] > 0 else None
            ax.bar(
                x[level_idx] + offset,
                mean_value,
                bar_width,
                yerr=yerr,
                capsize=2 if yerr else 0,
                color=plot_colors[model_name],
                edgecolor="#333333",
                linewidth=0.5,
                label=model_name if level_idx == 0 else None,
            )
            ax.text(
                x[level_idx] + offset,
                mean_value + 0.02,
                f"{mean_value:.2f}",
                ha="center",
                fontsize=6,
                rotation=90,
                color="#555555",
            )
        for level_idx, value in seed_points:
            if not np.isnan(value):
                ax.scatter(
                    x[level_idx] + offset + rng.uniform(-jitter, jitter),
                    value,
                    s=8,
                    c="#222222",
                    alpha=0.3,
                    edgecolors="none",
                    zorder=10,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_LABELS, fontsize=10)
    ax.set_ylabel(f"{ylabel} (macro)", fontsize=12)
    ax.set_title(f"MGnify MST - {ylabel}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, ncol=3, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1.08)
    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(
            OUTPUT_DIR / f"mgnify_{metric_key}.{fmt}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close()

print(f"\nDone. {OUTPUT_DIR}/")
