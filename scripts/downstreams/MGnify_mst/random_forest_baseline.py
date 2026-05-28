#!/usr/bin/env python3
"""
Random Forest baseline for MGnify MST downstream task.

Uses OTU abundance vectors as features, trained independently for each
environment hierarchy level. Runs 5 random seeds identically to the MGM2
and Expert pipelines for fair comparison.

Results are saved as:
  outputs_downstreams/MGnify_mst_rf/{level}/seed_{seed}/
      test_predictions.csv
      metrics.json

Usage:
  python scripts/downstreams/MGnify_mst/random_forest_baseline.py \
      --n_estimators 300 --n_jobs 8
"""

import argparse, json, logging, os, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_otu_data(data_pkl_path: str):
    """Load OTU data dict from pkl."""
    logger.info("Loading %s ...", data_pkl_path)
    with open(data_pkl_path, "rb") as f:
        data = pickle.load(f)
    otu_data = data.get("otu_data", {})
    if not otu_data:
        raise ValueError("No 'otu_data' found in pkl file")
    return otu_data


def build_abundance_matrix(otu_data: dict, sample_ids: list, otu_to_col: dict):
    """Build a dense OTU abundance matrix using a pre-computed OTU→column mapping."""
    n_samples = len(sample_ids)
    n_otus = len(otu_to_col)
    logger.info("Building %d × %d abundance matrix ...", n_samples, n_otus)

    X = np.zeros((n_samples, n_otus), dtype=np.float32)
    for i, sid in enumerate(sample_ids):
        if sid not in otu_data:
            continue
        ids = otu_data[sid]["input_ids"]
        abd = otu_data[sid]["abundance_values"]
        for otu_id, val in zip(ids, abd):
            if otu_id >= 0 and val > 0 and otu_id in otu_to_col:
                X[i, otu_to_col[otu_id]] = val
    return X


def build_unified_vocab(otu_data: dict, sample_ids_list: list):
    """Build a unified OTU→column mapping across all sample sets."""
    all_otus = set()
    for sample_ids in sample_ids_list:
        for sid in sample_ids:
            if sid in otu_data:
                all_otus.update(otu_data[sid]["input_ids"].tolist())
    all_otus = sorted(o for o in all_otus if o >= 0)
    return {otu: i for i, otu in enumerate(all_otus)}


def train_eval(
    X_train, y_train, sample_ids_train,
    X_test, y_test, sample_ids_test,
    label_names, output_dir, n_estimators, n_jobs, seed,
):
    """Train RF on train split, evaluate on test set."""
    os.makedirs(output_dir, exist_ok=True)

    X_tr = X_train
    y_tr = y_train
    logger.info("Train: %d, Test: %d", len(y_tr), len(y_test))

    # Remove zero-variance OTUs
    train_var = X_tr.var(axis=0)
    nonzero_var = train_var > 0
    X_tr = X_tr[:, nonzero_var]
    X_te = X_test[:, nonzero_var]
    n_features = X_tr.shape[1]
    logger.info("Features after variance filter: %d / %d", n_features, nonzero_var.sum())

    # Train Random Forest
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=0,
    )
    rf.fit(X_tr, y_tr)
    logger.info("Training took %.1f min", (time.time() - t0) / 60)

    # Predict on test set
    probs = rf.predict_proba(X_te)  # (N, C)
    preds = rf.classes_[probs.argmax(axis=1)]

    # Map label_id → column index in probs (classes missing from train won't be in rf.classes_)
    class_to_col = {cls: i for i, cls in enumerate(rf.classes_)}

    # Build prediction table
    df_pred = pd.DataFrame({"SampleID": sample_ids_test})
    df_pred["label_id"] = y_test
    df_pred["label"] = [label_names[l] for l in y_test]
    df_pred["pred_id"] = preds
    df_pred["prediction"] = [label_names[p] for p in preds]
    df_pred["correct"] = (df_pred["label_id"] == df_pred["pred_id"]).values

    # Probability columns
    for c in range(probs.shape[1]):
        df_pred[f"prob_{c}"] = probs[:, c]

    df_pred.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    # Metrics — filter classes with <5 test samples for ROC-AUC / F-max
    acc = df_pred["correct"].mean()
    from sklearn.metrics import precision_recall_curve

    vc = df_pred["label_id"].value_counts()
    valid_ids_all = sorted(vc[vc >= 5].index.tolist())
    # Only keep classes the RF knows about (present in training)
    valid_ids = [ci for ci in valid_ids_all if ci in class_to_col]
    if len(valid_ids) >= 2:
        mask = df_pred["label_id"].isin(valid_ids).values
        y_filt = df_pred.loc[mask, "label_id"].values
        mapping = {c: i for i, c in enumerate(valid_ids)}
        y_mapped = np.array([mapping[l] for l in y_filt])
        probs_filt = np.column_stack([probs[mask, class_to_col[ci]] for ci in valid_ids])
        nv = len(valid_ids)

        try:
            roc = roc_auc_score(np.eye(nv)[y_mapped], probs_filt, multi_class="ovr", average="macro")
        except Exception:
            roc = float("nan")

        fmax_pc = []
        for c in range(nv):
            yt = (y_mapped == c).astype(int); ys = probs_filt[:, c]
            if yt.sum() == 0 or yt.sum() == len(yt):
                continue
            prec, rec, _ = precision_recall_curve(yt, ys)
            with np.errstate(divide="ignore", invalid="ignore"):
                f1c = 2 * prec * rec / (prec + rec + 1e-12)
            fmax_pc.append(np.nanmax(f1c))
        fmax = np.mean(fmax_pc) if fmax_pc else float("nan")
    else:
        roc = float("nan"); fmax = float("nan"); nv = 0

    metrics = {
        "accuracy": float(acc),
        "roc_auc_macro": float(roc) if not np.isnan(roc) else None,
        "f_max_macro": float(fmax) if not np.isnan(fmax) else None,
        "n_classes": len(rf.classes_),
        "n_valid_classes": nv,
        "n_estimators": n_estimators,
        "n_features": n_features,
        "seed": seed,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Accuracy=%.4f  ROC-AUC=%.4f  F-max=%.4f", acc, roc, fmax)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="RF baseline for MGnify MST")
    parser.add_argument("--train_data", default="data/downstreams/MGnify_mst/train.pkl")
    parser.add_argument("--test_data", default="data/downstreams/MGnify_mst/test.pkl")
    parser.add_argument("--train_meta", default="data/downstreams/MGnify_mst/train_meta_split.csv")
    parser.add_argument("--test_meta", default="data/downstreams/MGnify_mst/test_meta_split.csv")
    parser.add_argument("--levels", nargs="+", default=["env_level_1", "env_level_2", "env_level_3", "env_level_4", "env_level_5"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--output_root", default="outputs_downstreams/MGnify_mst_rf")
    args = parser.parse_args()

    # Load metadata
    train_meta = pd.read_csv(args.train_meta)
    test_meta = pd.read_csv(args.test_meta)
    train_ids = train_meta["SampleID"].astype(str).str.strip().tolist()
    test_ids = test_meta["SampleID"].astype(str).str.strip().tolist()

    # Build unified OTU vocabulary and abundance matrices
    train_otu = load_otu_data(args.train_data)
    test_otu = load_otu_data(args.test_data)
    unified_vocab = build_unified_vocab(train_otu, [train_ids, test_ids])
    logger.info("Unified vocab size: %d OTUs", len(unified_vocab))

    X_train_all = build_abundance_matrix(train_otu, train_ids, unified_vocab)
    X_test_all = build_abundance_matrix(test_otu, test_ids, unified_vocab)

    for level in args.levels:
        logger.info("=" * 60)
        logger.info("Level: %s", level)
        logger.info("=" * 60)

        # Labels
        train_labels_raw = train_meta[level].astype(str).tolist()
        test_labels_raw = test_meta[level].astype(str).tolist()

        # Build label mapping
        all_labels = sorted(set(train_labels_raw) | set(test_labels_raw))
        all_labels = [l for l in all_labels if l.lower() != "unknown"]
        label_to_id = {l: i for i, l in enumerate(all_labels)}
        y_train = np.array([label_to_id[l] for l in train_labels_raw])
        y_test = np.array([label_to_id[l] for l in test_labels_raw])
        label_names = all_labels

        logger.info("Classes: %d, Train samples: %d, Test samples: %d",
                     len(all_labels), len(y_train), len(y_test))

        level_metrics = []
        for seed in args.seeds:
            output_dir = os.path.join(args.output_root, level, f"seed_{seed}")
            if os.path.exists(os.path.join(output_dir, "metrics.json")):
                logger.info("Seed %d: already done, skipping", seed)
                with open(os.path.join(output_dir, "metrics.json")) as f:
                    level_metrics.append(json.load(f))
                continue

            logger.info("--- Seed %d ---", seed)
            metrics = train_eval(
                X_train_all, y_train, train_ids,
                X_test_all, y_test, test_ids,
                label_names, output_dir,
                args.n_estimators, args.n_jobs, seed,
            )
            level_metrics.append(metrics)

        # Summary
        accs = [m["accuracy"] for m in level_metrics]
        rocs = [m["roc_auc_macro"] for m in level_metrics]
        fmaxs = [m["f_max_macro"] for m in level_metrics]
        print(f"\n{level} SUMMARY (5 seeds):")
        print(f"  Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  ROC-AUC:   {np.mean(rocs):.4f} ± {np.std(rocs):.4f}")
        print(f"  F-max:     {np.mean(fmaxs):.4f} ± {np.std(fmaxs):.4f}")
        print()


if __name__ == "__main__":
    main()
