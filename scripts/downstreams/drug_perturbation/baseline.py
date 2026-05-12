#!/usr/bin/env python3
"""Simple baselines for drug-microbiome perturbation prediction.

The task is framed as:
    matched control community + drug metadata -> drug-treated community

This script intentionally starts with low-complexity baselines so future MGM2
models can be compared against a stable leave-one-out protocol.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_DATA_DIR = Path("/data9/zhanghaohong/databases/MicrobeDrugPerturbation/data")


@dataclass
class RunConfig:
    data_dir: str
    output_dir: str
    split_modes: List[str]
    top_k_otus: int
    min_read_sum: int
    min_control_read_sum: int
    max_folds: Optional[int]
    knn_k: int
    pseudocount: float


def _decode_h5_strings(values) -> List[str]:
    decoded = []
    for value in values:
        if isinstance(value, (bytes, bytearray, np.bytes_)):
            decoded.append(value.decode("utf-8", errors="replace"))
        else:
            decoded.append(str(value))
    return decoded


def annotate_canonical_drugs(meta: pd.DataFrame, drug_index: pd.DataFrame) -> pd.DataFrame:
    """Add canonical drug keys and embedding indices to metadata.

    `Condition` is not stable across experiments for several salts/hydrates
    (`-` versus `_`). `Drug_Name_in_Library` is the safer key and gives complete
    coverage against the supplied drug embedding index.
    """
    annotated = meta.copy()
    annotated["drug_key"] = annotated["Drug_Name_in_Library"].fillna("")
    condition_fallback = annotated["Condition"].fillna("")
    annotated.loc[annotated["drug_key"] == "", "drug_key"] = condition_fallback

    index_cols = ["Drug_Name_in_Library", "embedding_idx"]
    deduped_index = drug_index[index_cols].drop_duplicates("Drug_Name_in_Library")
    annotated = annotated.merge(
        deduped_index,
        on="Drug_Name_in_Library",
        how="left",
        suffixes=("", "_from_index"),
    )
    return annotated


def build_matched_control_map(meta: pd.DataFrame) -> Dict[str, List[str]]:
    """Return drug Run -> matched control Runs.

    Exp01 uses plate-matched media controls. Exp06 uses donor/plate-matched DMSO
    controls and intentionally excludes `empty3` wells.
    """
    controls = meta[meta["Type"] == "control"].copy()
    matches: Dict[str, List[str]] = {}
    for _, row in meta[meta["Type"] == "drug"].iterrows():
        if row["Experiment"] == "Exp06":
            mask = (
                (controls["Experiment"] == row["Experiment"])
                & (controls["Donor"] == row["Donor"])
                & (controls["Plate"].astype(str) == str(row["Plate"]))
                & (controls["Condition"] == "DMSO")
            )
        else:
            mask = (
                (controls["Experiment"] == row["Experiment"])
                & (controls["Donor"] == row["Donor"])
                & (controls["Plate"].astype(str) == str(row["Plate"]))
                & (controls["Condition"].isin(["control", "Ethanol_control"]))
            )
        matches[row["Run"]] = controls.loc[mask, "Run"].tolist()
    return matches


def iter_leave_one_groups(
    examples: pd.DataFrame,
    mode: str = "pair",
) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
    """Yield train/test row positions for leave-one-group evaluation."""
    if mode == "pair":
        group_values = list(zip(examples["Donor"].astype(str), examples["drug_key"].astype(str)))
        group_labels = [f"{donor}::{drug}" for donor, drug in group_values]
    elif mode == "drug":
        group_labels = examples["drug_key"].astype(str).tolist()
    elif mode == "donor":
        group_labels = examples["Donor"].astype(str).tolist()
    else:
        raise ValueError("mode must be one of: pair, drug, donor")

    group_array = np.asarray(group_labels, dtype=object)
    for group in pd.unique(group_array):
        test_idx = np.flatnonzero(group_array == group)
        train_idx = np.flatnonzero(group_array != group)
        if len(train_idx) == 0:
            continue
        yield train_idx, test_idx, str(group)


def _mean_by_key(keys: Sequence[str], values: np.ndarray) -> Dict[str, np.ndarray]:
    frame = pd.DataFrame({"key": list(keys), "row": np.arange(len(keys))})
    means = {}
    for key, rows in frame.groupby("key")["row"]:
        means[str(key)] = values[rows.to_numpy()].mean(axis=0)
    return means


def predict_mean_delta_baselines(
    train_examples: pd.DataFrame,
    test_examples: pd.DataFrame,
    train_delta: np.ndarray,
    baseline_profiles: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Predict profiles with global, drug, donor, and additive mean deltas."""
    global_delta = train_delta.mean(axis=0)
    drug_means = _mean_by_key(train_examples["drug_key"].astype(str), train_delta)
    donor_means = _mean_by_key(train_examples["Donor"].astype(str), train_delta)

    predictions = {
        "no_effect": baseline_profiles.copy(),
        "global_mean_delta": baseline_profiles + global_delta,
    }

    drug_pred = []
    donor_pred = []
    additive_pred = []
    for _, row in test_examples.iterrows():
        drug_delta = drug_means.get(str(row["drug_key"]), global_delta)
        donor_delta = donor_means.get(str(row["Donor"]), global_delta)
        drug_pred.append(drug_delta)
        donor_pred.append(donor_delta)
        additive_pred.append(drug_delta + donor_delta - global_delta)

    predictions["drug_mean_delta"] = baseline_profiles + np.stack(drug_pred)
    predictions["donor_mean_delta"] = baseline_profiles + np.stack(donor_pred)
    predictions["additive_mean_delta"] = baseline_profiles + np.stack(additive_pred)
    return predictions


def load_drug_embeddings(data_dir: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    with h5py.File(data_dir / "drug_embeddings.h5", "r") as handle:
        embeddings = handle["embeddings"][:].astype(np.float32)
        drug_names = _decode_h5_strings(handle["drug_name"][:])
        conditions = _decode_h5_strings(handle["condition"][:])
    frame = pd.DataFrame(
        {
            "Drug_Name_in_Library": drug_names,
            "Condition": conditions,
            "embedding_idx": np.arange(len(drug_names), dtype=int),
        }
    )
    return embeddings, frame


def predict_knn_drug_delta(
    train_examples: pd.DataFrame,
    test_examples: pd.DataFrame,
    train_delta: np.ndarray,
    baseline_profiles: np.ndarray,
    drug_embeddings: np.ndarray,
    k: int,
) -> np.ndarray:
    if k <= 0 or "embedding_idx" not in train_examples or "embedding_idx" not in test_examples:
        return baseline_profiles + train_delta.mean(axis=0)

    train_idx = train_examples["embedding_idx"].astype(int).to_numpy()
    test_idx = test_examples["embedding_idx"].astype(int).to_numpy()
    train_emb = drug_embeddings[train_idx]
    test_emb = drug_embeddings[test_idx]
    train_norm = train_emb / np.clip(np.linalg.norm(train_emb, axis=1, keepdims=True), 1e-8, None)
    test_norm = test_emb / np.clip(np.linalg.norm(test_emb, axis=1, keepdims=True), 1e-8, None)
    sim = test_norm @ train_norm.T
    k_eff = min(k, train_delta.shape[0])
    neighbor_idx = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
    pred_delta = np.stack([train_delta[idx].mean(axis=0) for idx in neighbor_idx])
    return baseline_profiles + pred_delta


def summarize_h5_samples(otu_h5_path: Path) -> Tuple[List[str], Dict[str, int], pd.DataFrame]:
    with h5py.File(otu_h5_path, "r") as handle:
        sample_ids = _decode_h5_strings(handle["sample_ids"][:])
        counts = handle["counts"]
        read_sum = np.zeros(len(sample_ids), dtype=np.int64)
        nonzero_otus = np.zeros(len(sample_ids), dtype=np.int64)
        for start in range(0, counts.shape[1], 200):
            block = counts[:, start : start + 200][:]
            stop = start + block.shape[1]
            read_sum[start:stop] = block.sum(axis=0)
            nonzero_otus[start:stop] = (block > 0).sum(axis=0)
    sample_to_idx = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    stats = pd.DataFrame(
        {
            "Run": sample_ids,
            "sample_idx": np.arange(len(sample_ids), dtype=int),
            "read_sum": read_sum,
            "nonzero_otus": nonzero_otus,
        }
    )
    return sample_ids, sample_to_idx, stats


def build_examples(
    data_dir: Path,
    min_read_sum: int,
    min_control_read_sum: int,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], np.ndarray]:
    meta = pd.read_csv(data_dir / "sample_drug_mapping.tsv", sep="\t")
    drug_index = pd.read_csv(data_dir / "drug_embedding_index.tsv", sep="\t")
    drug_embeddings, embedding_frame = load_drug_embeddings(data_dir)
    if "embedding_idx" not in drug_index:
        drug_index = embedding_frame

    meta = annotate_canonical_drugs(meta, drug_index)
    _, sample_to_idx, sample_stats = summarize_h5_samples(data_dir / "otu_counts.h5")
    meta = meta.merge(sample_stats, on="Run", how="left")
    matched_controls = build_matched_control_map(meta)
    read_sum_by_run = dict(zip(meta["Run"], meta["read_sum"]))

    rows = []
    for _, row in meta[meta["Type"] == "drug"].iterrows():
        control_runs = matched_controls.get(row["Run"], [])
        good_controls = [
            run
            for run in control_runs
            if pd.notna(read_sum_by_run.get(run)) and int(read_sum_by_run[run]) >= min_control_read_sum
        ]
        if not good_controls:
            continue
        if pd.isna(row["sample_idx"]) or int(row["read_sum"]) < min_read_sum:
            continue
        if pd.isna(row["embedding_idx"]):
            continue
        enriched = row.to_dict()
        enriched["sample_idx"] = int(enriched["sample_idx"])
        enriched["embedding_idx"] = int(enriched["embedding_idx"])
        enriched["control_runs"] = ";".join(good_controls)
        enriched["control_indices"] = [sample_to_idx[run] for run in good_controls]
        rows.append(enriched)

    examples = pd.DataFrame(rows).reset_index(drop=True)
    if examples.empty:
        raise ValueError("No usable drug perturbation examples after filtering")
    return examples, matched_controls, drug_embeddings


def select_top_otus(otu_h5_path: Path, sample_indices: Sequence[int], top_k: int) -> pd.DataFrame:
    with h5py.File(otu_h5_path, "r") as handle:
        rel = handle["rel_abundance"]
        otu_ids = _decode_h5_strings(handle["otu_ids"][:])
        abundance_sum = np.zeros(rel.shape[0], dtype=np.float64)
        sample_indices = np.asarray(sorted(set(int(i) for i in sample_indices)), dtype=int)
        for start in range(0, rel.shape[0], 5000):
            block = rel[start : start + 5000, sample_indices]
            abundance_sum[start : start + block.shape[0]] = block.sum(axis=1)

    top_k = min(int(top_k), len(abundance_sum))
    top_idx = np.argpartition(-abundance_sum, kth=top_k - 1)[:top_k]
    top_idx = top_idx[np.argsort(-abundance_sum[top_idx])]
    return pd.DataFrame(
        {
            "otu_idx": top_idx.astype(int),
            "otu_id": [otu_ids[i] for i in top_idx],
            "abundance_sum": abundance_sum[top_idx],
        }
    )


def load_profile_matrix(otu_h5_path: Path, otu_indices: Sequence[int], sample_indices: Sequence[int]) -> Tuple[np.ndarray, List[int]]:
    """Load selected OTU profiles as sample x OTU matrix."""
    otu_indices = np.asarray(otu_indices, dtype=int)
    sorted_otu = np.sort(otu_indices)
    order = np.argsort(otu_indices)
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order))

    sample_indices = list(dict.fromkeys(int(i) for i in sample_indices))
    with h5py.File(otu_h5_path, "r") as handle:
        block = handle["rel_abundance"][sorted_otu, :][:, sample_indices]
    matrix = block[inverse_order].T.astype(np.float32)
    return matrix, sample_indices


def build_profile_arrays(
    data_dir: Path,
    examples: pd.DataFrame,
    selected_otus: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    needed_samples: List[int] = examples["sample_idx"].astype(int).tolist()
    for controls in examples["control_indices"]:
        needed_samples.extend(int(idx) for idx in controls)

    profile_matrix, loaded_sample_indices = load_profile_matrix(
        data_dir / "otu_counts.h5",
        selected_otus["otu_idx"].to_numpy(),
        needed_samples,
    )
    sample_pos = {sample_idx: pos for pos, sample_idx in enumerate(loaded_sample_indices)}

    post_profiles = np.stack([profile_matrix[sample_pos[int(idx)]] for idx in examples["sample_idx"]])
    baseline_profiles = []
    for controls in examples["control_indices"]:
        positions = [sample_pos[int(idx)] for idx in controls]
        baseline_profiles.append(profile_matrix[positions].mean(axis=0))
    baseline_profiles = np.stack(baseline_profiles).astype(np.float32)
    delta = post_profiles - baseline_profiles
    return baseline_profiles, post_profiles, delta


def _project_nonnegative(profile: np.ndarray) -> np.ndarray:
    return np.clip(profile, 0.0, None)


def bray_curtis(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = _project_nonnegative(pred)
    true = _project_nonnegative(true)
    denom = pred.sum(axis=1) + true.sum(axis=1)
    numer = np.abs(pred - true).sum(axis=1)
    return numer / np.clip(denom, 1e-12, None)


def clr_mse(pred: np.ndarray, true: np.ndarray, pseudocount: float) -> np.ndarray:
    pred = _project_nonnegative(pred) + pseudocount
    true = _project_nonnegative(true) + pseudocount
    pred_log = np.log(pred)
    true_log = np.log(true)
    pred_clr = pred_log - pred_log.mean(axis=1, keepdims=True)
    true_clr = true_log - true_log.mean(axis=1, keepdims=True)
    return ((pred_clr - true_clr) ** 2).mean(axis=1)


def delta_cosine(pred: np.ndarray, true: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    pred_delta = pred - baseline
    true_delta = true - baseline
    numer = (pred_delta * true_delta).sum(axis=1)
    denom = np.linalg.norm(pred_delta, axis=1) * np.linalg.norm(true_delta, axis=1)
    return numer / np.clip(denom, 1e-12, None)


def evaluate_predictions(
    predictions: Mapping[str, np.ndarray],
    true_profiles: np.ndarray,
    baseline_profiles: np.ndarray,
    pseudocount: float,
) -> pd.DataFrame:
    rows = []
    true_delta = true_profiles - baseline_profiles
    true_magnitude = np.abs(true_delta).sum(axis=1)
    for method, pred in predictions.items():
        pred = _project_nonnegative(pred)
        pred_delta = pred - baseline_profiles
        rows.append(
            {
                "method": method,
                "bray_curtis_mean": float(bray_curtis(pred, true_profiles).mean()),
                "clr_mse_mean": float(clr_mse(pred, true_profiles, pseudocount).mean()),
                "delta_cosine_mean": float(delta_cosine(pred, true_profiles, baseline_profiles).mean()),
                "delta_l1_mae_mean": float(np.abs(pred_delta - true_delta).sum(axis=1).mean()),
                "magnitude_mae_mean": float(np.abs(np.abs(pred_delta).sum(axis=1) - true_magnitude).mean()),
            }
        )
    return pd.DataFrame(rows)


def run_split_mode(
    examples: pd.DataFrame,
    baseline_profiles: np.ndarray,
    post_profiles: np.ndarray,
    delta: np.ndarray,
    drug_embeddings: np.ndarray,
    mode: str,
    max_folds: Optional[int],
    knn_k: int,
    pseudocount: float,
) -> pd.DataFrame:
    metric_frames = []
    for fold_idx, (train_idx, test_idx, group_label) in tqdm(enumerate(iter_leave_one_groups(examples, mode=mode)), total=examples.shape[0], desc=f"Split mode: {mode}"):
        if max_folds is not None and fold_idx >= max_folds:
            break
        train_examples = examples.iloc[train_idx].reset_index(drop=True)
        test_examples = examples.iloc[test_idx].reset_index(drop=True)
        predictions = predict_mean_delta_baselines(
            train_examples,
            test_examples,
            delta[train_idx],
            baseline_profiles[test_idx],
        )
        predictions[f"knn{kNN_LABEL(knn_k)}_drug_delta"] = predict_knn_drug_delta(
            train_examples,
            test_examples,
            delta[train_idx],
            baseline_profiles[test_idx],
            drug_embeddings,
            knn_k,
        )
        fold_metrics = evaluate_predictions(
            predictions,
            post_profiles[test_idx],
            baseline_profiles[test_idx],
            pseudocount=pseudocount,
        )
        fold_metrics.insert(0, "n_test", len(test_idx))
        fold_metrics.insert(0, "group", group_label)
        fold_metrics.insert(0, "fold", fold_idx)
        fold_metrics.insert(0, "split_mode", mode)
        metric_frames.append(fold_metrics)

    if not metric_frames:
        return pd.DataFrame()
    return pd.concat(metric_frames, ignore_index=True)


def kNN_LABEL(k: int) -> str:
    return str(int(k))


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "bray_curtis_mean",
        "clr_mse_mean",
        "delta_cosine_mean",
        "delta_l1_mae_mean",
        "magnitude_mae_mean",
    ]
    rows = []
    for (split_mode, method), group in fold_metrics.groupby(["split_mode", "method"]):
        weights = group["n_test"].to_numpy(dtype=float)
        row = {
            "split_mode": split_mode,
            "method": method,
            "n_folds": int(group["fold"].nunique()),
            "n_test": int(group["n_test"].sum()),
        }
        for col in metric_cols:
            row[col] = float(np.average(group[col].to_numpy(dtype=float), weights=weights))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split_mode", "bray_curtis_mean", "method"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simple leave-one-out baselines for drug perturbation prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR, help="Prepared MicrobeDrugPerturbation data directory")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_downstreams/drug_perturbation/baseline"), help="Output directory")
    parser.add_argument(
        "--split_modes",
        nargs="+",
        choices=["pair", "drug", "donor"],
        default=["pair", "drug", "donor"],
        help="Leave-one-group evaluation modes",
    )
    parser.add_argument("--top_k_otus", type=int, default=2048, help="Use globally most abundant OTUs as dense features")
    parser.add_argument("--min_read_sum", type=int, default=100, help="Drop drug samples below this read count")
    parser.add_argument("--min_control_read_sum", type=int, default=100, help="Drop matched controls below this read count")
    parser.add_argument("--max_folds", type=int, default=None, help="Optional fold cap for smoke tests")
    parser.add_argument("--knn_k", type=int, default=5, help="Number of nearest training samples for drug-embedding kNN baseline")
    parser.add_argument("--pseudocount", type=float, default=1e-6, help="Pseudocount for CLR MSE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        split_modes=list(args.split_modes),
        top_k_otus=args.top_k_otus,
        min_read_sum=args.min_read_sum,
        min_control_read_sum=args.min_control_read_sum,
        max_folds=args.max_folds,
        knn_k=args.knn_k,
        pseudocount=args.pseudocount,
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    print("Building examples...")
    examples, _, drug_embeddings = build_examples(
        data_dir,
        min_read_sum=args.min_read_sum,
        min_control_read_sum=args.min_control_read_sum,
    )
    examples_for_csv = examples.drop(columns=["control_indices"]).copy()
    examples_for_csv.to_csv(output_dir / "examples.csv", index=False)
    print(f"Usable examples: {len(examples)}")

    print(f"Selecting top {args.top_k_otus} OTUs...")
    sample_indices = examples["sample_idx"].astype(int).tolist()
    for controls in examples["control_indices"]:
        sample_indices.extend(int(idx) for idx in controls)
    selected_otus = select_top_otus(data_dir / "otu_counts.h5", sample_indices, args.top_k_otus)
    selected_otus.to_csv(output_dir / "selected_otus.tsv", sep="\t", index=False)

    print("Loading selected OTU profiles...")
    baseline_profiles, post_profiles, delta = build_profile_arrays(data_dir, examples, selected_otus)

    all_metrics = []
    for mode in args.split_modes:
        print(f"Running leave-one-{mode} baselines...")
        metrics = run_split_mode(
            examples=examples,
            baseline_profiles=baseline_profiles,
            post_profiles=post_profiles,
            delta=delta,
            drug_embeddings=drug_embeddings,
            mode=mode,
            max_folds=args.max_folds,
            knn_k=args.knn_k,
            pseudocount=args.pseudocount,
        )
        if not metrics.empty:
            all_metrics.append(metrics)

    if not all_metrics:
        raise RuntimeError("No folds were evaluated")

    fold_metrics = pd.concat(all_metrics, ignore_index=True)
    summary = summarize_metrics(fold_metrics)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print()
    print(summary.to_string(index=False))
    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    main()
