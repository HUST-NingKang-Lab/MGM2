#!/usr/bin/env python3
"""MLP delta baseline for drug-microbiome perturbation prediction.

By default this script reuses the outputs from
`scripts/downstreams/drug_perturbation/baseline.py`:

  outputs_downstreams/drug_perturbation/baseline/examples.csv
  outputs_downstreams/drug_perturbation/baseline/selected_otus.tsv

That keeps MLP iteration fast because we avoid rescanning the full HDF5 count
matrix for read-depth filtering on every run.
"""

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import baseline as baseline_utils  # noqa: E402


@dataclass
class MLPConfig:
    data_dir: str
    baseline_dir: str
    output_dir: str
    split_modes: List[str]
    top_k_otus: int
    max_folds: Optional[int]
    hidden_size: int
    dropout: float
    batch_size: int
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    val_fraction: float
    seed: int
    transform: str
    delta_l1_weight: float
    pseudocount: float
    prediction_space: str
    clip_nonnegative: bool
    normalize_prediction: str
    include_mean_baselines: bool
    device: str
    fold_selection: str
    fold_subset_seed: Optional[int]
    mgm2_cache: Optional[str]
    mgm2_feature_mode: str


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = values.mean(axis=0, keepdims=True).astype(np.float32)
        std = values.std(axis=0, keepdims=True).astype(np.float32)
        std[std < 1e-8] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std + self.mean).astype(np.float32)


class DrugPerturbationMLP(nn.Module):
    """Small fully connected model that predicts transformed community delta."""

    def __init__(
        self,
        community_dim: int,
        drug_dim: int,
        donor_dim: int,
        mgm2_dim: int = 0,
        hidden_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        community_hidden = max(hidden_size, 32)
        drug_hidden = max(hidden_size // 2, 32)
        donor_hidden = max(hidden_size // 8, 8)
        mgm2_hidden = max(hidden_size // 2, 32) if mgm2_dim > 0 else 0
        self.community_encoder = nn.Sequential(
            nn.Linear(community_dim, community_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, drug_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.donor_encoder = nn.Sequential(
            nn.Linear(donor_dim, donor_hidden),
            nn.ReLU(),
        )
        self.mgm2_encoder = (
            nn.Sequential(
                nn.Linear(mgm2_dim, mgm2_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            if mgm2_dim > 0
            else None
        )
        self.head = nn.Sequential(
            nn.Linear(community_hidden + drug_hidden + donor_hidden + mgm2_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, community_dim),
        )

    def forward(self, baseline_x: torch.Tensor, drug_x: torch.Tensor, donor_x: torch.Tensor, mgm2_x: torch.Tensor) -> torch.Tensor:
        features = [
            self.community_encoder(baseline_x),
            self.drug_encoder(drug_x),
            self.donor_encoder(donor_x),
        ]
        if self.mgm2_encoder is not None:
            features.append(self.mgm2_encoder(mgm2_x))
        return self.head(
            torch.cat(features, dim=-1)
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grouped_train_val_split(
    examples: pd.DataFrame,
    mode: str,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "pair":
        groups = np.asarray([f"{d}::{k}" for d, k in zip(examples["Donor"], examples["drug_key"])], dtype=object)
    elif mode == "drug":
        groups = examples["drug_key"].astype(str).to_numpy(dtype=object)
    elif mode == "donor":
        groups = examples["Donor"].astype(str).to_numpy(dtype=object)
    else:
        raise ValueError("mode must be one of: pair, drug, donor")

    unique_groups = np.asarray(pd.unique(groups), dtype=object)
    if len(unique_groups) < 2:
        raise ValueError("need at least two groups for grouped validation")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    n_val = max(1, int(round(len(unique_groups) * val_fraction)))
    n_val = min(n_val, len(unique_groups) - 1)
    val_groups = set(unique_groups[:n_val])
    val_idx = np.flatnonzero(np.asarray([group in val_groups for group in groups]))
    train_idx = np.flatnonzero(np.asarray([group not in val_groups for group in groups]))
    return train_idx, val_idx


def transform_profiles(values: np.ndarray, transform: str, pseudocount: float) -> np.ndarray:
    if transform == "relative":
        return values.astype(np.float32)
    if transform == "log1p":
        return np.log1p(values / pseudocount).astype(np.float32)
    raise ValueError("transform must be one of: relative, log1p")


def inverse_transform_profiles(values: np.ndarray, transform: str, pseudocount: float) -> np.ndarray:
    if transform == "relative":
        return values.astype(np.float32)
    if transform == "log1p":
        return (np.expm1(values) * pseudocount).astype(np.float32)
    raise ValueError("transform must be one of: relative, log1p")


def finalize_raw_predictions(
    predictions: Dict[str, np.ndarray],
    clip_nonnegative: bool = True,
    normalize_prediction: str = "none",
) -> Dict[str, np.ndarray]:
    """Post-process raw relative-abundance predictions for saving/evaluation."""
    if normalize_prediction not in {"none", "topk"}:
        raise ValueError("normalize_prediction must be one of: none, topk")

    finalized: Dict[str, np.ndarray] = {}
    for method, values in predictions.items():
        matrix = values.astype(np.float32, copy=True)
        if clip_nonnegative:
            matrix = np.clip(matrix, 0.0, None)
        if normalize_prediction == "topk":
            row_sum = matrix.sum(axis=1, keepdims=True)
            normalized = np.zeros_like(matrix, dtype=np.float32)
            np.divide(matrix, np.clip(row_sum, 1e-12, None), out=normalized, where=row_sum > 0)
            matrix = normalized
        finalized[method] = matrix.astype(np.float32)
    return finalized


def load_sample_index(data_dir: Path) -> Dict[str, int]:
    with h5py.File(data_dir / "otu_counts.h5", "r") as handle:
        sample_ids = baseline_utils._decode_h5_strings(handle["sample_ids"][:])
    return {sample_id: idx for idx, sample_id in enumerate(sample_ids)}


def load_cached_or_build_inputs(
    data_dir: Path,
    baseline_dir: Path,
    top_k_otus: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    examples_path = baseline_dir / "examples.csv"
    otus_path = baseline_dir / "selected_otus.tsv"
    _, drug_index = baseline_utils.load_drug_embeddings(data_dir)
    drug_embeddings, _ = baseline_utils.load_drug_embeddings(data_dir)

    if examples_path.exists() and otus_path.exists():
        examples = pd.read_csv(examples_path)
        sample_to_idx = load_sample_index(data_dir)
        examples["control_indices"] = examples["control_runs"].fillna("").apply(
            lambda raw: [sample_to_idx[item] for item in str(raw).split(";") if item]
        )
        if "embedding_idx" not in examples:
            examples = baseline_utils.annotate_canonical_drugs(examples, drug_index)
        selected_otus = pd.read_csv(otus_path, sep="\t").head(top_k_otus).copy()
        return examples.reset_index(drop=True), selected_otus, drug_embeddings

    examples, _, drug_embeddings = baseline_utils.build_examples(data_dir, min_read_sum=100, min_control_read_sum=100)
    sample_indices = examples["sample_idx"].astype(int).tolist()
    for controls in examples["control_indices"]:
        sample_indices.extend(int(idx) for idx in controls)
    selected_otus = baseline_utils.select_top_otus(data_dir / "otu_counts.h5", sample_indices, top_k_otus)
    return examples.reset_index(drop=True), selected_otus, drug_embeddings


def one_hot(values: pd.Series, categories: List[str]) -> np.ndarray:
    category_to_idx = {category: idx for idx, category in enumerate(categories)}
    result = np.zeros((len(values), len(categories)), dtype=np.float32)
    for row_idx, value in enumerate(values.astype(str)):
        result[row_idx, category_to_idx[value]] = 1.0
    return result


def load_mgm2_cache_features(
    cache_path: Optional[Path],
    examples: pd.DataFrame,
    feature_mode: str,
) -> Optional[np.ndarray]:
    if cache_path is None:
        return None
    if feature_mode == "none":
        return None
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    hidden = data["hidden_states"]
    if feature_mode == "cls":
        features = hidden[:, 0, :].float()
    elif feature_mode == "mean":
        features = hidden[:, 1:, :].mean(dim=1).float()
    elif feature_mode == "cls_mean":
        cls = hidden[:, 0, :].float()
        mean = hidden[:, 1:, :].mean(dim=1).float()
        features = torch.cat([cls, mean], dim=1)
    else:
        raise ValueError("mgm2_feature_mode must be one of: none, cls, mean, cls_mean")

    cache_examples = pd.DataFrame(
        {
            "Run": [item.get("sample_id", "") for item in data["examples"]],
            "Donor": [item.get("donor", "") for item in data["examples"]],
            "drug_key": [item.get("drug_key", "") for item in data["examples"]],
            "row": np.arange(len(data["examples"]), dtype=int),
        }
    )
    key_to_row = {
        (str(row.Run), str(row.Donor), str(row.drug_key)): int(row.row)
        for row in cache_examples.itertuples(index=False)
    }
    rows = []
    missing = []
    for row in examples.itertuples(index=False):
        key = (str(getattr(row, "Run")), str(getattr(row, "Donor")), str(getattr(row, "drug_key")))
        if key not in key_to_row:
            missing.append(key)
        else:
            rows.append(key_to_row[key])
    if missing:
        raise ValueError(f"MGM2 cache is missing {len(missing)} examples; first missing={missing[0]}")
    return features[rows].numpy().astype(np.float32)


def _h5_string_array(values: List[str]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=object)


def build_prediction_export(
    split_mode: str,
    fold_idx: int,
    group_label: str,
    test_examples: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    metadata = pd.DataFrame(
        {
            "split_mode": split_mode,
            "fold": int(fold_idx),
            "group": group_label,
            "Run": test_examples["Run"].astype(str).to_list(),
            "Donor": test_examples["Donor"].astype(str).to_list(),
            "drug_key": test_examples["drug_key"].astype(str).to_list(),
        }
    )
    matrices = {method: values.astype(np.float32) for method, values in predictions.items()}
    return metadata, matrices


def write_prediction_h5(
    output_path: Path,
    selected_otus: pd.DataFrame,
    metadata_frames: List[pd.DataFrame],
    prediction_matrices: Dict[str, List[np.ndarray]],
) -> None:
    if not metadata_frames:
        return
    metadata = pd.concat(metadata_frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset(
            "otu_ids",
            data=_h5_string_array(selected_otus["otu_id"].astype(str).to_list()),
            dtype=string_dtype,
        )
        handle.create_dataset("otu_idx", data=selected_otus["otu_idx"].astype(int).to_numpy())
        handle.attrs["prediction_space"] = "raw_relative_abundance"
        meta_group = handle.create_group("metadata")
        for column in ["split_mode", "group", "Run", "Donor", "drug_key"]:
            meta_group.create_dataset(
                column,
                data=_h5_string_array(metadata[column].astype(str).to_list()),
                dtype=string_dtype,
            )
        meta_group.create_dataset("fold", data=metadata["fold"].astype(int).to_numpy())

        pred_group = handle.create_group("predictions")
        for method, chunks in sorted(prediction_matrices.items()):
            if chunks:
                pred_group.create_dataset(method, data=np.concatenate(chunks, axis=0).astype(np.float32), compression="gzip")


def make_loader(
    baseline_x: np.ndarray,
    drug_x: np.ndarray,
    donor_x: np.ndarray,
    mgm2_x: Optional[np.ndarray],
    target_delta: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    if mgm2_x is None:
        mgm2_x = np.zeros((baseline_x.shape[0], 0), dtype=np.float32)
    dataset = TensorDataset(
        torch.from_numpy(baseline_x.astype(np.float32)),
        torch.from_numpy(drug_x.astype(np.float32)),
        torch.from_numpy(donor_x.astype(np.float32)),
        torch.from_numpy(mgm2_x.astype(np.float32)),
        torch.from_numpy(target_delta.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_predict_fold(
    examples: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    baseline_profiles: np.ndarray,
    post_profiles: np.ndarray,
    drug_embeddings: np.ndarray,
    donor_categories: List[str],
    mgm2_features: Optional[np.ndarray],
    mode: str,
    config: MLPConfig,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, float]]:
    train_examples = examples.iloc[train_idx].reset_index(drop=True)
    inner_train_rel, val_rel = grouped_train_val_split(train_examples, mode, config.val_fraction, config.seed)
    inner_train_idx = train_idx[inner_train_rel]
    val_idx = train_idx[val_rel]

    baseline_t = transform_profiles(baseline_profiles, config.transform, config.pseudocount)
    post_t = transform_profiles(post_profiles, config.transform, config.pseudocount)
    delta_t = post_t - baseline_t

    baseline_scaler = Standardizer.fit(baseline_t[inner_train_idx])
    delta_scaler = Standardizer.fit(delta_t[inner_train_idx])
    drug_scaler = Standardizer.fit(drug_embeddings[examples.iloc[inner_train_idx]["embedding_idx"].astype(int)])
    mgm2_scaler = Standardizer.fit(mgm2_features[inner_train_idx]) if mgm2_features is not None else None
    donor_x = one_hot(examples["Donor"], donor_categories)

    train_loader = make_loader(
        baseline_scaler.transform(baseline_t[inner_train_idx]),
        drug_scaler.transform(drug_embeddings[examples.iloc[inner_train_idx]["embedding_idx"].astype(int)]),
        donor_x[inner_train_idx],
        mgm2_scaler.transform(mgm2_features[inner_train_idx]) if mgm2_scaler is not None and mgm2_features is not None else None,
        delta_scaler.transform(delta_t[inner_train_idx]),
        config.batch_size,
        True,
    )
    val_loader = make_loader(
        baseline_scaler.transform(baseline_t[val_idx]),
        drug_scaler.transform(drug_embeddings[examples.iloc[val_idx]["embedding_idx"].astype(int)]),
        donor_x[val_idx],
        mgm2_scaler.transform(mgm2_features[val_idx]) if mgm2_scaler is not None and mgm2_features is not None else None,
        delta_scaler.transform(delta_t[val_idx]),
        config.batch_size,
        False,
    )

    model = DrugPerturbationMLP(
        community_dim=baseline_profiles.shape[1],
        drug_dim=drug_embeddings.shape[1],
        donor_dim=len(donor_categories),
        mgm2_dim=0 if mgm2_features is None else mgm2_features.shape[1],
        hidden_size=config.hidden_size,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    stale = 0
    epochs_trained = 0

    for epoch in range(config.max_epochs):
        model.train()
        for batch in train_loader:
            baseline_x, drug_x, donor_batch, mgm2_batch, target_delta = [item.to(device) for item in batch]
            pred_delta = model(baseline_x, drug_x, donor_batch, mgm2_batch)
            loss = criterion(pred_delta, target_delta)
            if config.delta_l1_weight > 0:
                loss = loss + config.delta_l1_weight * pred_delta.abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                baseline_x, drug_x, donor_batch, mgm2_batch, target_delta = [item.to(device) for item in batch]
                val_losses.append(float(criterion(model(baseline_x, drug_x, donor_batch, mgm2_batch), target_delta).item()))
        val_loss = float(np.mean(val_losses))
        epochs_trained = epoch + 1
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_loader = make_loader(
        baseline_scaler.transform(baseline_t[test_idx]),
        drug_scaler.transform(drug_embeddings[examples.iloc[test_idx]["embedding_idx"].astype(int)]),
        donor_x[test_idx],
        mgm2_scaler.transform(mgm2_features[test_idx]) if mgm2_scaler is not None and mgm2_features is not None else None,
        np.zeros_like(delta_t[test_idx], dtype=np.float32),
        config.batch_size,
        False,
    )
    pred_scaled = []
    with torch.no_grad():
        for batch in test_loader:
            baseline_x, drug_x, donor_batch, mgm2_batch, _ = [item.to(device) for item in batch]
            pred_scaled.append(model(baseline_x, drug_x, donor_batch, mgm2_batch).cpu().numpy())
    pred_delta_t = delta_scaler.inverse_transform(np.concatenate(pred_scaled, axis=0))
    pred_post_t = baseline_t[test_idx] + pred_delta_t
    pred_post = inverse_transform_profiles(pred_post_t, config.transform, config.pseudocount)
    return pred_post, {"best_val_loss": best_val, "epochs_trained": epochs_trained}


def run_mode(
    examples: pd.DataFrame,
    baseline_profiles: np.ndarray,
    post_profiles: np.ndarray,
    delta_profiles: np.ndarray,
    drug_embeddings: np.ndarray,
    donor_categories: List[str],
    mgm2_features: Optional[np.ndarray],
    mode: str,
    config: MLPConfig,
    device: torch.device,
    selected_otus: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    frames = []
    prediction_metadata: List[pd.DataFrame] = []
    prediction_matrices: Dict[str, List[np.ndarray]] = {}
    folds = list(baseline_utils.iter_leave_one_groups(examples, mode))
    if config.max_folds is not None:
        if config.fold_selection == "random":
            rng = np.random.default_rng(config.fold_subset_seed if config.fold_subset_seed is not None else config.seed)
            selected = np.sort(rng.choice(len(folds), size=min(config.max_folds, len(folds)), replace=False))
            folds = [folds[int(idx)] for idx in selected]
        else:
            folds = folds[: config.max_folds]
    for fold_idx, (train_idx, test_idx, group_label) in enumerate(folds):
        predictions = {}
        if config.include_mean_baselines:
            predictions.update(
                baseline_utils.predict_mean_delta_baselines(
                    examples.iloc[train_idx].reset_index(drop=True),
                    examples.iloc[test_idx].reset_index(drop=True),
                    delta_profiles[train_idx],
                    baseline_profiles[test_idx],
                )
            )
        pred_mlp, history = train_predict_fold(
            examples,
            train_idx,
            test_idx,
            baseline_profiles,
            post_profiles,
            drug_embeddings,
            donor_categories,
            mgm2_features,
            mode,
            config,
            device,
        )
        method_name = "mlp_delta" if mgm2_features is None else f"hybrid_mlp_{config.mgm2_feature_mode}"
        predictions[method_name] = pred_mlp
        predictions = finalize_raw_predictions(
            predictions,
            clip_nonnegative=config.clip_nonnegative,
            normalize_prediction=config.normalize_prediction,
        )
        export_metadata, export_matrices = build_prediction_export(
            split_mode=mode,
            fold_idx=fold_idx,
            group_label=group_label,
            test_examples=examples.iloc[test_idx].reset_index(drop=True),
            predictions=predictions,
        )
        prediction_metadata.append(export_metadata)
        for method, matrix in export_matrices.items():
            prediction_matrices.setdefault(method, []).append(matrix)
        metrics = baseline_utils.evaluate_predictions(
            predictions,
            post_profiles[test_idx],
            baseline_profiles[test_idx],
            config.pseudocount,
        )
        metrics.insert(0, "best_val_loss", history["best_val_loss"])
        metrics.insert(0, "epochs_trained", history["epochs_trained"])
        metrics.insert(0, "n_test", len(test_idx))
        metrics.insert(0, "group", group_label)
        metrics.insert(0, "fold", fold_idx)
        metrics.insert(0, "split_mode", mode)
        frames.append(metrics)
        print(f"{mode} fold {fold_idx + 1}: {group_label} n={len(test_idx)} val={history['best_val_loss']:.4f}", flush=True)
    write_prediction_h5(output_dir / f"predicted_abundance_{mode}.h5", selected_otus, prediction_metadata, prediction_matrices)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MLP delta predictor and compare against mean-delta baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, default=baseline_utils.DEFAULT_DATA_DIR)
    parser.add_argument("--baseline_dir", type=Path, default=Path("outputs_downstreams/drug_perturbation/baseline"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_downstreams/drug_perturbation/mlp_baseline"))
    parser.add_argument("--split_modes", nargs="+", choices=["pair", "drug", "donor"], default=["pair"])
    parser.add_argument("--top_k_otus", type=int, default=1024)
    parser.add_argument("--max_folds", type=int, default=20, help="Fold cap for quick runs; use 0 for full LOO")
    parser.add_argument("--fold_selection", choices=["first", "random"], default="first")
    parser.add_argument("--fold_subset_seed", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transform", choices=["relative", "log1p"], default="log1p")
    parser.add_argument("--delta_l1_weight", type=float, default=0.0)
    parser.add_argument("--pseudocount", type=float, default=1e-6)
    parser.add_argument("--no_clip_nonnegative", action="store_true", help="Disable nonnegative clipping before raw-space save/eval")
    parser.add_argument("--normalize_prediction", choices=["none", "topk"], default="none", help="Optional normalization after clipping")
    parser.add_argument("--no_mean_baselines", action="store_true")
    parser.add_argument("--mgm2_cache", type=Path, default=None, help="Optional MGM2 cached backbone file to append frozen features to the MLP")
    parser.add_argument("--mgm2_feature_mode", choices=["none", "cls", "mean", "cls_mean"], default="none")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_dir = args.data_dir.resolve()
    baseline_dir = args.baseline_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    max_folds = None if args.max_folds <= 0 else args.max_folds
    config = MLPConfig(
        data_dir=str(data_dir),
        baseline_dir=str(baseline_dir),
        output_dir=str(output_dir),
        split_modes=list(args.split_modes),
        top_k_otus=args.top_k_otus,
        max_folds=max_folds,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        seed=args.seed,
        transform=args.transform,
        delta_l1_weight=args.delta_l1_weight,
        pseudocount=args.pseudocount,
        prediction_space="raw_relative_abundance",
        clip_nonnegative=not args.no_clip_nonnegative,
        normalize_prediction=args.normalize_prediction,
        include_mean_baselines=not args.no_mean_baselines,
        device=str(device),
        fold_selection=args.fold_selection,
        fold_subset_seed=args.fold_subset_seed,
        mgm2_cache=str(args.mgm2_cache) if args.mgm2_cache is not None else None,
        mgm2_feature_mode=args.mgm2_feature_mode,
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    print("Loading cached baseline examples and OTU selection...", flush=True)
    examples, selected_otus, drug_embeddings = load_cached_or_build_inputs(data_dir, baseline_dir, args.top_k_otus)
    examples.drop(columns=["control_indices"]).to_csv(output_dir / "examples.csv", index=False)
    selected_otus.to_csv(output_dir / "selected_otus.tsv", sep="\t", index=False)
    print(f"Usable examples: {len(examples)} | selected OTUs: {len(selected_otus)}", flush=True)

    print("Loading profile arrays...", flush=True)
    baseline_profiles, post_profiles, delta_profiles = baseline_utils.build_profile_arrays(data_dir, examples, selected_otus)
    donor_categories = sorted(examples["Donor"].astype(str).unique())
    mgm2_features = load_mgm2_cache_features(args.mgm2_cache, examples, args.mgm2_feature_mode)
    if mgm2_features is not None:
        print(f"Loaded MGM2 features: {mgm2_features.shape} from {args.mgm2_cache}", flush=True)

    all_metrics = []
    for mode in args.split_modes:
        print(f"Running leave-one-{mode}...", flush=True)
        metrics = run_mode(
            examples,
            baseline_profiles,
            post_profiles,
            delta_profiles,
            drug_embeddings,
            donor_categories,
            mgm2_features,
            mode,
            config,
            device,
            selected_otus,
            output_dir,
        )
        if not metrics.empty:
            all_metrics.append(metrics)
    if not all_metrics:
        raise RuntimeError("No folds evaluated")

    fold_metrics = pd.concat(all_metrics, ignore_index=True)
    summary = baseline_utils.summarize_metrics(fold_metrics)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print()
    print(summary.to_string(index=False))
    print(f"\nWrote MLP results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
