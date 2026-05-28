"""Dataset utilities for FMT donor-recipient community prediction."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.Preprocessing import AbundancePreprocessor
from src.data.utils import load_preprocessed_data


def _decode_h5_strings(values) -> List[str]:
    decoded = []
    for value in values:
        if isinstance(value, (bytes, bytearray, np.bytes_)):
            decoded.append(value.decode("utf-8", errors="replace"))
        else:
            decoded.append(str(value))
    return decoded


def _load_resolved_embeddings(path: str | Path) -> Tuple[List[str], np.ndarray]:
    with h5py.File(path, "r") as handle:
        feature_ids = _decode_h5_strings(handle["otu_ids"][:])
        embeddings = handle["embeddings"][:].astype(np.float32)
    if embeddings.shape[0] != len(feature_ids):
        raise ValueError(f"Embedding row count does not match feature ids in {path}")
    return feature_ids, embeddings


class FMTCommunityPredictionDataset(Dataset):
    """FMT examples with donor, recipient-pre, and post-FMT communities aligned."""

    def __init__(
        self,
        community_data_file: str | Path,
        metadata_file: str | Path,
        ntv3_embedding_path: str | Path,
        min_abundance: float = 1e-4,
        output_min: float = -2.0,
        output_max: float = 2.0,
        normalize: bool = False,
        donor_sample_column: str = "donor_sample_id",
        recipient_sample_column: str = "recipient_sample_id",
        post_sample_column: str = "post_sample_id",
        target_id_column: str = "TargetID",
    ):
        self.community_data_file = Path(community_data_file)
        self.metadata_file = Path(metadata_file)
        self.ntv3_embedding_path = Path(ntv3_embedding_path)
        self.donor_sample_column = donor_sample_column
        self.recipient_sample_column = recipient_sample_column
        self.post_sample_column = post_sample_column
        self.target_id_column = target_id_column

        self.data = load_preprocessed_data(str(self.community_data_file))
        self.otu_data: Dict[str, Dict[str, np.ndarray]] = self.data["otu_data"]
        self.sample_ids = set(str(sample_id) for sample_id in self.data.get("samples", self.otu_data.keys()))
        self.feature_ids, resolved_embeddings = _load_resolved_embeddings(self.ntv3_embedding_path)
        self.num_features = len(self.feature_ids)
        self.input_ids = torch.cat(
            [
                torch.tensor([-1], dtype=torch.long),
                torch.arange(self.num_features, dtype=torch.long),
            ]
        )
        cls_embedding = np.zeros((1, resolved_embeddings.shape[1]), dtype=np.float32)
        self.fixed_embeddings = torch.from_numpy(np.vstack([cls_embedding, resolved_embeddings]).astype(np.float32))
        self.attention_mask = torch.ones(self.num_features + 1, dtype=torch.long)
        self.supervision_mask = torch.ones(self.num_features + 1, dtype=torch.bool)
        self.supervision_mask[0] = False

        self.preprocessor = AbundancePreprocessor(
            min_abundance=min_abundance,
            output_min=output_min,
            output_max=output_max,
            normalize=normalize,
        )
        self.preprocessor.fit([])
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> pd.DataFrame:
        metadata = pd.read_csv(self.metadata_file)
        required = {
            self.donor_sample_column,
            self.recipient_sample_column,
            self.post_sample_column,
        }
        missing = sorted(required.difference(metadata.columns))
        if missing:
            raise ValueError(f"FMT metadata is missing columns: {missing}")
        if self.target_id_column not in metadata.columns:
            metadata[self.target_id_column] = [
                f"{row.get('fmt_id', idx)}::{row[self.post_sample_column]}"
                for idx, row in metadata.iterrows()
            ]

        usable = []
        for _, row in metadata.iterrows():
            donor = str(row[self.donor_sample_column])
            recipient = str(row[self.recipient_sample_column])
            post = str(row[self.post_sample_column])
            if donor in self.sample_ids and recipient in self.sample_ids and post in self.sample_ids:
                usable.append(row)
        if not usable:
            raise ValueError("No FMT metadata rows have donor, recipient, and post samples in the community dataset")
        return pd.DataFrame(usable).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def _dense_raw_profile(self, sample_id: str) -> np.ndarray:
        sample = self.otu_data[str(sample_id)]
        dense = np.zeros(self.num_features, dtype=np.float32)
        input_ids = np.asarray(sample["input_ids"], dtype=np.int64)
        abundance_values = np.asarray(sample["abundance_values"], dtype=np.float32)
        for feature_idx, abundance in zip(input_ids[1:], abundance_values[1:]):
            if 0 <= int(feature_idx) < self.num_features:
                dense[int(feature_idx)] = float(abundance)
        return dense

    def _model_values(self, dense_raw: np.ndarray) -> np.ndarray:
        floored = np.where(dense_raw > 0, dense_raw, self.preprocessor.min_abundance).astype(np.float32)
        transformed = self.preprocessor.transform(floored)
        return np.concatenate([[0.0], transformed.astype(np.float32)])

    @staticmethod
    def _raw_values(dense_raw: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], dense_raw.astype(np.float32)])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[int(idx)]
        donor_sample_id = str(row[self.donor_sample_column])
        recipient_sample_id = str(row[self.recipient_sample_column])
        post_sample_id = str(row[self.post_sample_column])
        donor_raw = self._dense_raw_profile(donor_sample_id)
        recipient_raw = self._dense_raw_profile(recipient_sample_id)
        post_raw = self._dense_raw_profile(post_sample_id)

        timepoint = row.get("timepoint_fmt", np.nan)
        try:
            timepoint = float(timepoint)
        except (TypeError, ValueError):
            timepoint = np.nan

        return {
            "target_id": str(row[self.target_id_column]),
            "fmt_id": str(row.get("fmt_id", "")),
            "study": str(row.get("study", "")),
            "response_label": str(row.get("response_label", "")),
            "donor_sample_id": donor_sample_id,
            "recipient_sample_id": recipient_sample_id,
            "post_sample_id": post_sample_id,
            "timepoint_fmt": torch.tensor(timepoint, dtype=torch.float32),
            "input_ids": self.input_ids,
            "embeddings": self.fixed_embeddings,
            "attention_mask": self.attention_mask,
            "supervision_mask": self.supervision_mask,
            "donor_abundance_values": torch.from_numpy(self._model_values(donor_raw)).float(),
            "recipient_abundance_values": torch.from_numpy(self._model_values(recipient_raw)).float(),
            "post_abundance_values": torch.from_numpy(self._model_values(post_raw)).float(),
            "donor_abundance_raw": torch.from_numpy(self._raw_values(donor_raw)).float(),
            "recipient_abundance_raw": torch.from_numpy(self._raw_values(recipient_raw)).float(),
            "post_abundance_raw": torch.from_numpy(self._raw_values(post_raw)).float(),
        }


class FMTCommunityClassificationDataset(Dataset):
    """FMT donor-recipient examples for event-level response classification."""

    DEFAULT_CLASS_NAMES = ["non-responder", "responder"]

    def __init__(
        self,
        community_data_file: str | Path,
        metadata_file: str | Path,
        ntv3_embedding_path: str | Path,
        min_abundance: float = 1e-4,
        output_min: float = -2.0,
        output_max: float = 2.0,
        normalize: bool = False,
        donor_sample_column: str = "donor_sample_id",
        recipient_sample_column: str = "recipient_sample_id",
        label_column: str = "response_label",
        target_id_column: str = "TargetID",
        class_names: List[str] | None = None,
        deduplicate_group_column: str | None = "fmt_id",
    ):
        self.community_data_file = Path(community_data_file)
        self.metadata_file = Path(metadata_file)
        self.ntv3_embedding_path = Path(ntv3_embedding_path)
        self.donor_sample_column = donor_sample_column
        self.recipient_sample_column = recipient_sample_column
        self.label_column = label_column
        self.target_id_column = target_id_column
        self.class_names = list(class_names or self.DEFAULT_CLASS_NAMES)
        self.class_to_id = {label: idx for idx, label in enumerate(self.class_names)}
        self.deduplicate_group_column = deduplicate_group_column

        self.data = load_preprocessed_data(str(self.community_data_file))
        self.otu_data: Dict[str, Dict[str, np.ndarray]] = self.data["otu_data"]
        self.sample_ids = set(str(sample_id) for sample_id in self.data.get("samples", self.otu_data.keys()))
        self.feature_ids, resolved_embeddings = _load_resolved_embeddings(self.ntv3_embedding_path)
        self.num_features = len(self.feature_ids)
        self.input_ids = torch.cat(
            [
                torch.tensor([-1], dtype=torch.long),
                torch.arange(self.num_features, dtype=torch.long),
            ]
        )
        cls_embedding = np.zeros((1, resolved_embeddings.shape[1]), dtype=np.float32)
        self.fixed_embeddings = torch.from_numpy(np.vstack([cls_embedding, resolved_embeddings]).astype(np.float32))
        self.attention_mask = torch.ones(self.num_features + 1, dtype=torch.long)
        self.supervision_mask = torch.ones(self.num_features + 1, dtype=torch.bool)
        self.supervision_mask[0] = False

        self.preprocessor = AbundancePreprocessor(
            min_abundance=min_abundance,
            output_min=output_min,
            output_max=output_max,
            normalize=normalize,
        )
        self.preprocessor.fit([])
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> pd.DataFrame:
        metadata = pd.read_csv(self.metadata_file)
        required = {
            self.donor_sample_column,
            self.recipient_sample_column,
            self.label_column,
        }
        missing = sorted(required.difference(metadata.columns))
        if missing:
            raise ValueError(f"FMT response metadata is missing columns: {missing}")
        if self.target_id_column not in metadata.columns:
            metadata[self.target_id_column] = [
                str(row.get("fmt_id", idx))
                for idx, row in metadata.iterrows()
            ]

        metadata = metadata.copy()
        metadata[self.label_column] = metadata[self.label_column].astype(str)
        metadata = metadata[metadata[self.label_column].isin(self.class_to_id)]
        if self.deduplicate_group_column and self.deduplicate_group_column in metadata.columns:
            metadata = metadata.sort_values(self.target_id_column).drop_duplicates(self.deduplicate_group_column, keep="first")

        usable = []
        for _, row in metadata.iterrows():
            donor = str(row[self.donor_sample_column])
            recipient = str(row[self.recipient_sample_column])
            if donor in self.sample_ids and recipient in self.sample_ids:
                usable.append(row)
        if not usable:
            raise ValueError("No FMT response rows have donor and recipient samples in the community dataset")
        return pd.DataFrame(usable).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def _dense_raw_profile(self, sample_id: str) -> np.ndarray:
        sample = self.otu_data[str(sample_id)]
        dense = np.zeros(self.num_features, dtype=np.float32)
        input_ids = np.asarray(sample["input_ids"], dtype=np.int64)
        abundance_values = np.asarray(sample["abundance_values"], dtype=np.float32)
        for feature_idx, abundance in zip(input_ids[1:], abundance_values[1:]):
            if 0 <= int(feature_idx) < self.num_features:
                dense[int(feature_idx)] = float(abundance)
        return dense

    def _model_values(self, dense_raw: np.ndarray) -> np.ndarray:
        floored = np.where(dense_raw > 0, dense_raw, self.preprocessor.min_abundance).astype(np.float32)
        transformed = self.preprocessor.transform(floored)
        return np.concatenate([[0.0], transformed.astype(np.float32)])

    @staticmethod
    def _raw_values(dense_raw: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], dense_raw.astype(np.float32)])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[int(idx)]
        donor_sample_id = str(row[self.donor_sample_column])
        recipient_sample_id = str(row[self.recipient_sample_column])
        label_name = str(row[self.label_column])
        donor_raw = self._dense_raw_profile(donor_sample_id)
        recipient_raw = self._dense_raw_profile(recipient_sample_id)
        return {
            "target_id": str(row[self.target_id_column]),
            "fmt_id": str(row.get("fmt_id", "")),
            "study": str(row.get("study", "")),
            "indication": str(row.get("indication", "")),
            "response_label": label_name,
            "label": torch.tensor(self.class_to_id[label_name], dtype=torch.long),
            "donor_sample_id": donor_sample_id,
            "recipient_sample_id": recipient_sample_id,
            "input_ids": self.input_ids,
            "embeddings": self.fixed_embeddings,
            "attention_mask": self.attention_mask,
            "supervision_mask": self.supervision_mask,
            "donor_abundance_values": torch.from_numpy(self._model_values(donor_raw)).float(),
            "recipient_abundance_values": torch.from_numpy(self._model_values(recipient_raw)).float(),
            "donor_abundance_raw": torch.from_numpy(self._raw_values(donor_raw)).float(),
            "recipient_abundance_raw": torch.from_numpy(self._raw_values(recipient_raw)).float(),
        }


class FMTCommunityClassificationCollator:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "target_id": [item["target_id"] for item in batch],
            "fmt_id": [item["fmt_id"] for item in batch],
            "study": [item["study"] for item in batch],
            "indication": [item["indication"] for item in batch],
            "response_label": [item["response_label"] for item in batch],
            "donor_sample_id": [item["donor_sample_id"] for item in batch],
            "recipient_sample_id": [item["recipient_sample_id"] for item in batch],
        }
        for key in [
            "input_ids",
            "attention_mask",
            "supervision_mask",
            "donor_abundance_values",
            "recipient_abundance_values",
            "donor_abundance_raw",
            "recipient_abundance_raw",
            "label",
        ]:
            result[key] = torch.stack([item[key] for item in batch])
        result["embeddings"] = torch.stack([item["embeddings"] for item in batch]).contiguous()
        result["labels"] = result["label"]
        return result


class FMTCommunityPredictionCollator:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "target_id": [item["target_id"] for item in batch],
            "fmt_id": [item["fmt_id"] for item in batch],
            "study": [item["study"] for item in batch],
            "response_label": [item["response_label"] for item in batch],
            "donor_sample_id": [item["donor_sample_id"] for item in batch],
            "recipient_sample_id": [item["recipient_sample_id"] for item in batch],
            "post_sample_id": [item["post_sample_id"] for item in batch],
        }
        for key in [
            "input_ids",
            "attention_mask",
            "supervision_mask",
            "donor_abundance_values",
            "recipient_abundance_values",
            "post_abundance_values",
            "donor_abundance_raw",
            "recipient_abundance_raw",
            "post_abundance_raw",
            "timepoint_fmt",
        ]:
            result[key] = torch.stack([item[key] for item in batch])
        result["embeddings"] = torch.stack([item["embeddings"] for item in batch]).contiguous()
        return result


def grouped_train_val_indices(metadata: pd.DataFrame, val_ratio: float, seed: int, group_column: str = "fmt_id"):
    labels = metadata[group_column].astype(str).to_numpy() if group_column in metadata.columns else np.arange(len(metadata))
    unique = np.asarray(pd.unique(labels), dtype=object)
    if len(unique) < 2:
        indices = np.arange(len(metadata))
        return indices, indices[:0]
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_ratio)))
    n_val = min(n_val, len(unique) - 1)
    val_groups = set(unique[:n_val])
    val_idx = np.flatnonzero(np.asarray([label in val_groups for label in labels]))
    train_idx = np.flatnonzero(np.asarray([label not in val_groups for label in labels]))
    return train_idx, val_idx


def grouped_kfold_indices(
    metadata: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    group_column: str = "fmt_id",
    label_column: str = "response_label",
):
    """Return stratified group k-fold indices without splitting FMT events."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if group_column in metadata.columns:
        group_values = metadata[group_column].astype(str).to_numpy()
    else:
        group_values = np.asarray([str(idx) for idx in range(len(metadata))], dtype=object)
    if label_column in metadata.columns:
        label_values = metadata[label_column].astype(str).to_numpy()
    else:
        label_values = np.asarray([""] * len(metadata), dtype=object)

    group_to_label: Dict[str, str] = {}
    for group, label in zip(group_values, label_values):
        group_to_label.setdefault(str(group), str(label))

    label_to_groups: Dict[str, List[str]] = {}
    for group, label in group_to_label.items():
        label_to_groups.setdefault(label, []).append(group)

    rng = np.random.default_rng(seed)
    fold_groups: List[List[str]] = [[] for _ in range(n_splits)]
    for label in sorted(label_to_groups, key=str):
        groups = list(label_to_groups[label])
        rng.shuffle(groups)
        for offset, group in enumerate(groups):
            fold_groups[offset % n_splits].append(group)

    folds = []
    all_indices = np.arange(len(metadata))
    for groups in fold_groups:
        val_groups = set(groups)
        val_mask = np.asarray([str(group) in val_groups for group in group_values])
        val_idx = all_indices[val_mask]
        train_idx = all_indices[~val_mask]
        if len(val_idx) == 0 or len(train_idx) == 0:
            raise ValueError("Grouped k-fold split produced an empty train or validation fold")
        folds.append((train_idx, val_idx))
    return folds


def create_fmt_community_classification_dataloaders(
    train_dataset: FMTCommunityClassificationDataset,
    fold_index: int = 0,
    n_splits: int = 5,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    group_column: str = "fmt_id",
    label_column: str = "response_label",
):
    folds = grouped_kfold_indices(
        train_dataset.metadata,
        n_splits=n_splits,
        seed=seed,
        group_column=group_column,
        label_column=label_column,
    )
    if fold_index < 0 or fold_index >= len(folds):
        raise ValueError(f"fold_index must be in [0, {len(folds) - 1}]")
    train_idx, val_idx = folds[fold_index]
    collator = FMTCommunityClassificationCollator()
    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": collator,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(Subset(train_dataset, train_idx.tolist()), batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(train_dataset, val_idx.tolist()), batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, train_idx, val_idx


def create_fmt_community_classification_test_loader(
    test_dataset: FMTCommunityClassificationDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
):
    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": FMTCommunityClassificationCollator(),
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "shuffle": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(test_dataset, batch_size=batch_size, **loader_kwargs)


def create_fmt_community_prediction_dataloaders(
    train_dataset: FMTCommunityPredictionDataset,
    val_ratio: float = 0.1,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    group_column: str = "fmt_id",
):
    train_idx, val_idx = grouped_train_val_indices(train_dataset.metadata, val_ratio, seed, group_column=group_column)
    collator = FMTCommunityPredictionCollator()
    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": collator,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(Subset(train_dataset, train_idx.tolist()), batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(train_dataset, val_idx.tolist()), batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, train_idx, val_idx


def create_fmt_community_prediction_test_loader(
    test_dataset: FMTCommunityPredictionDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
):
    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": FMTCommunityPredictionCollator(),
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "shuffle": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(test_dataset, batch_size=batch_size, **loader_kwargs)
