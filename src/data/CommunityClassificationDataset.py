"""Datasets and dataloaders for downstream community classification."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.Dataset import BatchCollatorMGM2, MicrobiomeDatasetMGM2
from src.data.Preprocessing import AbundancePreprocessor, BinManager
from src.data.utils import load_preprocessed_data

logger = logging.getLogger(__name__)


def load_metadata_table(metadata_path: Path, h5_key: Optional[str] = None) -> pd.DataFrame:
    """Load sample metadata from CSV, TSV, or pandas-compatible HDF5."""
    metadata_path = Path(metadata_path)
    suffix = metadata_path.suffix.lower()
    if suffix in {'.tsv', '.txt'}:
        return pd.read_csv(metadata_path, sep='\t')
    if suffix in {'.h5', '.hdf5'}:
        return pd.read_hdf(metadata_path, key=h5_key)
    return pd.read_csv(metadata_path)


def build_label_mapping(
    metadata: pd.DataFrame,
    sample_id_column: str,
    label_column: str,
    unknown_label: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Return sample-to-label and deterministic label-to-id mappings."""
    if sample_id_column not in metadata.columns or label_column not in metadata.columns:
        raise KeyError(f'Metadata must contain columns {sample_id_column!r} and {label_column!r}')

    labels_by_sample = {}
    for _, row in metadata[[sample_id_column, label_column]].dropna().iterrows():
        sample_id = MicrobiomeDatasetMGM2._normalize_sample_id(row[sample_id_column])
        labels_by_sample[sample_id] = str(row[label_column])

    label_names = sorted(set(labels_by_sample.values()))
    if not label_names:
        raise ValueError(f'No valid labels found in metadata column {label_column!r}')
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    if unknown_label and unknown_label not in label_to_id:
        label_to_id[unknown_label] = len(label_to_id)
    return labels_by_sample, label_to_id


def stratified_split_sample_ids(
    sample_ids: Sequence[str],
    labels_by_sample: Dict[str, Any],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split sample ids by label groups while keeping singleton classes in train."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError('val_ratio must be between 0 and 1')

    grouped: Dict[Any, List[str]] = {}
    for sample_id in sample_ids:
        grouped.setdefault(labels_by_sample[sample_id], []).append(sample_id)

    rng = np.random.default_rng(seed)
    train_ids: List[str] = []
    val_ids: List[str] = []
    for label in sorted(grouped, key=str):
        ids = list(grouped[label])
        rng.shuffle(ids)
        if len(ids) <= 1:
            train_ids.extend(ids)
            continue
        val_count = int(round(len(ids) * val_ratio))
        val_count = min(max(1, val_count), len(ids) - 1)
        val_ids.extend(ids[:val_count])
        train_ids.extend(ids[val_count:])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    if not train_ids or not val_ids:
        raise ValueError('Stratified split produced an empty train or validation set')
    return train_ids, val_ids


def subset_preprocessed_data(data: Dict[str, Any], sample_ids: Sequence[str]) -> Dict[str, Any]:
    """Subset an MGM2 preprocessed payload by sample id."""
    wanted = set(sample_ids)
    subset = {}
    for key, value in data.items():
        if key == 'samples':
            subset[key] = [sample_id for sample_id in data['samples'] if sample_id in wanted]
        elif key == 'otu_data' and isinstance(value, dict):
            subset[key] = {sample_id: value[sample_id] for sample_id in subset['samples']}
        else:
            subset[key] = value
    return subset


def _collect_abundances(data: Dict[str, Any], sample_ids: Sequence[str]) -> List[np.ndarray]:
    if 'otu_data' in data:
        return [np.asarray(data['otu_data'][sample_id]['abundance_values'], dtype=np.float32) for sample_id in sample_ids]
    return [np.asarray(data['abundance_values'][idx], dtype=np.float32) for idx, _ in enumerate(sample_ids)]


class CommunityClassificationDataset(MicrobiomeDatasetMGM2):
    """MGM2 sample dataset with community-level classification labels."""

    def __init__(
        self,
        preprocessed_data: Dict[str, Any],
        labels_by_sample: Dict[str, int],
        abundance_preprocessor: AbundancePreprocessor,
        bin_manager: BinManager,
        ntv3_embedding_path: Optional[str] = None,
    ):
        super().__init__(
            preprocessed_data=preprocessed_data,
            abundance_preprocessor=abundance_preprocessor,
            bin_manager=bin_manager,
            mask_prob=0.0,
            ntv3_embedding_path=ntv3_embedding_path,
            is_training=False,
            deterministic_masking=False,
        )
        self.labels_by_sample = {
            self._normalize_sample_id(sample_id): int(label_id)
            for sample_id, label_id in labels_by_sample.items()
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._build_sample(idx)
        sample_id = self._normalize_sample_id(self.data['samples'][idx])
        if sample_id not in self.labels_by_sample:
            raise KeyError(f'Missing classification label for sample {sample_id!r}')
        sample['sample_id'] = sample_id
        sample['labels'] = torch.tensor(self.labels_by_sample[sample_id], dtype=torch.long)
        return sample


def _build_common_preprocessors(
    train_data: Dict[str, Any],
    abundance_bin_config: Dict[str, Any],
) -> Tuple[AbundancePreprocessor, BinManager]:
    abundance_preprocessor = AbundancePreprocessor(
        min_abundance=abundance_bin_config.get('min_abundance', 1e-4),
        output_min=abundance_bin_config.get('output_min', -2.0),
        output_max=abundance_bin_config.get('output_max', 2.0),
        normalize=abundance_bin_config.get('normalize', False),
    )
    train_abundances = _collect_abundances(train_data, train_data['samples']) if abundance_preprocessor.normalize else []
    abundance_preprocessor.fit(train_abundances)
    processed_train = [
        abundance_preprocessor.transform(np.asarray(abundance, dtype=np.float32))
        for abundance in train_abundances
    ] if abundance_preprocessor.normalize else []
    bin_manager = BinManager(
        num_bins=abundance_bin_config['num_bins'],
        method=abundance_bin_config['binning_method'],
        fixed_thresholds=abundance_bin_config.get('fixed_thresholds'),
        enable_regression=abundance_bin_config.get('enable_regression', False),
        regression_top_k_bins=abundance_bin_config.get('regression_top_k_bins', 1),
    )
    bin_manager.fit(processed_train)
    return abundance_preprocessor, bin_manager


def create_community_classification_dataloaders(
    data_path: Path,
    metadata_path: Path,
    ntv3_embedding_path: Path,
    abundance_bin_config: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    sample_id_column: str,
    label_column: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    metadata_h5_key: Optional[str] = None,
    unknown_label: str = 'unknown',
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Create stratified train/validation dataloaders from one labeled pkl."""
    data = load_preprocessed_data(str(data_path))
    data['ntv3_embedding_path'] = str(ntv3_embedding_path)
    metadata = load_metadata_table(Path(metadata_path), metadata_h5_key)
    raw_labels_by_sample, label_to_id = build_label_mapping(
        metadata,
        sample_id_column,
        label_column,
        unknown_label=unknown_label,
    )

    normalized_samples = [MicrobiomeDatasetMGM2._normalize_sample_id(sample_id) for sample_id in data['samples']]
    sample_lookup = dict(zip(normalized_samples, data['samples']))
    usable_sample_ids = [
        sample_lookup[sample_id]
        for sample_id in normalized_samples
        if sample_id in raw_labels_by_sample
    ]
    if not usable_sample_ids:
        raise ValueError('No samples overlap between preprocessed data and metadata labels')

    string_labels_by_sample = {
        sample_lookup[sample_id]: raw_labels_by_sample[sample_id]
        for sample_id in normalized_samples
        if sample_id in raw_labels_by_sample
    }
    train_ids, val_ids = stratified_split_sample_ids(usable_sample_ids, string_labels_by_sample, val_ratio, seed)
    train_data = subset_preprocessed_data(data, train_ids)
    val_data = subset_preprocessed_data(data, val_ids)
    abundance_preprocessor, bin_manager = _build_common_preprocessors(train_data, abundance_bin_config)
    labels_by_sample = {
        sample_id: label_to_id[label]
        for sample_id, label in string_labels_by_sample.items()
    }

    train_dataset = CommunityClassificationDataset(train_data, labels_by_sample, abundance_preprocessor, bin_manager, str(ntv3_embedding_path))
    val_dataset = CommunityClassificationDataset(val_data, labels_by_sample, abundance_preprocessor, bin_manager, str(ntv3_embedding_path))
    collator = BatchCollatorMGM2()
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        ),
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        ),
        label_to_id,
    )


def create_community_classification_test_loader(
    data_path: Path,
    metadata_path: Path,
    ntv3_embedding_path: Path,
    abundance_bin_config: Dict[str, Any],
    label_to_id: Dict[str, int],
    batch_size: int,
    num_workers: int,
    sample_id_column: str,
    label_column: str,
    metadata_h5_key: Optional[str] = None,
    abundance_preprocessor: Optional[AbundancePreprocessor] = None,
    bin_manager: Optional[BinManager] = None,
    unknown_label: str = 'unknown',
) -> DataLoader:
    """Create a labeled test dataloader using the training label mapping."""
    data = load_preprocessed_data(str(data_path))
    data['ntv3_embedding_path'] = str(ntv3_embedding_path)
    metadata = load_metadata_table(Path(metadata_path), metadata_h5_key)
    raw_labels_by_sample, _ = build_label_mapping(metadata, sample_id_column, label_column)
    normalized_samples = [MicrobiomeDatasetMGM2._normalize_sample_id(sample_id) for sample_id in data['samples']]
    sample_lookup = dict(zip(normalized_samples, data['samples']))
    usable_sample_ids = [
        sample_lookup[sample_id]
        for sample_id in normalized_samples
        if sample_id in raw_labels_by_sample
    ]
    if not usable_sample_ids:
        raise ValueError('No test samples overlap between preprocessed data and metadata labels')

    if unknown_label not in label_to_id:
        label_to_id[unknown_label] = len(label_to_id)
    unknown_labels = sorted({raw_labels_by_sample[sample_id] for sample_id in normalized_samples if sample_id in raw_labels_by_sample} - set(label_to_id))
    if unknown_labels:
        logger.warning(
            'Mapping test labels absent from training metadata to %r: %s',
            unknown_label,
            unknown_labels,
        )

    test_data = subset_preprocessed_data(data, usable_sample_ids)
    if abundance_preprocessor is None or bin_manager is None:
        abundance_preprocessor, bin_manager = _build_common_preprocessors(test_data, abundance_bin_config)
    labels_by_sample = {
        sample_lookup[sample_id]: label_to_id.get(raw_labels_by_sample[sample_id], label_to_id[unknown_label])
        for sample_id in normalized_samples
        if sample_id in raw_labels_by_sample
    }
    test_dataset = CommunityClassificationDataset(test_data, labels_by_sample, abundance_preprocessor, bin_manager, str(ntv3_embedding_path))
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=BatchCollatorMGM2(),
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
