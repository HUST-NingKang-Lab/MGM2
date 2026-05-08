"""
MGM2 dataset implementation.

Main dataset classes for dual-task masked microbiome pre-training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

from src.data.utils import load_ntv3_embeddings, load_preprocessed_data
from src.data.Preprocessing import AbundancePreprocessor, BinManager, preprocess_otu_sample

logger = logging.getLogger(__name__)


class MicrobiomeDatasetMGM2(Dataset):
    """
    Microbiome dataset for the current dual-task masked abundance objective.

    This dataset returns multiple targets for each sample:
    - bin_labels: Abundance bin classification (0=num_bins-1)
    - regression_targets: Continuous abundance values (for high abundance)
    - regression_mask: Mask indicating which positions to compute regression loss

    Runtime caches are kept at the class level so train/val/test splits inside the
    same process do not each re-read the full embedding files from disk. This cuts
    startup overhead substantially, especially before DDP forks or re-executes.
    """

    _ntv3_runtime_cache: Dict[str, Dict[str, Any]] = {}
    _qwen3_runtime_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        preprocessed_data: Dict[str, Any],
        abundance_preprocessor: AbundancePreprocessor,
        bin_manager: BinManager,
        mask_prob: float = 0.15,
        mask_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        ntv3_embedding_path: Optional[str] = None,
        qwen3_embedding_path: Optional[str] = None,
        qwen3_projection_dim: Optional[int] = None,
        enable_qwen3_alignment: bool = False,
        is_training: bool = True,
        deterministic_masking: bool = False,
        mask_seed: int = 42,
    ):
        """
        Args:
            preprocessed_data: Preprocessed data dictionary from BuildDataset
            abundance_preprocessor: Preprocessor for abundance values
            bin_manager: Bin manager for abundance binning
            mask_prob: Probability of masking
            mask_ratio: (mask, random, keep) ratio for masked positions
            ntv3_embedding_path: Path to NTv3 embeddings (if not in preprocessed_data)
            qwen3_embedding_path: Path to per-sample Qwen3 embedding HDF5 file
            qwen3_projection_dim: Expected Qwen3 projection dimension when alignment is enabled
            enable_qwen3_alignment: Whether to expose sample-level Qwen3 alignment targets
            is_training: Whether this is a training dataset (affects masking)
            deterministic_masking: Apply masking with a per-sample deterministic seed
            mask_seed: Base seed used for deterministic validation/test masking
        """
        self.data = preprocessed_data
        self.abundance_preprocessor = abundance_preprocessor
        self.bin_manager = bin_manager
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio
        self.is_training = is_training
        self.deterministic_masking = deterministic_masking
        self.mask_seed = int(mask_seed)
        self.mask_abundance_value = float(self.abundance_preprocessor.output_min - 1.0)

        # Load embeddings
        self.ntv3_embedding_path = ntv3_embedding_path or self.data.get('ntv3_embedding_path')
        self.ntv3_embeddings = None
        self.otu_id_to_idx = self.data.get('otu_id_to_idx', {})
        self.cls_embedding = None
        self.enable_qwen3_alignment = enable_qwen3_alignment
        self.qwen3_embedding_path = qwen3_embedding_path or self.data.get('qwen3_embedding_path')
        self.qwen3_projection_dim = qwen3_projection_dim
        self.qwen3_embeddings = None
        self.sample_id_to_qwen3_idx = {}
        self.qwen3_embedding_dim = 0
        self._load_embeddings()
        self._load_qwen3_embeddings()

        logger.info(
            f"MGM2Dataset initialized with {len(self)} samples | "
            f"Bins: {self.bin_manager.num_bins}, "
            f"Method: {self.bin_manager.method.value} | "
            f"Mask prob: {mask_prob}"
        )

    @classmethod
    def clear_runtime_caches(cls):
        """Clear process-local embedding caches. Mainly used by tests."""
        cls._ntv3_runtime_cache.clear()
        cls._qwen3_runtime_cache.clear()

    @staticmethod
    def _runtime_cache_key(path_like: Any) -> str:
        path = Path(path_like).resolve()
        stat = path.stat()
        return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"

    def _load_embeddings(self):
        """Load NTv3 embeddings, reusing a process-local cache when possible."""
        import h5py
        import os

        if self.ntv3_embedding_path and os.path.exists(self.ntv3_embedding_path):
            cache_key = self._runtime_cache_key(self.ntv3_embedding_path)
            cached = self._ntv3_runtime_cache.get(cache_key)
            if cached is None:
                with h5py.File(self.ntv3_embedding_path, 'r') as f:
                    embeddings = f['embeddings'][:]
                    otu_ids = f['otu_ids'][:].astype(str)

                cached = {
                    'embeddings': embeddings,
                    'otu_id_to_idx': {str(otu_id): idx for idx, otu_id in enumerate(otu_ids)},
                    'cls_embedding': np.zeros(embeddings.shape[1], dtype=np.float32),
                }
                self._ntv3_runtime_cache[cache_key] = cached
                logger.info('Loaded NTv3 embeddings into runtime cache: %s', embeddings.shape)
            else:
                logger.info('Reusing cached NTv3 embeddings from %s', self.ntv3_embedding_path)

            self.otu_id_to_idx = cached['otu_id_to_idx']
            self.cls_embedding = cached['cls_embedding']
            self.ntv3_embeddings = cached['embeddings']
            self._h5_file = None
        else:
            self._h5_file = None
            logger.warning(f"NTv3 embedding file not found: {self.ntv3_embedding_path}")
            if 'otu_id_to_idx' in self.data:
                dummy_dim = 1536
                n_otu = len(self.otu_id_to_idx)
                self.ntv3_embeddings = np.random.randn(n_otu, dummy_dim).astype(np.float32)
                self.cls_embedding = np.zeros(dummy_dim, dtype=np.float32)
                logger.warning(f"Using dummy embeddings with dim {dummy_dim}")


    def _load_qwen3_embeddings(self):
        """Load optional Qwen3 keyword embeddings keyed by sample ID."""
        import h5py
        import os

        if not self.enable_qwen3_alignment:
            return

        if not self.qwen3_embedding_path:
            raise FileNotFoundError('Qwen3 alignment is enabled but no qwen3 embedding path was provided')
        if not os.path.exists(self.qwen3_embedding_path):
            raise FileNotFoundError(f'Qwen3 embedding file not found: {self.qwen3_embedding_path}')

        cache_key = self._runtime_cache_key(self.qwen3_embedding_path)
        cached = self._qwen3_runtime_cache.get(cache_key)
        if cached is None:
            with h5py.File(self.qwen3_embedding_path, 'r') as f:
                embeddings = f['embeddings'][:].astype(np.float32)
                raw_sample_ids = f['sample_ids'][:]
                sample_ids = [self._normalize_sample_id(sample_id) for sample_id in raw_sample_ids]

            cached = {
                'embeddings': embeddings,
                'sample_id_to_qwen3_idx': {sample_id: idx for idx, sample_id in enumerate(sample_ids)},
                'embedding_dim': int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            }
            self._qwen3_runtime_cache[cache_key] = cached
            logger.info('Loaded Qwen3 embeddings into runtime cache for %s samples', len(sample_ids))
        else:
            logger.info('Reusing cached Qwen3 embeddings from %s', self.qwen3_embedding_path)

        self.qwen3_embeddings = cached['embeddings']
        self.sample_id_to_qwen3_idx = cached['sample_id_to_qwen3_idx']
        self.qwen3_embedding_dim = cached['embedding_dim']
        if self.qwen3_projection_dim is not None and self.qwen3_embedding_dim != self.qwen3_projection_dim:
            raise ValueError(
                f'Qwen3 embedding dimension ({self.qwen3_embedding_dim}) does not match '
                f'configured projection dim ({self.qwen3_projection_dim})'
            )
        logger.info(
            "Loaded Qwen3 embeddings for %s samples with dim %s",
            len(self.sample_id_to_qwen3_idx),
            self.qwen3_embedding_dim,
        )

    @staticmethod
    def _normalize_sample_id(sample_id: Any) -> str:
        """Normalize sample ids from HDF5 and dataset metadata into comparable strings."""
        if isinstance(sample_id, (bytes, bytearray, np.bytes_)):
            normalized = sample_id.decode('utf-8')
        else:
            normalized = str(sample_id)

        normalized = normalized.strip()
        while True:
            if normalized.startswith("b'") and normalized.endswith("'"):
                normalized = normalized[2:-1].strip()
                continue
            if normalized.startswith('b"') and normalized.endswith('"'):
                normalized = normalized[2:-1].strip()
                continue
            break
        return normalized

    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.data['samples'])

    def __del__(self):
        """Close the H5 file when done"""
        if hasattr(self, '_h5_file') and self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with all required information.

        For training: Apply dynamic masking
        For validation/test: Apply deterministic masking when requested
        """
        # Build sample using the preprocessing pipeline
        sample = self._build_sample(idx)

        if self.is_training:
            self._apply_abundance_masking(sample, idx=idx, deterministic=False)
        elif self.deterministic_masking:
            self._apply_abundance_masking(sample, idx=idx, deterministic=True)

        return sample

    def _apply_abundance_masking(self, sample: Dict[str, torch.Tensor], idx: int, deterministic: bool) -> None:
        """Mask abundance inputs and keep labels only at supervised masked positions."""
        # Only mask OTU positions; CLS at position 0 carries sample-level context.
        maskable_positions = torch.arange(len(sample['bin_labels']))[1:]
        if len(maskable_positions) == 0:
            return

        generator = None
        if deterministic:
            generator = torch.Generator()
            generator.manual_seed(self.mask_seed + int(idx))

        num_to_mask = max(1, int(len(maskable_positions) * self.mask_prob))
        mask_indices = torch.randperm(len(maskable_positions), generator=generator)[:num_to_mask]
        mask_positions = maskable_positions[mask_indices]

        supervised_mask = torch.zeros_like(sample['bin_labels'], dtype=torch.bool)
        supervised_mask[mask_positions] = True
        sample['bin_labels'][~supervised_mask] = -100
        sample['regression_mask'] = sample['regression_mask'] & supervised_mask

        # Corrupt only abundance inputs so masked tokens retain OTU identity.
        for pos in mask_positions:
            rand_val = float(torch.rand((), generator=generator).item())
            if rand_val < self.mask_ratio[0]:
                sample['abundance_values'][pos] = self.mask_abundance_value
            elif rand_val < self.mask_ratio[0] + self.mask_ratio[1]:
                random_pos = maskable_positions[torch.randint(len(maskable_positions), (1,), generator=generator)].item()
                sample['abundance_values'][pos] = sample['abundance_values'][random_pos]
            # keep branch leaves the original OTU embedding and abundance untouched

    def _build_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Build a sample using the new preprocessing pipeline.
        """
        # Extract data based on expected format
        if 'samples' in self.data:
            sample_id = self.data['samples'][idx]

            # Get OTU IDs and abundances for this sample
            # Format depends on how data was preprocessed
            if 'otu_data' in self.data:
                # New format (built by src/data/build_dataset.py):
                # data['otu_data'] is dict {sample_id: {'input_ids', 'attention_mask', 'abundance_values'}}
                sample_data = self.data['otu_data'][sample_id]
                input_ids = sample_data['input_ids']
                abundances = sample_data['abundance_values']
                attention_mask = sample_data['attention_mask']
            else:
                # Old format: try input_ids and abundance_values lists
                otu_ids = self.data.get('otu_ids', [])[:]
                if idx < len(self.data['input_ids']):
                    input_ids = self.data['input_ids'][idx]
                    abundances = self.data['abundance_values'][idx]
                    attention_mask = self.data['attention_mask'][idx]
                else:
                    logger.warning(f"Sample {idx} not found in data")
                    return self._create_empty_sample()
        else:
            # Fallback - use indices directly
            if idx < len(self.data['input_ids']):
                input_ids = self.data['input_ids'][idx]
                abundances = self.data['abundance_values'][idx]
                attention_mask = self.data['attention_mask'][idx]
            else:
                logger.warning(f"Sample {idx} not found in data")
                return self._create_empty_sample()

        # Use preprocessing pipeline
        # input_ids already contains the NTv3 index (-1 for CLS)
        sample = preprocess_otu_sample(
            input_ids=input_ids,
            abundances=np.array(abundances, dtype=np.float32),
            abundance_preprocessor=self.abundance_preprocessor,
            bin_manager=self.bin_manager,
            otu_id_to_idx=self.otu_id_to_idx,
            cls_embedding=self.cls_embedding,
            ntv3_embeddings=self.ntv3_embeddings
        )

        if self.enable_qwen3_alignment:
            sample_id_str = self._normalize_sample_id(sample_id) if 'samples' in self.data else None
            qwen3_idx = self.sample_id_to_qwen3_idx.get(sample_id_str) if sample_id_str is not None else None
            if qwen3_idx is not None and self.qwen3_embeddings is not None:
                sample['qwen3_embedding'] = torch.from_numpy(self.qwen3_embeddings[qwen3_idx]).float()
                sample['qwen3_alignment_mask'] = torch.tensor(True, dtype=torch.bool)
            else:
                sample['qwen3_embedding'] = torch.zeros(self.qwen3_embedding_dim, dtype=torch.float32)
                sample['qwen3_alignment_mask'] = torch.tensor(False, dtype=torch.bool)

        return sample

    def _create_empty_sample(self) -> Dict[str, torch.Tensor]:
        """Create an empty sample with only CLS token"""
        embedding_dim = len(self.cls_embedding)

        sample = {
            'input_ids': torch.tensor([-1], dtype=torch.long),
            'embeddings': torch.from_numpy(self.cls_embedding).unsqueeze(0),
            'attention_mask': torch.ones(1, dtype=torch.long),
            'abundance_values': torch.zeros(1, dtype=torch.float),
            'bin_labels': torch.zeros(1, dtype=torch.long),
            'regression_targets': torch.zeros(1, dtype=torch.float),
            'regression_mask': torch.zeros(1, dtype=torch.bool),
        }
        if self.enable_qwen3_alignment:
            sample['qwen3_embedding'] = torch.zeros(self.qwen3_embedding_dim, dtype=torch.float32)
            sample['qwen3_alignment_mask'] = torch.tensor(False, dtype=torch.bool)
        return sample


class BatchCollatorMGM2:
    """
    Collator for batching MGM2.0 samples with proper padding.
    """

    def __init__(self, pad_value: int = -100):
        """
        Args:
            pad_value: Value to use for padding labels (should be -100 for PyTorch to ignore in loss)
        """
        self.pad_value = pad_value

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Pad batch to maximum sequence length.
        """
        if not batch:
            raise ValueError("Cannot collate empty batch")

        # Find maximum length in batch
        max_len = max(len(sample['input_ids']) for sample in batch)

        padded_batch = {}

        for key in batch[0].keys():
            if key == 'input_ids':
                padded_batch[key] = self._pad_sequence(
                    [sample[key] for sample in batch],
                    max_len,
                    pad_value=-1  # Use -1 for input_ids padding
                )
            elif key == 'attention_mask':
                padded_batch[key] = self._pad_sequence(
                    [sample[key] for sample in batch],
                    max_len,
                    pad_value=0
                )
            elif key == 'bin_labels':
                padded_batch[key] = self._pad_sequence(
                    [sample[key] for sample in batch],
                    max_len,
                    pad_value=self.pad_value
                )
            elif key == 'regression_mask':
                padded_batch[key] = self._pad_sequence(
                    [sample[key] for sample in batch],
                    max_len,
                    pad_value=False
                )
            elif key == 'qwen3_embedding':
                padded_batch[key] = torch.stack([sample[key] for sample in batch])
            elif key == 'qwen3_alignment_mask':
                padded_batch[key] = torch.stack([sample[key] for sample in batch])
            elif key == 'labels':
                padded_batch[key] = torch.stack([sample[key] for sample in batch])
            elif key == 'sample_id':
                padded_batch[key] = [sample[key] for sample in batch]
            elif key in ['abundance_values', 'regression_targets', 'embeddings']:
                # Float tensors
                if key == 'embeddings':
                    embedding_dim = batch[0][key].shape[1]
                    padded_batch[key] = self._pad_embeddings(
                        [sample[key] for sample in batch],
                        max_len,
                        embedding_dim
                    )
                else:
                    padded_batch[key] = self._pad_sequence(
                        [sample[key] for sample in batch],
                        max_len,
                        pad_value=0.0
                    )
            else:
                # Default padding
                padded_batch[key] = self._pad_sequence(
                    [sample[key] for sample in batch],
                    max_len,
                    pad_value=0
                )

        return padded_batch

    def _pad_sequence(self, sequences: List[torch.Tensor], max_len: int, pad_value) -> torch.Tensor:
        """Pad 1D sequences to max_len"""
        padded = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            if pad_len > 0:
                if isinstance(pad_value, (int, float)):
                    pad_tensor = torch.full((pad_len,), pad_value, dtype=seq.dtype)
                else:
                    pad_tensor = torch.tensor([pad_value] * pad_len, dtype=seq.dtype)
                padded_seq = torch.cat([seq, pad_tensor])
            else:
                padded_seq = seq
            padded.append(padded_seq)
        return torch.stack(padded)

    def _pad_embeddings(self, embeddings: List[torch.Tensor], max_len: int, embedding_dim: int) -> torch.Tensor:
        """Pad embeddings to max_len"""
        padded = []
        for emb in embeddings:
            pad_len = max_len - emb.shape[0]
            if pad_len > 0:
                pad_tensor = torch.zeros(pad_len, embedding_dim, dtype=emb.dtype)
                padded_emb = torch.cat([emb, pad_tensor], dim=0)
            else:
                padded_emb = emb
            padded.append(padded_emb)
        return torch.stack(padded)


def create_dataloaders_mgm2(
    data_path: str,
    ntv3_embedding_path: str,
    abundance_bin_config: Dict[str, Any],
    training_config: Dict[str, Any],
    qwen3_alignment_config: Optional[Dict[str, Any]] = None,
    mask_prob: float = 0.15,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    include_test_loader: bool = True,
) -> Tuple[DataLoader, ...]:
    """
    Create train, validation, and test dataloaders for MGM2.0.

    Args:
        data_path: Path to preprocessed data
        ntv3_embedding_path: Path to NTv3 embeddings
        abundance_bin_config: Configuration for abundance binning
        training_config: Training configuration
        qwen3_alignment_config: Optional Qwen3 alignment configuration
        mask_prob: Masking probability
        num_workers: Number of workers for data loading
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        include_test_loader: Whether to materialize a test split/loader

    Returns:
        Tuple of (train_loader, val_loader) or (train_loader, val_loader, test_loader)
    """
    # Load data
    data = load_preprocessed_data(data_path)

    # Update data with embedding path if not present
    if 'ntv3_embedding_path' not in data:
        data['ntv3_embedding_path'] = ntv3_embedding_path

    # Create preprocessor and bin manager
    # MGM2.0 preprocessing: floor to min_abundance, log10 + 2, clip to [-2, 2]
    abundance_preprocessor = AbundancePreprocessor(
        min_abundance=abundance_bin_config.get('min_abundance', 1e-4),
        output_min=abundance_bin_config.get('output_min', -2.0),
        output_max=abundance_bin_config.get('output_max', 2.0),
        normalize=abundance_bin_config.get('normalize', False),
    )

    # Fit preprocessor and bin manager on training data
    # Note: We fit on the training split only, and avoid materializing/transforms
    # unless a configuration actually needs dataset-wide statistics.
    train_end = int(len(data['samples']) * train_ratio)
    train_abundances = []
    if abundance_preprocessor.normalize:
        if 'otu_data' in data:
            for i in range(train_end):
                sample_id = data['samples'][i]
                sample_data = data['otu_data'][sample_id]
                train_abundances.append(sample_data['abundance_values'])
        else:
            train_abundances = [data['abundance_values'][i] for i in range(train_end)]

    abundance_preprocessor.fit(train_abundances)
    processed_train_abundances = []
    if abundance_preprocessor.normalize:
        processed_train_abundances = [
            abundance_preprocessor.transform(np.asarray(a, dtype=np.float32))
            for a in train_abundances
        ]
    bin_manager = BinManager(
        num_bins=abundance_bin_config['num_bins'],
        method=abundance_bin_config['binning_method'],
        fixed_thresholds=abundance_bin_config.get('fixed_thresholds'),
        enable_regression=abundance_bin_config.get('enable_regression', False),
        regression_top_k_bins=abundance_bin_config.get('regression_top_k_bins', 1),
    )
    bin_manager.fit(processed_train_abundances)

    logger.info(f"Abundance binning: {bin_manager.get_bin_info()}")

    # Split dataset deterministically so every DDP process sees the same split.
    n_samples = len(data['samples'])
    split_seed = int(training_config.get('seed', 42))
    rng = np.random.default_rng(split_seed)
    indices = list(range(n_samples))
    rng.shuffle(indices)

    if include_test_loader:
        if train_ratio + val_ratio >= 1.0:
            raise ValueError('train_ratio + val_ratio must be less than 1.0 when include_test_loader=True')
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
    else:
        if not np.isclose(train_ratio + val_ratio, 1.0):
            raise ValueError('train_ratio + val_ratio must equal 1.0 when include_test_loader=False')
        train_end = int(n_samples * train_ratio)
        val_end = n_samples

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    if include_test_loader:
        logger.info(f"Dataset split - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    else:
        logger.info(f"Dataset split - Train: {len(train_indices)}, Val: {len(val_indices)} (test split skipped)")

    # Create subset data dictionaries
    def create_subset(indices):
        subset = {}
        for key, value in data.items():
            if key == 'samples' and isinstance(value, list) and len(value) == n_samples:
                subset[key] = [value[i] for i in indices]
            elif key == 'otu_data' and isinstance(value, dict):
                # New format: otu_data is dict {sample_id: data}, we need to subset based on indices
                subset[key] = {}
                for i in indices:
                    sample_id = data['samples'][i]
                    subset[key][sample_id] = value[sample_id]
            elif isinstance(value, list) and len(value) == n_samples:
                subset[key] = [value[i] for i in indices]
            else:
                subset[key] = value
        return subset

    train_data = create_subset(train_indices)
    val_data = create_subset(val_indices)
    test_data = create_subset(test_indices) if include_test_loader else None

    qwen3_alignment_config = qwen3_alignment_config or {}
    qwen3_enabled = qwen3_alignment_config.get('enabled', False)
    qwen3_embedding_path = qwen3_alignment_config.get('embedding_path') or None
    mask_ratio = tuple(training_config.get('mask_ratio', (0.8, 0.1, 0.1)))

    # Create datasets
    train_dataset = MicrobiomeDatasetMGM2(
        preprocessed_data=train_data,
        abundance_preprocessor=abundance_preprocessor,
        bin_manager=bin_manager,
        mask_prob=mask_prob,
        mask_ratio=mask_ratio,
        qwen3_embedding_path=qwen3_embedding_path,
        enable_qwen3_alignment=qwen3_enabled,
        is_training=True,
    )

    val_dataset = MicrobiomeDatasetMGM2(
        preprocessed_data=val_data,
        abundance_preprocessor=abundance_preprocessor,
        bin_manager=bin_manager,
        mask_prob=mask_prob,
        mask_ratio=mask_ratio,
        qwen3_embedding_path=qwen3_embedding_path,
        enable_qwen3_alignment=qwen3_enabled,
        is_training=False,
        deterministic_masking=True,
        mask_seed=split_seed,
    )

    test_dataset = None
    if include_test_loader:
        test_dataset = MicrobiomeDatasetMGM2(
            preprocessed_data=test_data,
            abundance_preprocessor=abundance_preprocessor,
            bin_manager=bin_manager,
            mask_prob=mask_prob,
            mask_ratio=mask_ratio,
            qwen3_embedding_path=qwen3_embedding_path,
            enable_qwen3_alignment=qwen3_enabled,
            is_training=False,
            deterministic_masking=True,
            mask_seed=split_seed + 1,
        )

    # Create collator
    collator = BatchCollatorMGM2()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    test_loader = None
    if include_test_loader:
        test_loader = DataLoader(
            test_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    # Save bin manager info for reproducibility
    output_dir = Path(training_config.get('output_dir', './outputs_mgm2'))
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_info = bin_manager.get_bin_info()
    with open(output_dir / 'bin_info.json', 'w') as f:
        import json
        json.dump(bin_info, f, indent=2)

    logger.info(f"Bin information saved to {output_dir / 'bin_info.json'}")

    if include_test_loader:
        return train_loader, val_loader, test_loader
    return train_loader, val_loader
