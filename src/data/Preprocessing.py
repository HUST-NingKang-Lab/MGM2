"""Abundance preprocessing and binning utilities for the current MGM2 pipeline."""

import numpy as np
import torch
from typing import List, Optional, Tuple, Dict, Union, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BinningMethod(Enum):
    QUANTILE = "quantile"
    FIXED_THRESHOLD = "fixed_threshold"


class AbundancePreprocessor:
    """
    Preprocesses abundance values for robust training.

    MGM2 specific steps:
    1. Handle missing/invalid values
    2. Floor values between 0 and 1e-4 to 1e-4
    3. Apply log10 transformation and shift by +2 → range [-2, 2]
    4. Clip to [-2, 2]

    Resulting range: 0 for absence and a clipped log-abundance range for positive values.
    """

    def __init__(
        self,
        min_abundance: float = 1e-4,
        output_min: float = -2.0,
        output_max: float = 2.0,
        normalize: bool = False
    ):
        """
        Args:
            min_abundance: Minimum non-zero abundance to keep (values < min_abundance get floored to this)
            output_min: Minimum output after log transform (default: -2)
            output_max: Maximum output after log transform (default: 2)
            normalize: Whether to apply additional normalization (default: False)
        """
        self.min_abundance = min_abundance
        self.output_min = output_min
        self.output_max = output_max
        self.normalize = normalize
        self.mean_ = None
        self.std_ = None

    def fit(self, abundance_values: List[np.ndarray]):
        """
        Fit preprocessing parameters on training data.

        Args:
            abundance_values: List of abundance arrays from all samples
        """
        if not self.normalize:
            self.mean_ = None
            self.std_ = None
            logger.info(
                "AbundancePreprocessor initialized without dataset-wide normalization; skipping global abundance scan"
            )
            return

        positive_abundances = [np.asarray(a, dtype=np.float32)[np.asarray(a, dtype=np.float32) > 0] for a in abundance_values]
        positive_abundances = [a for a in positive_abundances if a.size > 0]
        if not positive_abundances:
            self.mean_ = 0.0
            self.std_ = 1.0
            logger.warning('AbundancePreprocessor received no positive abundances; using mean=0 and std=1')
            return

        all_abundances = np.concatenate(positive_abundances)

        processed = all_abundances.copy()
        mask = (processed > 0) & (processed < self.min_abundance)
        processed[mask] = self.min_abundance
        processed[processed > 0] = np.log10(processed[processed > 0]) + 2
        processed = np.clip(processed, self.output_min, self.output_max)

        self.mean_ = np.mean(processed)
        self.std_ = np.std(processed)
        logger.info(f"AbundancePreprocessor fitted: mean={self.mean_:.4f}, std={self.std_:.4f}")

    def transform(self, abundance: np.ndarray) -> np.ndarray:
        """
        Transform abundance values according to MGM2 specification.

        Args:
            abundance: Raw relative abundance array (0 = absence)

        Returns:
            Preprocessed abundance array
        """
        # Copy to avoid modifying original
        processed = abundance.copy().astype(np.float32)

        # Handle invalid values
        processed = np.nan_to_num(processed, nan=0.0, posinf=self.output_max, neginf=0.0)

        # Clip negative values to 0
        processed[processed < 0] = 0

        # Floor values between 0 and min_abundance to min_abundance
        mask = (processed > 0) & (processed < self.min_abundance)
        processed[mask] = self.min_abundance

        # Apply log10 transform + shift to get range [-2, 2]
        # 1e-4 → log10(1e-4) + 2 = -4 + 2 = -2
        # 1 → log10(1) + 2 = 0 + 2 = 2
        positive_mask = processed > 0
        processed[positive_mask] = np.log10(processed[positive_mask]) + 2

        # Clip to output range
        processed[positive_mask] = np.clip(processed[positive_mask], self.output_min, self.output_max)

        # Additional normalization if enabled
        if self.normalize and self.mean_ is not None and self.std_ is not None:
            if positive_mask.any():
                processed[positive_mask] = (processed[positive_mask] - self.mean_) / (self.std_ + 1e-8)

        return processed

    def fit_transform(self, abundance_values: List[np.ndarray]) -> List[np.ndarray]:
        """Fit and transform in one step"""
        self.fit(abundance_values)
        return [self.transform(a) for a in abundance_values]


class BinManager:
    """
    Manages abundance binning for hierarchical classification.

    Creates bins based on either quantiles or fixed thresholds.
    Maps continuous abundance values to discrete bins.
    """

    def __init__(
        self,
        num_bins: int = 5,
        method: Union[str, BinningMethod] = BinningMethod.QUANTILE,
        fixed_thresholds: Optional[List[float]] = None,
        enable_regression: bool = False,
        regression_top_k_bins: int = 1
    ):
        """
        Args:
            num_bins: Number of bins (default: 5 for [absence, very_low, low, medium, high])
            method: Binning method - "quantile" or "fixed_threshold"
            fixed_thresholds: List of threshold values (only for fixed_threshold method)
            enable_regression: Whether to enable regression output (keeps continuous values)
            regression_top_k_bins: Number of highest abundance bins to include in regression supervision
        """
        self.num_bins = num_bins
        self.method = BinningMethod(method) if isinstance(method, str) else method
        self.fixed_thresholds = fixed_thresholds
        self.enable_regression = enable_regression
        self.regression_top_k_bins = regression_top_k_bins
        if not 1 <= self.regression_top_k_bins <= (self.num_bins - 1):
            raise ValueError('regression_top_k_bins must be between 1 and num_bins - 1')

        # Bin thresholds (only used for fixed-threshold binning in the current implementation)
        self.bin_thresholds_ = None
        self.bin_labels = self._create_bin_labels()

    def _create_bin_labels(self) -> List[str]:
        """Create human-readable bin labels"""
        if self.num_bins == 5:
            return ["absence", "very_low", "low", "medium", "high"]
        elif self.num_bins == 4:
            return ["absence", "low", "medium", "high"]
        elif self.num_bins == 3:
            return ["absence", "low", "high"]
        else:
            return [f"bin_{i}" for i in range(self.num_bins)]

    def fit(self, abundance_values: List[np.ndarray]):
        """
        Compute bin thresholds from training data.

        Args:
            abundance_values: List of PREPROCESSED abundance arrays from all samples
                (output from AbundancePreprocessor.transform)
        """
        if self.method == BinningMethod.QUANTILE:
            # scGPT-style sample-wise quantile binning: thresholds are computed independently per sample
            self.bin_thresholds_ = None
            logger.info(
                'Using sample-wise quantile binning with %s bins; regression supervises the top %s bins',
                self.num_bins,
                self.regression_top_k_bins,
            )

        elif self.method == BinningMethod.FIXED_THRESHOLD:
            if self.fixed_thresholds is None:
                raise ValueError("Fixed thresholds must be provided for fixed_threshold method")
            if len(self.fixed_thresholds) != self.num_bins:
                raise ValueError(f"Number of thresholds ({len(self.fixed_thresholds)}) must match num_bins ({self.num_bins})")

            self.bin_thresholds_ = np.array(self.fixed_thresholds)
            logger.info(f"Using fixed thresholds for binning: {self.bin_thresholds_}")

    def transform(self, abundance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform preprocessed abundance values into bin labels and regression masks.

        Args:
            abundance: Preprocessed abundance array from AbundancePreprocessor.
                `0` means absence and positive values lie in the configured log-abundance range.

        Returns:
            Tuple of (bin_labels, high_abundance_mask)
                - bin_labels: Integer bin labels (0 = absence, 1+ = abundance bins)
                - high_abundance_mask: Boolean mask used to select regression-supervised positions
        """
        # Initialize output arrays
        bin_labels = np.zeros_like(abundance, dtype=np.int64)
        high_abundance_mask = np.zeros_like(abundance, dtype=bool)

        positive_mask = abundance != 0
        if not positive_mask.any():
            return bin_labels, high_abundance_mask

        if self.method == BinningMethod.QUANTILE:
            non_zero_abundance = abundance[positive_mask]
            quantiles = np.percentile(non_zero_abundance, np.linspace(0, 100, self.num_bins + 1)[1:-1])
            thresholds = np.asarray(quantiles, dtype=np.float32)
        else:
            if self.bin_thresholds_ is None:
                raise ValueError('BinManager must be fitted before fixed-threshold transform')
            thresholds = np.asarray(self.bin_thresholds_, dtype=np.float32)

        positive_values = abundance[positive_mask]
        positive_labels = np.searchsorted(thresholds, positive_values, side='right')
        positive_labels = np.clip(positive_labels, 1, self.num_bins - 1)
        bin_labels[positive_mask] = positive_labels
        regression_cutoff = max(1, self.num_bins - self.regression_top_k_bins)
        high_abundance_mask = bin_labels >= regression_cutoff

        return bin_labels, high_abundance_mask

    def fit_transform(self, abundance_values: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Fit and transform in one step"""
        self.fit(abundance_values)
        return [self.transform(a) for a in abundance_values]

    def get_bin_info(self) -> Dict[str, Any]:
        """Get bin information for logging"""
        info = {
            "num_bins": self.num_bins,
            "method": self.method.value,
            "bin_labels": self.bin_labels,
            "regression_top_k_bins": self.regression_top_k_bins,
        }
        if self.bin_thresholds_ is not None:
            info["thresholds"] = self.bin_thresholds_.tolist()
        return info


def preprocess_otu_sample(
    input_ids: np.ndarray,
    abundances: np.ndarray,
    abundance_preprocessor: AbundancePreprocessor,
    bin_manager: BinManager,
    otu_id_to_idx: Dict[str, int],
    cls_embedding: np.ndarray,
    ntv3_embeddings: Union[np.ndarray, 'h5py.Dataset']
) -> Dict[str, torch.Tensor]:
    """
    Preprocess a single OTU sample for MGM2.0 training.

    Args:
        input_ids: NTv3 index array (-1 for CLS token)
        abundances: Raw abundance values
        abundance_preprocessor: Preprocessor for abundances
        bin_manager: Bin manager for binning
        otu_id_to_idx: Mapping from OTU ID to index (not used when input_ids already provided)
        cls_embedding: CLS token embedding
        ntv3_embeddings: NTv3 embeddings (numpy array or h5py dataset)

    Returns:
        Dictionary containing all necessary tensors for training
    """
    # 1. Preprocess abundances
    processed_abundance = abundance_preprocessor.transform(abundances)

    # 2. Create bins and labels
    bin_labels, high_abundance_mask = bin_manager.transform(processed_abundance)

    # 3. Get regression targets (original preprocessed abundance for high-abundance OTUs)
    regression_targets = processed_abundance.copy()
    # Only keep targets for high abundance OTUs, mask others
    regression_mask = high_abundance_mask
    regression_targets[~regression_mask] = 0.0

    # 4. Build input embeddings from input_ids (already converted to NTv3 indices)
    embeddings = []

    # input_ids already contains the list of OTU indices (with -1 for CLS if prepended)
    # Build embeddings by looking up each OTU index
    embeddings = []
    for otu_idx in input_ids:
        if otu_idx == -1:
            # CLS token
            embeddings.append(cls_embedding)
        else:
            # Regular OTU - get embedding
            # Handle h5py dataset or numpy array
            if hasattr(ntv3_embeddings, 'shape') and len(ntv3_embeddings.shape) == 2:
                embedding = ntv3_embeddings[otu_idx]
            else:
                # h5py dataset
                embedding = ntv3_embeddings[otu_idx]
            embeddings.append(embedding)

    if len(embeddings) == 0:
        # No valid OTUs found - create empty sample with only CLS token
        embeddings = [cls_embedding]
        input_ids = np.array([-1], dtype=np.int64)  # CLS token
        attention_mask = np.ones(1, dtype=np.int64)

        # Create empty labels
        bin_labels = np.array([0], dtype=np.int64)
        regression_targets = np.array([0.0], dtype=np.float32)
        regression_mask = np.array([False], dtype=bool)
        processed_abundance_reduced = np.array([0.0], dtype=np.float32)
    else:
        # Check if CLS is already at the beginning (input_ids starts with -1)
        if input_ids[0] != -1:
            # CLS not already present - add it
            embeddings = [cls_embedding] + embeddings
            input_ids = np.array([-1] + list(input_ids), dtype=np.int64)
            bin_labels = np.concatenate([[0], bin_labels])
            regression_targets = np.concatenate([[0.0], regression_targets])
            regression_mask = np.concatenate([[False], regression_mask])
            processed_abundance_reduced = np.concatenate([[0.0], processed_abundance])
        else:
            # CLS already present - keep as is
            processed_abundance_reduced = processed_abundance

        attention_mask = np.ones(len(embeddings), dtype=np.int64)

    embeddings = np.array(embeddings, dtype=np.float32)

    # Convert to tensors
    return {
        'input_ids': torch.from_numpy(input_ids),
        'embeddings': torch.from_numpy(embeddings),
        'attention_mask': torch.from_numpy(attention_mask),
        'abundance_values': torch.from_numpy(processed_abundance_reduced),  # Preprocessed abundance
        'bin_labels': torch.from_numpy(bin_labels),  # Classification target (0 = absence)
        'regression_targets': torch.from_numpy(regression_targets),  # Continuous targets
        'regression_mask': torch.from_numpy(regression_mask),  # Mask for regression loss
    }
