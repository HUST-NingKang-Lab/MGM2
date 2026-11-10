"""Compatibility imports for the released training implementation."""

from .models.config import CONFIG, VARIANTS
from .models.core import CheckpointEncoder, CLSDecoder, PilotModel, build_backbone
from .training.objectives import (
    alignment_loss,
    corrupt_abundance,
    objective,
    reconstruction_sums,
    summarize_statistics,
)

__all__ = [
    "CONFIG",
    "VARIANTS",
    "CheckpointEncoder",
    "build_backbone",
    "CLSDecoder",
    "PilotModel",
    "corrupt_abundance",
    "alignment_loss",
    "reconstruction_sums",
    "objective",
    "summarize_statistics",
]
