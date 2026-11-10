"""Compatibility imports; implementation lives in :mod:`mgm2.models.heads`."""

from .models.heads import AbundanceBinHead, Qwen3AlignmentHead, RegressionHead

__all__ = ["AbundanceBinHead", "RegressionHead", "Qwen3AlignmentHead"]
