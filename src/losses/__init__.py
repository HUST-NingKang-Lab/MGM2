"""
MGM2.0 Losses Module

Multi-task loss functions for hierarchical microbiome pre-training.
"""

from .MultiTaskLoss import MGM2MultiTaskLoss, FocalLoss

__all__ = [
    'MGM2MultiTaskLoss',
    'FocalLoss',
]
