"""
MGM2 - Dual-Task Microbiome Pre-training Framework

Current MGM2 package for masked abundance-bin prediction and masked log-abundance regression.

All code lives in `src/` and follows the strict dual-task configuration defined in `src.configs`.
"""

from .configs.Config import MGM2Config
from .models.Model import MGM2Model
from .training.TrainerModule import MGM2LightningModule
from .losses.MultiTaskLoss import MGM2MultiTaskLoss

__version__ = "2.0.0"
__author__ = "MGM2 Development Team"

__all__ = [
    'MGM2Config',
    'MGM2Model',
    'MGM2LightningModule',
    'MGM2MultiTaskLoss',
]
