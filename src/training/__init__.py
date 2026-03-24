"""
MGM2.0 Training Module

Contains PyTorch Lightning training logic and scripts.
"""

from .TrainerModule import MGM2LightningModule
from .Train import main

__all__ = [
    'MGM2LightningModule',
    'main'
]
