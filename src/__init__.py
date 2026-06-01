"""
MGM2 - Dual-Task Microbiome Pre-training Framework

Current MGM2 package for masked abundance-bin prediction and masked log-abundance regression.

All code lives in `src/` and follows the strict dual-task configuration defined in `src.configs`.
"""

__version__ = "2.0.0"
__author__ = "MGM2 Development Team"

__all__ = [
    'MGM2Config',
    'MGM2Model',
    'MGM2LightningModule',
    'MGM2MultiTaskLoss',
]


def __getattr__(name):
    """Lazily expose top-level classes without importing training dependencies."""
    if name == 'MGM2Config':
        from .configs.Config import MGM2Config
        return MGM2Config
    if name == 'MGM2Model':
        from .models.Model import MGM2Model
        return MGM2Model
    if name == 'MGM2LightningModule':
        from .training.TrainerModule import MGM2LightningModule
        return MGM2LightningModule
    if name == 'MGM2MultiTaskLoss':
        from .losses.MultiTaskLoss import MGM2MultiTaskLoss
        return MGM2MultiTaskLoss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
