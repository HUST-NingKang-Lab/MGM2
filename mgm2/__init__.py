"""Microbial General Model 2: community representations and sparse features."""

from .data import EmbeddingBank, collate_samples, prepare_sample
from .interpretability import load_sae
from .models import MGM2

__version__ = "2.0.1"
__all__ = ["MGM2", "EmbeddingBank", "prepare_sample", "collate_samples", "load_sae"]
