"""Microbial General Model 2: community representations and sparse features."""
from .model import MGM2
from .inputs import EmbeddingBank, prepare_sample, collate_samples
from .sae import load_sae

__version__ = "2.0.0"
__all__ = ["MGM2", "EmbeddingBank", "prepare_sample", "collate_samples", "load_sae"]
