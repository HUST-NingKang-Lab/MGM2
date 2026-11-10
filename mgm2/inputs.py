"""Compatibility imports; implementation lives in :mod:`mgm2.data`."""

from .data import EmbeddingBank, collate_samples, prepare_sample

__all__ = ["EmbeddingBank", "prepare_sample", "collate_samples"]
