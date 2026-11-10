"""On-demand HDF5 embedding access and taxonomy resolution."""

from pathlib import Path

import h5py
import numpy as np

from .taxonomy import TaxonomyLookup


class EmbeddingBank(TaxonomyLookup):
    """Read only requested HDF5 rows, keeping the full OTU matrix off RAM/GPU.

    Taxonomy fallback is enabled only by supplying its HDF5 file. Every resolved
    feature has a provenance label; unknown IDs without fallback raise an error.
    """

    def __init__(self, path, fallback_path=None):
        self.path = Path(path)
        self.handle = h5py.File(self.path, "r")
        try:
            self.ids = self._decode_h5_strings(self.handle["otu_ids"][:])
            self.index = {name: i for i, name in enumerate(self.ids)}
            if len(self.index) != len(self.ids):
                raise ValueError("Duplicate embedding identifiers")
            self.dimension = self.handle["embeddings"].shape[1]
            self.fallback_embedding_path = fallback_path
            self.fallback, self.global_vector, self.aliases = self._load_fallback_embeddings()
        except Exception:
            self.handle.close()
            raise

    def resolve(self, ids):
        vectors, sources = [], []
        for feature in map(str, ids):
            if feature in self.index:
                vectors.append(self.handle["embeddings"][self.index[feature]])
                sources.append("ntv3")
                continue
            match = self._resolve_taxonomy_match(feature, self.fallback, self.aliases)
            if match is not None:
                rank, key = match
                vectors.append(self.fallback[key])
                sources.append(f"fallback:{rank}")
            elif self.global_vector is not None:
                vectors.append(self.global_vector)
                sources.append("fallback:global")
            else:
                raise KeyError(f"No sequence or taxonomy embedding for {feature!r}")
        values = np.asarray(vectors, dtype=np.float32)
        if values.shape != (len(ids), self.dimension) or not np.isfinite(values).all():
            raise ValueError("Invalid embedding vectors")
        return values, sources

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
