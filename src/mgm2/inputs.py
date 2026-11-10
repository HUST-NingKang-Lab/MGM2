"""Per-community preprocessing; no global abundance statistics."""
from pathlib import Path
import h5py
import numpy as np
import torch
from ._taxonomy import TaxonomyLookup


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
                vectors.append(self.fallback[key]); sources.append(f"fallback:{rank}")
            elif self.global_vector is not None:
                vectors.append(self.global_vector); sources.append("fallback:global")
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


def prepare_sample(feature_ids, values, bank, top_k=768):
    """Normalize a complete count/relative-abundance vector, then select top OTUs.

    The top-K selection matches the research builder, including NumPy tie
    handling. Positive values are floored to 1e-4 after selection; neither the
    retained subset nor the floored abundances are renormalized. Zero features
    are omitted. Equal transformed abundances remain in the same lower bin.
    """
    ids = np.asarray(feature_ids, dtype=str)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or ids.shape != values.shape or len(set(ids)) != len(ids):
        raise ValueError("Expected equal-length vectors with unique feature identifiers")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Abundances must be finite and non-negative")
    if top_k < 1 or top_k > 768:
        raise ValueError("top_k must be between 1 and the trained limit of 768")
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("A community must contain a positive finite total abundance")
    keep = values > 0
    ids, relative = ids[keep], (values[keep] / total).astype(np.float32)
    if len(relative) > top_k:
        ix = np.argpartition(relative, -top_k)[-top_k:]
        ix = ix[np.argsort(relative[ix])[::-1]]
        ids, relative = ids[ix], relative[ix]
    vectors, sources = bank.resolve(ids)
    transformed = np.clip(np.log10(np.maximum(relative, np.float32(1e-4))) + 2, -2, 2)
    thresholds = np.quantile(transformed, [.2, .4, .6, .8]).astype(np.float32)
    bins = np.searchsorted(thresholds, transformed, side="left")
    return dict(feature_ids=ids.tolist(), resolution=sources, embeddings=vectors,
                abundance_values=transformed.astype(np.float32), bins=bins.astype(np.int64))


def collate_samples(samples, device="cpu"):
    """Prepend CLS and right-pad, without putting padding into bin 0."""
    if not samples:
        raise ValueError("An empty batch is not supported")
    lengths = [len(s["abundance_values"]) for s in samples]
    if min(lengths) < 1:
        raise ValueError("Empty communities are not supported")
    n, width, dim = len(samples), max(lengths) + 1, samples[0]["embeddings"].shape[1]
    ids = torch.full((n, width), -1, dtype=torch.long)
    mask = torch.zeros((n, width), dtype=torch.long)
    abundance = torch.zeros((n, width), dtype=torch.float32)
    vectors = torch.zeros((n, width, dim), dtype=torch.float32)
    bins = torch.full((n, width), -100, dtype=torch.long)
    for i, (sample, length) in enumerate(zip(samples, lengths)):
        ids[i, 1:length+1] = torch.arange(length)
        mask[i, :length+1] = 1
        abundance[i, 1:length+1] = torch.as_tensor(sample["abundance_values"])
        vectors[i, 1:length+1] = torch.as_tensor(sample["embeddings"])
        bins[i, 1:length+1] = torch.as_tensor(sample["bins"])
    return {k: v.to(device) for k, v in dict(input_ids=ids, embeddings=vectors,
            abundance_values=abundance, attention_mask=mask, bins=bins).items()}
