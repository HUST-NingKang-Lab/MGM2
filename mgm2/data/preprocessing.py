"""Per-community normalization, token selection and abundance binning."""

import numpy as np


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
    thresholds = np.quantile(transformed, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    bins = np.searchsorted(thresholds, transformed, side="left")
    return dict(
        feature_ids=ids.tolist(),
        resolution=sources,
        embeddings=vectors,
        abundance_values=transformed.astype(np.float32),
        bins=bins.astype(np.int64),
    )
