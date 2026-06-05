"""Feature-specific activation normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_normalized_activation_columns(events: pd.DataFrame) -> pd.DataFrame:
    """Add fmax, inverse-frequency weights, and combined normalized activation."""
    out = events.copy()
    if "activation" not in out:
        raise KeyError("events must contain an 'activation' column")
    max_activation = float(out["activation"].max()) if len(out) else 0.0
    denom = max(max_activation, 1e-12)
    out["fmax"] = out["activation"].astype(float) / denom

    n_tokens = max(1, len(out))
    token_counts = out["otu_id"].map(out["otu_id"].value_counts()).astype(float) if "otu_id" in out else 1.0
    sample_counts = (
        out["sample_id"].map(out["sample_id"].value_counts()).astype(float) if "sample_id" in out else 1.0
    )
    out["inverse_token_frequency"] = np.log1p(n_tokens / token_counts)
    out["inverse_sample_frequency"] = np.log1p(n_tokens / sample_counts)
    out["normalized_activation"] = (
        out["fmax"]
        * np.sqrt(out["inverse_token_frequency"].astype(float))
        * np.sqrt(out["inverse_sample_frequency"].astype(float))
    )
    return out

