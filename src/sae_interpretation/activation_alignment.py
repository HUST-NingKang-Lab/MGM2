"""Helpers for aligning token manifests to activation HDF5 rows."""

from __future__ import annotations

import json
from pathlib import Path


def infer_activation_token_type(activations_path: Path, activation_token_type: str = "auto") -> str:
    """Infer whether activation rows are aligned to all tokens or OTU-only rows."""

    if activation_token_type != "auto":
        if activation_token_type not in {"all", "otu"}:
            raise ValueError(f"activation_token_type must be 'auto', 'all', or 'otu', got {activation_token_type!r}")
        return activation_token_type
    sidecar = Path(str(activations_path) + ".json")
    if not sidecar.exists():
        return "all"
    data = json.loads(sidecar.read_text())
    token_type = str(data.get("token_type", "all"))
    if token_type not in {"all", "otu"}:
        return "all"
    return token_type
