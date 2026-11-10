"""Frozen layer-10 OTU dictionary; use contextual XLarge OTUs, not CLS."""

import json
from pathlib import Path

from safetensors.torch import load_file

from .dictionary import BatchTopKSAE


def load_sae(directory, device="cpu"):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    if config.get("format_version") != "mgm2-sae-v1":
        raise ValueError("Unsupported SAE export")
    model = BatchTopKSAE(config["dimension"], config["width"], config["k"])
    model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
    model.config = config
    return model.to(device).eval()
