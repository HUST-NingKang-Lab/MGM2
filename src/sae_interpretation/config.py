"""Configuration helpers for SAE interpretation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open() as handle:
        return json.load(handle)


def cfg(config: Dict[str, Any], key: str, default=None):
    return config.get(key, default)

