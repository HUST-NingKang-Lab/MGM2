#!/usr/bin/env python3
"""Validate Codex-authored JSON outputs for SAE feature interpretation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.sae_interpretation.config import cfg, load_config
from src.sae_interpretation.schemas import validate_interpretation_output_dir


def _parse_feature_ids(value: str | None):
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sae_interpretation/otu_xlarge_k64_aux0p03.json"))
    parser.add_argument("--outputs-dir", type=Path, default=None)
    parser.add_argument("--feature-ids", type=str, default=None, help="Comma-separated feature IDs to validate.")
    parser.add_argument("--strict-quality", action="store_true", help="Fail outputs that pass basic schema but look too shallow.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    outputs_dir = args.outputs_dir or Path(cfg(config, "output_dir")) / "codex_outputs"
    results = []
    feature_ids = _parse_feature_ids(args.feature_ids)
    if feature_ids is None:
        feature_dirs = sorted(outputs_dir.glob("feature_*"))
    else:
        feature_dirs = [outputs_dir / f"feature_{feature_id:06d}" for feature_id in feature_ids]
    for feature_dir in feature_dirs:
        if not feature_dir.is_dir():
            results.append(
                {
                    "feature_dir": str(feature_dir),
                    "valid": False,
                    "quality_pass": False,
                    "missing_files": ["feature_dir"],
                    "invalid_json": [],
                    "schema_errors": [],
                    "quality_warnings": [],
                }
            )
            continue
        results.append(validate_interpretation_output_dir(feature_dir, strict_quality=args.strict_quality))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
