#!/usr/bin/env python3
"""Build deterministic evidence packages from feature lookup tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.sae_interpretation.config import cfg, load_config
from src.sae_interpretation.evidence import build_evidence_package
from src.sae_interpretation.schemas import validate_evidence_package


def _parse_feature_ids(value: str | None):
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sae_interpretation/otu_xlarge_k64_aux0p03.json"))
    parser.add_argument("--feature-ids", type=str, default=None)
    parser.add_argument("--lookup-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--background-json", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    root = Path(cfg(config, "output_dir"))
    lookup_dir = args.lookup_dir or root / "feature_lookup"
    output_dir = args.output_dir or root / "evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    background_path = args.background_json or Path(cfg(config, "background_metadata", root / "background_metadata.json"))
    background = None
    if background_path.exists():
        background = json.loads(background_path.read_text())
    else:
        print(f"Warning: background metadata not found: {background_path}", file=sys.stderr)
    feature_ids = _parse_feature_ids(args.feature_ids)
    if feature_ids is None:
        feature_ids = sorted(int(path.name.split("_")[1].split(".")[0]) for path in lookup_dir.glob("feature_*.events.csv.gz"))
    summary = []
    for feature_id in feature_ids:
        events_path = lookup_dir / f"feature_{feature_id:06d}.events.csv.gz"
        summary_path = lookup_dir / f"feature_{feature_id:06d}.lookup_summary.json"
        lookup_stats = None
        lookup_activation_mode = cfg(config, "activation_mode", "dense")
        lookup_activation_source = cfg(config, "activation_source", "")
        if summary_path.exists():
            lookup_summary = json.loads(summary_path.read_text())
            lookup_stats = lookup_summary.get("stats")
            lookup_activation_mode = lookup_summary.get("activation_mode", lookup_activation_mode)
            lookup_activation_source = lookup_summary.get("activation_source", lookup_activation_source)
        events = pd.read_csv(events_path)
        evidence = build_evidence_package(
            feature_id=feature_id,
            events=events,
            max_examples=args.max_examples or int(cfg(config, "max_examples_per_feature", 100)),
            lookup_stats=lookup_stats,
            background=background,
        )
        evidence["global_stats"]["activation_mode"] = lookup_activation_mode
        evidence["global_stats"]["activation_source"] = lookup_activation_source
        validate_evidence_package(evidence)
        out = output_dir / f"feature_{feature_id:06d}.evidence.json"
        out.write_text(json.dumps(evidence, indent=2, sort_keys=True))
        summary.append({"feature_id": feature_id, "evidence_path": str(out)})
    (output_dir / "evidence_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
