#!/usr/bin/env python3
"""Summarize Codex-authored feature interpretation JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.sae_interpretation.config import cfg, load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sae_interpretation/otu_xlarge_k64_aux0p03.json"))
    parser.add_argument("--outputs-dir", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    root = Path(cfg(config, "output_dir"))
    outputs_dir = args.outputs_dir or root / "codex_outputs"
    output_csv = args.output_csv or root / "reports" / "feature_interpretation_summary.csv"
    output_json = args.output_json or root / "reports" / "feature_interpretation_summary.json"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for feature_dir in sorted(outputs_dir.glob("feature_*")):
        feature_id = int(feature_dir.name.split("_")[1])
        unified_path = feature_dir / "unified_description.json"
        classification_path = feature_dir / "classification.json"
        final_validation_path = feature_dir / "final_validation.json"
        if not unified_path.exists():
            continue
        unified = json.loads(unified_path.read_text())
        classification = json.loads(classification_path.read_text()) if classification_path.exists() else {}
        final_validation = json.loads(final_validation_path.read_text()) if final_validation_path.exists() else {}
        classes = classification.get("classes", classification.get("classification", []))
        if not classes:
            primary_class = classification.get("primary_class", "")
            secondary_classes = classification.get("secondary_classes", [])
            classes = [primary_class] + list(secondary_classes) if primary_class else list(secondary_classes)
        rows.append(
            {
                "feature_id": feature_id,
                "short_name": unified.get("short_name", ""),
                "summary": unified.get("summary", ""),
                "confidence": unified.get("confidence", ""),
                "classes": ";".join(classes) if isinstance(classes, list) else classes,
                "accepted": classification.get(
                    "accepted",
                    final_validation.get(
                        "acceptance",
                        final_validation.get("accepted", ""),
                    ),
                ),
            }
        )
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(json.dumps({"rows": len(rows), "csv": str(output_csv), "json": str(output_json)}, indent=2))


if __name__ == "__main__":
    main()
