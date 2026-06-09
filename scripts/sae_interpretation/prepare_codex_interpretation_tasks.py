#!/usr/bin/env python3
"""Render Codex task packets from evidence packages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.sae_interpretation.config import cfg, load_config
from src.sae_interpretation.prompt_rendering import render_codex_task_files
from src.sae_interpretation.schemas import validate_evidence_package


def _parse_feature_ids(value: str | None):
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_feature_ids_file(path: Path | None):
    if path is None:
        return None
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("feature_ids", data.get("selected_feature_ids", []))
    return [int(feature_id) for feature_id in data]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sae_interpretation/otu_xlarge_k64_aux0p03.json"))
    parser.add_argument("--feature-ids", type=str, default=None)
    parser.add_argument("--feature-ids-file", type=Path, default=None)
    parser.add_argument("--evidence-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    root = Path(cfg(config, "output_dir"))
    evidence_dir = args.evidence_dir or root / "evidence"
    output_dir = args.output_dir or root / "codex_tasks"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_ids = _parse_feature_ids(args.feature_ids)
    if args.feature_ids_file:
        if feature_ids is not None:
            raise SystemExit("Provide only one of --feature-ids or --feature-ids-file.")
        feature_ids = _parse_feature_ids_file(args.feature_ids_file)
    if feature_ids is None:
        feature_ids = sorted(int(path.name.split("_")[1].split(".")[0]) for path in evidence_dir.glob("feature_*.evidence.json"))
    summary = []
    for feature_id in feature_ids:
        evidence = json.loads((evidence_dir / f"feature_{feature_id:06d}.evidence.json").read_text())
        validate_evidence_package(evidence)
        feature_dir = render_codex_task_files(evidence, output_dir)
        summary.append({"feature_id": feature_id, "task_dir": str(feature_dir)})
    (output_dir / "codex_task_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
