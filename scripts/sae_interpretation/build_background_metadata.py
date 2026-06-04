#!/usr/bin/env python3
"""Build background metadata distributions for SAE feature evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.sae_interpretation.background import build_token_background
from src.sae_interpretation.config import cfg, load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sae_interpretation/otu_xlarge_k64_aux0p03.json"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=500000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    root = Path(cfg(config, "output_dir"))
    output_json = args.output_json or Path(cfg(config, "background_metadata", root / "background_metadata.json"))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if output_json.exists() and not args.overwrite:
        print(json.dumps({"background_metadata": str(output_json), "skipped_existing": True}, indent=2))
        return
    background = build_token_background(
        tokens_path=Path(cfg(config, "tokens")),
        samples_path=Path(cfg(config, "samples")),
        splits=set(cfg(config, "description_splits", ["train", "val"])),
        chunk_size=args.chunk_size,
    )
    output_json.write_text(json.dumps(background, indent=2, sort_keys=True))
    print(json.dumps({"background_metadata": str(output_json), "total_tokens": background["total_tokens"]}, indent=2))


if __name__ == "__main__":
    main()
