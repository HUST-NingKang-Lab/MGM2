#!/usr/bin/env python3
"""Score all SAE features for interpretation priority selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.sae_interpretation.config import cfg, load_config
from src.sae_interpretation.feature_scoring import score_sae_features
from src.sae_interpretation.sae_features import load_sae_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sae_interpretation/otu_xlarge_k64_aux0p03.json"))
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--feature-batch-size", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    root = Path(cfg(config, "output_dir"))
    output_csv = args.output_csv or Path(cfg(config, "feature_scores", root / "feature_scores.csv"))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists() and not args.overwrite and args.max_chunks is None:
        print(json.dumps({"feature_scores": str(output_csv), "skipped_existing": True}, indent=2))
        return

    device = args.device or cfg(config, "device", "cpu")
    num_threads = args.num_threads if args.num_threads is not None else cfg(config, "num_threads", None)
    if num_threads is not None and int(num_threads) > 0:
        torch.set_num_threads(int(num_threads))
        print(f"Using torch CPU threads: {torch.get_num_threads()}")
    model = load_sae_from_checkpoint(Path(cfg(config, "sae_checkpoint")), device=device)
    scores = score_sae_features(
        activations_path=Path(cfg(config, "activations")),
        tokens_path=Path(cfg(config, "tokens")),
        model=model,
        splits=set(cfg(config, "description_splits", ["train", "val"])),
        chunk_size=args.chunk_size or int(cfg(config, "chunk_size", 262144)),
        feature_batch_size=args.feature_batch_size or int(cfg(config, "feature_score_batch_size", 512)),
        device=device,
        activation_mode=cfg(config, "activation_mode", "dense"),
        activation_token_type=cfg(config, "activation_token_type", "auto"),
        max_chunks=args.max_chunks,
    )
    scores.to_csv(output_csv, index=False)
    summary = {
        "feature_scores": str(output_csv),
        "features": int(len(scores)),
        "selected_token_count": int(scores["selected_token_count"].iloc[0]) if len(scores) else 0,
        "metric_columns": [col for col in scores.columns if col not in {"feature_id"}],
    }
    (output_csv.parent / "feature_scores.summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
